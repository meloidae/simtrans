from typing import List, Dict, Union, Any, Callable
import os
import logging
import logging.config
import subprocess
import re
import shutil

import cloudpickle as pkl
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.singleton import Singleton
from hydra.core.utils import configure_log
from hydra.core.hydra_config import HydraConfig
import torch
import torch.cuda
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

import torch.cuda.amp as amp

from simtrans.model import build_model, DistributedDataParallelPassthrough
from simtrans.wait_k_transformer import WaitKTransformer
from simtrans.data import build_dataset, build_dataloader
from simtrans.optim import get_optimizer, get_lr_scheduler
from simtrans.utils import iterable_to_device, set_seed, sort_by_index, save_checkpoint, write_lines_to_file
from simtrans.tokenize import get_sp_tokenizer, SPTokenizer

from reinforce import train as train_rl

def get_bleu(cfg: DictConfig, data_type: str, candidate_path: str) -> Dict[str, Union[float, str]]:
    def parse_output(output: str) -> Dict[str, Union[float, str]]:
        split_idx = output.find('=')
        options = output[:split_idx].strip()
        values = output[split_idx:].strip()
        matches = re.findall('([0-9.]+)', values)
        keys = ['bleu', 'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'bp', 'ratio', 'hyp_len', 'ref_len']
        result = {}
        for k, v in zip(keys, matches):
            result[k] = v
        result['option'] = options
        return result

    language = cfg.eval.language
    sacrebleu_path = cfg.eval.sacrebleu_path
    reference_path = os.path.join(hydra.utils.get_original_cwd(), cfg['data'][data_type]['tgt'])
    full_command = f'cat {candidate_path} | {sacrebleu_path} -l {language} {reference_path}'
    proc = subprocess.run(full_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    text_output = proc.stdout.decode('UTF-8').strip()
    if proc.returncode == 0:
        bleu_result = parse_output(text_output)
    else:
        bleu_result = {}
    bleu_result['text'] = text_output
    return bleu_result


def infer(gpu: int,
          cfg: DictConfig,
          model: nn.Module,
          dataloader: torch.utils.data.DataLoader,
          tokenizer: SPTokenizer,
          device: torch.device) -> Dict:
    if gpu == 0:
        output_ids = []
        idxs = []
    special_ids = {'bos': tokenizer['<s>'],
                   'eos': tokenizer['</s>'],
                   'unk': tokenizer['<unk>']}
    model.eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            src_ids, src_lens, _, _, _, global_idxs = iterable_to_device(batch, device=device)
            output = model(mode='infer',
                           src_ids=src_ids,
                           src_lens=src_lens,
                           wait_k=cfg.model.wait_k,
                           max_output_len=cfg.misc.max_output_length,
                           tok_ids=special_ids,
                           decode=cfg.misc.decode)
            if gpu == 0:
                output_ids.extend(output['output'].tolist())
                idxs.extend(global_idxs)
    result = {}
    if gpu == 0:
        output_ids = sort_by_index(output_ids, idxs)
        result['output'] = output_ids
    return result


# TODO
def test(gpu: int,
         cfg: DictConfig):
    if gpu == 0:
        logger = logging.getLogger(__name__)
        logger.info("Loading vocab...")

    tokenizer = get_sp_tokenizer(cfg)

    if gpu == 0:
        logger.info("Done")
        logger.info("Building dataset...")
    dataset = build_dataset(cfg, tokenizer)

    if gpu == 0:
        logger.info("Done")
        logger.info("Building dataloader...")
    dataloader = build_dataloader(cfg, dataset)

    if gpu == 0:
        logger.info("Done")
        logger.info("Building model...")
    model = build_model(cfg)

    if gpu == 0:
        logger.info("Done")
        logger.info("Loading checkpoint...")
    checkpoint = os.path.join(hydra.utils.get_original_cwd(), cfg.misc.checkpoint)
    save_state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(save_state['model'])
    device = torch.device('cuda', gpu)
    model.to(device)

    output_path = os.path.join(os.getcwd(), "output.test.txt")


def train(gpu: int, cfg: DictConfig, n_gpu: int, singleton_state) -> None:
    if cfg.batch.distributed:
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=n_gpu,
                                rank=gpu)
    Singleton.set_state(pkl.loads(singleton_state))

    seed = cfg.misc.seed
    set_seed(seed)
    
    if gpu == 0:
        # Initialize logger, very hacky but works!!
        logging_cfg = HydraConfig.get().job_logging
        logging_cfg = OmegaConf.to_container(logging_cfg, resolve=False)
        logging_cfg['handlers']['file']['filename'] = f'process_{gpu}.log'
        logging.config.dictConfig(logging_cfg)
        logger = logging.getLogger(__name__)
        logger.info("Loading vocab...")

    tokenizer = get_sp_tokenizer(cfg)

    if gpu == 0:
        logger.info("Done")
        logger.info("Building dataset...")
    dataset = build_dataset(cfg, tokenizer)

    if gpu == 0:
        logger.info("Done")
        logger.info("Building dataloader...")
    dataloader = build_dataloader(cfg, dataset)

    if gpu == 0:
        logger.info("Done")
        logger.info("Building model...")
    model = build_model(cfg)
    device = torch.device('cuda', gpu)
    model.to(device)
    if cfg.batch.distributed:
        model = DistributedDataParallelPassthrough(model, device_ids=[gpu], output_device=gpu)

    if gpu == 0:
        logger.info("Done")
        logger.info("Initialize optimizer/scheduler...")
    optimizer = get_optimizer(cfg, model)
    scheduler = get_lr_scheduler(cfg)
    scheduler(optimizer)
    scaler = amp.GradScaler()
    if gpu == 0:
        logger.info("Done")

    # Load params
    max_iteration = cfg.misc.max_iterations
    validate_every = cfg.misc.validate_every
    acc_steps = cfg.optim.accumulation_steps
    smoothing_eps = cfg.optim.label_smoothing
    wait_k = cfg.model.wait_k

    update = 0
    iteration = 0
    epoch = 0
    acc_loss = 0
    total_loss = 0
    update_count = 0
    best_eval_score = -float('inf')
    special_ids = {'bos': tokenizer['<s>'],
                   'eos': tokenizer['</s>'],
                   'unk': tokenizer['<unk>']}

    if gpu == 0:
        logger.info('Start training')
    model.train()
    while True:
        for batch_idx, batch in enumerate(dataloader['train']):
            src_ids, src_lens, tgt_ids, tgt_lens, _, _ = iterable_to_device(batch, device=device)
            batch_size = src_ids.size(0)

            with amp.autocast():
                result = model(mode='train',
                               src_ids=src_ids,
                               src_lens=src_lens,
                               tgt_ids=tgt_ids,
                               tgt_lens=tgt_lens,
                               wait_k=wait_k,
                               tok_ids=special_ids,
                               smoothing_eps=smoothing_eps)
                loss = result['loss']
                loss /= (tgt_lens.sum() - batch_size) * acc_steps
                acc_loss += loss.item()
            scaler.scale(loss).backward()
            iteration += 1

            if iteration % acc_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler(optimizer)
                optimizer.zero_grad()
                total_loss += acc_loss
                acc_loss = 0
                update += 1
                update_count += 1

                # Do evaluation
                if update % validate_every == 0:
                    model.eval()
                    eval_output = infer(gpu=gpu,
                                        cfg=cfg,
                                        model=model,
                                        dataloader=dataloader['dev'],
                                        tokenizer=tokenizer,
                                        device=device)
                    if gpu == 0:
                        logger.info(f'iteration={update}, avg_train_loss={total_loss / update_count}')
                        # Calculate bleu scores
                        eval_texts = [tokenizer.convert_ids_to_text(s) for s in eval_output['output']]
                        curr_output_path = os.path.join(os.getcwd(), 'output.curr.txt')
                        write_lines_to_file(path=curr_output_path, lines=eval_texts) 
                        bleu_result = get_bleu(cfg, data_type='dev', candidate_path=curr_output_path)
                        logger.info(bleu_result['text'])
                        bleu_score = float(bleu_result['bleu'])
                        if bleu_score > best_eval_score:
                            logger.info(f'current_eval_score={bleu_score} > best_eval_score={best_eval_score}')
                            best_eval_score = bleu_score
                            logger.info(f'Updating best_eval_score={bleu_score} and saving checkpoint')
                            best_output_path = os.path.join(os.getcwd(), 'output.best.txt')
                            # Keep current output as best output
                            shutil.copyfile(curr_output_path, best_output_path)
                            if hasattr(model, 'module'):
                                model_state = model.module.state_dict()
                            else:
                                model_state = model.state_dict()
                            save_state = {'model': model_state,
                                          'optimizer': optimizer.state_dict(),
                                          'scheduler': scheduler.state_dict(),
                                          'epoch': epoch,
                                          'batch': batch,
                                          'update': update}
                            save_checkpoint(params=save_state)
                    update_count = 0
                    total_loss = 0
                    model.train()
        epoch += 1
        if update >= max_iteration:
            break


@hydra.main(config_path='cfg', config_name='config')
def main(cfg: DictConfig) -> None:
    if cfg.misc.do_train:
        if cfg.misc.mode == 'wait_k':
            os.environ['MASTER_ADDR'] = cfg.env.address
            os.environ['MASTER_PORT'] = cfg.env.port
            n_gpu = torch.cuda.device_count()
            singleton_state = Singleton.get_state()
            singleton_state = pkl.dumps(singleton_state)
            mp.spawn(train, nprocs=n_gpu, args=(cfg, n_gpu, singleton_state))
        elif cfg.misc.mode == 'reinforce':
            n_gpu = 1
            singleton_state = Singleton.get_state()
            singleton_state = pkl.dumps(singleton_state)
            mp.spawn(train_rl, nprocs=n_gpu, args=(cfg, n_gpu, singleton_state))

    if cfg.misc.do_test:
        test(gpu=0, cfg=cfg)


if __name__ == '__main__':
    main()
