from typing import Callable, Dict
import os
import shutil
import logging
import logging.config

import cloudpickle as pkl
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.singleton import Singleton
from hydra.core.hydra_config import HydraConfig
import torch
import torch.nn as nn
import torch.utils.data 
import torch.cuda.amp as amp
import numpy as np

from simtrans.utils import iterable_to_device, set_seed, sort_by_index, save_checkpoint, write_lines_to_file
from simtrans.utils import gleu
from simtrans.data import build_dataset, build_dataloader
from simtrans.tokenize import get_sp_tokenizer, SPTokenizer

from simtrans.reinforce.optim import get_optimizer
from simtrans.reinforce.optim import get_lr_scheduler
from simtrans.reinforce.model import build_model
from simtrans.reinforce.reward import get_reward_fn


def evaluate(gpu: int,
             cfg: DictConfig,
             model: nn.Module,
             dataloader: torch.utils.data.DataLoader,
             tokenizer: SPTokenizer,
             reward_fn: Callable,
             device: torch.device) -> Dict:
    def calculate_gleu(out_ids, out_lens, ref_ids, ref_lens):
        # Expect tensors
        out_ids = out_ids.cpu().numpy()
        out_lens = out_lens.cpu().numpy()
        ref_ids = ref_ids.cpu().numpy()
        ref_lens = ref_lens.cpu().numpy()
        gleus = gleu(out_ids, out_lens, ref_ids, ref_lens)
        return gleus

    def calculate_src_tgt(src_ids, out_ids):
        return reward_fn(src_ids, out_ids)

    special_ids = {'bos': tokenizer['<s>'],
                   'eos': tokenizer['</s>'],
                   'unk': tokenizer['<unk>']}
    smoothing_eps = cfg.optim.label_smoothing
    lam = cfg.optim.lam
    use_ce = lam > 0.0
    use_rl = lam < 1.0
    use_bse = use_rl and cfg.model.baseline != 'self'
    score_from_tgt_and_output = cfg.model.reward == 'gleu'
    score_from_src_and_output = cfg.model.reward in ['xribes', 'kendall']
    if gpu == 0:
        output_ids = []
        all_scores = []
        idxs = []
        total_ce_loss = 0
        total_rl_loss = 0
        total_bse_loss = 0
        n_batches = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            src_ids, src_lens, tgt_ids, tgt_lens, _, global_idxs = iterable_to_device(batch, device=device)
            batch_size = src_ids.size(0)
            eval_result = model(mode='rl',
                                src_ids=src_ids,
                                src_lens=src_lens,
                                tgt_ids=tgt_ids,
                                tgt_lens=tgt_lens,
                                max_output_len=cfg.misc.max_output_length,
                                tok_ids=special_ids,
                                smoothing_eps=smoothing_eps,
                                lam=lam,
                                reward_fn=reward_fn,
                                reward_type=cfg.model.reward)
            if gpu == 0:
                if use_ce:
                    # Cross-entropy loss
                    ce_loss = eval_result['ce_loss']
                    ce_loss /= (tgt_lens.sum() - batch_size)
                    total_ce_loss += ce_loss.item()

                if use_rl:
                    # RL loss
                    sample_size = eval_result['output_len'].sum()
                    rl_loss = eval_result['rl_loss']
                    rl_loss /= sample_size
                    total_rl_loss += rl_loss.item()
                    if use_bse:
                        # Baseline loss
                        bse_loss = eval_result['bse_loss']
                        bse_loss /= sample_size
                        total_bse_loss += bse_loss.item()
            if not use_rl:
                output = model(mode='infer',
                               src_ids=src_ids,
                               src_lens=src_lens,
                               max_output_len=cfg.misc.max_output_length,
                               tok_ids=special_ids,
                               decode='greedy')
                if gpu == 0:
                    output_ids.extend(output['output'].tolist())
                    if score_from_tgt_and_output:
                        scores = calculate_gleu(output['output'],
                                                 output['output_len'],
                                                 tgt_ids[:, 1:],
                                                 tgt_lens - 1)
                        all_scores.extend(scores.tolist())
                    elif score_from_src_and_output:
                        scores = calculate_src_tgt(src_ids, output['output'])
                        all_scores.extend(scores.tolist())
                    else:
                        all_scores.extend([0] * batch_size)
            elif gpu == 0:
                output_ids.extend(eval_result['output_ids'].tolist())
                if score_from_tgt_and_output:
                    scores = calculate_gleu(eval_result['output_ids'],
                                             eval_result['output_len'],
                                             tgt_ids[:, 1:],
                                             tgt_lens - 1)
                    all_scores.extend(scores.tolist())
                elif score_from_src_and_output:
                    scores = calculate_src_tgt(src_ids, eval_result['output_ids'])
                    all_scores.extend(scores.tolist())
                else:
                    all_scores.extend([0] * batch_size)
            if gpu == 0:
                idxs.extend(global_idxs)
                n_batches += 1

    result = {}
    if gpu == 0:
        id_tuples = zip(output_ids, all_scores)
        id_tuples = sort_by_index(id_tuples, idxs)
        output_ids, all_scores = zip(*id_tuples)
        if cfg.model.reward in ['gleu', 'xribes', 'kendall']:
            result['score'] = all_scores
        else:
            result['score'] = None
        result['output'] = output_ids
        result['ce_loss'] = total_ce_loss / n_batches
        result['rl_loss'] = total_rl_loss / n_batches
        result['bse_loss'] = total_bse_loss / n_batches
    return result


def train(gpu: int, cfg: DictConfig, n_gpu: int, singleton_state) -> None:
    assert not cfg.batch.distributed
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
    else:
        logger = None

    tokenizer = get_sp_tokenizer(cfg)

    if logger is not None:
        logger.info("Done")
        logger.info("Building dataset...")
    dataset = build_dataset(cfg, tokenizer)

    if logger is not None:
        logger.info("Done")
        logger.info("Building dataloader...")
    dataloader = build_dataloader(cfg, dataset)

    if logger is not None:
        logger.info("Done")
        logger.info("Building model...")
    model = build_model(cfg)
    device = torch.device('cuda', gpu)
    pretrain_checkpoint = os.path.join(hydra.utils.get_original_cwd(), cfg.model.pretrain_path)
    pretrain_state = torch.load(pretrain_checkpoint, map_location='cpu')
    model_state = model.state_dict()
    model_state.update(pretrain_state['model'])
    model.load_state_dict(model_state)
    model.to(device)

    if logger is not None:
        logger.info("Done")
        logger.info("Initialize optimizer/scheduler...")
    opt_nmt, opt_bse = get_optimizer(cfg, model)
    scheduler_nmt, scheduler_bse = get_lr_scheduler(cfg, opt_nmt, opt_bse)
    scaler = amp.GradScaler()
    if logger is not None:
        logger.info("Done")


    # Load params
    max_iteration = cfg.misc.max_iterations
    validate_every = cfg.misc.validate_every
    acc_steps = cfg.optim.accumulation_steps
    smoothing_eps = cfg.optim.label_smoothing
    rl_lam = cfg.optim.lam
    rl_start = cfg.optim.rl_start
    decay_start = cfg.optim.decay_start if cfg.optim.schedule == 'reduce_on_plateau' else -1
    clip_grad = cfg.optim.clip_grad

    update = 0
    iteration = 0
    epoch = 0
    acc_ce_loss = 0
    acc_rl_loss = 0
    acc_bse_loss = 0
    total_losses = {'ce_loss': 0,
                    'rl_loss': 0,
                    'bse_loss': 0}
    update_count = 0
    best_eval_loss = float('inf')
    special_ids = {'bos': tokenizer['<s>'],
                   'eos': tokenizer['</s>'],
                   'unk': tokenizer['<unk>']}
    reward_fn = get_reward_fn(cfg, tokenizer, hydra.utils.get_original_cwd())

    if logger is not None:
        logger.info('Start Training')
    model.train()
    while True:
        if epoch < rl_start:
            lam = 1.0
        else:
            lam = rl_lam
        use_ce = lam > 0.0
        use_rl = lam < 1.0
        use_bse = use_rl and cfg.model.baseline != 'self'
        for batch_idx, batch in enumerate(dataloader['train']):
            src_ids, src_lens, tgt_ids, tgt_lens, _, _ = iterable_to_device(batch, device=device)
            batch_size = src_ids.size(0)

            with amp.autocast():
                result = model(mode='rl',
                               src_ids=src_ids,
                               src_lens=src_lens,
                               tgt_ids=tgt_ids,
                               tgt_lens=tgt_lens,
                               max_output_len=cfg.misc.max_output_length,
                               tok_ids=special_ids,
                               smoothing_eps=smoothing_eps,
                               lam=lam,
                               reward_type=cfg.model.reward,
                               reward_fn=reward_fn)
                if use_ce:
                    # Cross-entropy loss
                    ce_loss = result['ce_loss']
                    ce_loss /= (tgt_lens.sum() - batch_size) * acc_steps
                    acc_ce_loss += ce_loss.item()
                else:
                    ce_loss = 0.0

                if use_rl:
                    # RL loss
                    sample_size = result['output_len'].sum()
                    rl_loss = result['rl_loss']
                    rl_loss /= sample_size * acc_steps
                    acc_rl_loss += rl_loss.item()
                    # Baseline loss
                    if use_bse:
                        bse_loss = result['bse_loss']
                        bse_loss /= sample_size * acc_steps
                        acc_bse_loss += bse_loss.item()
                    else:
                        bse_loss = 0.0
                else:
                    rl_loss = 0.0
                    bse_loss = 0.0
                # Combine all losses
                loss = lam * ce_loss + (1.0 - lam) * rl_loss + bse_loss
            scaler.scale(loss).backward()
            iteration += 1

            if iteration % acc_steps == 0:
                if clip_grad > 0.0:
                    scaler.unscale_(opt_nmt)
                    nn.utils.clip_grad_norm_(model.non_bse_parameters(), clip_grad)
                scaler.step(opt_nmt)
                if use_rl and use_bse:
                    scaler.step(opt_bse)
                scaler.update()
                total_losses['ce_loss'] += acc_ce_loss
                total_losses['rl_loss'] += acc_rl_loss
                total_losses['bse_loss'] += acc_bse_loss
                acc_ce_loss = 0
                acc_rl_loss = 0
                acc_bse_loss = 0
                update += 1
                update_count += 1

                # Do evaluation
                if update % validate_every == 0:
                    model.eval()
                    eval_output = evaluate(gpu=gpu,
                                           cfg=cfg,
                                           model=model,
                                           dataloader=dataloader['dev'],
                                           tokenizer=tokenizer,
                                           reward_fn=reward_fn,
                                           device=device)
                    nmt_loss = lam * total_losses['ce_loss'] + (1.0 - lam) * total_losses['rl_loss']
                    if logger is not None:
                        logger.info(f'iteration={update}, avg_nmt_loss={nmt_loss / update_count:.5f}')
                        logger.info(f'avg_bse_loss={total_losses["bse_loss"] / update_count:.5f}')
                        if eval_output['score'] is not None:
                            score = sum(eval_output['score']) / len(eval_output['score'])
                            logger.info(f'avg_eval_score={score:.5f}')
                        # Spit current text outputs to file
                        eval_texts = [tokenizer.convert_ids_to_text(s) for s in eval_output['output']]
                        curr_output_path = os.path.join(os.getcwd(), 'output.curr.txt')
                        write_lines_to_file(path=curr_output_path, lines=eval_texts) 
                    # Examine evaluation loss
                    eval_nmt_loss = lam * eval_output['ce_loss'] + (1.0 - lam) * eval_output['rl_loss']
                    if logger is not None:
                        logger.info(f'eval_nmt_loss={eval_nmt_loss:.5f}')
                    if epoch > decay_start:
                        if cfg.optim.schedule == 'reduce_on_plateau':
                            scheduler_nmt.step(eval_nmt_loss)
                            if use_rl and use_bse:
                                scheduler_bse.step(eval_nmt_loss)
                        else:
                            scheduler_nmt.step(opt_nmt)
                            if use_rl and use_bse:
                                scheduler_bse.step(opt_bse)

                    if logger is not None:
                        if eval_nmt_loss < best_eval_loss:
                            logger.info(f'current_eval_loss={eval_nmt_loss:.5f} < best_eval_loss={best_eval_loss:.5f}')
                            best_eval_loss = eval_nmt_loss
                            logger.info(f'Updating best_eval_loss and saving checkpoint')
                            best_output_path = os.path.join(os.getcwd(), 'output.best.txt')
                            shutil.copyfile(curr_output_path, best_output_path)
                            model_state = model.state_dict()
                            save_state = {'model': model_state,
                                          'optimizer_nmt': opt_nmt.state_dict(),
                                          'optimizer_bse': opt_bse.state_dict() if opt_bse is not None else None,
                                          'scheduler_nmt': scheduler_nmt.state_dict(),
                                          'scheduler_bse': scheduler_bse.state_dict() if scheduler_bse is not None else None,
                                          'epoch': epoch,
                                          'batch': batch,
                                          'update': update}
                            save_checkpoint(params=save_state)
                    update_count = 0
                    total_losses['ce_loss'] = 0
                    total_losses['rl_loss'] = 0
                    total_losses['bse_loss'] = 0
                    model.train()

                opt_nmt.zero_grad()
                if use_rl and use_bse:
                    opt_bse.zero_grad()
        epoch += 1
        if update >= max_iteration:
            break
