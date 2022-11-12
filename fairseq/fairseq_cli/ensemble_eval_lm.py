#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""

import logging
import math
import pandas as pd
import os
import numpy as np
import sys
from argparse import Namespace
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm
import fairseq
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter
from fairseq.sequence_scorer import SequenceScorer
from omegaconf import DictConfig
from fairseq.trainer import Trainer
from fairseq.data import ResamplingDataset
from fairseq.models.moe_model import MoETransformer
from fairseq.modules.moe.top1gate import Top1Gate



logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.eval_lm")


def evaluate(models, sample, device, gen_timer, scorer, bpe_toks, output_word_probs, output_word_stats, remove_bos_token=False, ensemble_type="cached_prior", prior=None):
    score_sum = 0
    count = 0

    sample = utils.move_to_cuda(sample, device=device % 8)

    gen_timer.start()

    if isinstance(prior, list):
        prior = torch.tensor(prior).float()
    if ensemble_type == "cached_prior":
        if prior is None:
            raise ValueError("if using cached_prior, please supply a prior.")
            sys.exit(1)
        ensemble_weighted_average = True
        ensemble_average = None
    elif ensemble_type == "updating_prior":
        ensemble_weighted_average = True
        ensemble_average = None
    elif ensemble_type == "simple_average":
        ensemble_weighted_average = None
        ensemble_average = True
    else:
        raise ValueError("unrecognized ensembling type.")
        sys.exit(1)

    hypos = scorer.generate(models, sample, ensemble=True, ensemble_weighted_average=ensemble_weighted_average, ensemble_average=ensemble_average, prior=prior,all_reduce=torch.distributed.is_initialized())

    gen_timer.stop(sample["ntokens"])
    expert_probs = []
    for i, hypos_i in enumerate(hypos):
        hypo = hypos_i[0]
        sample_id = sample["id"][i]

        tokens = hypo["tokens"]
        tgt_len = tokens.numel()
        pos_scores = hypo["positional_scores"].float()
        if remove_bos_token:
            tokens = tokens[1:]
            pos_scores = pos_scores[1:]

        skipped_toks = 0
        if bpe_toks is not None:
            for i in range(tgt_len - 1):
                if tokens[i].item() in bpe_toks:
                    skipped_toks += 1
                    pos_scores[i + 1] += pos_scores[i]
                    pos_scores[i] = 0

        inf_scores = pos_scores.eq(float("inf")) | pos_scores.eq(float("-inf"))
        if inf_scores.any():
            logger.info(
                "skipping tokens with inf scores:",
                target_dictionary.string(tokens[inf_scores.nonzero()]),
            )
            pos_scores = pos_scores[(~inf_scores).nonzero()]
        score_sum += pos_scores.sum().cpu()
        count += pos_scores.numel() - skipped_toks
        if hypo['expert_probs'] is not None:
            expert_ps = hypo['expert_probs'].mean(1).unsqueeze(0).cpu().numpy()
            expert_probs.append(expert_ps)

        if output_word_probs or output_word_stats:
            w = ""
            word_prob = []
            is_bpe = False
            for i in range(len(tokens)):
                w_ind = tokens[i].item()
                w += source_dictionary[w_ind]
                if bpe_toks is not None and w_ind in bpe_toks:
                    w = w[:-bpe_len]
                    is_bpe = True
                else:
                    word_prob.append((w, pos_scores[i].item()))

                    next_prob = None
                    ind = i + 1
                    while ind < len(tokens):
                        if pos_scores[ind].item() != 0:
                            next_prob = pos_scores[ind]
                            break
                        ind += 1

                    word_stats.setdefault(w, WordStat(w, is_bpe)).add(
                        pos_scores[i].item(), next_prob
                    )
                    is_bpe = False
                    w = ""
            if output_word_probs:
                logger.info(
                    str(int(sample_id))
                    + " "
                    + (
                        "\t".join(
                            "{} [{:2f}]".format(x[0], x[1]) for x in word_prob
                        )
                    )
                )
    if expert_probs:
        expert_probs = np.concatenate(expert_probs, 0)
        expert_probs = np.expand_dims(expert_probs.mean(0), 0)
    else:
        expert_probs = None
    ppl = 2 ** (-score_sum / count / math.log(2) if count > 0 else 0)
    return score_sum, count, ppl, expert_probs


def dynamic_eval_lm(
    models: List[fairseq.models.FairseqModel],
    source_dictionary: fairseq.data.Dictionary,
    dev_batch_iterator: Iterable,
    trainer: Trainer=None,
    train_batch_iterator: Iterable=None,
    undomain_dataset = None,
    undomain_replay_dataset = None,
    post_process: Optional[str] = None,
    output_word_probs: bool = False,
    output_word_stats: bool = False,
    target_dictionary: Optional[fairseq.data.Dictionary] = None,
    softmax_batch: int = False,
    remove_bos_token: bool = False,
    device: Optional[torch.device] = None,
    use_expert: int = None,
    cfg=None,
    num_shards: int=None,
    shard_id: int=None,
    task=None,
    eval_only=False,
    max_documents = None,
    max_train_documents=None,
    precomputed_prior = None,
    ensemble_type: str = None
):
    """
    Args:
        models (List[~fairseq.models.FairseqModel]): list of models to
            evaluate. Models are essentially `nn.Module` instances, but
            must be compatible with fairseq's `SequenceScorer`.
        source_dictionary (~fairseq.data.Dictionary): dictionary for
            applying any relevant post processing or outputing word
            probs/stats.
        batch_iterator (Iterable): yield batches of data
        post_process (Optional[str]): post-process text by removing BPE,
            letter segmentation, etc. Valid options can be found in
            fairseq.data.utils.post_process, although not all options
            are implemented here.
        output_word_probs (Optional[bool]): output words and their
            predicted log probabilities
        output_word_stats (Optional[bool]): output word statistics such
            as word count and average probability
        target_dictionary (Optional[~fairseq.data.Dictionary]): output
            dictionary (defaults to *source_dictionary*)
        softmax_batch (Optional[bool]): if BxT is more than this, will
            batch the softmax over vocab to this amount of tokens, in
            order to fit into GPU memory
        remove_bos_token (Optional[bool]): if True, confirm that the
            first token is the beginning-of-sentence symbol (according
            to the relevant dictionary) and remove it from the output
        device (Optional[torch.device]): device to use for evaluation
            (defaults to device of first model parameter)
    """
    if target_dictionary is None:
        target_dictionary = source_dictionary
    if device is None:
        device = next(models[0].parameters()).device

    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(target_dictionary, softmax_batch)
    score_sum = 0.0
    count = 0
    if post_process is not None:
        if post_process in {"subword_nmt", "@@ "}:
            bpe_cont = post_process.rstrip()
            bpe_toks = {
                i
                for i in range(len(source_dictionary))
                if source_dictionary[i].endswith(bpe_cont)
            }
        else:
            raise NotImplementedError(
                "--post-process={post_process} is not implemented"
            )
        bpe_len = len(bpe_cont)
    else:
        bpe_toks = None
        bpe_len = 0

    word_stats = dict()
    running_perplexities = []
    steps = 0
    target_score_sum = 0
    target_count = 0

    target_ppls = []
    expert_probs_all = []
    priors_all = []
    if not eval_only:
        for ix, sample in enumerate(train_batch_iterator):
            if max_train_documents and ix > max_train_documents:
                break
            if not sample or "net_input" not in sample:
                continue
            if ix > 0:
                steps += 1
                trainer.train_step([sample])

    if precomputed_prior:
        prior = precomputed_prior
    else:
        prior = None

    if ensemble_type == 'updating_prior':
        precomputed_prior = None
        prior = None

    for ix, sample in enumerate(dev_batch_iterator):
        if max_documents and ix > max_documents:
            break
        if not sample or "net_input" not in sample:
            continue
        target_score_sum_, target_count_, _, expert_probs = evaluate(models, sample, device, gen_timer, scorer, bpe_toks, output_word_probs, output_word_stats, remove_bos_token=remove_bos_token, ensemble_type=ensemble_type, prior=prior)
        target_score_sum += target_score_sum_
        target_count += target_count_
        target_ppl = 2 ** (-target_score_sum / target_count / math.log(2) if target_count > 0 else 0)
        target_ppls.append(target_ppl.item())
        if expert_probs is not None:
            expert_probs_all.append(expert_probs)
            if not precomputed_prior:
                prior = pd.DataFrame(np.concatenate(expert_probs_all,0)).ewm(alpha=0.3, adjust=False).mean().tail(n=1).to_numpy().squeeze(0)
            priors_all.append(prior)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            dev_batch_iterator.tqdm.set_description(f"ppl: {target_ppl.item()}")
    return target_ppls, expert_probs_all, priors_all




class WordStat(object):
    def __init__(self, word, is_bpe):
        self.word = word
        self.is_bpe = is_bpe
        self.log_prob = 0
        self.next_word_prob = 0
        self.count = 0
        self.missing_next_words = 0

    def add(self, log_prob, next_word_prob):
        """increments counters for the sum of log probs of current word and next
        word (given context ending at current word). Since the next word might be at the end of the example,
        or it might be not counted because it is not an ending subword unit,
        also keeps track of how many of those we have seen"""
        if next_word_prob is not None:
            self.next_word_prob += next_word_prob
        else:
            self.missing_next_words += 1
        self.log_prob += log_prob
        self.count += 1

    def __str__(self):
        return "{}\t{}\t{}\t{}\t{}\t{}".format(
            self.word,
            self.count,
            self.log_prob,
            self.is_bpe,
            self.next_word_prob,
            self.count - self.missing_next_words,
        )


def main(cfg: DictConfig, **unused_kwargs):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    logger.info(cfg)

    if cfg.dynamic_eval_lm.context_window > 0:
        # reduce tokens per sample by the required context window size
        cfg.task.tokens_per_sample -= cfg.dynamic_eval_lm.context_window

    # Initialize the task using the current *cfg*
    task = tasks.setup_task(cfg.task)

    # Load precomputed prior
    if cfg.common_eval.precomputed_prior is not None:
        precomputed_prior = [float(x) for x in cfg.common_eval.precomputed_prior.split(',')]
    else:
        precomputed_prior = None

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    if cfg.common_eval.is_moe and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        cfg.checkpoint.checkpoint_suffix = f"-rank-{torch.distributed.get_rank()}"
        moe_freq = 1
    else:
        moe_freq = 0
    model_ = cfg.common_eval.path.split(':')
    if precomputed_prior and cfg.common_eval.eval_topk and cfg.common_eval.eval_topk < len(model_):
        prior_inds = sorted(sorted(range(len(precomputed_prior)), key=lambda i: precomputed_prior[i])[-cfg.common_eval.eval_topk:])
        precomputed_prior = [precomputed_prior[i] if i in prior_inds else 0 for i in range(len(precomputed_prior))]
        precomputed_prior = [p/sum(precomputed_prior) for p in precomputed_prior]
        # model_ = [model_[j] for j in prior_inds]
    model_ = [model_[torch.distributed.get_rank()]] if torch.distributed.is_initialized() else model_
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        model_,
        arg_overrides=eval(cfg.common_eval.model_overrides),
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
        task=task,
        moe_freq=moe_freq,
        desynchronize=cfg.common_eval.is_moe and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1,
        partial_load=cfg.common_eval.partial_load
    )

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    # Optimize ensemble for generation and set the source and dest dicts on the model
    # (required by scorer)
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()

    assert len(models) > 0

    logger.info(
        "num. model params: {:,}".format(sum(p.numel() for p in models[0].parameters()))
    )

    # Load dataset splits
    for subset in cfg.dataset.gen_subset.split(','):
        task.load_dataset(subset)

    eval_dataset = task.dataset(cfg.dataset.target_eval)

    logger.info(
        "{} {} {:,} examples".format(
            cfg.task.data, cfg.dataset.target_eval, len(eval_dataset)
        )
    )

    num_shards = max(
        cfg.dataset.num_shards,
        cfg.distributed_training.distributed_world_size,
    )
    shard_id = max(
        cfg.dataset.shard_id,
        cfg.distributed_training.distributed_rank,
    )

    if cfg.common_eval.is_moe:
        num_shards = 1
        shard_id = 0
    num_shards = 1
    shard_id = 0

    dev_itr = task.eval_lm_dataloader(
        dataset=eval_dataset,
        max_tokens=cfg.dataset.max_tokens or 36000,
        batch_size=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            *[model.max_positions() for model in models]
        ),
        #
        num_shards=num_shards,
        shard_id=shard_id,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
        context_window=cfg.eval_lm.context_window,
    )
    if torch.distributed.get_rank() == 0:
        dev_itr = progress_bar.progress_bar(
            dev_itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            level=0,
            default_log_format="tqdm",
        )

    criterion = task.build_criterion(cfg.criterion)

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per GPU = {} and batch size per GPU = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    results, expert_probs,priors = dynamic_eval_lm(
        # trainer=trainer,
        models=models,
        source_dictionary=task.source_dictionary,
        train_batch_iterator=None,
        dev_batch_iterator=dev_itr,
        post_process=cfg.common_eval.post_process,
        output_word_probs=cfg.eval_lm.output_word_probs,
        output_word_stats=cfg.eval_lm.output_word_stats,
        target_dictionary=task.target_dictionary,
        softmax_batch=cfg.eval_lm.softmax_batch,
        remove_bos_token=False,
        use_expert=cfg.common_eval.use_expert,
        cfg=cfg,
        num_shards=num_shards,
        shard_id=shard_id,
        task=task,
        device=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
        eval_only=True,
        max_documents=cfg.common_eval.max_samples,
        max_train_documents=0,
        precomputed_prior=precomputed_prior,
        ensemble_type=cfg.common_eval.ensemble_type
    )
    if results and expert_probs:
        expert_probs = [x.tolist()[0] for x in expert_probs]
        res = []
        for x,y,z in zip(results, expert_probs, priors):
            res.append({"ppl": x, "posterior": y, "exp_avg_posterior": z})
        df = pd.DataFrame(res)
        df.to_json(cfg.common_eval.results_path, lines=True, orient='records')
        logger.info(f"final ppl: {results[-1]}")

    return results


def cli_main():
    parser = options.get_dynamic_eval_lm_parser()
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)


if __name__ == "__main__":
    cli_main()
