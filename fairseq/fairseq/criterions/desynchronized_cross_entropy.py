# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from collections import defaultdict
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("desynchronized_cross_entropy", dataclass=CrossEntropyCriterionConfig)
class CrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=True)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            'is_training': model.training,
            'rank': torch.distributed.get_rank(),
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "domain": sample['net_input']['src_domain_idx'][0]
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        # """Aggregate logging outputs from data parallel training."""
        logs_ = defaultdict(list)
        if not logging_outputs[0]['is_training'] and torch.distributed.is_initialized():
            gpus  = range(torch.distributed.get_world_size())
            if len(gpus) >= 8:
                num_gpus_per_domain = torch.distributed.get_world_size() // 8
            else:
                num_gpus_per_domain = 1
            gpu_mappings = [list(gpus[n:n+num_gpus_per_domain]) for n in range(0, len(gpus), num_gpus_per_domain)]
            mappings = defaultdict(list)
            for ix, gpus in enumerate(gpu_mappings):
                for gpu in gpus:
                    mappings[ix].append(gpu)
            # if evaluating, we only want the outputs of GPUs assigned to the domain of interest
            for gpu, log in enumerate(logging_outputs):
                if log['rank'] in mappings[log['domain']]:
                    logs_[log['domain']].append(dict((k,v) for k,v in log.items() if k != 'domain'))
        else:
            for log in logging_outputs:        
                logs_[log['domain']].append(dict((k,v) for k,v in log.items() if k != 'domain'))
        
        logs = {}
            
        for domain, group in logs_.items():
            logs[domain]  = {'loss': sum(x['loss'] for x in group), 
                            'ntokens': sum(x['ntokens'] for x in group),
                            'sample_size': sum(x['sample_size'] for x in group)}        
        
        
        loss_sum = sum(logs[domain]['loss'] for domain in logs)
        ntokens = sum(logs[domain]['ntokens'] for domain in logs)
        sample_size = sum(logs[domain]['sample_size'] for domain in logs)

        # we divide by log(2) to convert the loss from base e to base 2
        try:
            metrics.log_scalar(
                "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
            ) 
        except:
            return  
        for domain in logs:
            # metrics.log_scalar(
                # f"loss_{domain}", logs[domain]['loss'] / logs[domain]['sample_size'] / math.log(2), logs[domain]['sample_size'], round=3
            # )
            if logs[domain]['sample_size'] != logs[domain]['ntokens']:
                # metrics.log_scalar(
                #     f"nll_loss_{domain}", logs[domain]['loss']/ logs[domain]['ntokens'] / math.log(2), logs[domain]['ntokens'], round=3
                # )
                
                metrics.log_scalar(
                    f"ppl_{domain}", 2 ** (logs[domain]['loss'] / logs[domain]['ntokens'] / math.log(2)), logs[domain]['ntokens'], round=3
                )
            else:
                metrics.log_scalar(f"ppl_{domain}", 2 ** (logs[domain]['loss'] / logs[domain]['sample_size'] / math.log(2)), logs[domain]['sample_size'], round=3)


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
