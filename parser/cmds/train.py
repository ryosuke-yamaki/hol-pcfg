# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
from parser.cmds.cmd import CMD
from parser.helper.metric import Metric
from parser.helper.loader_wrapper import DataPrefetcher
import torch
import numpy as np
from parser.helper.util import *
from parser.helper.data_module import DataModule
from pathlib import Path
import wandb

class Train(CMD):

    def __call__(self, args):

        self.args = args
        self.device = args.device

        dataset = DataModule(args)
        self.model = get_model(args.model, dataset)
        create_save_path(args)
        log = get_logger(args)
        self.optimizer = get_optimizer(args.optimizer, self.model)
        log.info("Create the model")
        log.info(f"{self.model}\n")
        total_time = timedelta()
        best_e, best_metric = 1, Metric()
        log.info(self.optimizer)
        log.info(args)
        eval_loader = dataset.val_dataloader

        # Initialize W&B
        wandb_cfg = getattr(args, 'wandb', None)
        use_wandb = wandb_cfg is not None and getattr(wandb_cfg, 'enabled', True)
        seed_val = getattr(args, 'seed', None)
        if seed_val is not None and wandb_cfg is not None:
            base_name = getattr(wandb_cfg, 'run_name', '') or ''
            if base_name:
                wandb_cfg.run_name = f"{base_name}-seed{seed_val}"
        if use_wandb:
            wandb.init(
                project=getattr(wandb_cfg, 'project', 'hol-pcfg'),
                entity=getattr(wandb_cfg, 'entity', None),
                name=getattr(wandb_cfg, 'run_name', None),
                tags=getattr(wandb_cfg, 'tags', None),
                config={
                    'model': dict(args.model),
                    'train': dict(args.train),
                    'optimizer': dict(args.optimizer),
                    'data': dict(args.data),
                    'n_params': sum(p.numel() for p in self.model.parameters()),
                },
            )
            wandb.watch(self.model, log='gradients', log_freq=100)

        '''
        Training
        '''
        train_arg = args.train
        self.train_arg = train_arg

        for epoch in range(1, train_arg.max_epoch + 1):

            # KL annealing for VAE models
            kl_warmup = getattr(train_arg, 'kl_warmup_epochs', 0)
            if kl_warmup > 0 and hasattr(self.model, '_current_beta'):
                self.model._current_beta = min(1.0, (epoch - 1) / kl_warmup)
            elif hasattr(self.model, '_current_beta'):
                self.model._current_beta = 1.0

            # curriculum learning. Used in compound PCFG.
            if train_arg.curriculum:
                train_loader = dataset.train_dataloader(max_len=min(train_arg.start_len + epoch - 1, train_arg.max_len))
            else:
                train_loader = dataset.train_dataloader(max_len=train_arg.max_len)

            train_loader_autodevice = DataPrefetcher(train_loader, device=self.device)
            eval_loader_autodevice = DataPrefetcher(eval_loader, device=self.device)
            start = datetime.now()
            avg_loss = self.train(train_loader_autodevice)
            log.info(f"Epoch {epoch} / {train_arg.max_epoch}:")


            dev_f1_metric, dev_ll = self.evaluate(eval_loader_autodevice)
            log.info(f"{'dev f1:':6}   {dev_f1_metric}")
            log.info(f"{'dev ll:':6}   {dev_ll}")

            t = datetime.now() - start

            # save the model if it is the best so far
            is_best = dev_ll > best_metric
            if is_best:
                best_metric = dev_ll
                best_e = epoch
                torch.save(
                   obj=self.model.state_dict(),
                   f = args.save_dir + "/best.pt"
                )
                log.info(f"{t}s elapsed (saved)\n")
            else:
                log.info(f"{t}s elapsed\n")

            if use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train/loss': avg_loss,
                    'val/sentence_f1': dev_f1_metric.sentence_uf1,
                    'val/corpus_f1': dev_f1_metric.corpus_uf1,
                    'val/likelihood': dev_ll.avg_likelihood.item(),
                    'val/perplexity': dev_ll.perplexity.item(),
                    'best/epoch': best_e,
                    'best/likelihood': best_metric.avg_likelihood.item(),
                }
                if hasattr(self.model, 'get_monitoring_metrics'):
                    log_dict.update(self.model.get_monitoring_metrics())
                wandb.log(log_dict)

            total_time += t
            if train_arg.patience > 0 and epoch - best_e >= train_arg.patience:
                break

        if use_wandb:
            wandb.finish()
