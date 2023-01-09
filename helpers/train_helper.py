import os
import tqdm
import numpy as np
import torch

from helpers.checkpoint_helper import save_checkpoint
from helpers.checkpoint_helper import load_checkpoint


class Trainer(object):
    def __init__(self, cfg, model, optimizer, lr_scheduler, warmup_lr_scheduler, train_loader, test_loader,
                 logger, tb_logger):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.logger = logger
        self.tb_logger = tb_logger
        self.epoch = 0
        self.iter = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        if cfg.get('resume_checkpoint') is not None:
            assert os.path.exists(cfg['resume_checkpoint'])
            self.epoch = load_checkpoint(
                file_name=cfg['resume_checkpoint'],
                model=self.model,
                optimizer=self.optimizer,
                map_location=self.device,
                logger=self.logger,
            )
            assert self.epoch is not None
            self.lr_scheduler.last_epoch = self.epoch - 1

    def train(self):
        start_epoch = self.epoch
        progress_bar = tqdm.tqdm(
            range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True,
            desc='epochs'
        )

        for epoch in range(start_epoch, self.cfg['max_epoch']):
            np.random.seed(np.random.get_state()[1][0] + epoch)
            self.train_one_epoch()
            self.epoch += 1

            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            if self.epoch % self.cfg['save_frequency'] == 0:
                ckpt_dir = 'checkpoints'
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_name = os.path.join(ckpt_dir, 'checkpoint_epoch_%d' % self.epoch)
                save_checkpoint(ckpt_name, self.model, self.optimizer, self.epoch)

            progress_bar.update()
        progress_bar.close()

    def train_one_epoch(self):
        self.model.train()
        progress_bar = tqdm.tqdm(
            total=len(self.train_loader), dynamic_ncols=True, leave=(self.epoch + 1 == self.cfg['max_epoch']),
            desc='iters'
        )

        for batch_idx, (inputs, targets, infos, lidar_maps) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            lidar_maps = lidar_maps.to(self.device)
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            total_loss, stats_dict = self.model.compute_loss(outputs, targets, lidar_maps)
            total_loss.backward()
            self.optimizer.step()

            self.tb_logger.add_scalar('learning_rate/learning_rate', self.optimizer.param_groups[0]['lr'], self.iter)
            self.tb_logger.add_scalar('loss/loss', total_loss.item(), self.iter)
            for key, val in stats_dict.items():
                self.tb_logger.add_scalar('sub_loss/' + key, val, self.iter)
            self.iter += 1

            progress_bar.update()
        progress_bar.close()
