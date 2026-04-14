import math
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWithLinearWarmupLR(LRScheduler):

    def __init__(self, optimizer, warmup_iters, max_iters, lr_max, cos_lr_min=None, warmup_lr_min=0, last_epoch=-1):
        """
        Warms the LR from warmup_lr_min to lr_max over t_warmup steps, then anneals lr_max down to 
        cos_lr_min over t_max - t_warmup steps.

        If cos_lr_min is None, sets cos_lr_min = lr_max / 10 as rec'd by the Chinchilla paper
        """
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.anneal_steps = max_iters - warmup_iters
        self.lr_max = lr_max
        self.cos_lr_min = cos_lr_min
        self.warmup_lr_min = warmup_lr_min
        if self.cos_lr_min is None:
            self.cos_lr_min = lr_max / 10.0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # last epoch is like the init counter value for step num
        lr_list = None
        t = self.last_epoch
        if t < self.warmup_iters:
            lr_list = self._get_lr_linear_warmup(t)
        elif self.warmup_iters <= t < self.max_iters:
            lr_list = self._get_lr_cos_anneal(t)
        else:
            return [self.cos_lr_min for _ in self.optimizer.param_groups]
        return lr_list

    def _get_lr_linear_warmup(self, t):
        c = t / float(self.warmup_iters)
        lr = self.warmup_lr_min + (self.lr_max - self.warmup_lr_min) * c
        return [lr for _ in self.optimizer.param_groups]

    def _get_lr_cos_anneal(self, t):
        t_anneal = t - self.warmup_iters
        lr = self.cos_lr_min + 0.5*(self.lr_max - self.cos_lr_min)*(1 + math.cos(math.pi * t_anneal / self.anneal_steps))
        return [lr for _ in self.optimizer.param_groups]

