from torch.optim.lr_scheduler import _LRScheduler
import math

class WarmupCosineLR(_LRScheduler):

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):

        self.warmup_epochs = warmup_epochs

        self.total_epochs = total_epochs

        self.min_lr = min_lr

        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch < self.warmup_epochs:

            return [base_lr * ((self.last_epoch + 1) / self.warmup_epochs) for base_lr in self.base_lrs]

        else:

            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)

            return [self.min_lr + (base_lr - self.min_lr) * 

                    (1 + math.cos(math.pi * progress)) / 2

                    for base_lr in self.base_lrs]

