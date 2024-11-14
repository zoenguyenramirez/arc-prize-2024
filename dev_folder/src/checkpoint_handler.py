import torch
import math
import logging

from pathlib import Path

from torch.nn.parallel import DataParallel

from src.model import Transformer
from src.utils.logger_helper import setup_logger
from src.token import VOCAB_SIZE, SpecialToken

logger = setup_logger(overwrite_line=False)

latest_checkpoint_saving_frequency = 10
periodic_checkpoint_saving_frequency = 100
class CheckpointHandler:

    def __init__(self, save_dir, model_name="Transformer", minimize_checkpoints=False):
        self.model_name = model_name
        self.best_val_loss = float('inf')
        self.minimize_checkpoints = minimize_checkpoints

        self.update_save_dir(save_dir)

    def update_save_dir(self, save_dir):
        self.save_dir = Path(save_dir)
        self.latest_path = self.save_dir / f"{self.model_name}_latest.pt"
        self.save_dir = Path(save_dir)

    def is_quadratic_checkpoint(self, epoch):        
        if epoch < 50 and epoch > 0:
            return False
        # Solve quadratic equation: x^2 + x - (2 * (epoch - initial) / 100) = 0
        a = 1
        b = 1
        c = -2 * (epoch - 50) / 100
        
        # Quadratic formula: (-b + sqrt(b^2 - 4ac)) / (2a)
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return True
        
        x = (-b + math.sqrt(discriminant)) / (2*a)
        
        # Check if x is very close to an integer
        return abs(x - round(x)) < 1e-6
    
    @staticmethod
    def validate_total_epoch(total_epoch: int):
        assert total_epoch % latest_checkpoint_saving_frequency == 0        
    
    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_loss, args, start_epoch, seed, relay_seed):
        checkpoint = {
            'epoch': start_epoch + epoch,
            'model_state_dict': model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'hyperparameters': args, # the key name was wrong, too late to fix
            'best_val_loss': self.best_val_loss,
            'epoch_in_session': epoch,
            'seed': seed,
            'relay_seed': relay_seed
        }

        new_best_validation_loss = False

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss            
            checkpoint['best_val_loss'] = val_loss
            new_best_validation_loss = True

        if (epoch + 1) % latest_checkpoint_saving_frequency == 0:
            # Save the latest checkpoint
            torch.save(checkpoint, self.latest_path)

        if not self.minimize_checkpoints:
            # Save periodic checkpoint
            if (epoch + 1) % periodic_checkpoint_saving_frequency == 0:
                periodic_path = self.save_dir / f"{self.model_name}_epoch_{epoch:04d}.pt"
                torch.save(checkpoint, periodic_path)

        # Save the best checkpoint
        if new_best_validation_loss and start_epoch + epoch > 50:
            best_path = self.save_dir / f"{self.model_name}_best_{start_epoch + epoch}.pt"
            logger.info(f'best_saved @: {best_path}')
            torch.save(checkpoint, best_path)

            return self.best_val_loss

        return None
    
    @staticmethod
    def load_and_fixup_checkpoint(path, device, *, adjust_max_length = 0):
        checkpoint = torch.load(path, map_location=device, weights_only=True)

        positional_encoding = checkpoint['model_state_dict']['positional_encoding']
        if checkpoint['hyperparameters']['grid_encoder'] and adjust_max_length and adjust_max_length != positional_encoding.shape[1]:
            assert adjust_max_length > positional_encoding.shape[1]
            logging.info(f'positional_encoding will be changed from {positional_encoding.shape} to [:, {adjust_max_length}, :]')
            checkpoint['model_state_dict']['positional_encoding'] = torch.zeros(positional_encoding.shape[0], adjust_max_length, positional_encoding.shape[2])

        return checkpoint


    @staticmethod
    def load_checkpoint(path, model, *, device, optimizer=None, initial_lr=None, adjust_max_length = 0):
        checkpoint = CheckpointHandler.load_and_fixup_checkpoint(path, device, adjust_max_length = adjust_max_length)
        
        if isinstance(model, DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        if optimizer:
            for param_group in checkpoint['optimizer_state_dict']['param_groups']:
                param_group['initial_lr'] = initial_lr
                
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint, checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
    
    @staticmethod
    def load_checkpoint_in_production(checkpoint_path, device, *, adjust_max_length=0):
        checkpoint = CheckpointHandler.load_and_fixup_checkpoint(checkpoint_path, adjust_max_length=adjust_max_length, device = device)

        args = checkpoint['hyperparameters']
        model = Transformer(VOCAB_SIZE, args['embed_size'], args['num_layers'], args['heads'], use_grid_encoder=args['grid_encoder'], progressive_head=args['progressive_head'], max_length=adjust_max_length if adjust_max_length > 0 else args['max_seq_length']).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])

        logging.info('The checkpoint was saved at epoch %d, train_loss: %f, val_loss: %f, args: %s', 
                    checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss'], args)
        return model, adjust_max_length or args['max_seq_length'], args
    

    @staticmethod
    def restore_model_state(model, checkpoint_path, device, *, adjust_max_length=0):
        checkpoint = CheckpointHandler.load_and_fixup_checkpoint(checkpoint_path, adjust_max_length=adjust_max_length, device = device)
        model.load_state_dict(checkpoint['model_state_dict'])

