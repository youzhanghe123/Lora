import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import time
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoRATrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        test_encodings,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        num_epochs: int = 1,
        warmup_steps: int = 0,
        save_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_encodings = test_encodings
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.save_dir = save_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize tracking variables
        self.best_perplexity = float('inf')
        self.training_stats = []

    def _create_optimizer(self):
        """Create optimizer for LoRA parameters"""
        # Only optimize LoRA parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        return AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        total_steps = len(self.train_dataloader) * self.num_epochs
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

    def train(self):
        """Main training loop"""
        logger.info(f"Starting training on device: {self.device}")
        
        for epoch in range(self.num_epochs):
            logger.info(f"Beginning epoch {epoch+1}/{self.num_epochs}")
            
            # Training phase
            train_loss = self._train_epoch()
            
            # Evaluation phase
            perplexity = self.evaluate()
            
            # Save stats
            self.training_stats.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'perplexity': perplexity,
                'learning_rate': self.scheduler.get_last_lr()[0]
            })
            
            # Save best model
            if perplexity < self.best_perplexity and self.save_dir:
                self.best_perplexity = perplexity
                self.save_checkpoint(f"best_model_epoch_{epoch+1}")
            
            logger.info(f"Epoch {epoch+1}: Loss = {train_loss:.4f}, Perplexity = {perplexity:.4f}")

    def _train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, attention_mask, labels = batch
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Update
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_dataloader)

    def evaluate(self) -> float:
        """Calculate perplexity on test set"""
        self.model.eval()
        max_length = 1024
        stride = 256
        seq_len = self.test_encodings.input_ids.size(1)
        
        nlls = []
        prev_end_loc = 0
        
        for begin_loc in tqdm(range(0, seq_len, stride), desc="Evaluating"):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            
            input_ids = self.test_encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
            
            nlls.append(neg_log_likelihood)
            
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
                
        return torch.exp(torch.stack(nlls).mean())

    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            checkpoint_path = os.path.join(self.save_dir, f"{name}.pt")
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'training_stats': self.training_stats,
                'best_perplexity': self.best_perplexity
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_stats = checkpoint['training_stats']
        self.best_perplexity = checkpoint['best_perplexity']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
