from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union
import torch

class WikiTextDataset:
    def __init__(
        self,
        tokenizer,
        split: str = "train",
        chunk_size: int = 256,
        subset_fraction: float = 0.1,
        max_length: int = 256
    ):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_length = max_length
        
        # Load dataset
        print(f"Loading {split} split of WikiText...")
        self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        
        # Process dataset
        self.chunks = self._prepare_chunks(subset_fraction)
        print(f"Created {len(self.chunks)} chunks")

    def _prepare_chunks(self, subset_fraction: float) -> List[str]:
        chunks = []
        
        # Split texts into chunks
        for example in self.dataset:
            text = example['text']
            if len(text.strip()) > 0:  # Skip empty texts
                text_chunks = [text[i:i+self.chunk_size] 
                             for i in range(0, len(text), self.chunk_size)]
                chunks.extend(text_chunks)

        # Take subset if requested
        if subset_fraction < 1.0:
            num_chunks = int(len(chunks) * subset_fraction)
            chunks = chunks[:num_chunks]

        return chunks

    def tokenize_function(self, examples: Union[str, List[str]]) -> Dict:
        """Tokenize texts and prepare for language modeling"""
        # Ensure examples is a list
        if isinstance(examples, str):
            examples = [examples]
            
        # Tokenize
        result = self.tokenizer(
            examples,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Copy input_ids to labels for language modeling
        result["labels"] = result["input_ids"].clone()
        
        return result

    def create_dataloader(
        self,
        batch_size: int = 8,
        shuffle: bool = True
    ) -> DataLoader:
        """Create a DataLoader for the chunks"""
        # Tokenize all chunks
        tokenized_chunks = self.tokenize_function(self.chunks)
        
        # Create tensor dataset
        tensor_dataset = torch.utils.data.TensorDataset(
            tokenized_chunks["input_ids"],
            tokenized_chunks["attention_mask"],
            tokenized_chunks["labels"]
        )
        
        # Create dataloader
        return DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def prepare_test_data(self) -> Dict[str, torch.Tensor]:
        """Prepare test data for perplexity calculation"""
        test_text = "\n\n".join(self.dataset["text"])
        return self.tokenizer(test_text, return_tensors="pt")

def load_and_prepare_data(
    tokenizer,
    batch_size: int = 8,
    chunk_size: int = 256,
    subset_fraction: float = 0.1,
    max_length: int = 256
) -> tuple:
    """Helper function to load and prepare all required data"""
    # Prepare training data
    train_dataset = WikiTextDataset(
        tokenizer=tokenizer,
        split="train",
        chunk_size=chunk_size,
        subset_fraction=subset_fraction,
        max_length=max_length
    )
    train_dataloader = train_dataset.create_dataloader(batch_size=batch_size)
    
    # Prepare test data
    test_dataset = WikiTextDataset(
        tokenizer=tokenizer,
        split="test",
        chunk_size=chunk_size,
        subset_fraction=1.0,  # Use full test set
        max_length=max_length
    )
    test_encodings = test_dataset.prepare_test_data()
    
    return train_dataloader, test_encodings
