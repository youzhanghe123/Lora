from transformers import AutoModelForCausalLM, AutoTokenizer
from src.lora import add_lora_to_model
from src.dataset import load_and_prepare_data
from src.trainer import LoRATrainer

def main():
    # Initialize model and tokenizer
    model_name = "facebook/opt-125m"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add LoRA to model
    model, lora_modules = add_lora_to_model(
        model,
        use_attention_lora=True,
        use_ffn_lora=True,
        save_path="./lora_checkpoints"
    )

    # Prepare data
    train_dataloader, test_encodings = load_and_prepare_data(
        tokenizer,
        batch_size=8,
        chunk_size=256,
        subset_fraction=0.1
    )

    # Initialize trainer
    trainer = LoRATrainer(
        model=model,
        train_dataloader=train_dataloader,
        test_encodings=test_encodings,
        learning_rate=1e-4,
        num_epochs=3,
        save_dir="./checkpoints"
    )

    # Train model
    trainer.train()

if __name__ == "__main__":
    main()
