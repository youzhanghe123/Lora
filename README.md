The src file contains trainer.py, lora.py and dataset.py 
You can try Lora_experiment.ipynb to experiment, and try the saved lora matrixes. 

For loranew.py, please use the inference format as this style (you can also replace * with the number of layer you want to make modification): 
# Add LoRA
model, _ = add_lora_to_model(
    model,

    target_modules=[
        "*.self_attn.q_proj",  # Match q_proj in any self_attn
        "*.fc1"              # Match any FFN layers
    ],
    auto_detect=False,

    save_path = output_dir
)
