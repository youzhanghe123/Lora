The src file contains trainer.py, lora.py,loranew.py and dataset.py 

lora.py is based on the original Lora code of huggigface.

loranew.py adding the function of deciding which layer and which module to add lora matrix.

You can try Lora_experiment.ipynb ( for lora.py) and Lora_experiment_new.ipynb ( for loranew.py) to experiment, and try the saved lora matrixes. 

For loranew.py, please use the inference format as this style (you can also replace * with the number of layer you want to make modification): 

'''
model, _ = add_lora_to_model(
    model,

    target_modules=[
        "*.self_attn.q_proj",  # Match q_proj in any self_attn
        "*.fc1"              # Match any FFN layers
    ],
    auto_detect=False,

    save_path = output_dir
)
'''
