import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Dict, Optional
import os
import json

class LoRALayer:
    """Base class for LoRA layers"""
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        layer_type: str  # 'attention' or 'ffn'
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.layer_type = layer_type
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

class LoRAAdapter(nn.Module, LoRALayer):
    """LoRA adapter implementation with layer type tracking"""
    def __init__(
        self,
        existing_layer: nn.Module,
        layer_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        layer_type: str = 'attention'  # 'attention' or 'ffn'
    ):
        nn.Module.__init__(self)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, layer_type=layer_type)
        self.existing_layer = existing_layer
        self.layer_name = layer_name
        
        self.in_features = existing_layer.in_features
        self.out_features = existing_layer.out_features
        
        existing_dtype = next(existing_layer.parameters()).dtype

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features, dtype=existing_dtype))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=existing_dtype))
            self.scaling = self.lora_alpha / self.r
            self.reset_parameters()

    def reset_parameters(self):
        if self.r > 0:
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        if self.r > 0:
            original = self.existing_layer(x)
            lora = self.lora_dropout(x)
            lora = F.linear(lora, self.lora_A)
            lora = F.linear(lora, self.lora_B)
            return original + (lora * self.scaling)
        return self.existing_layer(x)

    def get_lora_state_dict(self):
        """Get state dict for saving LoRA parameters"""
        if self.r > 0:
            return {
                'lora_A': self.lora_A.data,
                'lora_B': self.lora_B.data,
                'layer_type': self.layer_type,  # Explicitly save whether this is attention or FFN
                'layer_name': self.layer_name,
                'r': self.r,
                'alpha': self.lora_alpha,
                'scaling': self.scaling,
                'in_features': self.in_features,
                'out_features': self.out_features
            }
        return {
            'layer_type': self.layer_type,
            'layer_name': self.layer_name,
            'r': 0,
            'alpha': 0,
            'in_features': self.in_features,
            'out_features': self.out_features
        }

class LoRAConfig:
    """Configuration class for LoRA settings"""
    def __init__(
        self,
        use_attention_lora: bool = True,
        use_ffn_lora: bool = True,
        attention_r: int = 8,
        ffn_r: int = 4,
        attention_alpha: int = 8,
        ffn_alpha: int = 4,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        attention_targets: List[str] = None,
        ffn_targets: List[str] = None
    ):
        self.use_attention_lora = use_attention_lora
        self.use_ffn_lora = use_ffn_lora
        
        self.attention_targets = attention_targets or ['q_proj', 'k_proj', 'v_proj']
        self.ffn_targets = ffn_targets or ['fc1', 'fc2']
        
        self.attention_config = {
            'r': attention_r,
            'lora_alpha': attention_alpha,
            'lora_dropout': attention_dropout
        }
        
        self.ffn_config = {
            'r': ffn_r,
            'lora_alpha': ffn_alpha,
            'lora_dropout': ffn_dropout
        }

def inject_adapter(model: nn.Module, config: LoRAConfig):
    """Inject LoRA adapters into the model based on configuration."""
    lora_modules = {}  # Keep track of added LoRA modules

    if config.use_attention_lora:
        # Add LoRA to attention layers
        for target in config.attention_targets:
            for name, _ in model.named_modules():
                if target in name:
                    original_module = model.get_submodule(name)
                    lora_adapter = LoRAAdapter(
                        original_module,
                        layer_name=name,
                        layer_type='attention',
                        **config.attention_config
                    )
                    
                    if next(original_module.parameters(), None) is not None:
                        lora_adapter = lora_adapter.to(next(original_module.parameters()).device)
                    
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    if parent_name:
                        parent_module = model.get_submodule(parent_name)
                        setattr(parent_module, child_name, lora_adapter)
                    else:
                        setattr(model, name, lora_adapter)
                    
                    lora_modules[name] = lora_adapter

    if config.use_ffn_lora:
        # Add LoRA to FFN layers
        for target in config.ffn_targets:
            for name, _ in model.named_modules():
                if target in name:
                    original_module = model.get_submodule(name)
                    lora_adapter = LoRAAdapter(
                        original_module,
                        layer_name=name,
                        layer_type='ffn',
                        **config.ffn_config
                    )
                    
                    if next(original_module.parameters(), None) is not None:
                        lora_adapter = lora_adapter.to(next(original_module.parameters()).device)
                    
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    if parent_name:
                        parent_module = model.get_submodule(parent_name)
                        setattr(parent_module, child_name, lora_adapter)
                    else:
                        setattr(model, name, lora_adapter)
                    
                    lora_modules[name] = lora_adapter

    return model, lora_modules

def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """
    Freeze all parameters except LoRA parameters.
    Handles both attention and FFN LoRA layers.
    """
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Then unfreeze LoRA parameters for both attention and FFN
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            if hasattr(module, 'lora_A'):
                module.lora_A.requires_grad = True
                print(f"Unfreezing LoRA A for {module.layer_type} layer: {name}")
            if hasattr(module, 'lora_B'):
                module.lora_B.requires_grad = True
                print(f"Unfreezing LoRA B for {module.layer_type} layer: {name}")
            if hasattr(module, 'lora_dropout') and isinstance(module.lora_dropout, nn.Module):
                for param in module.lora_dropout.parameters():
                    param.requires_grad = True
                    print(f"Unfreezing LoRA dropout for {module.layer_type} layer: {name}")

def save_lora_matrices(model: nn.Module, save_path: str):
    """Save LoRA matrices and configuration"""
    os.makedirs(save_path, exist_ok=True)
    
    lora_state = {}
    config_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LoRAAdapter):
            lora_state[name] = module.get_lora_state_dict()
            config_dict[name] = {
                'layer_type': module.layer_type,
                'r': module.r,
                'alpha': module.lora_alpha
            }
    
    # Save LoRA matrices
    torch.save(lora_state, os.path.join(save_path, 'lora_weights.pt'))
    
    # Save configuration
    with open(os.path.join(save_path, 'lora_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_lora_matrices(model: nn.Module, load_path: str):
    """Load saved LoRA matrices"""
    lora_state = torch.load(os.path.join(load_path, 'lora_weights.pt'))
    
    for name, module in model.named_modules():
        if isinstance(module, LoRAAdapter) and name in lora_state:
            state = lora_state[name]
            if module.r > 0:
                module.lora_A.data.copy_(state['lora_A'])
                module.lora_B.data.copy_(state['lora_B'])
                print(f"Loaded LoRA weights for {state['layer_type']} layer: {name}")

def add_lora_to_model(
    model,
    use_attention_lora: bool = True,
    use_ffn_lora: bool = True,
    save_path: Optional[str] = None
):
    """Helper function to add LoRA to a model with common configurations"""
    config = LoRAConfig(
        use_attention_lora=use_attention_lora,
        use_ffn_lora=use_ffn_lora
    )
    
    model, lora_modules = inject_adapter(model, config)
    mark_only_lora_as_trainable(model)
    
    if save_path:
        save_lora_matrices(model, save_path)
    
    return model, lora_modules
