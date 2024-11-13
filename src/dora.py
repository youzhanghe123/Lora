import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Dict, Optional, Union
import os
import json
import re

class DoRALayer:
    """Base class for DoRA layers"""
    def __init__(
        self,
        r: int,
        dora_alpha: int,
        dora_dropout: float,
        layer_type: str  # 'attention' or 'ffn' or 'unknown'
    ):
        self.r = r
        self.dora_alpha = dora_alpha
        self.layer_type = layer_type
        if dora_dropout > 0.:
            self.dora_dropout = nn.Dropout(p=dora_dropout)
        else:
            self.dora_dropout = lambda x: x

class DoRAAdapter(nn.Module, DoRALayer):
    """DoRA adapter implementation with layer type tracking"""
    def __init__(
        self,
        existing_layer: nn.Module,
        layer_name: str,
        r: int = 0,
        dora_alpha: int = 1,
        dora_dropout: float = 0.,
        layer_type: str = 'attention'
    ):
        nn.Module.__init__(self)
        DoRALayer.__init__(self, r=r, dora_alpha=dora_alpha, dora_dropout=dora_dropout, layer_type=layer_type)
        self.existing_layer = existing_layer
        self.layer_name = layer_name
        self.in_features = existing_layer.in_features
        self.out_features = existing_layer.out_features
        self.m = nn.Parameter(
            self.existing_layer.weight.norm(p=2, dim=0, keepdim=True))
        existing_dtype = next(existing_layer.parameters()).dtype

        if r > 0:
            self.dora_A = nn.Parameter(torch.zeros(r, self.in_features, dtype=existing_dtype))
            self.dora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=existing_dtype))
            self.scaling = self.dora_alpha / self.r
            self.reset_parameters()

    def reset_parameters(self):
        if self.r > 0:
            nn.init.normal_(self.dora_A, mean=0.0, std=0.02)
            nn.init.zeros_(self.dora_B)

    def forward(self, x: torch.Tensor):
        if self.r > 0:
            dora = self.dora_B @ self.dora_A
            numerator = self.existing_layer.weight + self.dora_alpha * dora
            denominator = numerator.norm(p=2, dim=0, keepdim=True)
            directional_component = numerator / denominator
            new_weight = self.m * directional_component
        else:
            new_weight = self.existing_layer.weight
        return F.linear(x, new_weight, self.existing_layer.bias)

    def get_dora_state_dict(self):
        """Get state dict for saving DoRA parameters"""
        if self.r > 0:
            return {
                'dora_A': self.dora_A.data,
                'dora_B': self.dora_B.data,
                'layer_type': self.layer_type,
                'layer_name': self.layer_name,
                'r': self.r,
                'alpha': self.dora_alpha,
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

class ModelAnalyzer:
    """Analyzes model architecture to detect attention and FFN modules"""
    
    COMMON_PATTERNS = {
        'attention': {
            'query': ['q_proj', 'q_lin', 'query', 'self_attn.q_proj', 'self_attention.query'],
            'key': ['k_proj', 'k_lin', 'key', 'self_attn.k_proj', 'self_attention.key'],
            'value': ['v_proj', 'v_lin', 'value', 'self_attn.v_proj', 'self_attention.value'],
            'output': ['o_proj', 'out_lin', 'output', 'self_attn.o_proj', 'self_attention.output']
        },
        'ffn': {
            'ffn1': ["up_proj",'fc1', 'lin1', 'w1', 'ffn.lin1', 'mlp.fc1', 'feed_forward.fc1', 'ffn.0', 'mlp.0'],
            'ffn2': ["down_proj",'fc2', 'lin2', 'w2', 'ffn.lin2', 'mlp.fc2', 'feed_forward.fc2', 'ffn.2', 'mlp.2']
        }
    }
    
    @staticmethod
    def get_module_type(module_name: str) -> str:
        """
        Determine if a module is attention or FFN based on its name
        Returns 'attention', 'ffn', or 'unknown'
        """
        # Clean the module name from any parent path
        clean_name = module_name.split('.')[-1]
        
        # First check exact matches
        if clean_name == 'fc1' or clean_name == 'fc2':
            return 'ffn'
            
        # Check attention patterns
        for patterns in ModelAnalyzer.COMMON_PATTERNS['attention'].values():
            if any(pattern in module_name for pattern in patterns):
                return 'attention'
        
        # Check FFN patterns
        for patterns in ModelAnalyzer.COMMON_PATTERNS['ffn'].values():
            if any(pattern in module_name for pattern in patterns):
                return 'ffn'
        
        # Additional FFN detection logic
        if any(name in module_name for name in ['mlp', 'ffn', 'feed_forward']):
            if 'fc' in module_name or 'linear' in module_name.lower():
                return 'ffn'
        
        return 'unknown'
    
    @staticmethod
    def detect_module_patterns(model: nn.Module) -> Dict[str, List[str]]:
        """
        Analyze model architecture to detect actual module names
        Returns a dictionary mapping module types to their actual names in the model
        """
        detected_patterns = {'attention': [], 'ffn': [], 'unknown': []}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module_type = ModelAnalyzer.get_module_type(name)
                detected_patterns[module_type].append(name)
                
                # Debug output
                print(f"Detected module: {name} as type: {module_type}")
                if module_type == 'ffn':
                    print(f"  Input features: {module.in_features}, Output features: {module.out_features}")
        
        return detected_patterns

class ModuleMatcher:
    """Handles module matching for LoRA target modules"""
    
    @staticmethod
    def create_pattern(target: str) -> str:
        """Convert a target string into a matching pattern"""
        # Handle special cases for fc1/fc2
        if target in ['fc1', 'fc2']:
            return f".*{target}$"  # Match fc1/fc2 at the end of the name
        return target.replace('*', '.*')
    
    @staticmethod
    def matches_target(module_name: str, target: str) -> bool:
        """Check if a module name matches a target pattern"""
        pattern = ModuleMatcher.create_pattern(target)
        
        # Exact match
        if pattern == module_name:
            return True
            
        # Special case for fc1/fc2
        if target in ['fc1', 'fc2'] and module_name.endswith(target):
            return True
            
        # Wildcard match
        if '*' in target or target in ['fc1', 'fc2']:
            return bool(re.match(f"^{pattern}", module_name))
            
        # Substring match (legacy behavior)
        return target in module_name
    
    @staticmethod
    def find_matching_modules(model: nn.Module, target_modules: List[str]) -> Dict[str, nn.Module]:
        """Find all modules that match the target patterns"""
        matches = {}
        
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
                
            for target in target_modules:
                if ModuleMatcher.matches_target(name, target):
                    matches[name] = module
                    # Debug output
                    print(f"Matched module: {name} for target: {target}")
                    break
                    
        return matches

class DoRAConfig:
    """Configuration class for DoRA settings with flexible target modules"""
    def __init__(
        self,
        target_modules: List[str] = None,
        attention_r: int = 8,
        ffn_r: int = 4,
        attention_alpha: int = 8,
        ffn_alpha: int = 4,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        auto_detect_modules: bool = True,
        default_r: int = 8,
        default_alpha: int = 8,
        default_dropout: float = 0.1
    ):
        self.target_modules = target_modules or []
        self.auto_detect_modules = auto_detect_modules
        
        # Configurations for different module types
        self.module_configs = {
            'attention': {
                'r': attention_r,
                'dora_alpha': attention_alpha,
                'dora_dropout': attention_dropout
            },
            'ffn': {
                'r': ffn_r,
                'dora_alpha': ffn_alpha,
                'dora_dropout': ffn_dropout
            },
            'unknown': {
                'r': default_r,
                'dora_alpha': default_alpha,
                'dora_dropout': default_dropout
            }
        }
    
    def get_module_config(self, module_name: str) -> Dict:
        """Get DoRA configuration for a specific module based on its type"""
        module_type = ModelAnalyzer.get_module_type(module_name)
        return self.module_configs[module_type]

def mark_only_dora_as_trainable(model: nn.Module) -> None:
    """
    Freeze all parameters except DoRA parameters.
    This includes both attention and FFN DoRA layers.
    """
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Then unfreeze DoRA parameters
    for name, module in model.named_modules():
        if isinstance(module, DoRALayer):
            if hasattr(module, 'dora_A'):
                module.dora_A.requires_grad = True
                print(f"Unfreezing DoRA A for {module.layer_type} layer: {name}")
            if hasattr(module, 'dora_B'):
                module.dora_B.requires_grad = True
                print(f"Unfreezing DoRA B for {module.layer_type} layer: {name}")
            if hasattr(module, 'dora_dropout') and isinstance(module.dora_dropout, nn.Module):
                for param in module.dora_dropout.parameters():
                    param.requires_grad = True
                    print(f"Unfreezing DoRA dropout for {module.layer_type} layer: {name}")

def save_dora_matrices(model: nn.Module, save_path: str):
    """Save DoRA matrices and configuration"""
    os.makedirs(save_path, exist_ok=True)
    
    dora_state = {}
    config_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, DoRAAdapter):
            dora_state[name] = module.get_dora_state_dict()
            config_dict[name] = {
                'layer_type': module.layer_type,
                'r': module.r,
                'alpha': module.dora_alpha
            }
    
    torch.save(dora_state, os.path.join(save_path, 'dora_weights.pt'))
    with open(os.path.join(save_path, 'dora_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_dora_matrices(model: nn.Module, load_path: str):
    """Load saved DoRA matrices"""
    dora_state = torch.load(os.path.join(load_path, 'dora_weights.pt'))
    
    for name, module in model.named_modules():
        if isinstance(module, DoRAAdapter) and name in dora_state:
            state = dora_state[name]
            if module.r > 0:
                module.dora_A.data.copy_(state['dora_A'])
                module.dora_B.data.copy_(state['dora_B'])
                print(f"Loaded DoRA weights for {state['layer_type']} layer: {name}")

def inject_adapter(model: nn.Module, config: DoRAConfig):
    """Inject DoRA adapters into the model based on configuration"""
    dora_modules = {}
    
    # Find target modules based on configuration
    if config.auto_detect_modules:
        detected_patterns = ModelAnalyzer.detect_module_patterns(model)
        print("Auto-detected modules:")
        for module_type, modules in detected_patterns.items():
            if modules:
                print(f"  {module_type}: {modules}")
        
        # Combine auto-detected modules with explicit targets
        all_targets = set(config.target_modules)
        all_targets.update([name for names in detected_patterns.values() for name in names])
        matching_modules = ModuleMatcher.find_matching_modules(model, all_targets)
    else:
        if not config.target_modules:
            raise ValueError("When auto_detect is False, target_modules must be provided")
            
        matching_modules = ModuleMatcher.find_matching_modules(model, config.target_modules)
        print(f"Found {len(matching_modules)} matching modules for targets {config.target_modules}")
    
    # Add DoRA adapters to matching modules
    for name, module in matching_modules.items():
        module_type = ModelAnalyzer.get_module_type(name)
        module_config = config.get_module_config(name)
        
        dora_adapter = DoRAAdapter(
            module,
            layer_name=name,
            layer_type=module_type,
            **module_config
        )
        
        if next(module.parameters(), None) is not None:
            dora_adapter = dora_adapter.to(next(module.parameters()).device)
        
        # Update the model with the DoRA adapter
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        if parent_name:
            parent_module = model.get_submodule(parent_name)
            setattr(parent_module, child_name, dora_adapter)
        else:
            setattr(model, name, dora_adapter)
        
        dora_modules[name] = dora_adapter
        print(f"Added DoRA adapter to {module_type} module: {name}")
    
    return model, dora_modules

def add_dora_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    auto_detect: bool = True,
    save_path: Optional[str] = None
) -> tuple[nn.Module, Dict]:
    """
    Helper function to add DoRA to a model with flexible targeting
    
    Args:
        model: The model to add DoRA to
        target_modules: List of specific module names to target
        auto_detect: Whether to automatically detect module patterns
        save_path: Optional path to save DoRA weights
    """
    config = DoRAConfig(
        target_modules=target_modules,
        auto_detect_modules=auto_detect
    )
    
    model, dora_modules = inject_adapter(model, config)
    mark_only_dora_as_trainable(model)
    
    if save_path:
        save_dora_matrices(model, save_path)
    
    return model, dora_modules
