from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from src.model_loading.common.model_config import ModelConfig
from src.model_loading.loaders.interface import ModelLoaderInterface
import os

class QuipModelLoader(ModelLoaderInterface):
    """Loader for QuIP quantized models using the original QUIP loading method"""
    
    def load_model(
        self,
        config: ModelConfig,
        compile_mode: str = "reduce-overhead",
        fullgraph: bool = True,
        dynamic: bool = False
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        
        # Make a copy of max_memory with integer keys for GPU devices
        max_memory = None
        if config.max_memory is not None:
            max_memory = {}
            for key, value in dict(config.max_memory).items():
                if isinstance(key, str) and key.isdigit():
                    max_memory[int(key)] = value
                else:
                    max_memory[key] = value
        
        # Check if we should use multi-GPU loading
        is_multi_gpu = (
            max_memory is not None and
            isinstance(max_memory, dict) and
            len(max_memory) > 1
        )
        
        # Use the original QUIP loading method
        model = self._load_quip_model_original_method(config.model_path, config.device, is_multi_gpu)
        
        # Compile model if CUDA is available (only for single-GPU case)
        if torch.cuda.is_available() and config.apply_compile and not is_multi_gpu:
            print(f"Compiling model with mode: {compile_mode}")
            model = torch.compile(
                model,
                mode=compile_mode,
                fullgraph=fullgraph,
                dynamic=dynamic
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path or config.model_path,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code
        )
        
        return model, tokenizer
    
    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve HuggingFace cache paths to actual model directories"""
        from huggingface_hub import snapshot_download
        
        # If it's already a local path that exists, use it
        if os.path.exists(model_path) and os.path.isdir(model_path):
            return model_path
        
        # If it looks like a HuggingFace model ID, download/locate it
        if not os.path.exists(model_path):
            try:
                # This will either download or find the cached version
                resolved_path = snapshot_download(repo_id=model_path, local_files_only=True)
                return resolved_path
            except Exception as e:
                print(f"Failed to resolve HuggingFace path with local_files_only: {e}")
                try:
                    # If local_files_only fails, try downloading
                    resolved_path = snapshot_download(repo_id=model_path)
                    return resolved_path
                except Exception as e2:
                    print(f"Failed to download model: {e2}")
                    return model_path
        
        return model_path
    
    def _load_quip_model_original_method(self, model_path: str, device: str, is_multi_gpu: bool):
        """Load QUIP model using the original method from the QUIP repository"""
        
        # Resolve the model path first
        resolved_model_path = self._resolve_model_path(model_path)
        print(f"Resolved model path: {resolved_model_path}")
        
        # Disable all initialization functions
        def noop(*args, **kwargs):
            pass
        
        # Store original functions to restore later
        original_kaiming_uniform = torch.nn.init.kaiming_uniform_
        original_uniform = torch.nn.init.uniform_
        original_normal = torch.nn.init.normal_
        
        torch.nn.init.kaiming_uniform_ = noop
        torch.nn.init.uniform_ = noop
        torch.nn.init.normal_ = noop
        
        # Disable modeling_utils initialization
        from transformers import modeling_utils
        original_init_weights = modeling_utils._init_weights
        modeling_utils._init_weights = False
        
        # Store original default dtype
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.half)
        
        try:
            # Load config from the resolved model path
            config = LlamaConfig.from_pretrained(resolved_model_path)
            
            # Create model with config (this will use half precision and no initialization)
            model = LlamaForCausalLM(config)
            
            # Load state dict from checkpoint
            checkpoint_file = self._find_checkpoint_file(model_path)
            print(f"Loading checkpoint from: {checkpoint_file}")
            
            if checkpoint_file.endswith('.safetensors'):
                from safetensors.torch import load_file as safe_load
                state_dict = safe_load(checkpoint_file)
            else:
                state_dict = torch.load(checkpoint_file, map_location='cpu')
            
            # Load the state dict
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
            
            # Move to device
            if is_multi_gpu:
                # For multi-GPU, let PyTorch handle device placement
                print("Multi-GPU setup detected, using automatic device placement")
                model = model.cuda()  # Move to CUDA, let PyTorch handle distribution
            else:
                print(f"Moving model to device: {device}")
                model = model.to(device)
            
            # Set model to eval mode
            model = model.eval()
            
            # Set sequence length (from original QUIP code)
            model.seqlen = 4096
            
            print('QUIP model loading complete.')
            
        finally:
            # Restore original functions and settings
            torch.set_default_dtype(original_dtype)
            torch.nn.init.kaiming_uniform_ = original_kaiming_uniform
            torch.nn.init.uniform_ = original_uniform
            torch.nn.init.normal_ = original_normal
            modeling_utils._init_weights = original_init_weights
        
        return model
    
    def _find_checkpoint_file(self, model_path: str) -> str:
        """Find the checkpoint file in the model directory or HuggingFace cache"""
        # First try to use snapshot_download to get the cached path
        try:
            from huggingface_hub import snapshot_download
            resolved_path = snapshot_download(repo_id=model_path, local_files_only=True)
            print(f"Resolved HuggingFace cache path: {resolved_path}")
        except Exception as e:
            print(f"Failed to resolve HuggingFace path, using original: {e}")
            resolved_path = model_path
        
        # Look for common checkpoint file patterns in the resolved path
        possible_files = [
            'model.safetensors',
            'pytorch_model.bin',
            'model.bin',
            'checkpoint.safetensors',
            'checkpoint.bin'
        ]
        
        for filename in possible_files:
            full_path = os.path.join(resolved_path, filename)
            if os.path.exists(full_path):
                return full_path
        
        # If none of the standard files exist, look for any .safetensors or .bin file
        if os.path.isdir(resolved_path):
            for file in os.listdir(resolved_path):
                if file.endswith('.safetensors') or file.endswith('.bin'):
                    return os.path.join(resolved_path, file)
        
        # If model_path is a file itself
        if os.path.isfile(resolved_path):
            return resolved_path
        
        raise FileNotFoundError(f"No checkpoint file found in {resolved_path}. Contents: {os.listdir(resolved_path) if os.path.exists(resolved_path) else 'Path does not exist'}")