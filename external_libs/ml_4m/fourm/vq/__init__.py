import sys, os
import torch
import json
from safetensors.torch import load_file as load_safetensors_file

from .vqvae import VQ, VQVAE, DiVAE, VQControlNet
from .scheduling import *


def get_image_tokenizer(tokenizer_id: str, 
                        tokenizers_root: str = './tokenizer_ckpts', 
                        encoder_only: bool = False, 
                        device: str = 'cuda', 
                        verbose: bool = True,
                        return_None_on_fail: bool = False,):
    """
    Load a pretrained image tokenizer.
    Tries to load from Hugging Face Hub format (model.safetensors + config.json) first.
    If not found, falls back to the original .pth loading mechanism.

    Args:
        tokenizer_id (str): ID of the tokenizer to load (name of the checkpoint file without ".pth").
        tokenizers_root (str): Path to the directory containing the tokenizer checkpoints.
        encoder_only (bool): Set to True to load only the encoder part of the tokenizer.
        device (str): Device to load the tokenizer on.
        verbose (bool): Set to True to print load_state_dict warning/success messages
        return_None_on_fail (bool): Set to True to return None if the tokenizer fails to load (e.g. doesn't exist)

    Returns:
        model (nn.Module): The loaded tokenizer.
    """
    # --- Start of NEW LOGIC for .safetensors + config.json ---
    hf_model_dir = os.path.join(tokenizers_root, tokenizer_id)
    safetensors_path = os.path.join(hf_model_dir, 'model.safetensors')
    config_json_path = os.path.join(hf_model_dir, 'config.json')

    model_loaded_successfully = False
    config_data_loaded = None # To store the config from either path

    if os.path.exists(hf_model_dir) and os.path.isdir(hf_model_dir) and \
       os.path.exists(safetensors_path) and os.path.exists(config_json_path):
        
        if verbose:
            print(f"Attempting to load tokenizer '{tokenizer_id}' from Hugging Face Hub format (safetensors & config.json)...")
        try:
            with open(config_json_path, 'r') as f:
                config_data_dict = json.load(f)
            config_data_loaded = config_data_dict # Store for return

            is_divae = False
            if config_data_dict.get("architectures"):
                if "DiVAE" in config_data_dict["architectures"][0]:
                    is_divae = True
            elif "divae" in tokenizer_id.lower() or "DiVAE" in tokenizer_id:
                 is_divae = True
            
            model_config_for_instantiation = config_data_dict 

            if is_divae:
                if verbose: print(f"Instantiating DiVAE for {tokenizer_id} from HF config...")
                model = DiVAE(config=model_config_for_instantiation)
            else:
                if verbose: print(f"Instantiating VQVAE for {tokenizer_id} from HF config...")
                model = VQVAE(config=model_config_for_instantiation)

            if verbose: print(f"Loading state_dict from {safetensors_path}...")
            state_dict = load_safetensors_file(safetensors_path, device='cpu')
            
            if encoder_only:
                keys_to_remove = [k for k in state_dict.keys() if 'decoder' in k or 'post_quant_proj' in k or 'post_quant_conv' in k]
                for k in keys_to_remove:
                    del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            if verbose:
                print(f"Successfully loaded tokenizer '{tokenizer_id}' from safetensors.")
                print(msg)
            
            model_loaded_successfully = True
            return model.to(device).eval(), config_data_loaded

        except Exception as e:
            if verbose:
                print(f"Error loading tokenizer '{tokenizer_id}' from Hugging Face Hub format: {e}")
            if return_None_on_fail:
                return None, None
            print("Fallback to .pth loading mechanism...")
    # --- End of NEW LOGIC ---


    # --- Start of ORIGINAL-LIKE LOGIC for .pth (adapted) ---
    if not model_loaded_successfully:
        if return_None_on_fail and not os.path.exists(os.path.join(tokenizers_root, f'{tokenizer_id}.pth')):
            return None, None
        
        if verbose:
            print(f'Loading tokenizer {tokenizer_id} from .pth ... ', end='')
        
        pth_file_path = os.path.join(tokenizers_root, f'{tokenizer_id}.pth')
        try:
            ckpt = torch.load(pth_file_path, map_location='cpu', weights_only=False)
        except FileNotFoundError as e:
            if return_None_on_fail:
                if verbose: print(f"File not found: {pth_file_path}")
                return None, None
            else:
                raise e 

        if not (isinstance(ckpt, dict) and 'args' in ckpt and 'model' in ckpt):
            error_msg = f"Checkpoint .pth for '{tokenizer_id}' has an unexpected structure. Expected a dict with 'args' and 'model' keys."
            if return_None_on_fail:
                if verbose: print(error_msg)
                return None, None
            else:
                raise ValueError(error_msg)

        args_from_ckpt = ckpt['args'] 
        config_data_loaded = args_from_ckpt # Store for return

        def get_arg_val(args_obj, key, default=None):
            if isinstance(args_obj, dict): return args_obj.get(key, default)
            return getattr(args_obj, key, default)

        def setattr_func(args_obj, key, val):
            if val is None: return # Do not set if val is None, to avoid overriding with None if key not present
            if isinstance(args_obj, dict): args_obj[key] = val
            else: setattr(args_obj, key, val)

        domain_value = get_arg_val(args_from_ckpt, 'domain', '')
        if 'CLIP' in domain_value or 'DINO' in domain_value or 'ImageBind' in domain_value:
            setattr_func(args_from_ckpt, 'patch_proj', False)
        elif 'sam' in domain_value:
            mask_size_val = get_arg_val(args_from_ckpt, 'mask_size')
            setattr_func(args_from_ckpt, 'input_size_min', mask_size_val)
            setattr_func(args_from_ckpt, 'input_size_max', mask_size_val)
            setattr_func(args_from_ckpt, 'input_size', mask_size_val)
        
        setattr_func(args_from_ckpt, 'quant_type', get_arg_val(args_from_ckpt, 'quantizer_type'))
        setattr_func(args_from_ckpt, 'enc_type', get_arg_val(args_from_ckpt, 'encoder_type'))
        setattr_func(args_from_ckpt, 'dec_type', get_arg_val(args_from_ckpt, 'decoder_type'))
        image_size_val = get_arg_val(args_from_ckpt, 'input_size') or get_arg_val(args_from_ckpt, 'input_size_max')
        setattr_func(args_from_ckpt, 'image_size', image_size_val)
        setattr_func(args_from_ckpt, 'image_size_enc', get_arg_val(args_from_ckpt, 'input_size_enc'))
        setattr_func(args_from_ckpt, 'image_size_dec', get_arg_val(args_from_ckpt, 'input_size_dec'))
        setattr_func(args_from_ckpt, 'image_size_sd', get_arg_val(args_from_ckpt, 'input_size_sd'))
        setattr_func(args_from_ckpt, 'ema_decay', get_arg_val(args_from_ckpt, 'quantizer_ema_decay'))
        setattr_func(args_from_ckpt, 'enable_xformer', get_arg_val(args_from_ckpt, 'use_xformer'))
        
        state_dict_model_pth = ckpt['model']
        if 'cls_emb.weight' in state_dict_model_pth:
            n_labels_val, n_channels_val = state_dict_model_pth['cls_emb.weight'].shape
            setattr_func(args_from_ckpt, 'n_labels', n_labels_val)
            setattr_func(args_from_ckpt, 'n_channels', n_channels_val)
        elif 'encoder.linear_in.weight' in state_dict_model_pth:
            setattr_func(args_from_ckpt, 'n_channels', state_dict_model_pth['encoder.linear_in.weight'].shape[1])
        elif 'encoder.proj.weight' in state_dict_model_pth:
            setattr_func(args_from_ckpt, 'n_channels', state_dict_model_pth['encoder.proj.weight'].shape[1])
        
        setattr_func(args_from_ckpt, 'sync_codebook', False)
        
        model_type_class = None
        model_state_to_load = state_dict_model_pth

        if encoder_only:
            model_type_class = VQ 
            model_state_to_load = {k: v for k, v in state_dict_model_pth.items() if 'decoder' not in k and 'post_quant_proj' not in k and 'post_quant_conv' not in k}
        else:
            model_type_str = get_arg_val(args_from_ckpt, 'model_type')
            if not model_type_str:
                if any(['controlnet' in k for k in state_dict_model_pth.keys()]):
                    model_type_str = 'VQControlNet'
                elif get_arg_val(args_from_ckpt, 'beta_schedule') is not None:
                    model_type_str = 'DiVAE'
                else:
                    model_type_str = 'VQVAE'
                setattr_func(args_from_ckpt, 'model_type', model_type_str)
            model_type_class = getattr(sys.modules[__name__], model_type_str)
        
        instantiation_args = args_from_ckpt if isinstance(args_from_ckpt, dict) else vars(args_from_ckpt)
        try:
            # Try instantiating with config= first, as this seems to be a pattern in ml-4m for VQVAE/DiVAE
            model = model_type_class(config=args_from_ckpt) 
        except TypeError: # If config= is not right or type is an issue, try unpacking
            try:
                model = model_type_class(**instantiation_args)
            except Exception as e_init:
                if verbose: print(f"Failed to instantiate {model_type_class.__name__} with provided args: {e_init}")
                raise e_init
                
        msg = model.load_state_dict(model_state_to_load, strict=False)
        if verbose:
            print(" done.") 
            print(msg)

        return model.to(device).eval(), config_data_loaded # config_data_loaded is args_from_ckpt here
    # --- End of ORIGINAL-LIKE LOGIC --- 
