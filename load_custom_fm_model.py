import torch
import yaml
from pathlib import Path
from functools import partial
import torch.nn as nn
import argparse # Ajout pour torch.load si Namespace doit être autorisé (bien que weights_only=False soit la solution ici)

# --- Imports spécifiques à ml_4m ---
from fourm.models.fm import FourM  # Classe de base du modèle
from fourm.data.modality_info import MODALITY_INFO
from fourm.models.generate import GenerationSampler  # Correction: GenerationSampler est dans models/generate.py
from fourm.models.fm_utils import LayerNorm # Import direct si utilisé par la config

# --- Configuration Utilisateur --- 
TRAIN_CONFIG_PATH = 'external_libs/ml_4m/cfgs/default/4m/models/specialized/4m-b_mod7_500b--nutri_snap.yaml'
CHECKPOINT_PATH = '/work/com-304/nutri-snap/output/nutrisnap_finetune/checkpoint-final.pth'
PROJECT_ROOT = Path('.') 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

actual_train_config_path = PROJECT_ROOT / TRAIN_CONFIG_PATH
with open(actual_train_config_path, 'r') as f:
    train_cfg_dict = yaml.safe_load(f)
train_cfg = AttrDict(train_cfg_dict)
print(f"Training config '{actual_train_config_path}' loaded.")

actual_data_config_path = PROJECT_ROOT / train_cfg.data_config
with open(actual_data_config_path, 'r') as f:
    data_cfg_dict = yaml.safe_load(f)
data_cfg = AttrDict(data_cfg_dict)

active_domains = set(data_cfg.train.datasets.n5k.in_domains.split('-')) | \
                 set(data_cfg.train.datasets.n5k.out_domains.split('-'))
print(f"Active domains for the model: {active_domains}")

# ---- START DEBUG ----
print("DEBUG: Keys available in global MODALITY_INFO at runtime:")
print(sorted(list(MODALITY_INFO.keys())))
# ---- END DEBUG ----

effective_modality_info = {
    domain: MODALITY_INFO[domain]
    for domain in active_domains if domain in MODALITY_INFO
}
print("Effective modality_info prepared.")
if not all(domain in effective_modality_info for domain in active_domains):
    missing_in_eff = active_domains - set(effective_modality_info.keys())
    print(f"ERROR: Some active_domains are not in effective_modality_info after filtering MODALITY_INFO: {missing_in_eff}")
    print("This usually means these domains are missing or misspelled in your main MODALITY_INFO definition in fourm/data/modality_info.py")
    # exit() # Décommentez pour arrêter si des erreurs critiques sont détectées ici

encoder_embeddings = {}
for mod_name in active_domains:
    if mod_name not in effective_modality_info:
        print(f"ERROR: Modality '{mod_name}' is in active_domains but NOT in effective_modality_info. Cannot create encoder embedding. Check MODALITY_INFO definition.")
        continue
    info = effective_modality_info[mod_name]
    if info.get("encoder_embedding", None) is not None:
        emb_fn = info["encoder_embedding"]
        emb_args = {}
        if info["type"] == "img" or 'patch_size' in info:
            emb_args['patch_size'] = info.get('patch_size', train_cfg.patch_size)
            emb_args['image_size'] = info.get('input_size', train_cfg.input_size)
            if 'num_channels' in info: emb_args['num_channels'] = info['num_channels']
            if 'vocab_size' in info: emb_args['vocab_size'] = info['vocab_size']
        elif info["type"] in ["seq", "seq_token"]:
            emb_args['vocab_size'] = info['vocab_size']
            emb_args['max_length'] = info.get('max_tokens', info.get('max_length')) 
            if 'padding_idx' in info: emb_args['padding_idx'] = info['padding_idx']
        elif info["type"] == "seq_emb":
            emb_args['max_length'] = info.get('max_tokens', info.get('max_length'))
            emb_args['orig_emb_dim'] = info['orig_emb_dim']
            if 'use_bottleneck' in info: emb_args['use_bottleneck'] = info['use_bottleneck']
            if 'bottleneck_dim' in info: emb_args['bottleneck_dim'] = info['bottleneck_dim']
        try:
            encoder_embeddings[mod_name] = emb_fn(**emb_args)
        except Exception as e:
            print(f"ERROR creating encoder_embedding for {mod_name} with fn {emb_fn} and args {emb_args}: {e}")

decoder_embeddings = {}
for mod_name in active_domains:
    if mod_name not in effective_modality_info:
        print(f"ERROR: Modality '{mod_name}' is in active_domains but NOT in effective_modality_info. Cannot create decoder embedding. Check MODALITY_INFO definition.")
        continue
    info = effective_modality_info[mod_name]
    if info.get("decoder_embedding", None) is not None:
        emb_fn = info["decoder_embedding"]
        emb_args = {}
        if info["type"] == "img" or 'patch_size' in info:
            emb_args['patch_size'] = info.get('patch_size', train_cfg.patch_size)
            emb_args['image_size'] = info.get('input_size', train_cfg.input_size)
            emb_args['vocab_size'] = info['vocab_size']
        elif info["type"] in ["seq", "seq_token"]:
            emb_args['vocab_size'] = info['vocab_size']
            emb_args['max_length'] = info.get('max_tokens', info.get('max_length'))
            if 'padding_idx' in info: emb_args['padding_idx'] = info['padding_idx']
        emb_args['share_embedding'] = False 
        try:
            decoder_embeddings[mod_name] = emb_fn(**emb_args)
        except Exception as e:
            print(f"ERROR creating decoder_embedding for {mod_name} with fn {emb_fn} and args {emb_args}: {e}")

print(f"Constructed {len(encoder_embeddings)} encoder embeddings: {sorted(list(encoder_embeddings.keys()))}")
print(f"Constructed {len(decoder_embeddings)} decoder embeddings: {sorted(list(decoder_embeddings.keys()))}")

is_swiglu = "swiglu" in train_cfg.model
is_nobias = "nobias" in train_cfg.model
is_qknorm = "qknorm" in train_cfg.model
qkv_b = not is_nobias
proj_b = not is_nobias
mlp_b = not is_nobias
if hasattr(train_cfg, 'act_layer') and train_cfg.act_layer:
    act_l = getattr(nn, train_cfg.act_layer)
else: act_l = nn.SiLU if is_swiglu else nn.GELU
norm_bias_val = train_cfg.norm_bias if hasattr(train_cfg, 'norm_bias') else (not is_nobias)
norm_l = partial(LayerNorm, eps=1e-6, bias=norm_bias_val)
gated_m = is_swiglu
qk_n = is_qknorm

fm_constructor_args = {
    'encoder_embeddings': encoder_embeddings,
    'decoder_embeddings': decoder_embeddings,
    'modality_info': effective_modality_info,
    'mlp_ratio': train_cfg.get('mlp_ratio', 4.0),
    'qkv_bias': train_cfg.get('qkv_bias', qkv_b),
    'proj_bias': train_cfg.get('proj_bias', proj_b),
    'mlp_bias': train_cfg.get('mlp_bias', mlp_b),
    'act_layer': act_l,
    'norm_layer': norm_l,
    'gated_mlp': train_cfg.get('gated_mlp', gated_m),
    'qk_norm': train_cfg.get('qk_norm', qk_n),
    'drop_path_rate_encoder': train_cfg.get('drop_path_rate_encoder', 0.0),
    'drop_path_rate_decoder': train_cfg.get('drop_path_rate_decoder', 0.0),
    'shared_drop_path': train_cfg.get('shared_drop_path', False),
    'decoder_causal_mask': train_cfg.get('decoder_causal_mask', False),
    'decoder_sep_mask': train_cfg.get('decoder_sep_mask', True),
    'num_register_tokens': train_cfg.get('num_register_tokens', 0),
    'use_act_checkpoint': train_cfg.get('use_act_checkpoint', False),
    'share_modality_embeddings': train_cfg.get('share_modality_embeddings', True)
}
model_parts = train_cfg.model.split('_')
model_size_map = {'tiny': 384, 'small': 512, 'base': 768, 'large': 1024, 'xlarge': 2048}
model_heads_map = {'tiny': 6, 'small': 8, 'base': 12, 'large': 16, 'xlarge': 32}
fm_constructor_args['dim'] = train_cfg.get('dim', model_size_map.get(model_parts[1], 768))
fm_constructor_args['num_heads'] = train_cfg.get('num_heads', model_heads_map.get(model_parts[1], 12))
fm_constructor_args['encoder_depth'] = train_cfg.get('encoder_depth', 12)
fm_constructor_args['decoder_depth'] = train_cfg.get('decoder_depth', 12)
for part in model_parts:
    if part.endswith('e') and part[:-1].isdigit(): fm_constructor_args['encoder_depth'] = int(part[:-1])
    if part.endswith('d') and part[:-1].isdigit(): fm_constructor_args['decoder_depth'] = int(part[:-1])
print(f"FourM constructor arguments inferred/set: {sorted(list(fm_constructor_args.keys()))}")

fm = FourM(**fm_constructor_args)
print(f"Model FourM instantiated.")

# --- Correction pour UnpicklingError ---
# Ajout de l'import et modification de torch.load
import pickle

# Solution 1: weights_only=False car le checkpoint est fiable
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

# Solution 2 (Alternative, si on voulait garder weights_only=True et que argparse.Namespace était le seul problème):
# from torch.serialization import add_safe_globals, safe_globals
# add_safe_globals([argparse.Namespace])
# OU avec un context manager:
# with safe_globals([argparse.Namespace]):
#     checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True)

print(f"Checkpoint '{CHECKPOINT_PATH}' loaded.")

state_dict_key = None
if 'model' in checkpoint: state_dict_key = 'model'
elif 'model_state_dict' in checkpoint: state_dict_key = 'model_state_dict'
elif 'state_dict' in checkpoint: state_dict_key = 'state_dict'
else:
    if all(isinstance(k, str) for k in checkpoint.keys()) and not any(k in ['epoch', 'optimizer_state_dict'] for k in checkpoint.keys()):
         state_dict_key = None
    else:
        raise KeyError(f"Cannot find model state_dict. Available keys in checkpoint: {list(checkpoint.keys())}")
actual_state_dict = checkpoint if state_dict_key is None else checkpoint[state_dict_key]
new_state_dict = {}
ddp_prefix = 'module.'
is_ddp_checkpoint = any(key.startswith(ddp_prefix) for key in actual_state_dict.keys())
if is_ddp_checkpoint:
    print("Checkpoint appears to be from DDP training, removing 'module.' prefix.")
    for k, v in actual_state_dict.items():
        name = k[len(ddp_prefix):] if k.startswith(ddp_prefix) else k
        new_state_dict[name] = v
else: new_state_dict = actual_state_dict

load_result = fm.load_state_dict(new_state_dict, strict=False)
print(f"Load state_dict result: missing_keys={load_result.missing_keys}, unexpected_keys={load_result.unexpected_keys}")

fm = fm.eval().to(device)
print(f"Weights loaded into model. Model is on {device} and in eval mode.")

sampler = GenerationSampler(fm)
print("GenerationSampler initialized successfully.")

print("\n--- Ready for Generation ---") 