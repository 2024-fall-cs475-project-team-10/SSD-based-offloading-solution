from typing import List, Dict
from models.models import MODELS_MAX_LAYER, MODELS
from models.modeling_phi3 import Phi3ForCausalLM
from models.modeling_llama import LlamaForCausalLM
import json
import argparse
from contextlib import contextmanager
import torch

def set_device_map(
    model_id: str,
    skipped_attn_layers: List[int] = [],
    skipped_mlp_layers: List[int] = [],
    max_skip_attn_layers: int = 4,
    max_skip_mlp_layers: int = 4,
    cuda_devices: List[int] = [0],
) -> Dict[str, int | str]:
    max_layers = MODELS_MAX_LAYER(model_id)
    skipped_attn_layers = sorted(skipped_attn_layers, reverse=True)
    skipped_mlp_layers = sorted(skipped_mlp_layers, reverse=True)
    skipped_attn_layers = [x for x in skipped_attn_layers if x < max_layers]
    skipped_mlp_layers = [x for x in skipped_mlp_layers if x < max_layers]
    if len(skipped_attn_layers) < max_skip_attn_layers:
        skipped_attn_layers += [
            i for i in range(max_layers - 1, -1, -1) if i not in skipped_attn_layers
        ]
    skipped_attn_layers = skipped_attn_layers[:max_skip_attn_layers]

    if len(skipped_mlp_layers) < max_skip_mlp_layers:
        skipped_mlp_layers += [
            i for i in range(max_layers - 1, -1, -1) if i not in skipped_mlp_layers
        ]
    skipped_mlp_layers = skipped_mlp_layers[:max_skip_mlp_layers]

    if MODELS(model_id) == Phi3ForCausalLM:
        device_map_cuda, device_map_auto = _device_mapping_phi(max_layers, skipped_attn_layers, skipped_mlp_layers)
    elif MODELS(model_id) == LlamaForCausalLM:
        device_map_cuda, device_map_auto = _device_mapping_llama(max_layers, skipped_attn_layers, skipped_mlp_layers)
    else:
        raise NotImplementedError("지원하지 않는 모델")
        
    device_map = {}
    chunk_size = len(device_map_cuda) // len(cuda_devices)
    remaining = len(device_map_cuda) % len(cuda_devices)
    start_idx = 0
    for i, cuda_device in enumerate(cuda_devices):
        end_idx = start_idx + chunk_size + (1 if i < remaining else 0)
        for name in device_map_cuda[start_idx:end_idx]:
            device_map[name] = cuda_device
        start_idx = end_idx

    for name in device_map_auto:
        device_map[name] = "cpu"
    return device_map

def _device_mapping_phi(max_layers: int, skipped_attn_layers: List[int], skipped_mlp_layers: List[int]):
    device_map_cuda = ["model.embed_tokens"]
    device_map_auto = []
    for i in range(max_layers):
        device_map_cuda.append(f"model.layers.{i}.input_layernorm")
        if i in skipped_attn_layers:
            device_map_auto.append(f"model.layers.{i}.self_attn")
        else:
            device_map_cuda.append(f"model.layers.{i}.self_attn")
        device_map_cuda.append(f"model.layers.{i}.post_attention_layernorm")
        if i in skipped_mlp_layers:
            device_map_auto.append(f"model.layers.{i}.mlp")
        else:
            device_map_cuda.append(f"model.layers.{i}.mlp")
    device_map_cuda.append("model.norm")
    device_map_cuda.append("lm_head")
    return device_map_cuda, device_map_auto

def _device_mapping_llama(max_layers: int, skipped_attn_layers: List[int], skipped_mlp_layers: List[int]):
    device_map_cuda = ["model.embed_tokens"]
    device_map_auto = []
    for i in range(max_layers):
        device_map_cuda.append(f"model.layers.{i}.input_layernorm")
        if i in skipped_attn_layers:
            device_map_auto.append(f"model.layers.{i}.self_attn")
        else:
            device_map_cuda.append(f"model.layers.{i}.self_attn")
        device_map_cuda.append(f"model.layers.{i}.post_attention_layernorm")
        if i in skipped_mlp_layers:
            device_map_auto.append(f"model.layers.{i}.mlp")
        else:
            device_map_cuda.append(f"model.layers.{i}.mlp")
    device_map_cuda.append("model.norm")
    device_map_cuda.append("lm_head")
    return device_map_cuda, device_map_auto

def preprocess_json_evaluation(input_file):
    keys = ["model_id", "num_offload_layers", "skip_attn_layers", "skip_mlp_layers", "search_iteration"]
    output = []
    with open(input_file, "r") as file:
        data = json.load(file)
        for key in keys:
            output.append(data.get(key))
    return output

def preprocess_json_searching(input_file):
    keys = ["model_id", "num_offload_layers", "skip_attn_layers", "skip_mlp_layers", "search_iteration", "csv_filename", "json_filename"]
    output = []
    with open(input_file, "r") as file:
        data = json.load(file)
        for key in keys:
            output.append(data.get(key))
    return output

def postprocess_json(input_file, output_file, **kwargs):
    input_json = {}
    with open(input_file, "r") as file:
        data = json.load(file)
        for key, value in data.items():
            input_json[key] = value
    for key, item in kwargs.items():
        item_formatted = "{:.3f}".format(item) if isinstance(item, (float)) else item
        print(f"{key.replace('_', ' ').title()}: {item_formatted}")
    kwargs["inputs"] = input_json
    with open(output_file, "w") as file:
        json.dump(kwargs, file, indent=2)

def get_args_parser_evaluation(name: str):
    parser = argparse.ArgumentParser(name, add_help=False)
    parser.add_argument('-c', "--hf_cache_dir", default='.hf_cache/', metavar='.hf_cache/', type=str, help="huggingface의 cache dir입니다.")
    parser.add_argument('-i', "--input_json_path", default='config/debug/debug.json', metavar='config/debug/debug.json', type=str, help="searching이 완료된 json config을 넣어주세요.")
    parser.add_argument('-o', "--output_json_path", default="config/debug/debug_result.json", metavar='config/debug/debug_result.json', type=str, help="evaluation이 완료된 결과를 저장할 위치. json 형식")
    parser.add_argument('-t', "--task", type=str, choices=['xsum', 'cnn_dailymail'], metavar='xsum', help="할 task. xsum, cnn_dailymail 중 선택", required=True)
    parser.add_argument('-n', "--n_shot", type=int, default=1, metavar=1, help="evaluation 중 몇 샷을 할건지")
    parser.add_argument('-l', "--length", type=int, default=100, metavar=100, help="evaluation 중 몇개의 data를 사용할 것인지")
    return parser

def get_args_parser_searching(name: str):
    parser = argparse.ArgumentParser(name, add_help=False)
    parser.add_argument('-c', "--hf_cache_dir", default='.hf_cache/', metavar='.hf_cache/', type=str, help="huggingface의 cache dir입니다.")
    parser.add_argument('-i', "--input_json_path", metavar='config/debug/debug.json', type=str, help="searching에 사용할 json config을 넣어주세요.", required=True)
    parser.add_argument('-t', "--offloading_type", metavar='optimal', type=str, choices=['optimal', 'naive'], help="decoding offloading policy")
    return parser

def print_args(**kwargs):
    print("===== Configs =====")
    for key, item in kwargs.items():
        key_formatted = key.replace("_", " ")
        if isinstance(item, (list, tuple)) and len(item) > 0:
            item = ", ".join([str(i) for i in item] if not isinstance(item[0], str) else item)
        print(f"{key_formatted}: {item}")
    print("===================")

_model_memory = 0
_initial_memory = 0
_peak_memory = 0

@contextmanager
def trace_model_memory():
    global _model_memory, _initial_memory
    try:
        initial_model_memory = 0
        _model_memory = 0
        _initial_memory = 0
        torch.cuda.reset_peak_memory_stats()
        initial_model_memory += torch.cuda.memory_allocated()
        _initial_memory = initial_model_memory
        yield
    finally:
        final_model_memory = 0
        final_model_memory += torch.cuda.memory_allocated()
        _model_memory = final_model_memory - _initial_memory

@contextmanager
def trace_peak_memory():
    global _peak_memory, _initial_memory
    try:
        _peak_memory = 0
        torch.cuda.reset_peak_memory_stats()
        yield
    finally:
        final_peak_memory = 0
        final_peak_memory += torch.cuda.max_memory_allocated()
        _peak_memory = final_peak_memory - _initial_memory

def get_memory_info_gb():
    global _model_memory, _peak_memory
    return _model_memory / 1024 ** 3, _peak_memory / 1024 ** 3