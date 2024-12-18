import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from models.ssd_pipeline import SSDPipeline
from models.models import MODELS
from utils import set_device_map
from dataset_evaluator import DatasetEvaluator

hf_cache_dir = '.hf_cache/'

model_ids = [
    "microsoft/Phi-3-medium-4k-instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

model_id = model_ids[2]
def memory_size(param):
    return (
        round((param.shape.numel() * param.element_size() / 1024 / 1024) * 1000) / 1000
    )


def get_model_param_size(model):
    total_size = 0
    for name, param in model.named_parameters():
        total_size += param.numel()
    return total_size


def get_model_memory_size(model):
    total_size = 0
    for name, param in model.named_parameters():
        total_size += param.numel() * param.element_size()
    return total_size


def print_model_hierarchy(model, depth=0):
    for name, module in model.named_children():
        print("  " * depth + f"|-- {name}: {module.__class__.__name__}")
        for param_name, param in module.named_parameters(recurse=False):
            print(
                "  " * (depth + 1)
                + f"|-- Param: {param_name} | Shape: {list(param.shape)} ({memory_size(param)}MB)"
            )
        if list(module.children()):
            print_model_hierarchy(module, depth + 1)


model = MODELS(model_id).from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    cache_dir=hf_cache_dir,
)

print("Model Hierarchy:")
print_model_hierarchy(model)

print(model.config)
print("Total Parameters: {:.2f}B".format(get_model_param_size(model) / 1e9))
print("Total Memory: {:.2f}GB".format(get_model_memory_size(model) / 1024 ** 3))