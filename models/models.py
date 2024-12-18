from models.modeling_phi3 import Phi3ForCausalLM
from models.modeling_llama import LlamaForCausalLM

def MODELS(model_id: str):
    if model_id == "microsoft/Phi-3-mini-4k-instruct":
        return Phi3ForCausalLM
    elif model_id == "microsoft/Phi-3-medium-4k-instruct":
        return Phi3ForCausalLM
    elif model_id == "meta-llama/Llama-3.1-8B-Instruct":
        return LlamaForCausalLM
    elif model_id == "meta-llama/Llama-3.2-3B-Instruct":
        return LlamaForCausalLM
    raise NotImplementedError("지원하지 않는 모델")

def MODELS_MAX_LAYER(model_id: str):
    if model_id == "microsoft/Phi-3-mini-4k-instruct":
        return 32
    elif model_id == "microsoft/Phi-3-medium-4k-instruct":
        return 40
    elif model_id == "meta-llama/Llama-3.1-8B-Instruct":
        return 32
    elif model_id == "meta-llama/Llama-3.2-3B-Instruct":
        return 28
    raise NotImplementedError("지원하지 않는 모델")

MODEL_DICT = {
    "microsoft/Phi-3-mini-4k-instruct": Phi3ForCausalLM,
    "microsoft/Phi-3-medium-4k-instruct": Phi3ForCausalLM,
    "meta-llama/Llama-3.1-8B-Instruct": LlamaForCausalLM,
    "meta-llama/Llama-3.2-3B-Instruct": LlamaForCausalLM,
}