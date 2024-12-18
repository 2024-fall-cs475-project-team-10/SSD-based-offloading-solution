import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from models.ssd_pipeline import SSDPipeline
from models.models import MODELS
from utils import set_device_map, preprocess_json_evaluation, postprocess_json, get_args_parser_evaluation, print_args, trace_model_memory, trace_peak_memory, get_memory_info_gb
from utils.dataset_evaluator import DatasetEvaluator
import argparse
import gc

TYPE = "Baseline"

def main(hf_cache_dir, input_json_path, output_json_path, task, n_shot, length):
    model_id, num_offload_layers, skip_attn_layers, skip_mlp_layers, _ = preprocess_json_evaluation(input_json_path)
    print_args(
        type=TYPE,
        hf_cache_dir=hf_cache_dir,
        input_json_path=input_json_path,
        output_json_path=output_json_path,
        task=task,
        n_shot=n_shot,
        length=length,
        model_id=model_id,
        num_offload_layers=num_offload_layers
    )
    gc.collect()
    torch.cuda.empty_cache()

    cuda_devices = [i for i in range(torch.cuda.device_count())]
    device_map = set_device_map(
        model_id=model_id,
        skipped_attn_layers=[],
        skipped_mlp_layers=[],
        max_skip_attn_layers=num_offload_layers,
        max_skip_mlp_layers=num_offload_layers,
        cuda_devices=cuda_devices,
    )

    ####### Section 1. Set up #######
    torch.random.manual_seed(0)
    evaluator = DatasetEvaluator(hf_cache_dir=hf_cache_dir, task=task, n_shot=n_shot, length=length)
    with trace_model_memory():
        model = MODELS(model_id).from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype="auto",
            cache_dir=hf_cache_dir,
        ).eval()

        for parameter in model.parameters():
            parameter.requires_grad = False

        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=hf_cache_dir)

    pipe = SSDPipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        generate_fn="base",
    )

    generation_args = {
        "max_new_tokens": 500,
        "temperature": 0.0,
        "early_stop": True,
        "do_sample": False,
    }

    ####### Section 2. GPU Warm up #######
    print("===== Warm up =====")
    output = pipe(evaluator.warmup_data(), **generation_args)
    print(output['completion'])

    ####### Section 3. Load data and Inference -> Performance evaluation part #######
    print("===== Inference =====")
    with trace_peak_memory():
        outs = pipe(evaluator.data(), **generation_args)
    model_memory, peak_memory = get_memory_info_gb()
    ####### Section 4. Accuracy #######
    result = evaluator.evaluate(outs['completion'])

    print("===== Perf result =====")
    postprocess_json(
        input_file=input_json_path,
        output_file=output_json_path,
        type=TYPE,
        **result,
        elapsed_time=outs['time'],
        generated_tokens=outs['generated_tokens_length'],
        model_memory=model_memory,
        peak_memory=peak_memory,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("python3 test_script_base.py", parents=[get_args_parser_evaluation("Base Evaluation Script")])
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)