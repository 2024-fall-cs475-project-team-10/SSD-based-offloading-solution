import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from layer_searching.searching import LayerSkippingSearching
from utils import preprocess_json_searching, print_args, get_args_parser_searching
import argparse
import json

torch.nn.Linear.reset_parameters = lambda x: None

def get_prompts(hf_cache_dir:str, length:int = 4):
    cache_dir = f"{hf_cache_dir}.searching_length_{length}.jsonl"
    prompts = None
    try:
        prompts = load_dataset("json", data_files=cache_dir)["train"].to_list()
        print(f"{cache_dir} loaded")
    except:
        prompts = []
        xsum = load_dataset("EdinburghNLP/xsum", cache_dir=hf_cache_dir).shuffle(4242)
        cnn = load_dataset("abisee/cnn_dailymail", "3.0.0",
                        cache_dir=hf_cache_dir).shuffle(4242)

        for i in range(length):
            item = xsum["train"][i + 100]
            prompt = []
            prompt.append({
                "role": "user",
                "content": "Article: " + item["document"] + "\nSummary:"
            })
            prompt.append({
                "role": "assistant",
                "content": item["summary"].replace("\n", "")
            })
            item = xsum["train"][i]
            prompt.append({
                "role": "user",
                "content": "Article: " + item["document"] + "\nSummary:"
            })
            prompts.append({
                "message": prompt,
                "answer": item["summary"].replace("\n", "")
            })
        for i in range(length):
            item = cnn["train"][i + 100]
            prompt = []
            prompt.append({
                "role": "user",
                "content": "Article: " + item["article"] + "\nSummary:"
            })
            prompt.append({
                "role": "assistant",
                "content": item["highlights"].replace("\n", "")
            })
            item = cnn["train"][i]
            prompt.append({
                "role": "user",
                "content": "Article: " + item["article"] + "\nSummary:"
            })
            prompts.append({
                "message": prompt,
                "answer": item["highlights"].replace("\n", "")
            })
        with open(cache_dir, "w") as f:
            for prompt in prompts:
                f.write(json.dumps(prompt, ensure_ascii=False) + "\n")
        print(f"{cache_dir} created")
    return prompts

def main(hf_cache_dir, input_json_path, offloading_type):
    model_id, num_offload_layers, skip_attn_layers, skip_mlp_layers, search_iteration, csv_filename, json_filename = preprocess_json_searching(input_json_path)
    probes = []
    probes.append(skip_attn_layers if skip_attn_layers is not None else [])
    probes.append(skip_mlp_layers if skip_mlp_layers is not None else [])
    print_args(
        type="Searching",
        hf_cache_dir=hf_cache_dir,
        csv_filename=csv_filename,
        json_filename=json_filename,
        model_id=model_id,
        num_offload_layers=num_offload_layers,
        search_iteration=search_iteration,
        probed_skip_attention_layers=skip_attn_layers,
        probed_skip_mlp_layers=skip_mlp_layers,
        offloading_type=offloading_type
    )

    cuda_devices = [i for i in range(torch.cuda.device_count())]
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=hf_cache_dir)

    prompts = get_prompts(hf_cache_dir=hf_cache_dir, length=4)

    naive_offloading = offloading_type == "naive"
    layer_searching = LayerSkippingSearching(
        model_id,
        tokenizer,
        prompts,
        hf_cache_dir=hf_cache_dir,
        cuda_devices=cuda_devices,
        num_offload_layers=num_offload_layers,
        generate_fn="ssd",
        evaluate_config={"max_new_tokens": 32},
        csv_filename=csv_filename,
        naive_offloading=naive_offloading,
    )

    if len(probes[0]) > 0 and len(probes[1]) > 0:
        layer_searching.probe(attn_skip_layers=probes[0], mlp_skip_layers=probes[1])

    skip_attn_layers, skip_mlp_layers = layer_searching.search(search_iteration)
    with open(json_filename, "w") as outfile:
        result = {
            "model_id": model_id,
            "num_offload_layers": num_offload_layers,
            "search_iteration": search_iteration,
            "csv_filename": csv_filename,
            "json_filename": json_filename,
            "skip_attn_layers": skip_attn_layers,
            "skip_mlp_layers": skip_mlp_layers,
        }
        json.dump(result, outfile, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("python3 search_layers.py", parents=[get_args_parser_searching("Searching Script")])
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)