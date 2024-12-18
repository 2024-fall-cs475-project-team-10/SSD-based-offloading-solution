import argparse
import time
import sys
from scripts.utils import redirect_output, redirect_output_subproccess, validate_config_json_dir, save_config_json, validate_process
import os
import subprocess
import json
from utils import print_args
from copy import deepcopy

def get_args_parse(name: str):
    parser = argparse.ArgumentParser(name, add_help=False)
    parser.add_argument('-c', "--hf_cache_dir", default='.hf_cache/', metavar='.hf_cache/', type=str, help="huggingface의 cache dir입니다.")
    parser.add_argument('-i', "--json_dir", metavar='search_models_a100.jsonl', type=str, help="searching이 필요한 jsonl", required=True)
    return parser

def main(hf_cache_dir, json_dir):
    validate_process(2)
    configs = validate_config_json_dir(json_dir)
    for i in range(len(configs)):
        config = deepcopy(configs[i])
        is_motivation = config.pop("is_motivation")
        is_done = config.pop("is_done")
        if not is_done:
            if is_motivation:
                script_type = "Motivation Script"
                with redirect_output("search_script_log.out", script_type, "", True):
                    motivation_searching_and_evaluation(script_type=script_type, hf_cache_dir=hf_cache_dir, **config)
            else:
                script_type = "Train and Evaluation Script"
                with redirect_output("search_script_log.out", script_type, "", True):
                    searching_and_evaluation(script_type=script_type, hf_cache_dir=hf_cache_dir, **config)
            configs[i]["is_done"] = True
            save_config_json(configs, json_dir)
    return

def motivation_searching_and_evaluation(script_type, hf_cache_dir, config_dir, model_id, search_iter, n_shot, length, num_iter, num_offloads):
    func_types = ['base', 'ssd_naive', 'ssd']
    tasks = ['xsum', 'cnn_dailymail']

    _config_dir = os.path.join(config_dir, "motivation")
    print_args(type=script_type, hf_cache_dir=hf_cache_dir, config_dir=_config_dir, model_id=model_id, num_offloads=num_offloads, search_iter=search_iter, n_shot=n_shot, length=length, num_iter=num_iter)
    num_offload = 0
    config_dir, input_data_json, search_result_json = make_search_config(_config_dir, model_id, search_iter, num_offload)
    if not os.path.isfile(search_result_json):
        search_layers(hf_cache_dir, input_data_json, os.path.join(config_dir, "searching.out"), config_dir)
    configs = duplicate_search_result_json(_config_dir, search_result_json, search_iter, num_offloads)
    for config in configs:
        config_dir = config["config_dir"]
        search_result_json = config["search_result_json"]
        num_offload = config["num_offload"]
        for i in range(num_iter):
            for func_type in func_types:
                for task in tasks:
                    evaluate(hf_cache_dir, config_dir, search_result_json, func_type, task, n_shot, length, i + 1, True)
        save_logging_to_csv(config_dir, "motivation", model_id, num_offload, func_types, tasks, n_shot, length, num_iter)

def searching_and_evaluation(script_type, hf_cache_dir, config_dir, model_id, search_iter, n_shot, length, num_iter, num_offload, search_offload_policy):
    func_types = ['base', 'ssd_naive', 'ssd']
    tasks = ['xsum', 'cnn_dailymail']

    print_args(type=script_type, hf_cache_dir=hf_cache_dir, config_dir=config_dir, model_id=model_id, num_offload=num_offload, search_iter=search_iter, n_shot=n_shot, length=length, num_iter=num_iter, search_offload_policy=search_offload_policy)
    config_dir, input_data_json, search_result_json = make_search_config(config_dir, model_id, search_iter, num_offload, search_offload_policy)
    log_name = f"searching{'' if search_offload_policy == 'optimal' else '-naive_offload'}.out"
    if not os.path.isfile(search_result_json):
        search_layers(hf_cache_dir, input_data_json, os.path.join(config_dir, log_name), config_dir, search_offload_policy)
    if search_offload_policy=="optimal":
        for i in range(num_iter):
            for func_type in func_types:
                for task in tasks:
                    evaluate(hf_cache_dir, config_dir, search_result_json, func_type, task, n_shot, length, i + 1)
        save_logging_to_csv(config_dir, "evaluation", model_id, num_offload, func_types, tasks, n_shot, length, num_iter)

def evaluate(hf_cache_dir, config_dir, input_data_json, func_type, task, n_shot, length, num_iter, is_motivation=False):
    output_config_dir = os.path.join(config_dir, f"{func_type}_{task}_n_shot_{n_shot}_length_{length}")
    os.makedirs(output_config_dir, exist_ok=True)
    output_json_path = os.path.join(output_config_dir, f"iter_{num_iter}.json")
    output_log_path = os.path.join(output_config_dir, f"iter_{num_iter}.out")
    if not os.path.isfile(output_json_path):
        cmd_dict = {"hf_cache_dir": hf_cache_dir, "input_json_path": input_data_json, "output_json_path": output_json_path, "task": task, "n_shot": n_shot, "length": length}
        cmd = ["python3", "-m", f"test_script_{func_type}"]
        for key, value in cmd_dict.items():
            cmd.append(f"--{key}")
            cmd.append(str(value))
        redirect_output_subproccess(cmd, output_log_path, f"{'' if not is_motivation else 'Motivation '}Evaluation", f"{'' if not is_motivation else 'Motivation '}Evaluation: {output_json_path}")

def search_layers(hf_cache_dir, input_data_json, output_log_path, config_dir, search_offload_policy="optimal"):
    cmd_dict = {"hf_cache_dir": hf_cache_dir, "input_json_path": input_data_json, "offloading_type": search_offload_policy}
    cmd = ["python3", "-m", f"search_layers"]
    for key, value in cmd_dict.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))
    redirect_output_subproccess(cmd, output_log_path, "Searching", f"Searching: {config_dir}")

def save_logging_to_csv(config_dir, eval_config, model_id, num_offload, func_types, tasks, n_shot, length, num_iter):
    log_dir = os.path.join(config_dir, f"evaluation_n_shot_{n_shot}_length_{length}.csv")
    with open(log_dir, "w") as f:
        f.write(f"config,model_id,num_offload,type,task,latency(item/s),elapsed_time(s),length,rouge-1,rouge-2,rouge-l,rouge-lsum,model_memory(gb),peak_memory(gb),n_shot,num_iter\n")
    for func_type in func_types:
        for task in tasks:
            rouge_1, rouge_2, rouge_l, rouge_lsum, elapsed_time, model_memory, peak_memory = [], [], [], [], [], [], []
            for i in range(num_iter):
                output_json = os.path.join(config_dir, f"{func_type}_{task}_n_shot_{n_shot}_length_{length}", f"iter_{i + 1}.json")
                with open(output_json, "r") as file:
                    output = json.load(file)
                rouge_1.append(output["rouge-1"])
                rouge_2.append(output["rouge-2"])
                rouge_l.append(output["rouge-l"])
                rouge_lsum.append(output["rouge-lsum"])
                elapsed_time.append(output["elapsed_time"])
                model_memory.append(output["model_memory"])
                peak_memory.append(output["peak_memory"])
            rouge_1 = sum(rouge_1) / len(rouge_1)
            rouge_2 = sum(rouge_2) / len(rouge_2)
            rouge_l = sum(rouge_l) / len(rouge_l)
            rouge_lsum = sum(rouge_lsum) / len(rouge_lsum)
            elapsed_time = sum(elapsed_time) / len(elapsed_time)
            model_memory = sum(model_memory) / len(model_memory)
            peak_memory = sum(peak_memory) / len(peak_memory)
            with open(log_dir, "a") as f:
                f.write(f"{eval_config},{model_id},{num_offload},{func_type},{task},{elapsed_time / length},{elapsed_time},{length},{rouge_1},{rouge_2},{rouge_l},{rouge_lsum},{model_memory},{peak_memory},{n_shot},{num_iter}\n")

def make_search_config(config_dir, model_id, search_iter, num_offload, search_offload_policy):
    postfix = "" if search_offload_policy=="optimal" else "-naive_offload"
    config_dir = os.path.join(config_dir, f"offload_{num_offload}-search_{search_iter}")
    os.makedirs(config_dir, exist_ok=True)
    input_data_json = os.path.join(config_dir, f"search_config{postfix}.json")
    output_data_json = os.path.join(config_dir, f"skipped_layers{postfix}.json")
    with open(input_data_json, "w") as file:
        json.dump({
            "model_id": model_id,
            "num_offload_layers": num_offload,
            "search_iteration": search_iter,
            "csv_filename": os.path.join(config_dir, f'searching{postfix}.csv'),
            "json_filename": os.path.join(config_dir, f'skipped_layers{postfix}.json'),
        }, file, indent=4)
    return config_dir, input_data_json, output_data_json

def duplicate_search_result_json(config_dir, src_result_json, search_iter, num_offloads):
    src = None
    with open(src_result_json, "r") as file:
        src = json.load(file)
    search_result_jsons = []
    for num_offload in num_offloads:
        search_result_json = deepcopy(src)
        search_result_json["num_offload_layers"] = num_offload
        search_result_json_path = os.path.join(config_dir, f"offload_{num_offload}-search_{search_iter}")
        os.makedirs(search_result_json_path, exist_ok=True)
        search_result_json_dir = os.path.join(search_result_json_path, f"skipped_layers.json")
        with open(search_result_json_dir, "w") as file:
            json.dump(search_result_json, file, indent=4)
        search_result_jsons.append({"num_offload": num_offload, "config_dir": search_result_json_path, "search_result_json": search_result_json_dir})
    return search_result_jsons



if __name__ == "__main__":
    parser = argparse.ArgumentParser("python3 automated_script.py", parents=[get_args_parse("Automated Searching/Evaluation Script")])
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)