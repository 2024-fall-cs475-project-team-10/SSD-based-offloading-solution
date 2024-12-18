import sys
from contextlib import contextmanager
from datetime import timedelta, datetime
from datasets import load_dataset
from models.models import MODEL_DICT
import json
import psutil
import os
import subprocess

@contextmanager
def redirect_output(log_file_path:str, description: str, description_root: str, root=False):
    log_file = open(log_file_path, "w" if not root else "a")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    start_time = None
    end_time = None
    try:
        start_time = datetime.now() + timedelta(hours=9)
        if description_root!="":
            print(f"[Time log] {description_root}", flush = True)
            print(f" - Start Time: {format(start_time, '%Y.%m.%d %H:%M:%S')} (KST)", flush = True)
        sys.stdout = log_file
        sys.stderr = log_file
        if description!="":
            print(f"[Time log] {description} Start Time: {format(start_time, '%Y.%m.%d %H:%M:%S')} (KST)", flush = True)
        yield
    finally:
        end_time = datetime.now() + timedelta(hours=9)
        if description!="":
            print(f"[Time log] {description} End Time: {format(end_time, '%Y.%m.%d %H:%M:%S')} (KST)", flush = True)
            print(f"[Time log] {description} Takes " + str(timedelta(seconds=int((end_time - start_time).total_seconds()))), flush = True)
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        if description_root!="":
            print(f" - End Time: {format(end_time, '%Y.%m.%d %H:%M:%S')} (KST)", flush = True)
            print(f" - Takes " + str(timedelta(seconds=int((end_time - start_time).total_seconds()))), flush = True)
            print(f"")

def redirect_output_subproccess(cmd, log_file_path:str, description: str, description_root: str, root=False):
    log_file = open(log_file_path, "w" if not root else "a")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    start_time = None
    end_time = None
    try:
        start_time = datetime.now() + timedelta(hours=9)
        if description_root!="":
            print(f"[Time log] {description_root}", flush = True)
            print(f" - Start Time: {format(start_time, '%Y.%m.%d %H:%M:%S')} (KST)", flush = True)
        sys.stdout = log_file
        sys.stderr = log_file
        if description!="":
            print(f"[Time log] {description} Start Time: {format(start_time, '%Y.%m.%d %H:%M:%S')} (KST)", flush = True)
        subprocess.run(cmd, cwd=os.getcwd(), stdout = log_file, stderr = log_file)
    finally:
        end_time = datetime.now() + timedelta(hours=9)
        if description!="":
            print(f"[Time log] {description} End Time: {format(end_time, '%Y.%m.%d %H:%M:%S')} (KST)", flush = True)
            print(f"[Time log] {description} Takes " + str(timedelta(seconds=int((end_time - start_time).total_seconds()))), flush = True)
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        if description_root!="":
            print(f" - End Time: {format(end_time, '%Y.%m.%d %H:%M:%S')} (KST)", flush = True)
            print(f" - Takes " + str(timedelta(seconds=int((end_time - start_time).total_seconds()))), flush = True)
            print(f"")

def validate_config_json_dir(config_json_dir):
    try:
        configs = load_dataset("json", data_files=config_json_dir)["train"].to_list()
    except:
        raise f"{config_json_dir} 파일이 잘못되었습니다."
    key_preprocess = {
        "is_done": lambda x: isinstance(x, bool),
        "is_motivation": lambda x: isinstance(x, bool),
    }
    key_common = {
        "config_dir": lambda x: isinstance(x, str),
        "model_id": lambda x: x in MODEL_DICT.keys(),
        "search_iter": lambda x: isinstance(x, int) and x >= 0,
        "n_shot": lambda x: isinstance(x, int) and x > 0,
        "num_iter": lambda x: isinstance(x, int) and x > 0,
        "length": lambda x: isinstance(x, int) and x > 0,
    }
    key_evaluate = {
        "num_offload": lambda x: isinstance(x, int) and x >= 0,
        "search_offload_policy": lambda x: x in ["optimal", "naive"],
    }
    key_motivation = {
        "num_offloads": lambda x: isinstance(x, (list, tuple)) and all(isinstance(e, int) and e >= 0 for e in x),
    }
    results = []
    for config in configs:
        result={}
        if config.get("is_done") is None:
            config["is_done"] = False
        if config.get("is_motivation") is None:
            config["is_motivation"] = False
        if config.get("is_motivation") is False and config.get("search_offload_policy") is None:
            config["search_offload_policy"] = "optimal"
        for key, validate_func in key_preprocess.items():
            assert key in config.keys(), f"{key} 없음 - {str(config)}"
            value = config[key]
            assert validate_func(value), f"올바르지 않은 {key}: {value} - {str(config)}"
            result[key] = value
        for key, validate_func in key_common.items():
            assert key in config.keys(), f"{key} 없음 - {str(config)}"
            value = config[key]
            assert validate_func(value), f"올바르지 않은 {key}: {value} - {str(config)}"
            result[key] = value
        if config["is_motivation"]:
            for key, validate_func in key_motivation.items():
                assert key in config.keys(), f"{key} 없음 - {str(config)}"
                value = config[key]
                assert validate_func(value), f"올바르지 않은 {key}: {value} - {str(config)}"
                result[key] = value
        else:
            for key, validate_func in key_evaluate.items():
                assert key in config.keys(), f"{key} 없음 - {str(config)}"
                value = config[key]
                assert validate_func(value), f"올바르지 않은 {key}: {value} - {str(config)}"
                result[key] = value
        results.append(result)
    save_config_json(results, config_json_dir)
    return results
        
def save_config_json(configs, config_json_dir):
    with open(config_json_dir, "w") as f:
            for config in configs:
                f.write(json.dumps(config, ensure_ascii=False) + "\n")

def validate_process(num_count=2):
    catch_count = 0
    pids = []
    for proc in psutil.process_iter():
        cmdline = proc.cmdline()
        for cmd in cmdline:
            if "automated_script_core" in cmd:
                catch_count += 1
                pids.append(proc.pid)
                break
    assert not (len(pids) > 0 and catch_count >= num_count), f"Searching script가 pid {pids[0]} 에서 실행 중입니다."