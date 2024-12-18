import subprocess
import argparse
import os
from scripts.utils import validate_config_json_dir, validate_process

def get_args_parse(name: str):
    parser = argparse.ArgumentParser(name, add_help=False)
    parser.add_argument('-c', "--hf_cache_dir", default='.hf_cache/', metavar='.hf_cache/', type=str, help="huggingface의 cache dir입니다.")
    parser.add_argument('-i', "--json_dir", metavar='search_models_a100.jsonl', type=str, help="searching이 필요한 jsonl", required=True)
    parser.add_argument('-bg', "--background", choices=['true', 'false'], type=str, default='true', metavar='true', help="background에서 실행할 것인지")
    return parser

def main(hf_cache_dir, json_dir, background):
    validate_process(1)
    validate_config_json_dir(json_dir)
    background = True if background == 'true' else False
    cmd = ['python3', '-m', 'scripts.automated_script_core',
                    '--hf_cache_dir', hf_cache_dir,
                    '--json_dir', json_dir
    ]
    if background:
        cmd = ['nohup'] + cmd
        subprocess.Popen(cmd,cwd=os.getcwd(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setpgrp).pid
    else:
        subprocess.run(cmd, cwd=os.getcwd(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("python3 automated_script.py", parents=[get_args_parse("Automated Searching/Evaluation Script")])
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)