from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from evaluate import load
import re
import json

class DatasetEvaluator:
    def __init__(self, task, hf_cache_dir, **args):
        self.hf_cache_dir = hf_cache_dir
        self.task = task
        self.n_shot = None
        self.length = None
        if task == 'xsum':
            self.evaluate_metric = "rouge"
            self.n_shot = args['n_shot']
            self.length = args['length']
            self._data = self._try_get_cached_file()
            if self._data is None:
                self._data = self._prepare_xsum()
                self._save_cached_file()
        elif task == 'cnn_dailymail':
            self.evaluate_metric = "rouge"
            self.n_shot = args['n_shot']
            self.length = args['length']
            self._data = self._try_get_cached_file()
            if self._data is None:
                self._data = self._prepare_cnn_dailymail()
                self._save_cached_file()
        else:
            raise NotImplementedError("없는 Task")

    def data(self):
        return KeyDataset(self._data, "message")
    
    def evaluate(self, output):
        if self.evaluate_metric == "rouge":
            return self._evaluate_rouge(output)


    def _prepare_xsum(self):
        assert self.evaluate_metric == "rouge", "xsum은 rouge로 평가해야 합니다."

        data = load_dataset('EdinburghNLP/xsum', split='test', cache_dir=self.hf_cache_dir).shuffle(4242).select(range(self.length))
        shots = load_dataset('EdinburghNLP/xsum',split='train', cache_dir=self.hf_cache_dir).shuffle(4242).select(range(self.n_shot))
        preprocessed_shot = []
        for i in range(self.n_shot):
            preprocessed_shot.append({
                "role": "user",
                "content": "\nArticle: " + shots[i]["document"] + "\nSummary:"
            })
            preprocessed_shot.append({
                "role": "assistant",
                "content": shots[i]["summary"].replace("\n", "")
            })
        prompts = []
        for i in range(self.length):
            prompt = []
            prompt += preprocessed_shot
            prompt.append({
                "role": "user",
                "content": "\nArticle: " + data[i]["document"] + "\nSummary:"
            })
            prompts.append({
                "message": prompt,
                "answer": data[i]["summary"],
            })
        return prompts

    def _prepare_cnn_dailymail(self):
        assert self.evaluate_metric == "rouge", "cnn_dailymail은 rouge로 평가해야 합니다."

        data = load_dataset('abisee/cnn_dailymail', '3.0.0', split='test', cache_dir=self.hf_cache_dir).shuffle(4242).select(range(self.length))
        shots = load_dataset('abisee/cnn_dailymail', '3.0.0', split='train', cache_dir=self.hf_cache_dir).shuffle(4242).select(range(self.n_shot))
        preprocessed_shot = []
        for i in range(self.n_shot):
            preprocessed_shot.append({
                "role": "user",
                "content": "\nArticle: " + shots[i]["article"] + "\nSummary:"
            })
            preprocessed_shot.append({
                "role": "assistant",
                "content": shots[i]["highlights"].replace("\n", "")
            })
        prompts = []
        for i in range(self.length):
            prompt = []
            prompt += preprocessed_shot
            prompt.append({
                "role": "user",
                "content": "\nArticle: " + data[i]["article"] + "\nSummary:"
            })
            prompts.append({
                "message": prompt,
                "answer": data[i]["highlights"],
            })
        return prompts
    
    def _evaluate_rouge(self, output):
        rouge = load("rouge", cache_dir=self.hf_cache_dir)
        predictions = []
        references = []
        for i, out in enumerate(output):
            predictions.append(out.replace("\n", " ").strip())
            references.append(self._data[i]['answer'].replace("\n", " ").strip())
        rouge_scores = rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
        )
        result = {"task": self.task}
        if self.n_shot is not None:
            result['n_shot'] = self.n_shot
        if self.length is not None:
            result['length'] = self.length
        result['rouge-1'] = rouge_scores["rouge1"].item()
        result['rouge-2'] = rouge_scores["rouge2"].item()
        result['rouge-l'] = rouge_scores["rougeL"].item()
        result['rouge-lsum'] = rouge_scores["rougeLsum"].item()
        return result


    def warmup_data(self):
        messages = [
            {
                "role": "user",
                "content": "Can you provide ways to eat combinations of bananas and dragonfruits?",
            },
            {
                "role": "assistant",
                "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey.",
            },
            {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
        ]
        return messages

    def _get_cached_filename(self):
        return f"{self.hf_cache_dir}.{self.task}_n_shot_{self.n_shot}_length_{self.length}.jsonl"
    
    def _try_get_cached_file(self):
        cached_file = None
        try:
            cached_file = load_dataset("json", data_files=self._get_cached_filename())["train"]
            print(f"{self._get_cached_filename()} loaded")
        except:
            cached_file = None
        return cached_file
    
    def _save_cached_file(self):
        print(f"{self._get_cached_filename()} created")
        with open(self._get_cached_filename(), "w") as f:
            for prompt in self._data:
                f.write(json.dumps(prompt, ensure_ascii=False) + "\n")