from transformers.pipelines.pt_utils import KeyDataset
from layer_searching.decoding import infer
from models.modeling_phi3 import Phi3ForCausalLM
from models.modeling_llama import LlamaForCausalLM
from tqdm import tqdm

class SSDPipeline:
    def __init__(self, task: str, model, tokenizer, generate_fn = "base", auto_th_stop_draft = True, th_stop_draft = 0.8):
        assert task in ["text-generation"], \
            "Only 'text-generation' task is supported."
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.generate_fn = generate_fn
        self.auto_th_stop_draft = auto_th_stop_draft if generate_fn != "base" else False
        self.th_stop_draft = th_stop_draft

    def preprocess(self, inputs):
        if isinstance(
            inputs, (list, tuple, KeyDataset)
        ) and isinstance(inputs[0], (list, tuple, dict)):
            if isinstance(inputs[0], dict):
                chats = self.parse_input(inputs)
            else:
                chats = [self.parse_input(item) for item in inputs]
        else:
            chats = inputs
        
        return {"prompts": chats}

    def postprocess(self, outputs):
        if isinstance(outputs, (list, tuple)):
            completion = []
            time = 0
            generated_tokens_length = 0
            for output in outputs:
                completion.append(self.parse_output(output['completion']))
                time += output['time']
                generated_tokens_length += output['generated_tokens_length']
            return {
                "time": time,
                "th_stop_draft": self.th_stop_draft,
                "completion": completion,
                "generated_tokens_length": generated_tokens_length,
            }
        else:
            completion = self.parse_output(outputs['completion'])
            time = outputs['time']
            generated_tokens_length = outputs['generated_tokens_length']
            return {
                "time": time,
                "th_stop_draft": self.th_stop_draft,
                "completion": completion,
                "generated_tokens_length": generated_tokens_length,
            }


    def __call__(self, inputs, **generate_args):
        preprocessed_input = self.preprocess(inputs)["prompts"]
        if self.generate_fn != "base":
            generate_args['th_stop_draft'] = self.th_stop_draft
            generate_args['auto_th_stop_draft'] = self.auto_th_stop_draft
        if isinstance(preprocessed_input, (list, tuple)):
            model_output = []
            for text in tqdm(preprocessed_input, desc="Model Inference..."):
                output = infer(self.model, self.tokenizer, text, self.generate_fn, **generate_args)
                if self.auto_th_stop_draft:
                    self.th_stop_draft = output['th_stop_draft']
                    generate_args['th_stop_draft'] = self.th_stop_draft
                model_output.append(output)
        else:
            model_output = infer(self.model, self.tokenizer, preprocessed_input, self.generate_fn, **generate_args)
        
        model_output = self.postprocess(model_output)

        return model_output

    def parse_input(self, inputs):
        if isinstance(self.model, (Phi3ForCausalLM)):
            rendered_input = ""
            for text in inputs:
                rendered_input += f"<|{text['role']}|>\n{text['content']}<|end|>\n"
            rendered_input += "<|assistant|>\n"
            return rendered_input
        elif isinstance(self.model, (LlamaForCausalLM)):
            rendered_input = "<|begin_of_text|>"
            for text in inputs:
                rendered_input += f"<|start_header_id|>{text['role']}<|end_header_id|>\n"
                rendered_input += f"{text['content']}<|eot_id|>"
            rendered_input += "<|start_header_id|>assistant<|end_header_id|>\n"
            return rendered_input
        else:
            raise NotImplementedError("Model에 맞는 적절한 preprocess 구현")

    def parse_output(self, outputs):
        if isinstance(self.model, (Phi3ForCausalLM)):
            text = outputs.replace("<|end|>", "").replace("<|endoftext|>", "")
        elif isinstance(self.model, (LlamaForCausalLM)):
            text = outputs.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").replace("<|start_header_id|>assistant<|end_header_id|>", "").replace("<|start_header_id|>assistant", "").replace("<|start_header_id|>", "")
        else:
            raise NotImplementedError("Model에 맞는 적절한 postprocess 구현")
        if "<|" in text:
            print(text)
        return text