from layer_searching.decoding import infer
from models.models import MODELS, MODELS_MAX_LAYER
from bayes_opt import BayesianOptimization
from transformers.utils import logging
from transformers.pipelines.pt_utils import KeyDataset
from models.ssd_pipeline import SSDPipeline
import torch
import gc
from utils import set_device_map

logger = logging.get_logger(__name__)

solution_iter = 0

csv_header = "type,iter,elapsed_time,token/s,th_stop_draft,skipped_attn,skipped_mlp\n"

class LayerSkippingSearching:
    def __init__(
        self,
        model_id,
        tokenizer,
        evaluate_prompts,
        hf_cache_dir,
        cuda_devices,
        num_offload_layers,
        csv_filename,
        naive_offloading=False,
        generate_fn="ssd",
        evaluate_config={"max_new_tokens": 32},
    ):
        self.model_id = model_id
        self.cuda_devices = cuda_devices
        self.hf_cache_dir = hf_cache_dir
        self.tokenizer = tokenizer
        self.num_hidden_layers = MODELS_MAX_LAYER(model_id)
        self.generate_fn = generate_fn
        self.evaluate_prompts = evaluate_prompts
        self.evaluate_config = evaluate_config
        self.num_offload_layers = num_offload_layers
        self.csv_filename = csv_filename
        self.naive_offloading = naive_offloading
        if self.naive_offloading:
            print("Warning: naive offloading option is enabled.")
        self.model = None
        self.pipe = None
        self.candidates = []

        self.pbounds = {
            f"x{i}": (0, 1) for i in range(self.num_hidden_layers * 2)
        }
        with open(self.csv_filename, "w") as f:
            f.write(csv_header)

        self.optimizer = BayesianOptimization(
            f=self._black_box_evaluate_function,
            pbounds=self.pbounds,
            random_state=1,
            verbose=1,
        )

        self.optimizer.set_gp_params(alpha=1e-2)
        self.calculate_base()

    def gpu_warmup(self):
        self.get_new_model([], [])

        logger.warning(f"[GPU Warmup Start]")

        self.inference(self.evaluate_prompts[:2])

        logger.warning(f"[GPU Warmup End]")

    def calculate_base(self):
        self.gpu_warmup()
        global solution_iter
        self.get_new_model([], [], generate_fn="base")
        logger.warning(
            f"[Iter {solution_iter}] Baseline: Skip attn({0}), Skip mlp({0})"
        )

        total_tokens, total_time, th_stop_draft = self.inference(self.evaluate_prompts)

        logger.warning(
            f"Elapsed_time: {total_time}, Log: {total_tokens / total_time} tokens/s, Skipped attn: {0}, Skipped mlp: {0}\n"
        )
        with open(self.csv_filename, "a") as f:
            f.write(
                f"base,{solution_iter},{total_time},{total_tokens / total_time},{th_stop_draft},{0},{0}\n")

    def _black_box_evaluate_function(self, **kargs):
        global solution_iter
        attn_skip_layers = []
        for i in range(self.num_hidden_layers):
            if kargs[f"x{i}"] > 0.5:
                attn_skip_layers.append(i)
        mlp_skip_layers = []
        for i in range(
            self.num_hidden_layers, self.num_hidden_layers * 2
        ):
            if kargs[f"x{i}"] > 0.5:
                mlp_skip_layers.append(i - self.num_hidden_layers)

        self.get_new_model(attn_skip_layers, mlp_skip_layers, naive_offloading=self.naive_offloading)

        logger.warning(
            f"[Iter {solution_iter}] Skip attn({len(attn_skip_layers)}): {attn_skip_layers}, Skip mlp({len(mlp_skip_layers)}): {mlp_skip_layers}"
        )

        total_tokens, total_time, th_stop_draft = self.inference(self.evaluate_prompts)

        logger.warning(
            f"Elapsed_time: {total_time}, Log: {total_tokens / total_time} tokens/s, Skipped attn: {len(attn_skip_layers)}, Skipped mlp: {len(mlp_skip_layers)}\n"
        )
        with open(self.csv_filename, "a") as f:
            f.write(
                f"ssd,{solution_iter},{total_time},{total_tokens / total_time},{th_stop_draft},{len(attn_skip_layers)},{len(mlp_skip_layers)}\n"
            )
        self.candidates.append({
            "attn_skip_layers": attn_skip_layers,
            "mlp_skip_layers": mlp_skip_layers,
            "token_per_sec": total_tokens / total_time,
        })
        solution_iter += 1

        return total_tokens / total_time

    def inference(self, dataset):
        outs = self.pipe(KeyDataset(dataset, "message"), **self.evaluate_config)
        total_tokens = outs['generated_tokens_length']
        total_time = outs['time']
        th_stop_draft = outs['th_stop_draft']
        return total_tokens, total_time, th_stop_draft

    def probe(self, attn_skip_layers, mlp_skip_layers):
        """
        Add some good points to accelerate searching
        """

        params = {f"x{i}": 0.0 for i in range(self.num_hidden_layers * 2)}
        for i in attn_skip_layers:
            params[f"x{i}"] = 1.0
        for i in mlp_skip_layers:
            params[f"x{i+self.num_hidden_layers}"] = 1.0
        self.optimizer.probe(params=params, lazy=True)

    def search(self, n_iter=1000):
        self.optimizer.maximize(init_points=0, n_iter=n_iter)
        return self.get_solution()

    def get_solution(self):

        self.candidates.sort(key = lambda x:x["token_per_sec"], reverse=True)
        optimal = self.candidates[0]
        skip_attn_layers = optimal["attn_skip_layers"]
        skip_mlp_layers = optimal["mlp_skip_layers"]

        return skip_attn_layers, skip_mlp_layers

    def get_new_model(self, skip_attn_layers, skip_mlp_layers, generate_fn = "ssd", naive_offloading = False):
        del self.pipe
        if self.num_offload_layers != 0:
            del self.model
        gc.collect()
        torch.cuda.empty_cache()

        if self.num_offload_layers != 0 or self.model == None:
            self.model = MODELS(self.model_id).from_pretrained(
                self.model_id,
                device_map=set_device_map(
                    model_id=self.model_id,
                    skipped_attn_layers=[] if naive_offloading else skip_attn_layers,
                    skipped_mlp_layers=[] if naive_offloading else skip_mlp_layers,
                    max_skip_attn_layers=self.num_offload_layers,
                    max_skip_mlp_layers=self.num_offload_layers,
                    cuda_devices=self.cuda_devices,
                ),
                torch_dtype="auto",
                cache_dir=self.hf_cache_dir,
            ).eval()
            for parameter in self.model.parameters():
                parameter.requires_grad = False
        self.model.set_skip_layers(
            attn_skip_layer_id_set=skip_attn_layers,
            mlp_skip_layer_id_set=skip_mlp_layers,
        )
        self.pipe = SSDPipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                generate_fn=generate_fn,
                th_stop_draft=0.6,
                auto_th_stop_draft=True,
        )
