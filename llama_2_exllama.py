import time
from pathlib import Path

from exllama.generator import ExLlamaGenerator
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer

path_to_model = Path(
    "C:/Users/Roy/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-GPTQ/snapshots/98ffa0d89723ce1e3214f477469b2db67c6c4586"
)
tokenizer_model_path = path_to_model / "tokenizer.model"
model_config_path = path_to_model / "config.json"
model_path = path_to_model / "gptq_model-4bit-128g.safetensors"

config = ExLlamaConfig(str(model_config_path))
config.model_path = str(model_path)

model = ExLlama(config)
tokenizer = ExLlamaTokenizer(str(tokenizer_model_path))
cache = ExLlamaCache(model)
generator = ExLlamaGenerator(model, tokenizer, cache)

t0 = time.time()

t1 = time.time()

# print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens)')
