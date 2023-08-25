import glob
import os
import time

from exllama.generator import ExLlamaGenerator
from exllama.lora import ExLlamaLora
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer

from llm import Generation, BaseLLM


class Llama2ExLlama(BaseLLM):
    def __init__(
        self,
        model="llama-2",
        model_directory="../Llama-2-7b-Chat-GPTQ",
    ):
        super().__init__(model)

        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        model_config_path = os.path.join(model_directory, "config.json")
        st_pattern = os.path.join(model_directory, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]

        # Create config, model, tokenizer and generator
        config = ExLlamaConfig(model_config_path)  # create config from config.json
        config.model_path = model_path  # supply path to model weights file

        model = ExLlama(config)  # create ExLlama instance and load the weights
        tokenizer = ExLlamaTokenizer(
            tokenizer_path
        )  # create tokenizer from tokenizer model file

        cache = ExLlamaCache(model)  # create cache for inference
        self.generator = ExLlamaGenerator(model, tokenizer, cache)  # create generator

        # Configure generator

        self.generator.disallow_tokens([tokenizer.eos_token_id])

        self.generator.settings.token_repetition_penalty_max = 1.2
        self.generator.settings.temperature = 0.95
        self.generator.settings.top_p = 0.65
        self.generator.settings.top_k = 100
        self.generator.settings.typical = 0.5

    def generate(self, messages, temperature, max_tokens):
        self.generator.settings.temperature = temperature
        t0 = time.time()
        output = self.generator.generate_simple(messages, max_new_tokens=max_tokens)
        t1 = time.time()

        return Generation(
            messages=messages,
            completion=output[len(messages) :],
            time=t1 - t0,
            completion_tokens=self.generator.gen_num_tokens(),
        )


class Llama2PokemonExLlama(BaseLLM):
    def __init__(
        self,
        model="llama-2",
        model_directory="../Llama-2-7b-Chat-GPTQ",
        lora_directory="../Llama-2-7b-Chat-Pokemon-GPTQ",
    ):
        super().__init__(model)

        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        model_config_path = os.path.join(model_directory, "config.json")
        st_pattern = os.path.join(model_directory, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]

        lora_config_path = os.path.join(lora_directory, "adapter_config.json")
        lora_path = os.path.join(lora_directory, "adapter_model.bin")

        # Create config, model, tokenizer and generator
        config = ExLlamaConfig(model_config_path)  # create config from config.json
        config.model_path = model_path  # supply path to model weights file

        model = ExLlama(config)  # create ExLlama instance and load the weights
        tokenizer = ExLlamaTokenizer(
            tokenizer_path
        )  # create tokenizer from tokenizer model file

        cache = ExLlamaCache(model)  # create cache for inference
        self.generator = ExLlamaGenerator(model, tokenizer, cache)  # create generator

        # Load LoRA

        lora = ExLlamaLora(model, lora_config_path, lora_path)
        self.generator.lora = lora

        # Configure generator

        self.generator.disallow_tokens([tokenizer.eos_token_id])

        self.generator.settings.token_repetition_penalty_max = 1.2
        self.generator.settings.temperature = 0.95
        self.generator.settings.top_p = 0.65
        self.generator.settings.top_k = 100
        self.generator.settings.typical = 0.5

    def generate(self, messages, temperature, max_tokens):
        self.generator.settings.temperature = temperature
        t0 = time.time()
        output = self.generator.generate_simple(messages, max_new_tokens=max_tokens)
        t1 = time.time()

        return Generation(
            messages=messages,
            completion=output[len(messages) :],
            time=t1 - t0,
            completion_tokens=self.generator.gen_num_tokens(),
        )
