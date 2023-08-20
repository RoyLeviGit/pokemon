import time

from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM

from llm import Generation, BaseLLM


class Llama2GPTQ(BaseLLM):
    def __init__(
        self,
        model_name_or_path="TheBloke/Llama-2-7b-Chat-GPTQ",
        model_basename="model",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True
        )
        self.model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
        )
        # Supposedly more accurate model
        # self.model = AutoGPTQForCausalLM.from_quantized(
        #     model_name_or_path,
        #     revision="gptq-4bit-32g-actorder_True",
        #     model_basename=model_basename,
        #     use_safetensors=True,
        #     trust_remote_code=True,
        #     inject_fused_attention=False,
        #     device="cuda:0",
        # )

    def generate(self, prompt, temperature, max_tokens):
        system_message = "You are a helpful assistant."
        prompt_template = f'''[INST] <<SYS>>
        {system_message}
        <</SYS>>

        {prompt} [/INST]'''
        input_ids = self.tokenizer(
            prompt_template, return_tensors="pt"
        ).input_ids.cuda()

        t0 = time.time()
        output = self.model.generate(
            inputs=input_ids, temperature=temperature, max_new_tokens=max_tokens
        )
        response = self.tokenizer.decode(output[0])
        t1 = time.time()

        total_tokens = len(output[0]) - input_ids.shape[1]

        return Generation(
            completion=response,
            time=t1 - t0,
            completion_tokens=total_tokens,
        )

# Inference can also be done using transformers' pipeline

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
# logging.set_verbosity(logging.CRITICAL)

# print("*** Pipeline:")
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=200,
#     temperature=0.7,
#     top_p=0.95,
#     repetition_penalty=1.15
# )
#
# t0 = time.time()
# result = pipe(prompt_template)
# t1 = time.time()
# new_tokens = len(output[0])
# print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens)')
#
# print(result[0]['generated_text'])
