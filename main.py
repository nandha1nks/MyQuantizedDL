import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from src.main.w8a16_linear_quantizer import replace_linear_with_target
from src.layers.linear import W8A16LinearLayer

model_id = "Salesforce/codegen-350M-mono"
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)


pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe("def hello_world():", max_new_tokens=200, do_sample=False))

replace_linear_with_target(model, W8A16LinearLayer, ["lm_head"])

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe("def hello_world():", max_new_tokens=200, do_sample=False))
