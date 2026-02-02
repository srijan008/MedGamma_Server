import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "google/medgemma-1.5-4b-it"   # example name (depends what you use)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

prompt = "Patient has fever and cough for 3 days. What are possible causes?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

out = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7
)

print(tokenizer.decode(out[0], skip_special_tokens=True))
