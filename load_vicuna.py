# load_vicuna.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "TheBloke/vicuna-7B-1.1-HF"

def main():
    prompt = input("Enter your prompt: ")
    prompt = str(prompt)  # luôn convert sang string

    print(f"Loading tokenizer and model from {MODEL_NAME} on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
    )
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(f"\nResponse: {response}")

if __name__ == "__main__":
    main()
