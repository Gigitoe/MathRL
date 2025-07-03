from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Correct model identifier (adjust based on the exact model you have access to)
organization_name = "deepseek-ai"
model_name = "DeepSeek-R1-Distill-Qwen-7B"

# Load model with bfloat16 precision
model = AutoModelForCausalLM.from_pretrained(
    f"{organization_name}/{model_name}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(f"{organization_name}/{model_name}")

# (Optional) Save locally
model.save_pretrained(f"models/{model_name}")
tokenizer.save_pretrained(f"models/{model_name}")

print("Download successful!")