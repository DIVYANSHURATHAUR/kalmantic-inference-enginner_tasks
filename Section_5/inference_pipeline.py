"""
Custom Inference Pipeline Demo
--------------------------------
Demonstrates an optimized local text-generation inference pipeline:
- Uses Hugging Face Transformers
- Includes caching & batching
- Simulates concurrent inference requests
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import asyncio

# ---------------------------
# Load Model Efficiently
# ---------------------------
MODEL_NAME = "distilgpt2"  # small, fast model for demo

print(f"Loading model '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on: {device}")

# ---------------------------
# Simple Cache
# ---------------------------
cache = {}

def cached_inference(prompt: str, max_new_tokens=30):
    """Check cache, else run model and store result"""
    if prompt in cache:
        return cache[prompt], True  # cached response
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cache[prompt] = text
    return text, False

# ---------------------------
# Batching Simulation
# ---------------------------
async def process_batch(prompts):
    """Simulate batch inference asynchronously"""
    start = time.time()
    results = []
    for p in prompts:
        text, cached = cached_inference(p)
        results.append((p, text, cached))
    end = time.time()
    print(f"Batch processed in {end - start:.3f}s for {len(prompts)} requests")
    return results

# ---------------------------
# Run Simulation
# ---------------------------
async def main():
    prompts = [
        "AI will transform healthcare by",
        "The future of machine learning is",
        "Python is the best language for",
        "AI will transform healthcare by",  # duplicate (tests cache)
        "Data science is powerful because"
    ]

    results = await process_batch(prompts)
    print("\n=== Sample Outputs ===")
    for p, t, cached in results:
        print(f"\nPrompt: {p}\nCached: {cached}\nResponse: {t[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
