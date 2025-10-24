from transformers import DistilBertModel, DistilBertTokenizer
import torch
import time

# Load pretrained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Prepare sample input
text = "Quantization helps reduce inference latency."
inputs = tokenizer(text, return_tensors="pt")

# Measure baseline latency
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(**inputs)
end = time.time()
print(f"Baseline Latency: {end - start:.2f}s")

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Measure latency after quantization
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = quantized_model(**inputs)
end = time.time()
print(f"Quantized Latency: {end - start:.2f}s")
