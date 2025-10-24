import torch
import time

def matrix_multiplication(size=1000):
    # Create random matrices
    A = torch.randn(size, size)
    B = torch.randn(size, size)

    start = time.time()
    C = torch.mm(A, B)
    end = time.time()

    print(f"CPU Time: {end - start:.4f}s")

def matrix_multiplication_gpu(size=1000):
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU...")
        return matrix_multiplication(size)

    # Move matrices to GPU
    A = torch.randn(size, size, device='cuda')
    B = torch.randn(size, size, device='cuda')

    torch.cuda.synchronize()  # Ensure accurate timing
    start = time.time()
    C = torch.mm(A, B)
    torch.cuda.synchronize()
    end = time.time()

    print(f"GPU Time: {end - start:.4f}s")

# Run
matrix_multiplication()
matrix_multiplication_gpu()
