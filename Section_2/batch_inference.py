import asyncio, random, time

# Simulated text generation function
async def fake_inference(user_id: int):
    await asyncio.sleep(random.uniform(0.05, 0.1))  # simulate model latency
    return f"Response from model for user {user_id}"

async def run_batch_inference(num_requests=100):
    start = time.time()
    tasks = [asyncio.create_task(fake_inference(i)) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)
    end = time.time()

    print(f"Processed {num_requests} requests in {end - start:.2f}s")
    print("Sample responses:", results[:5])

# Run the batch inference simulation
asyncio.run(run_batch_inference(100))
