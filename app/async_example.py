import asyncio
import time

async def slow_task(name, delay):
    print(f"Starting task {name}: {time.strftime('%X')}")
    await asyncio.sleep(delay)
    print(f"Finished task {name}: {time.strftime('%X')}")

async def main():
    await slow_task("Task 1", 3)
    await slow_task("Task 2", 7)
    print("All tasks finished")

asyncio.run(main())