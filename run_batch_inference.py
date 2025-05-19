import subprocess
import threading
import time
import os
import sys
from pathlib import Path

"""
This script allows for batch processing of multiple videos. 
Highly recommend to use GPU to process the videos.
CPU processing can be very slow.
"""


# Define two groups of jobs
group_1 = [
    {"video_input": Path("video_input/video_1.mp4")}
]

group_2 = [
    {"video_input": Path("video_input/video_2.mp4")}
]

# Arguments for the WASB model (matching original .bat call)
BASE_ARGS = [
    "--weights", "your_sport",
    "--model", "choose_a_model",
    "--overlay"
]


def run_jobs(jobs):
    for job in jobs:
        name = job["video_input"].name
        print(f"▶ Running: {name}")
        cmd = [
            sys.executable,
            "main.py",
            "--input", str(job["video_input"])
        ] + BASE_ARGS

        subprocess.run(cmd)  # Run directly in blocking mode so tqdm output is clean
        print(f"✔ Completed: {name}")


def shutdown_machine():
    print("💤 All jobs done. Shutting down in 60 seconds...")
    os.system("shutdown /s /t 60")  # Abort with `shutdown /a`


# Launch both groups concurrently
t1 = threading.Thread(target=run_jobs, args=(group_1,))
t2 = threading.Thread(target=run_jobs, args=(group_2,))

start_time = time.time()
t1.start()
t2.start()

t1.join()
t2.join()

print(f"\n✅ All jobs completed in {time.time() - start_time:.2f}s")

# Uncomment this if you want to your machine to turn off automatically after batch processing.
# shutdown_machine()
