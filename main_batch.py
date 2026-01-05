import subprocess
import time
import os
import sys
from pathlib import Path
from tqdm import tqdm
from model_definitions.wasb import HRNet
from memory_utils import estimate_batch_size
import torch

# ─── CONFIG ─────────────────────────────────────────────
VIDEO_DIR = Path("video_input")
BASE_ARGS = [
    "--weights", "badminton",
    "--model", "wasb",
    "--overlay"
]
SHUTDOWN = False  # Set to True to shut down after processing

# ─── Load Video Jobs ─────────────────────────────────────
video_paths = sorted(VIDEO_DIR.glob("*.mp4"))
all_jobs = [{"video_input": p} for p in video_paths]

# ─── Main Execution ──────────────────────────────────────
def main():
    if not all_jobs:
        print("No videos found in", VIDEO_DIR)
        return

    print(f"Starting sequential processing of {len(all_jobs)} job(s).\n")
    start_time = time.time()

    with tqdm(total=len(all_jobs), desc="Processing videos", ncols=80) as progress:
        for job in all_jobs:
            name = job["video_input"].name
            print(f"\n▶ Running: {name}")

            dynamic_args = BASE_ARGS.copy()

            if "wasb" in [a.lower() for a in dynamic_args]:
                dummy_cfg = {
                    "name": "hrnet",
                    "frames_in": 3,
                    "frames_out": 3,
                    "inp_height": 288,
                    "inp_width": 512,
                    "out_height": 288,
                    "out_width": 512,
                    "rgb_diff": False,
                    "out_scales": [0],
                    "MODEL": {
                        "EXTRA": {
                            "FINAL_CONV_KERNEL": 1,
                            "PRETRAINED_LAYERS": ['*'],
                            "STEM": {
                                "INPLANES": 64,
                                "STRIDES": [1, 1]
                            },
                            "STAGE1": {
                                "NUM_MODULES": 1,
                                "NUM_BRANCHES": 1,
                                "BLOCK": 'BOTTLENECK',
                                "NUM_BLOCKS": [1],
                                "NUM_CHANNELS": [32],
                                "FUSE_METHOD": 'SUM'
                            },
                            "STAGE2": {
                                "NUM_MODULES": 1,
                                "NUM_BRANCHES": 2,
                                "BLOCK": 'BASIC',
                                "NUM_BLOCKS": [2, 2],
                                "NUM_CHANNELS": [16, 32],
                                "FUSE_METHOD": 'SUM'
                            },
                            "STAGE3": {
                                "NUM_MODULES": 1,
                                "NUM_BRANCHES": 3,
                                "BLOCK": 'BASIC',
                                "NUM_BLOCKS": [2, 2, 2],
                                "NUM_CHANNELS": [16, 32, 64],
                                "FUSE_METHOD": 'SUM'
                            },
                            "STAGE4": {
                                "NUM_MODULES": 1,
                                "NUM_BRANCHES": 4,
                                "BLOCK": 'BASIC',
                                "NUM_BLOCKS": [2, 2, 2, 2],
                                "NUM_CHANNELS": [16, 32, 64, 128],
                                "FUSE_METHOD": 'SUM'
                            },
                            "DECONV": {
                                "NUM_DECONVS": 0,
                                "KERNEL_SIZE": [],
                                "NUM_BASIC_BLOCKS": 2
                            }
                        },
                        "INIT_WEIGHTS": True
                    },
                    "model_path": f"model_weights/wasb_{dynamic_args[1]}_best.pth.tar",  # Update with your model path
                }
                estimator_model = HRNet(dummy_cfg)
                if torch.cuda.is_available():
                    print("torch cuda is available")
                    batch_size = estimate_batch_size(estimator_model)
                    del estimator_model                         # free immediately
                    torch.cuda.empty_cache()
                else:
                    batch_size = 16
                dynamic_args += ["--batch_size", str(batch_size)]
                print(f"Using estimated batch size: {batch_size}")

            cmd = [sys.executable, "main.py", "--input", str(job["video_input"])] + dynamic_args
            subprocess.run(cmd)
            print(f"Completed: {name}")
            progress.update(1)
            torch.cuda.empty_cache()

    print(f"\nAll jobs completed in {time.time() - start_time:.2f} seconds.")

    if SHUTDOWN:
        print("All jobs done. Shutting down in 60 seconds...")
        os.system("shutdown /s /t 60")  # Use `shutdown /a` to abort

if __name__ == "__main__":
    main()
