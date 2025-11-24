import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

import json
import uuid
import random
import requests
import torch
import gc
import subprocess
from google.cloud import storage
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import save_video

import runpod

# ---------------------------
# GPU MEMORY CLEAR
# ---------------------------
def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
clear_gpu_memory()

# ---------------------------
# LOGGING INIT
# ---------------------------
def init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    else:
        logging.basicConfig(level=logging.ERROR)

# ---------------------------
# GCS FUNCTIONS
# ---------------------------
def fetch_gcs_json_from_drive(file_id: str) -> dict:
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def upload_to_gcs_public(source_file, bucket_name="runpod_bucket_testing"):
    gcs_json_dict = fetch_gcs_json_from_drive("1leNukepERYsBmoKSYTbqUjGb-pQvwQlz")
    creds_path = "/tmp/gcs_creds.json"

    with open(creds_path, "w") as f:
        json.dump(gcs_json_dict, f)

    client = storage.Client.from_service_account_json(creds_path)
    bucket = client.bucket(bucket_name)

    destination_blob = f"t2v_videos/{uuid.uuid4()}.mp4"
    blob = bucket.blob(destination_blob)

    blob.upload_from_filename(source_file)

    url = blob.generate_signed_url(expiration=timedelta(hours=1))
    return url

# ---------------------------
# MODEL MANAGEMENT
# ---------------------------
WAN_MODEL = None  # global model cache

def ensure_model_weights(local_dir: str):
    """Download model from HuggingFace only if not exists"""
    if os.path.exists(local_dir) and os.listdir(local_dir):
        logging.info(f"Model already exists at {local_dir}, skipping download.")
        return
    logging.info("Downloading WAN model weights from HuggingFace...")
    os.makedirs(local_dir, exist_ok=True)
    subprocess.run([
        "huggingface-cli", "download",
        "Wan-AI/Wan2.2-T2V-A14B",
        "--local-dir", local_dir,
        "--local-dir-use-symlinks", "False"
    ], check=True)
    logging.info("Model download complete.")

def load_model_once(task, ckpt_dir, device):
    global WAN_MODEL
    if WAN_MODEL is not None:
        logging.info("Reusing loaded WAN model.")
        return WAN_MODEL

    logging.info("Loading WAN model into GPU...")
    cfg = WAN_CONFIGS[task]
    WAN_MODEL = wan.WanT2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=device,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        convert_model_dtype=True
    )
    logging.info("Model loaded and cached in memory.")
    return WAN_MODEL

# ---------------------------
# HANDLER
# ---------------------------
def handler(event):
    input_data = event.get("input", {})

    # Mandatory
    try:
        SIZE = input_data["size"]
        FRAME_NUM = input_data["frame_num"]
        PROMPT = input_data["prompt"]
    except KeyError as e:
        return {"status": "error", "message": f"Missing mandatory key: {e}"}

    # Optional
    TASK = input_data.get("task", "t2v-A14B")
    CKPT_DIR = input_data.get("ckpt_dir", "./Wan2.2-T2V-A14B")
    OFFLOAD_MODEL = input_data.get("offload_model", True)
    SAMPLE_SOLVER = input_data.get("sample_solver", "unipc")
    SAMPLE_STEPS = input_data.get("sample_steps", None)
    SAMPLE_SHIFT = input_data.get("sample_shift", None)
    GUIDE_SCALE = input_data.get("guide_scale", None)
    BASE_SEED = input_data.get("base_seed", -1)

    rank = 0
    device = 0
    init_logging(rank)

    # Ensure model exists locally
    ensure_model_weights(CKPT_DIR)

    # Config
    cfg = WAN_CONFIGS[TASK]
    cfg.sample_neg_prompt = """disconnected body parts, unnatural poses, skipped frames, jittery motion, sliding feet, floating limbs, stiff movement, teleporting body, blurred motion, chopped animation, broken anatomy, misaligned joints, unnatural limb rotation, gliding instead of running, distorted motion, frame skipping, laggy or uneven motion, unnatural physics, clipping through objects, body stretching, unrealistic timing, broken continuity, sudden pops in position, unnatural acceleration, unnatural deceleration, frozen frames, awkward transitions"""
    cfg.sample_fps = 20

    if SAMPLE_STEPS is None:
        SAMPLE_STEPS = cfg.sample_steps
    if SAMPLE_SHIFT is None:
        SAMPLE_SHIFT = cfg.sample_shift
    if GUIDE_SCALE is None:
        GUIDE_SCALE = cfg.sample_guide_scale

    seed = BASE_SEED if BASE_SEED >= 0 else random.randint(0, sys.maxsize)

    logging.info(f"Task = {TASK}, Prompt = {PROMPT}")

    # Create results folder
    os.makedirs("results", exist_ok=True)

    # Load model once globally
    wan_model = load_model_once(TASK, CKPT_DIR, device)

    # Generate video
    video = wan_model.generate(
        PROMPT,
        size=SIZE_CONFIGS[SIZE],
        frame_num=FRAME_NUM,
        shift=SAMPLE_SHIFT,
        sample_solver=SAMPLE_SOLVER,
        sampling_steps=SAMPLE_STEPS,
        guide_scale=GUIDE_SCALE,
        seed=seed,
        offload_model=OFFLOAD_MODEL
    )

    # Save locally
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = PROMPT.replace(" ", "_")[:30]
    SAVE_FILE = f"results/{TASK}_{SIZE}_{short}_{ts}.mp4"

    logging.info(f"Saving video to: {SAVE_FILE}")

    save_video(
        tensor=video[None],
        save_file=SAVE_FILE,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )

    del video
    torch.cuda.synchronize()

    # Upload to GCS
    gcs_url = upload_to_gcs_public(SAVE_FILE)
    logging.info(f"Uploaded to GCS: {gcs_url}")

    return {
        "status": "success",
        "seed": seed,
        "gcs_url": gcs_url,
        "local_file": SAVE_FILE
    }

runpod.serverless.start({"handler": handler})
