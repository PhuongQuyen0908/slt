import modal
import subprocess
import sys
import os
from pathlib import Path

image = (
    modal.Image.debian_slim(python_version="3.11")
    # v3 - torch>=2.6 for CVE-2025-32434
    .pip_install(
        "torch>=2.6.0",
        "torchvision>=0.21.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.38.0",
        "mediapipe>=0.10.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "sacrebleu==1.4.14",
        "tqdm>=4.65.0",
        "pyyaml",
        "sentencepiece",
    )
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=[
            "data/features/",
            "data/extracted/",
            "data/raw/",
            "data/extract/",
            "logs/",
            "__pycache__/",
            "*.pyc",
            ".git/",
        ],
    )
)

app = modal.App("sign-musketeers", image=image)

# Volume cho logs/checkpoints (output)
logs_volume = modal.Volume.from_name("sign-musketeers-logs", create_if_missing=True)
# Volume cho full data features (input)
data_volume = modal.Volume.from_name("sign-musketeers-data", create_if_missing=True)
# Volume cho small data features (10% subset)
data_small_volume = modal.Volume.from_name("sign-musketeers-data-small", create_if_missing=True)


@app.function(
    gpu="A10G",  # ~$1.10/hr, 24GB VRAM, 600GB/s bandwidth
    timeout=3600 * 6,
    volumes={
        "/root/logs": logs_volume,
        "/root/data": data_volume,
    },
)
def train(config: str = "configs/baseline.yaml"):
    import yaml

    os.chdir("/root/project")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    with open(config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]

    # Override paths để dùng Volume
    cfg["logging"]["output_dir"] = f"/root/logs/{model_name}"
    cfg["data"]["data_dir"] = "/root/project/data"
    cfg["data"]["features_dir"] = "/root/data/features/how2sign"

    tmp_config = f"/tmp/{model_name}_config.yaml"
    with open(tmp_config, "w") as f:
        yaml.dump(cfg, f)

    subprocess.run([sys.executable, "train.py", "--config", tmp_config], check=True)
    logs_volume.commit()


@app.function(
    gpu="A10G",
    timeout=3600 * 2,  # 10% subset → nhanh hơn nhiều
    volumes={
        "/root/logs": logs_volume,
        "/root/data_small": data_small_volume,
    },
)
def train_small(config: str = "configs/baseline_small.yaml"):
    """Train trên 10% subset — dùng để kiểm tra pipeline nhanh."""
    import yaml

    os.chdir("/root/project")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    with open(config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    run_name = f"{model_name}_small"

    # CSV từ project (bundle cùng code), features từ Volume
    cfg["logging"]["output_dir"] = f"/root/logs/{run_name}"
    cfg["data"]["data_dir"] = "/root/project/data/small"
    cfg["data"]["features_dir"] = "/root/data_small/features/how2sign_small"

    tmp_config = f"/tmp/{run_name}_config.yaml"
    with open(tmp_config, "w") as f:
        yaml.dump(cfg, f)

    subprocess.run([sys.executable, "train.py", "--config", tmp_config], check=True)
    logs_volume.commit()


@app.function(volumes={"/root/data": data_volume})
def upload_data():
    """Chạy 1 lần để upload data lên Modal Volume."""
    print(
        "Data volume ready. Upload từ local bằng: modal volume put sign-musketeers-data data/ /"
    )


@app.local_entrypoint()
def main(
    config: str = "configs/baseline.yaml",
    model: str = "",
    small: bool = False,
    download: bool = True,
):
    """
    Chạy baseline:        modal run modal_train.py
    Chạy CA-CSA:          modal run modal_train.py --model ca_csa
    Chạy small (10%):     modal run modal_train.py --small
    """
    if small:
        config = "configs/baseline_small.yaml"
        train_small.remote(config=config)
    else:
        if model == "ca_csa":
            config = "configs/ca_csa.yaml"
        train.remote(config=config)

    if download:
        print("\nDownloading results from Modal Volume...")
        vol = modal.Volume.from_name("sign-musketeers-logs")
        local_out = Path("logs")
        local_out.mkdir(exist_ok=True)
        for entry in vol.listdir("/", recursive=True):
            if entry.path.endswith("/"):
                continue
            dest = local_out / entry.path
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in vol.read_file(f"/{entry.path}"):
                    f.write(chunk)
        print(f"Saved to: logs/")
