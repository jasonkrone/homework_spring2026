from pathlib import Path

import modal
import numpy as np

from hw1_imitation.train import TrainConfig, parse_train_config, run_training


APP_NAME = "hw1-imitation"
NETRC_PATH = Path("~/.netrc").expanduser()
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"
DEFAULT_GPU = "T4"
DEFAULT_CPU = 2.0
volume = modal.Volume.from_name("hw1-imitation-volume", create_if_missing=True)


def load_gitignore_patterns() -> list[str]:
    """Translate .gitignore entries into Modal ignore globs."""

    if not modal.is_local():
        return []

    root = Path(__file__).resolve().parents[2]
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []

    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns


# Build a container image with the project's dependencies using uv.
image = modal.Image.debian_slim().apt_install("libgl1", "libglib2.0-0").uv_sync()
if NETRC_PATH.is_file():
    image = image.add_local_file(
        NETRC_PATH,
        remote_path="/root/.netrc",
        copy=True,
    )
image = image.add_local_dir(
    ".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns()
)


app = modal.App(APP_NAME)

env = {
    "PYTHONPATH": f"{PROJECT_DIR}/src",
    "WANDB_DIR": f"{VOLUME_PATH}/wandb",
}

# gpu=DEFAULT_GPU,
@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 4,
    env=env,
    image=image,
    cpu=DEFAULT_CPU,
)
def train_remote(config: dict) -> None:
    defaults = TrainConfig(**config)
    defaults.data_dir = Path(VOLUME_PATH) / "data"
    config = parse_train_config(
        defaults=defaults,
        description="Train on Modal.",
    )
    run_training(config)
    volume.commit()


@app.local_entrypoint()
def main():
    n_trials = 1
    lr_list = np.logspace(-15, -6, num=n_trials, base=2)
    bs_list = [int(bs) for bs in np.logspace(6, 15, num=n_trials, base=2)]

    config_list = []
    for lr in lr_list:
        for bs in bs_list:
            exp_name = f"modal_bs-{bs}_lr-{lr}"
            config_list.append({
                "batch_size": bs,
                "lr": lr,
                "exp_name": exp_name,
            })

    for config in config_list:
        train_remote.spawn(config)