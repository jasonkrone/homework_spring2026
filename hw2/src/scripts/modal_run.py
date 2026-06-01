from pathlib import Path

import modal

from scripts.run import main, setup_arguments


APP_NAME = "hw2-pg"
NETRC_PATH = Path("~/.netrc").expanduser()
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/root/exp"
DEFAULT_GPU = "T4"
DEFAULT_CPU = 2.0
DEFAULT_MEMORY = 4096  # MB
volume = modal.Volume.from_name("hw2-pg-volume", create_if_missing=True)


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
image = modal.Image.debian_slim().apt_install("libgl1", "libglib2.0-0", "swig").uv_sync()
# Copy .netrc for wandb logging.
if NETRC_PATH.is_file():
    image = image.add_local_file(
        NETRC_PATH,
        remote_path="/root/.netrc",
        copy=True,
    )
# Copy the current directory.
image = image.add_local_dir(
    ".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns()
)


app = modal.App(APP_NAME)

env = {
    "PYTHONPATH": f"{PROJECT_DIR}/src",
}


@app.function(volumes={VOLUME_PATH: volume}, timeout=60 * 60 * 1, env=env, image=image, gpu=DEFAULT_GPU, cpu=DEFAULT_CPU, memory=DEFAULT_MEMORY)
def hw2_modal_remote(*args: str) -> None:
    args = setup_arguments(args)
    main(args)
    volume.commit()


@app.local_entrypoint()
def run_grid() -> None:
    n_layers_list = [2, 4, 8]
    layer_sizes = [64, 128, 256]
    batch_sizes = [1000, 2000, 4000]
    learning_rates = ["1e-2", "3.16e-3", "1e-3", "3.16e-4", "1e-4"]
    normalize_options = [True]

    all_args = []
    for n_layers in n_layers_list:
        for layer_size in layer_sizes:
            for batch_size in batch_sizes:
                n = 100000 // batch_size
                for lr in learning_rates:
                    for na in normalize_options:
                        na_tag = "_na" if na else ""
                        exp_name = f"pendulum_nl{n_layers}_ls{layer_size}_bs{batch_size}_lr{lr}{na_tag}"
                        args = [
                            "--env_name", "InvertedPendulum-v4",
                            "-n", str(n),
                            "-eb", "1000",
                            "--exp_name", exp_name,
                            "--n_layers", str(n_layers),
                            "--layer_size", str(layer_size),
                            "--batch_size", str(batch_size),
                            "--learning_rate", lr,
                            "--use_reward_to_go",
                        ]
                        if na:
                            args.append("--normalize_advantages")
                        all_args.append(args)

    for _ in hw2_modal_remote.starmap(all_args):
        pass
