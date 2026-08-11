import argparse
import os.path
import platform
import subprocess
import sys
from pathlib import Path

from utils.gpu import get_gpu_name, get_compute_cap
from utils.utils import download

OS_TYPE = platform.system()

if getattr(sys, 'frozen', False):
    WORKING_DIR = os.path.dirname(sys.executable)
else:
    WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

APP_DIR = os.path.join(WORKING_DIR, 'sd-node')


def install_git():
    try:
        if OS_TYPE == 'Windows':
            subprocess.run(["winget", "install", "--id", "Git.Git", "-e"], check=True)
        elif OS_TYPE == 'Darwin':
            subprocess.run(["brew", "install", "git"], check=True)
        else:
            print("Unsupported OS, install git by yourself and try again.")
            sys.exit(1)
    except subprocess.CalledProcessError:
        print("Failed to install Git.")
        sys.exit(1)


def check_git() -> None:
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Git is available: {result.stdout.strip()}")

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Git is not available, trying to install...")
        install_git()


def confirm_action(prompt: str) -> bool:
    user_input = input(prompt).strip().lower()
    # Check for correct input
    if user_input not in ('y', 'n', 'yes', 'no'): return confirm_action(prompt)

    return user_input in ('y', 'yes')


def install_sd(path: str) -> None:
    print(f"Installing SD to {path}")
    try:
        subprocess.run(["git", "clone", "https://github.com/AUTOMATIC1111/stable-diffusion-webui.git", path],
                       check=True)
    except subprocess.CalledProcessError:
        print(
            "Failed to install SD. Try installing it manually: https://github.com/automatic1111/stable-diffusion-webui#installation-and-running")
        sys.exit(1)

    try:
        subprocess.run([sys.executable, "-m", "venv", os.path.join(path, "venv")], check=True)

        if os.name == "nt":
            pip_path = os.path.join(path, "venv", "Scripts", "pip.exe")
            python_path = os.path.join(path, "venv", "Scripts", "python")
        else:
            pip_path = os.path.join(path, "venv", "bin", "pip")
            python_path = os.path.join(path, "venv", "bin", "python")

        print("Upgrading SD pip...")
        subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

        print("Installing SD requirements...")
        subprocess.run([pip_path, "install", "torch==2.1.2", "torchvision==0.16.2", "torchaudio==2.1.2", "--index-url", "https://download.pytorch.org/whl/cu121"])
        subprocess.run([pip_path, "install", "-r", os.path.join(path, "requirements_versions.txt")], check=True)

        if get_compute_cap() >= 7.0:
            print("Installing xformers...")
            subprocess.run([pip_path, "install", "xformers==0.0.23.post1", "--index-url", "https://download.pytorch.org/whl/cu121"])

        # CLIP installation fix
        print("Installing CLIP...")
        subprocess.run([pip_path, "install", "setuptools==69.5.1"])
        subprocess.run([pip_path, "install", "--no-build-isolation", "https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip"])
    except subprocess.CalledProcessError:
        print("Failed to install venv for SD!")

def sd_poll() -> None:
    is_enabled = args.sd_enabled if args.sd_enabled is not None else confirm_action("Do you want to enable Stable Diffusion?: ")
    config.SD_CONFIG.CONFIG.enable = is_enabled

    if not is_enabled:
        print("Stable Diffusion will not be enabled.")
        return

    if args.sd_path:
        sd_path = args.sd_path
    elif confirm_action("Do you want to install Stable Diffusion? Answer 'No' if you already installed it (Yes/No): "):
        sd_path = Path(input("Enter path where to install Stable Diffusion: "))
        install_sd(str(sd_path))
    else:
        sd_path = Path(input("Enter path where Stable Diffusion is installed (Default: current working directory): "))

    if not sd_path.exists():
        print("Stable Diffusion folder doesn't exist!")
        sys.exit(1)

    for child in sd_path.iterdir():
        if Path(child).is_file() and "webui-user" in str(child).lower():
            print("Stable Diffusion found.")
            break
    else:
        print("Stable Diffusion not found, check provided path and try again!")
        sys.exit(1)

    config.SD_CONFIG.CONFIG.path = sd_path
    config.SD_CONFIG.CONFIG.auto_start = args.sd_auto_start


def worker_poll() -> None:
    if config.MOTHER_NODE.url in (None, '', ' ') or confirm_action("URL of main server already choosed, do you want to change it? (Yes/No): "):
        config.MOTHER_NODE.url = input("Enter URL of main server: ")
    if config.MOTHER_NODE.key in (None, '', ' ') or confirm_action("Node key already exists, do you want to change it? (Yes/No): "):
        config.MOTHER_NODE.key = input("Enter your node key: ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="node-installer",
        description="Node installer assistant"
    )

    parser.add_argument(
        "--skip-credits",
        action="store_true",
        help="Skip credits poll."
    )

    parser.add_argument(
        "--download",
        action="append",
        metavar=('URL', 'DEST'),
        nargs=2,
        help="Download a file from URL to DEST."
    )

    parser.add_argument(
        "--llm-enabled",
        action="store_true",
        help="Enables Ollama if passed. !SKIPPED FOR NOW!"
    )

    parser.add_argument(
        "--sd-enabled",
        action="store_true",
        default=None,
        help="Enables SD requests if passed. If passed will skip question about SD WebUI path."
    )
    parser.add_argument(
        "--sd-path",
        type=Path,
        help="Path to SD WebUI directory. If passed will skip question about SD WebUI path."
    )
    parser.add_argument(
        "--sd-auto-start",
        action="store_true",
        help="Automatically start SD WebUI."
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Polls will not be used. Use if you want to update node."
    )


    args = parser.parse_args()

    if args.download:
        for url, dest in args.download:
            download(url, dest)

    if args.update:
        sys.exit(0)

    from utils import Config, download

    config = Config(allow_missing=True).load()

    if get_gpu_name() is None:
        print("GPU drivers may be incompatible or missing!")

    sd_poll()

    if not args.skip_credits:
        worker_poll()

    config.save()
