import os.path
import platform
import subprocess
import sys
from pathlib import Path

from utils.gpu import get_gpu_name

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


def create_venv() -> str:
    if os.path.exists("venv") or os.path.exists(".venv"):
        print("Found existing venv, skipping...")
        return "venv" if os.path.exists("venv") else ".venv"

    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    if not os.path.exists("venv") and not os.path.exists(".venv"): raise RuntimeError(
        "Venv wasn't created! Try creating it by yourself with 'python -m venv' or open github issue.")

    return "venv"


def install_requirements() -> None:
    # Updating pip
    subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True)

    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)


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


def sd_poll() -> None:
    is_enabled = confirm_action("Do you want to enable Stable Diffusion?: ")
    config.SD_CONFIG.CONFIG.enable = is_enabled
    if not is_enabled:
        print("Stable Diffusion will not be enabled.")
        return

    if confirm_action("Do you want to install Stable Diffusion? Answer 'No' if you already installed it (Yes/No): "):
        sd_path = input("Enter path where to install Stable Diffusion: ")
        install_sd(sd_path)
    else:
        sd_path = input("Enter path where Stable Diffusion is installed (Default: current working directory): ")

    for child in Path(sd_path).iterdir():
        if Path(child).is_file() and "webui-user" in str(child).lower():
            print("Stable Diffusion found.")
            break
    else:
        print("Stable Diffusion not found, check provided path and try again!")
        sys.exit(1)

    config.SD_CONFIG.CONFIG.path = sd_path
    config.SD_CONFIG.CONFIG.auto_start = True


def worker_poll() -> None:
    if config.MOTHER_NODE.url in (None, '', ' ') or confirm_action("URL of main server already choosed, do you want to change it? (Yes/No): "):
        config.MOTHER_NODE.url = input("Enter URL of main server: ")
    if config.MOTHER_NODE.key is (None, '', ' ') or confirm_action("Node key already exists, do you want to change it? (Yes/No): "):
        config.MOTHER_NODE.key = input("Enter your node key: ")


if __name__ == "__main__":
    # venv_name = create_venv()
    #
    # if os.name == "nt":
    #     pip_path = os.path.join(venv_name, "Scripts", "pip.exe")
    #     python_path = os.path.join(venv_name, "Scripts", "python.exe")
    # else:
    #     pip_path = os.path.join(venv_name, "bin", "pip")
    #     python_path = os.path.join(venv_name, "bin", "python")
    #
    # install_requirements()

    from utils import Config

    config = Config(allow_missing=True).load()

    if get_gpu_name() is None:
        print("GPU drivers may be incompatible or missing!")

    sd_poll()
    worker_poll()

    config.save()
