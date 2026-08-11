#!/bin/bash

cd /workspace/
set -eo pipefail

# Cloning repo
git clone https://github.com/SyperAI/some-ai-project-test.git
cd some-ai-project-test

uv venv venv --python 3.11
source venv/bin/activate
uv pip install -r requirements.txt

# Cloning ADetailer
cd /workspace/stable-diffusion-webui-forge/extensions
git clone https://github.com/Haoming02/ADetailer-Neo.git