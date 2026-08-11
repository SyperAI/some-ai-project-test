#!/bin/bash

cd /workspace/
set -eo pipefail

# Cloning repo
git clone https://github.com/SyperAI/some-ai-project-test.git
cd some-ai-project-test

sh install.sh --skip-credits --sd-enabled --sd-path "/workspace/stable-diffusion-webui-forge" "$@"

# Cloning ADetailer
cd /workspace/stable-diffusion-webui-forge/extensions
git clone https://github.com/Haoming02/ADetailer-Neo.git