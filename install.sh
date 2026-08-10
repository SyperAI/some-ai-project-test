#!/bin/bash

git pull

if command -v uv >/dev/null 2>&1; then
    echo "Found: $(uv --version)"
else
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ -d "venv" ]; then
    echo "Found venv"
else
    echo "Creating venv..."
    uv venv venv --python 3.11
fi

source venv/bin/activate
uv pip install -r requirements.txt

python installer.py