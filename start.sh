#!/bin/bash

git fetch origin

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse @{u})

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "Up to date."
else
    echo "Update available: $LOCAL -> $REMOTE"
    sh install.sh
fi

source venv/bin/activate
python main.py