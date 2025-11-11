#!/bin/bash

# Runpod start command script to set up environment and start training SeeAction model
# Should work on spot instances when a network share is provided with a name saved in e.g. $SHARED_DIR_NAME=/workspace
# Just paste wget -O - https://raw.githubusercontent.com/blkdfdr/SeeAction/main/runpod_startcmd.sh | sh into the start command field
set -e

pip install uv

# --- Clone SeeAction repo ---
# Store it in a temporary location since we will save data and models on the network share
if [ ! -d "/content/SeeAction" ]; then
    git clone https://github.com/blkdfdr/SeeAction.git /content/SeeAction
fi
cd /content/SeeAction
git clean -xdf
git pull

# --- Download training data from Google Drive ---
if [ ! -d "$SHARED_DIR_NAME/data" ]; then
    mkdir -p $SHARED_DIR_NAME/data
    cd $SHARED_DIR_NAME/data
fi
if [ ! -f "$SHARED_DIR_NAME/data/dataset_seeaction.tar.gz" ]; then
    uvx gdown --id 1KXA4SDEfrFP-1GGYtmqE63eWSBTgxDKE -O dataset_seeaction.tar.gz
fi
if [ ! -d "$SHARED_DIR_NAME/data/dataset" ]; then
    tar -xzvf $SHARED_DIR_NAME/data/dataset_seeaction.tar.gz --skip-old-files --no-same-owner -C $SHARED_DIR_NAME/data
    cd /content/SeeAction
fi

# --- Run training ---
export DATA_DIR="$SHARED_DIR_NAME/data/dataset"
uv run train.py
