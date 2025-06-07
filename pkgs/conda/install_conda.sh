#!/usr/bin/env bash

set -ev

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters."
    echo "Usage: install_runtime_dependencies.sh CONDA_PATH CONDA_ENV"
    exit 2
fi

CONDA_PATH=$1
CONDA_ENV_NAME=$2

# ---------- Install dependencies ----------
# Conda
CONDA_INSTALLATION_PATH=/tmp/conda_installation.sh
rm -f "$CONDA_INSTALLATION_PATH"


wget --quiet -O "$CONDA_INSTALLATION_PATH" https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh

sh "$CONDA_INSTALLATION_PATH" -b -f -p "$CONDA_PATH"
rm -f "$CONDA_INSTALLATION_PATH"

# Upgrade conda
"$CONDA_PATH/bin/conda" install --yes --quiet -n base -c conda-forge conda=22.11.1

# Create the conda environment
"$CONDA_PATH/bin/conda" create --quiet --name "$CONDA_ENV_NAME"
