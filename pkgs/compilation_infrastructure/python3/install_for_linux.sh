#!/bin/bash -v

set -ex

apt-get update; apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-wheel \
    ;rm -rf /var/lib/apt/lists/*; apt-get clean; apt-get autoclean; apt-get -y autoremove

which python3
python3 --version
python3 -m venv "$VIRTUAL_ENV"
mkdir -p /opt/cache/python/pip
chmod -R 777 /opt/cache
python3 -m pip install --upgrade \
    pip==24.3.1 \
    setuptools \
    wheel \
    --cache-dir /opt/cache/python/pip
chmod -R 777 "$VIRTUAL_ENV"
