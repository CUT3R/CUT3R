#!/bin/bash -v

set -ex

echo "Using PUBLIC APT mirrors."
wget --quiet -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null

apt-get update; apt-get install -y --no-install-recommends cmake
rm -rf /var/lib/apt/lists/*; apt-get clean; apt-get autoclean
mv /etc/apt/sources.list.d/kitware.list /etc/apt/sources.list.d/kitware.list.bak

