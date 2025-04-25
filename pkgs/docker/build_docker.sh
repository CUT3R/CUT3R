#! /bin/bash

set -ex

DOCKERFILE=$1
IMAGE_TAG=$2

docker build --progress=plain -f "scripts/docker/$DOCKERFILE" -t "$IMAGE_TAG" .