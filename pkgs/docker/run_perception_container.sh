#! /bin/bash

set -ex

if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters."
    echo "Usage: run_perception_container.sh IMAGE_TAG CONTAINER_NAME EXEC_SHELL SUDO_LESS"
    exit 2
fi

IMAGE_TAG=$1
CONTAINER_NAME=$2
EXEC_SHELL=$3
SUDO_LESS=$4

# ----- Fix permissions -----
# Create folders used by docker volume to ensure the owner is the current user
# Reason: As far as 2021, docker will automatically use root to create a folder if the folder doesn't exist
if ! $SUDO_LESS; then
    FILES_TO_TOUCH=("$HOME/.gitconfig"             \
                    "$HOME/.bashrc"                \
                    "$HOME/.bash_aliases")
    for file_to_touch in "${FILES_TO_TOUCH[@]}"; do
        if ! { [ -f "$file_to_touch" ] && [ "$(stat -c "%u" "$file_to_touch")" == "$(id -u "$USER")" ] && [ "$(stat -c "%g" "$file_to_touch")" == "$(id -g "$USER")" ]; }; then
            sudo mkdir -p "$(dirname "$file_to_touch")"
            sudo touch "$file_to_touch"
            sudo chown "$(id -u "$USER")":"$(id -g "$USER")" -R "$file_to_touch"
        fi
    done
fi

# ----- Docker run -----
printf "\n--- Run the Perception container ---\n"
# --name summer_robotics_perception sets the name of the container.
# --pull always is used to pull latest image before running the container.
# -d runs container in background and print container ID. Therefore, killing a terminal session will not kill the running container.
# --rm automatically remove the container when it exits.
# -it instructs Docker to allocate a pseudo-TTY connected to the container’s stdin; creating an interactive bash shell in the container.
# --privileged gives extended privileges to this container.
# --ipc=host is set to use the host system’s IPC (Inter-process communication) namespace.
# /dev is the location of device files. This is used to communicate with the cameras.
# --user is used to bind the current host user to the container user (including the UID and GID).
# --env="HOME" is used to set the $HOME environment variable.
# --volume "$PWD":/summer_robotics is used to bind the current folder (summer_robotics_customers) to /summer_robotics in the container.
# /etc/group file stores group information or defines the user groups.
# /etc/passwd file stores essential user information, which required during login.
# /etc/shadow file stores actual password in encrypted format.
# /etc/sudoers.d/ directory stores user permissions.
# --env="DISPLAY" is used by all X clients to determine what X server to display on.
# /tmp/.X11-unix/ directory stores Unix-domain sockets (stream of bytes) used by the X11 server.
docker run \
    --name "$CONTAINER_NAME" \
    -p 8080:8080 \
    --rm -itd \
    --runtime nvidia \
    --env USER="$USER" \
    --user="$(id -u "$USER")":"$(id -g "$USER")" \
    --group-add sudo --group-add dialout --group-add video \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    --env="DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env HOME="$HOME" \
    --workdir="$PWD" \
    --env HISTFILE="$PWD/artifacts/shell_history" \
    --volume "$HOME"/.bashrc:"$HOME"/.bashrc --volume "$HOME"/.bash_aliases:"$HOME"/.bash_aliases \
    --volume "$PWD"/scripts/docker/shell_config/zshrc_sr:"$HOME"/.zshrc \
    --volume "$HOME"/.sudo_as_admin_successful:"$HOME"/.sudo_as_admin_successful \
    --volume "/home/seb/dev/personal/art_in_3d/photogrammetry/cut3r":"/home/seb/dev/personal/art_in_3d/photogrammetry/cut3r" \
    --volume "/home/seb/dev/personal/art_in_3d/photogrammetry/datasets/2025-01-20-Julie/Video-raw/video_raw":"/home/seb/dev/personal/art_in_3d/photogrammetry/datasets/2025-01-20-Julie/Video-raw/video_raw" \
    "$IMAGE_TAG" \
    "$EXEC_SHELL"

# # Documentation:
# # - "$HOME/.aws for AWS
# # - "$HOME/.gitconfig" for Git
# # - "$HOME/.ssh" for SSH key used by Git
# # - "$HOME/.bashrc" and "$HOME/.bash_aliases" for bash
# # - "$HOME/.data_search_env" for Data Search credentials
# # - "$HOME/.cache" for ZSH P10k git plugin
# # Ensure the owner is the current user
# # Reason: As far as 2021, docker will automatically use root to create a folder if the folder doesn't exist
# docker exec -it summer_robotics_perception sudo sh -c " \
#     mkdir -p \
#         $HOME/.cache \
#     ;chown $USER:$USER -R \
#         $HOME/.aws \
#         $HOME/.bashrc \
#         $HOME/.bash_aliases \
#         $HOME/.cache \
#         $HOME/.data_search_env \
#         $HOME/.gitconfig \
#     "