#!/bin/bash
# Run first time setup

set -e

# Create the XDG Runtime folder
export XDG_RUNTIME_DIR="/tmp/xdg_runtime"
if ! { [ -d "$XDG_RUNTIME_DIR" ] && [ "$(stat -c "%u" "$XDG_RUNTIME_DIR")" == "$(id -u "$USER")" ] && [ "$(stat -c "%g" "$XDG_RUNTIME_DIR")" == "$(id -g "$USER")" ] && [ "$(stat -c "%a" "$XDG_RUNTIME_DIR")" == 700 ]; }; then
  mkdir -p -m 700 "$XDG_RUNTIME_DIR"
fi

# Remove the note for sudo usage
if ! [ -f "$HOME/.sudo_as_admin_successful" ]; then
  sudo touch "$HOME/.sudo_as_admin_successful"
fi


# ----- Fix permissions -----
# Ensure the owner is the current user
# Reason: As far as now, docker will automatically use root to create a folder if the folder doesn't exist
#
# Documentation:
## mkdir/chown
# - "$HOME/.aws for AWS
# - "$HOME/.dbus for DBUS
# - "$HOME/.meld" for the meld tool (YAML files diff viewer)
# - "/opt/resources/settings" for the Summer Robotics resource settings (scanners, etc...)
FOLDERS_TO_MKDIR=("$HOME/.cache"                \
                  "$HOME/.local"                \
                  "$HOME/.jupyter"                \
                  "$HOME")
for file_to_mkdir in "${FOLDERS_TO_MKDIR[@]}"; do
    if ! { [ -d "$file_to_mkdir" ] && [ "$(stat -c "%u" "$file_to_mkdir")" == "$(id -u "$USER")" ] && [ "$(stat -c "%g" "$file_to_mkdir")" == "$(id -g "$USER")" ]; }; then
        sudo mkdir -p "$file_to_mkdir"
        sudo chown "$(id -u "$USER")":"$(id -g "$USER")" -R "$file_to_mkdir"
    fi
done

# Execute the command sent by the CLI
exec "$@"
