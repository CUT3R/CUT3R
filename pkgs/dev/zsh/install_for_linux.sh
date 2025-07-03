#! /bin/bash

set -ex

if [ "$#" -ne 0 ]; then
    echo "Illegal number of parameters."
    echo "Usage: install_for_linux.sh"
    exit 2
fi

apt-get update && apt-get install -y --no-install-recommends \
  zsh \
  git \
  curl \
  && rm -rf /var/lib/apt/lists/* && apt-get clean && apt-get autoclean

# Install oh my ZSH
curl --create-dirs --silent https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh --output /tmp/sr/pkgs/zsh/install_oh_my_zsh.sh && \
ZSH=/opt/shell_config/.oh-my-zsh sh /tmp/sr/pkgs/zsh/install_oh_my_zsh.sh && \
rm /tmp/sr/pkgs/zsh/install_oh_my_zsh.sh
# Install powerlevel10k theme
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git /opt/shell_config/.oh-my-zsh/themes/powerlevel10k
# Add zsh-autosuggestions
git clone --depth=1 https://github.com/zsh-users/zsh-autosuggestions /opt/shell_config/.oh-my-zsh/plugins/zsh-autosuggestions
# Add zsh-syntax-highlighting
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git /opt/shell_config/.oh-my-zsh/plugins/zsh-syntax-highlighting

