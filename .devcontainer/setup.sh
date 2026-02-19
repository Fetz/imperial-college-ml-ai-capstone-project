#!/bin/bash
set -e 

NIX_BIN="/nix/var/nix/profiles/default/bin/nix-env"
NIX_REPO="https://github.com/NixOS/nixpkgs/archive/nixos-unstable.tar.gz"

# 1. Download ImageMagick 7
echo "Installing ImageMagick 7..."
mkdir -p ~/.local/bin
curl -L https://download.imagemagick.org/ImageMagick/download/binaries/magick -o ~/.local/bin/magick
chmod +x ~/.local/bin/magick

# 2. Build Python Environment
# Using -iA on a specific channel URL is the most robust method
echo "Building Python environment with pynvim..."
$NIX_BIN -f "$NIX_REPO" -iA python3Packages.pynvim python3Packages.jupyter-client python3Packages.ipykernel

# 3. Install Claude AI
echo "Installing Claude Code..."
curl -fsSL https://claude.ai/install.sh | bash

# 4. Setup Neovim Config
echo "Syncing Neovim configuration..."
mkdir -p ~/.config/nvim
cp -r .devcontainer/nvim/. ~/.config/nvim/
cp .devcontainer/.zshrc ~/.zshrc

# 5. Bootstrap Neovim Plugins
echo "Bootstrapping Neovim..."
/home/vscode/.nix-profile/bin/nvim +"Lazy! sync" +qa

# 6. Install kitty/kitten binaries for Linux inside the container
curl -L https://sw.kovidgoyal.net/kitty/installer.sh | sh /dev/stdin launch=n

# Link them into your path
mkdir -p ~/.local/bin
ln -sf ~/.local/kitty.app/bin/kitty ~/.local/bin/kitty
ln -sf ~/.local/kitty.app/bin/kitten ~/.local/bin/kitten

echo "Setup Complete!"
