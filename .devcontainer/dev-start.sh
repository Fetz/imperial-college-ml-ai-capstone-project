#!/bin/bash

# 1. Launch Neovim in the current main window
# We use 'kitten' to talk to the current instance
kitty @ send-text "nvim .\r"

# 2. Create a horizontal split for Claude
# '--location=hsplit' puts it at the bottom
# '--cwd=current' ensures it stays in your project
kitty @ launch --location=hsplit --cwd=current

# 3. Rename the new window and launch Claude
sleep 0.5 # Give the window a moment to initialize
kitty @ send-text --match "recent:0" "claude\r"

# 4. Focus back on Neovim (the top window)
kitty @ focus-window --match "recent:1"
