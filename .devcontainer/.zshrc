# Aliases
alias v="nvim"
alias lg="lazygit"
alias ask="claude"
alias work="bash .devcontainer/dev-start.sh"

# Nix Pathing
export PATH=$PATH:/home/vscode/.local/bin

# Terminal fixes for SSH into container
# Force a universally-supported TERM to fix backspace/key issues
# (Kitty sets TERM=xterm-kitty but the container lacks that terminfo)
export TERM=xterm-256color

# Point kitty remote control at the reverse-forwarded TCP port
export KITTY_LISTEN_ON=tcp:127.0.0.1:45876

# Fix backspace/delete key behavior over SSH
stty erase '^?' 2>/dev/null
bindkey "^?" backward-delete-char
bindkey "^[[3~" delete-char
bindkey "^H" backward-delete-char

# Run setup only if it hasn't been done yet
if [ ! -f "$HOME/.setup_done" ]; then
    echo "First time setup detected. Running setup.sh..."
    bash .devcontainer/setup.sh && touch "$HOME/.setup_done"
    echo "Setup complete."
fi
