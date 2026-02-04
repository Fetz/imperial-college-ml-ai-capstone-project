# Aliases
alias v="nvim"
alias lg="lazygit"
alias ask="claude"
alias work="bash .devcontainer/dev-start.sh"

# Nix Pathing
export PATH=$PATH:/home/vscode/.local/bin

# Run setup only if it hasn't been done yet
if [ ! -f "$HOME/.setup_done" ]; then
    echo "🐣 First time setup detected. Running setup.sh..."
    bash .devcontainer/setup.sh && touch "$HOME/.setup_done"
fi

# --- Setup Trigger ---
if [ ! -f "$HOME/.setup_done" ]; then
    echo "🚀 Environment initialized. Syncing Neovim plugins..."
    # This will open Neovim normally so Git credentials can pass through the SSH tunnel
    v +Lazy! sync
    touch "$HOME/.setup_done"
    echo "✅ Setup complete."
fi
