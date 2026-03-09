#!/bin/bash
# Host-side script: starts DevPod and opens kitty tabs with nvim, claude, and shell
set -e

SOCKET="tcp:127.0.0.1:45876"
WORKSPACE="imperial-college-ml-ai-capstone-project"

# ── 1. Verify kitty is listening ──
if ! kitty @ --to "$SOCKET" ls &>/dev/null; then
  echo "Error: kitty is not listening on $SOCKET"
  echo "Launch kitty with: kitty --listen-on $SOCKET"
  exit 1
fi

# ── 2. Start DevPod (idempotent — no-op if already running) ──
echo "Starting DevPod workspace..."
devpod up "$WORKSPACE" --ide none

# ── 3. Tab 1 (current window) — SSH in and launch Neovim ──
kitty @ --to "$SOCKET" send-text "devpod ssh $WORKSPACE\r"
sleep 2
kitty @ --to "$SOCKET" send-text "nvim .\r"

# ── 4. Tab 2 — Claude ──
kitty @ --to "$SOCKET" launch --type=tab --title claude
sleep 0.3
kitty @ --to "$SOCKET" send-text --match "title:claude" "devpod ssh $WORKSPACE\r"
sleep 2
kitty @ --to "$SOCKET" send-text --match "title:claude" "claude\r"

# ── 5. Tab 3 — General shell ──
kitty @ --to "$SOCKET" launch --type=tab --title shell
sleep 0.3
kitty @ --to "$SOCKET" send-text --match "title:shell" "devpod ssh $WORKSPACE\r"

# ── 6. Focus back on the Neovim tab ──
sleep 1
kitty @ --to "$SOCKET" focus-tab --match "index:0"

echo "Done — nvim, claude, and shell tabs are ready."
