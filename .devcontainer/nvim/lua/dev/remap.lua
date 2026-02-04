vim.g.mapleader = " "

-- The "Project View" (File Explorer)
vim.keymap.set("n", "<leader>pv", vim.cmd.Ex)

-- Move highlighted lines up and down (The "Greatest Hack")
vim.keymap.set("v", "J", ":m '>+1<CR>gv=gv")
vim.keymap.set("v", "K", ":m '<-2<CR>gv=gv")

-- Keep cursor centered when jumping or searching
vim.keymap.set("n", "J", "mzJ`z")
vim.keymap.set("n", "<C-d>", "<C-d>zz")
vim.keymap.set("n", "<C-u>", "<C-u>zz")
vim.keymap.set("n", "n", "nzzzv")
vim.keymap.set("n", "N", "Nzzzv")

-- The "Greatest Remap Ever": Paste over selection without losing your register
vim.keymap.set("x", "<leader>p", [["_dP]])

-- The "Next Greatest Remap": Copy to system clipboard (MacOS/Linux)
-- This works perfectly inside DevPod containers
vim.keymap.set({"n", "v"}, "<leader>y", [["+y]])
vim.keymap.set("n", "<leader>Y", [["+Y]])

-- Delete to void register (doesn't overwrite what you've copied)
vim.keymap.set({"n", "v"}, "<leader>d", [["_d]])

-- Escape with Ctrl-C (handy for muscle memory)
vim.keymap.set("i", "<C-c>", "<Esc>")

-- Disable Q (The most annoying key in Vim)
vim.keymap.set("n", "Q", "<nop>")

-- Quickfix navigation
vim.keymap.set("n", "<C-k>", "<cmd>cnext<CR>zz")
vim.keymap.set("n", "<C-j>", "<cmd>cprev<CR>zz")
vim.keymap.set("n", "<leader>k", "<cmd>lnext<CR>zz")
vim.keymap.set("n", "<leader>j", "<cmd>lprev<CR>zz")

-- Search and replace the word you are currently on
vim.keymap.set("n", "<leader>s", [[:%s/\<<C-r><C-w>\>/<C-r><C-w>/gI<Left><Left><Left>]])

-- Jupyter Notebook
vim.keymap.set("n", "<leader>mi", ":MoltenInit<CR>", { desc = "Initialize Molten" })
vim.keymap.set("n", "<leader>er", ":MoltenEvaluateOperator<CR>", { desc = "Evaluate operator" })
vim.keymap.set("v", "<leader>er", ":<C-u>MoltenEvaluateVisual<CR>gv", { desc = "Evaluate visual selection" })
