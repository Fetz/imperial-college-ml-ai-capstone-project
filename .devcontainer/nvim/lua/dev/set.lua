-- Relative numbers make jumping with <C-d> and <C-u> much easier
vim.opt.nu = true
vim.opt.relativenumber = true

-- Tab behavior (Standard for Web/Python/C++)
vim.opt.tabstop = 4
vim.opt.softtabstop = 4
vim.opt.shiftwidth = 4
vim.opt.expandtab = true

vim.opt.smartindent = true

-- Don't wrap lines (use horizontal scrolling)
vim.opt.wrap = false

-- Keep a history of undos even after closing the file
-- These will persist in your DevPod container volume
vim.opt.swapfile = false
vim.opt.backup = false
vim.opt.undodir = os.getenv("HOME") .. "/.vim/undodir"
vim.opt.undofile = true

-- Incremental search and highlighting
vim.opt.hlsearch = false
vim.opt.incsearch = true

-- Colors and UI
vim.opt.termguicolors = true
vim.opt.scrolloff = 8
vim.opt.signcolumn = "yes"
vim.opt.isfname:append("@-@")

-- Faster update time (better for LSP and Git gutter)
vim.opt.updatetime = 50

-- Fixed column for the numbers
vim.opt.colorcolumn = "80"
