-- 1. Bootstrap lazy.nvim
local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not vim.loop.fs_stat(lazypath) then
  vim.fn.system({
    "git",
    "clone",
    "--filter=blob:none",
    "https://github.com/folke/lazy.nvim.git",
    "--branch=stable",
    lazypath,
  })
end

-- 2. IMPORTANT: Add lazy to the runtime path so 'require("lazy")' works
vim.opt.rtp:prepend(lazypath)

-- 3. Now load your 'lua/dev/' folder
require("dev")