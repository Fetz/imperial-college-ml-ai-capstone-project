require("dev.set")
require("dev.remap")
require("dev.lazy")

-- Colorscheme
vim.cmd("colorscheme rose-pine")

-- 
vim.keymap.set("n", "s", "<cmd>Pounce<CR>")
vim.keymap.set("n", "S", "<cmd>PounceRepeat<CR>")

vim.keymap.set("n", "<leader>zz", function()
    require("zen-mode").toggle({
        window = { width = .85 }
    })
end)
