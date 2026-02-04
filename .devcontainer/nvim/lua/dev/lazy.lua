local plugins = {
    -- The Core Essentials
    "theprimeagen/harpoon",
    "mbbill/undotree",
    "nvim-treesitter/nvim-treesitter",
    
    -- Navigation & UI
    "folke/zen-mode.nvim",
    "folke/pounce.nvim",
    {
        "kdheepak/lazygit.nvim",
        cmd = { "LazyGit" },
        keys = { { "<leader>gg", "<cmd>LazyGit<cr>", desc = "LazyGit" } }
    },

    -- LSP Setup (C++, JS/TS, Python)
    {
        "neovim/nvim-lspconfig",
        dependencies = {
            "williamboman/mason.nvim",
            "williamboman/mason-lspconfig.nvim",
            "hrsh7th/nvim-cmp",
            "hrsh7th/cmp-nvim-lsp",
        },
    },

    -- Notebooks & Data (AI/Python)
    {
        "benlubas/molten-nvim",
        version = "^1.0.0", -- Avoid breaking changes
        build = ":UpdateRemotePlugins",
        init = function()
            -- Requirements for Molten
            vim.g.molten_image_provider = "image.nvim"
            vim.g.molten_output_win_max_height = 20
        end,
    },

    {
    "GCBallesteros/jupytext.nvim",
        config = true, -- Automatically converts .ipynb to markdown for editing
    },

    {
    "3rd/image.nvim",
        opts = {
            backend = "kitty", -- Kitty is your terminal, so this enables image rendering!
            max_width = 100,
            max_height = 12,
            max_height_window_percentage = math.huge,
            max_width_window_percentage = math.huge,
            window_overlap_clear_enabled = true,
        },
    }
}
require("lazy").setup(plugins)
