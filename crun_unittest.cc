# include <gtest/gtest.h>
# include "crun.cpp"

TEST(Config, loadFromFile) {
    Config config;
    
    config.loadFromFile("/home/caohe/Workspace/hecao/llama2.c/stories42M.bin");
    EXPECT_EQ(config.dim, 512);
    EXPECT_EQ(config.n_heads, 8);
    EXPECT_EQ(config.hidden_dim, 1376);
}