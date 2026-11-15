// Checkpoint file (doc 7.3): save, load, same content; bad files give clear errors.
#include <cstdio>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "temper/common/checkpoint.h"

using temper::Checkpoint;

namespace {

std::string tmp(const std::string& name) { return std::string(CRUCIBLE_TEST_TMP) + "_" + name; }

TEST(Checkpoint, RoundTrip) {
    Checkpoint c;
    c.algo = "ppo";
    c.task = "cartpole";
    c.obs_dim = 4;
    c.act_dim = 1;
    c.layer_sizes["actor"] = {4, 256, 256, 256, 1};
    c.layer_sizes["critic"] = {4, 256, 256, 256, 1};
    c.iteration = 77;
    c.seed = 12345;
    c.optimizer_steps = 1540;
    c.arrays["log_std"] = {-0.5f};
    c.arrays["obs.mean"] = {1, 2, 3, 4};
    c.arrays["empty"] = {};
    const std::string path = tmp("ckpt.bin");
    c.save(path);
    const Checkpoint d = Checkpoint::load(path);
    EXPECT_EQ(d.algo, "ppo");
    EXPECT_EQ(d.task, "cartpole");
    EXPECT_EQ(d.obs_dim, 4);
    EXPECT_EQ(d.act_dim, 1);
    EXPECT_EQ(d.layer_sizes, c.layer_sizes);
    EXPECT_EQ(d.iteration, 77);
    EXPECT_EQ(d.seed, 12345u);
    EXPECT_EQ(d.optimizer_steps, 1540);
    EXPECT_EQ(d.arrays, c.arrays);
    EXPECT_THROW(d.array("nope"), std::runtime_error);
    std::remove(path.c_str());
}

TEST(Checkpoint, BadFilesThrow) {
    EXPECT_THROW(Checkpoint::load(tmp("does_not_exist.bin")), std::runtime_error);
    const std::string junk = tmp("junk.bin");
    { std::ofstream(junk) << "hello"; }
    EXPECT_THROW(Checkpoint::load(junk), std::runtime_error);
    // A real file cut in half.
    Checkpoint c;
    c.algo = "ppo";
    c.arrays["w"] = std::vector<float>(1000, 1.0f);
    const std::string full = tmp("full.bin");
    c.save(full);
    std::ifstream in(full, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(in)), {});
    const std::string half = tmp("half.bin");
    { std::ofstream(half, std::ios::binary) << bytes.substr(0, bytes.size() / 2); }
    EXPECT_THROW(Checkpoint::load(half), std::runtime_error);
    std::remove(junk.c_str());
    std::remove(full.c_str());
    std::remove(half.c_str());
}

}  // namespace
