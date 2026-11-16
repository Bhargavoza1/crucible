// Phase 9b: demonstration files (forge/env/demo_file.h): write, read back.
#include <cstdio>
#include <string>

#include <gtest/gtest.h>

#include "forge/env/demo_file.h"

using namespace forge;

TEST(DemoFile, WriteAndReadBack) {
    const std::string path = ::testing::TempDir() + "crucible_demo_test.bin";
    DemoEpisode a, b;
    for (int t = 0; t < 3; ++t) {
        for (int k = 0; k < 2; ++k) a.obs.push_back(float(10 * t + k));
        a.expert.push_back(0.5f * t);
        a.executed.push_back(0.5f * t + 0.1f);
        a.reward.push_back(float(t));
    }
    a.final_obs = {7.0f, 8.0f};
    a.success = true;
    a.noise = 0.25f;
    b.final_obs = {1.0f, 2.0f};  // an empty episode is allowed
    {
        DemoWriter w(path, "panda_pick", 2, 1);
        w.write(a);
        w.write(b);
        EXPECT_EQ(w.episodes(), 2);
        DemoEpisode bad = a;
        bad.expert.pop_back();
        EXPECT_THROW(w.write(bad), std::runtime_error);
    }
    const DemoFile d = read_demo_file(path);
    EXPECT_EQ(d.task, "panda_pick");
    EXPECT_EQ(d.obs_dim, 2);
    EXPECT_EQ(d.act_dim, 1);
    ASSERT_EQ(d.episodes.size(), 2u);
    EXPECT_EQ(d.episodes[0].obs, a.obs);
    EXPECT_EQ(d.episodes[0].expert, a.expert);
    EXPECT_EQ(d.episodes[0].executed, a.executed);
    EXPECT_EQ(d.episodes[0].reward, a.reward);
    EXPECT_EQ(d.episodes[0].final_obs, a.final_obs);
    EXPECT_TRUE(d.episodes[0].success);
    EXPECT_EQ(d.episodes[0].noise, 0.25f);
    EXPECT_EQ(d.episodes[1].steps(), 0);
    EXPECT_FALSE(d.episodes[1].success);
    std::remove(path.c_str());
    EXPECT_THROW(read_demo_file(path), std::runtime_error);
}
