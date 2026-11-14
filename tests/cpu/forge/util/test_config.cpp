#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forge/util/config.h"

using forge::Config;
using forge::ConfigError;
namespace fs = std::filesystem;

namespace {

// Builds argv from strings for apply_args / from_cli.
struct Args {
    std::vector<std::string> s;
    std::vector<const char*> p;
    explicit Args(std::vector<std::string> v) : s(std::move(v)) {
        p.push_back("app");
        for (auto& x : s) p.push_back(x.c_str());
    }
    int argc() const { return static_cast<int>(p.size()); }
    const char* const* argv() const { return p.data(); }
};

fs::path tmp_dir() {
    fs::path d = fs::path(CRUCIBLE_TEST_TMP) / ::testing::UnitTest::GetInstance()->current_test_info()->name();
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

}  // namespace

TEST(Config, ParsesKeyValueCommentsAndSpaces) {
    Config c;
    c.load_string(
        "# comment line\n"
        "task = ant\n"
        "  num_envs=4096   # trailing comment\n"
        "\n"
        "lr = 3e-4\n"
        "headless = true\n");
    EXPECT_EQ(c.get_string("task", ""), "ant");
    EXPECT_EQ(c.get_int("num_envs", 0), 4096);
    EXPECT_FLOAT_EQ(c.get_float("lr", 0.0f), 3e-4f);
    EXPECT_TRUE(c.get_bool("headless", false));
}

TEST(Config, DefaultsWhenMissing) {
    Config c;
    EXPECT_EQ(c.get_int("num_envs", 1024), 1024);
    EXPECT_FALSE(c.has("num_envs"));
}

TEST(Config, LaterValueWins) {
    Config c;
    c.load_string("seed = 1\nseed = 7\n");
    EXPECT_EQ(c.get_int("seed", 0), 7);
    EXPECT_EQ(c.entries().size(), 1u);
}

TEST(Config, BadLineHasLineNumber) {
    Config c;
    try {
        c.load_string("task = ant\nthis line is wrong\n", "my.txt");
        FAIL() << "expected ConfigError";
    } catch (const ConfigError& e) {
        EXPECT_NE(std::string(e.what()).find("my.txt:2"), std::string::npos) << e.what();
    }
}

TEST(Config, BadNumbersThrow) {
    Config c;
    c.load_string("a = 12x\nb = abc\nc = maybe\nd = 99999999999\n");
    EXPECT_THROW(c.get_int("a", 0), ConfigError);
    EXPECT_THROW(c.get_float("b", 0.0f), ConfigError);
    EXPECT_THROW(c.get_bool("c", false), ConfigError);
    EXPECT_THROW(c.get_int("d", 0), ConfigError);  // out of int range
}

TEST(Config, CommandLineForms) {
    Config c;
    Args a({"--task", "ant", "--num_envs=4096", "--headless", "--lr", "-1e-4", "--viz"});
    c.apply_args(a.argc(), a.argv());
    EXPECT_EQ(c.get_string("task", ""), "ant");
    EXPECT_EQ(c.get_int("num_envs", 0), 4096);
    EXPECT_TRUE(c.get_bool("headless", false));      // bare flag before another flag
    EXPECT_FLOAT_EQ(c.get_float("lr", 0.0f), -1e-4f);  // negative value is a value
    EXPECT_TRUE(c.get_bool("viz", false));           // bare flag at the end
}

TEST(Config, NonFlagArgumentThrows) {
    Config c;
    Args a({"ant"});
    EXPECT_THROW(c.apply_args(a.argc(), a.argv()), ConfigError);
}

TEST(Config, FlagsOverrideFile) {
    fs::path d = tmp_dir();
    fs::path f = d / "cfg.txt";
    std::ofstream(f) << "task = ant\nnum_envs = 4096\nlr = 3e-4\n";
    Args a({"--num_envs", "16", "--config", f.string()});  // --config can be anywhere
    Config c = Config::from_cli(a.argc(), a.argv());
    EXPECT_EQ(c.get_string("task", ""), "ant");      // from file
    EXPECT_EQ(c.get_int("num_envs", 0), 16);          // flag wins
    EXPECT_FLOAT_EQ(c.get_float("lr", 0.0f), 3e-4f);
    EXPECT_FALSE(c.has("config"));
}

TEST(Config, MissingFileThrows) {
    Config c;
    EXPECT_THROW(c.load_file("/no/such/file.txt"), ConfigError);
}

TEST(Config, UnusedKeysShowTypos) {
    Config c;
    c.load_string("gamma = 0.99\nlamda = 0.95\n");  // typo: lamda
    c.get_float("gamma", 0.0f);
    c.get_float("lambda", 0.95f);
    auto unused = c.unused_keys();
    ASSERT_EQ(unused.size(), 1u);
    EXPECT_EQ(unused[0], "lamda");
}

TEST(Config, SaveAndLoadGiveSameConfig) {
    fs::path d = tmp_dir();
    Config a;
    a.load_string("task = ant\nnum_envs = 4096\nlr = 3e-4\n");
    a.save((d / "config.txt").string());
    Config b;
    b.load_file((d / "config.txt").string());
    EXPECT_EQ(a.entries(), b.entries());
}

TEST(Config, MakeRunDirIsUnique) {
    fs::path d = tmp_dir();
    std::string r1 = forge::make_run_dir(d.string(), "ant", 1);
    std::string r2 = forge::make_run_dir(d.string(), "ant", 1);
    EXPECT_TRUE(fs::is_directory(r1));
    EXPECT_TRUE(fs::is_directory(r2));
    EXPECT_NE(r1, r2);
    EXPECT_NE(fs::path(r1).filename().string().find("ant_"), std::string::npos);
    EXPECT_NE(fs::path(r1).filename().string().find("_seed1"), std::string::npos);
    EXPECT_EQ(fs::path(r2).filename().string().substr(fs::path(r1).filename().string().size()), "_2");
}
