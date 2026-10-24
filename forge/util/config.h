#pragma once
// Plain text config: one "key = value" per line, '#' starts a comment (doc 7.2).
// Command-line flags override the file. A copy is saved in the run folder.
//
//   Config cfg = Config::from_cli(argc, argv);   // reads --config FILE, then flags
//   int n = cfg.get_int("num_envs", 1024);
//
// Flags: "--key value", "--key=value", or "--flag" alone (means true).
// Errors (bad line, bad number, missing file) throw ConfigError with a clear message.
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace forge {

struct ConfigError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class Config {
public:
    // Parse text. 'source' is only used in error messages (file name).
    void load_string(const std::string& text, const std::string& source = "<string>");
    void load_file(const std::string& path);

    // Apply command-line flags (argv[0] is skipped). --config is handled by from_cli.
    void apply_args(int argc, const char* const* argv);

    // Load --config FILE first (if given), then apply all other flags on top.
    static Config from_cli(int argc, const char* const* argv);

    void set(const std::string& key, const std::string& value);
    bool has(const std::string& key) const;

    // get_* return the default when the key is missing.
    // They throw ConfigError when the value cannot be parsed.
    std::string get_string(const std::string& key, const std::string& def) const;
    int get_int(const std::string& key, int def) const;
    float get_float(const std::string& key, float def) const;
    bool get_bool(const std::string& key, bool def) const;

    // Keys that were set but never read. Print them as a warning: they are
    // usually typos ("lamda" instead of "lambda").
    std::vector<std::string> unused_keys() const;

    // "key = value" lines in insertion order. Loading this text gives the same config.
    std::string to_string() const;
    void save(const std::string& path) const;

    const std::vector<std::pair<std::string, std::string>>& entries() const { return entries_; }

private:
    int find(const std::string& key) const;
    const std::string* lookup(const std::string& key) const;

    std::vector<std::pair<std::string, std::string>> entries_;
    mutable std::vector<bool> used_;
};

// Creates runs/<task>_<YYYY-MM-DD>_seed<seed>/ (adds _2, _3 ... if it exists)
// and returns its path. Doc section 7.3.
std::string make_run_dir(const std::string& base, const std::string& task, int seed);

}  // namespace forge
