#include "forge/util/config.h"

#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace forge {

namespace {

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

bool valid_key(const std::string& k) {
    if (k.empty()) return false;
    for (char c : k)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == '-'))
            return false;
    return true;
}

}  // namespace

int Config::find(const std::string& key) const {
    for (size_t i = 0; i < entries_.size(); ++i)
        if (entries_[i].first == key) return static_cast<int>(i);
    return -1;
}

const std::string* Config::lookup(const std::string& key) const {
    int i = find(key);
    if (i < 0) return nullptr;
    used_[i] = true;
    return &entries_[i].second;
}

void Config::set(const std::string& key, const std::string& value) {
    if (!valid_key(key)) throw ConfigError("config: invalid key '" + key + "'");
    int i = find(key);
    if (i >= 0) {
        entries_[i].second = value;
    } else {
        entries_.emplace_back(key, value);
        used_.push_back(false);
    }
}

bool Config::has(const std::string& key) const { return find(key) >= 0; }

void Config::load_string(const std::string& text, const std::string& source) {
    std::istringstream in(text);
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos)
            throw ConfigError(source + ":" + std::to_string(line_no) + ": expected 'key = value', got '" +
                              line + "'");
        std::string key = trim(line.substr(0, eq));
        std::string value = trim(line.substr(eq + 1));
        if (!valid_key(key))
            throw ConfigError(source + ":" + std::to_string(line_no) + ": invalid key '" + key + "'");
        set(key, value);
    }
}

void Config::load_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw ConfigError("config: cannot open file '" + path + "'");
    std::stringstream ss;
    ss << f.rdbuf();
    load_string(ss.str(), path);
}

void Config::apply_args(int argc, const char* const* argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--", 0) != 0)
            throw ConfigError("command line: expected a flag starting with --, got '" + a + "'");
        a = a.substr(2);
        std::string key, value;
        size_t eq = a.find('=');
        if (eq != std::string::npos) {
            key = a.substr(0, eq);
            value = a.substr(eq + 1);
        } else {
            key = a;
            // A value follows unless the next word is another flag. Negative
            // numbers like "-1" are values, because they start with one dash only.
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                value = argv[++i];
            } else {
                value = "true";
            }
        }
        if (key == "config") continue;  // handled by from_cli
        set(key, value);
    }
}

Config Config::from_cli(int argc, const char* const* argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            cfg.load_file(argv[i + 1]);
            break;
        }
        if (a.rfind("--config=", 0) == 0) {
            cfg.load_file(a.substr(9));
            break;
        }
    }
    cfg.apply_args(argc, argv);
    return cfg;
}

std::string Config::get_string(const std::string& key, const std::string& def) const {
    const std::string* v = lookup(key);
    return v ? *v : def;
}

int Config::get_int(const std::string& key, int def) const {
    const std::string* v = lookup(key);
    if (!v) return def;
    errno = 0;
    char* end = nullptr;
    long x = std::strtol(v->c_str(), &end, 10);
    if (v->empty() || *end != '\0' || errno == ERANGE || x < INT32_MIN || x > INT32_MAX)
        throw ConfigError("config: '" + key + "' must be an integer, got '" + *v + "'");
    return static_cast<int>(x);
}

float Config::get_float(const std::string& key, float def) const {
    const std::string* v = lookup(key);
    if (!v) return def;
    errno = 0;
    char* end = nullptr;
    float x = std::strtof(v->c_str(), &end);
    if (v->empty() || *end != '\0' || errno == ERANGE)
        throw ConfigError("config: '" + key + "' must be a number, got '" + *v + "'");
    return x;
}

bool Config::get_bool(const std::string& key, bool def) const {
    const std::string* v = lookup(key);
    if (!v) return def;
    if (*v == "true" || *v == "1" || *v == "yes" || *v == "on") return true;
    if (*v == "false" || *v == "0" || *v == "no" || *v == "off") return false;
    throw ConfigError("config: '" + key + "' must be true or false, got '" + *v + "'");
}

std::vector<std::string> Config::unused_keys() const {
    std::vector<std::string> out;
    for (size_t i = 0; i < entries_.size(); ++i)
        if (!used_[i]) out.push_back(entries_[i].first);
    return out;
}

std::string Config::to_string() const {
    std::string s;
    for (const auto& [k, v] : entries_) s += k + " = " + v + "\n";
    return s;
}

void Config::save(const std::string& path) const {
    std::ofstream f(path);
    if (!f) throw ConfigError("config: cannot write file '" + path + "'");
    f << to_string();
}

std::string make_run_dir(const std::string& base, const std::string& task, int seed) {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char date[16];
    std::strftime(date, sizeof(date), "%Y-%m-%d", &tm);

    std::string name = task + "_" + date + "_seed" + std::to_string(seed);
    fs::path dir = fs::path(base) / name;
    for (int k = 2; fs::exists(dir); ++k) dir = fs::path(base) / (name + "_" + std::to_string(k));
    fs::create_directories(dir);
    return dir.string();
}

}  // namespace forge
