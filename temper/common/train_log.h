#pragma once
// log.csv (doc 7.4): one row per iteration. Columns are fixed when the file is
// opened; every row must give a value for each (NaN prints as "nan").
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace temper {

class TrainLog {
public:
    TrainLog(const std::string& path, const std::vector<std::string>& columns) : columns_(columns) {
        f_ = std::fopen(path.c_str(), "w");
        if (!f_) throw std::runtime_error("cannot write " + path);
        for (size_t i = 0; i < columns.size(); ++i) std::fprintf(f_, "%s%s", i ? "," : "", columns[i].c_str());
        std::fprintf(f_, "\n");
        std::fflush(f_);
    }
    ~TrainLog() {
        if (f_) std::fclose(f_);
    }
    TrainLog(const TrainLog&) = delete;
    TrainLog& operator=(const TrainLog&) = delete;

    void row(const std::vector<double>& values) {
        if (values.size() != columns_.size()) throw std::runtime_error("log.csv: wrong number of values in a row");
        for (size_t i = 0; i < values.size(); ++i) std::fprintf(f_, "%s%.6g", i ? "," : "", values[i]);
        std::fprintf(f_, "\n");
        std::fflush(f_);  // the sandbox live curve reads this file while training runs
    }

private:
    std::vector<std::string> columns_;
    std::FILE* f_ = nullptr;
};

}  // namespace temper
