#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace Drp {

// One bounded sample per process. Never retains pointers into DMA/pebble pools.
// The data path only copies into preallocated memory; EndRun starts the dump.
class EpixQuadDiagnostic {
public:
    static constexpr size_t MaxBytes = 2 * 1024 * 1024;
    EpixQuadDiagnostic(const std::string& parent, unsigned frames);
    ~EpixQuadDiagnostic();
    void arm();
    void configuration(const void* data, size_t size);
    void input(uint64_t pulse, uint64_t timestamp, const void* data, size_t size);
    void prewrite(uint64_t pulse, uint64_t timestamp, const void* data, size_t size, bool persist);
    void finish(const std::string& reason);
    void join();
    const std::string& directory() const { return m_directory; }
private:
    struct Sample {
        uint64_t pulse = 0, timestamp = 0;
        size_t inputSize = 0, prewriteSize = 0;
        bool persist = false;
        std::vector<char> input, prewrite;
    };
    void dump(const std::string& reason);
    std::string m_directory;
    std::vector<Sample> m_samples;
    std::vector<char> m_config;
    std::mutex m_mutex;
    std::thread m_writer;
    std::atomic<bool> m_accept{false}, m_finished{false}, m_paired{false};
    bool m_started = false;
    unsigned m_count = 0, m_pairs = 0, m_errors = 0;
};
}
