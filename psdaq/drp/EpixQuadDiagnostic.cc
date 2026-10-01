#include "EpixQuadDiagnostic.hh"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

using namespace Drp;
namespace fs = std::filesystem;

namespace {
void writeFile(const fs::path& path, const void* data, size_t size)
{
    std::ofstream out;
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.open(path, std::ios::binary);
    out.write(static_cast<const char*>(data), size);
    out.close();
}
}

EpixQuadDiagnostic::EpixQuadDiagnostic(const std::string& parent, unsigned frames)
{
    if (!frames || frames > 64 || !fs::path(parent).is_absolute() || !fs::is_directory(parent))
        throw std::invalid_argument("epixDiagDir must be an existing absolute directory; epixDiagFrames must be 1..64");
    // Allocate and touch all event pages before any detector initialization.
    m_samples.resize(frames);
    for (auto& s : m_samples) {
        s.input.resize(MaxBytes);
        s.prewrite.resize(MaxBytes);
    }
    const auto stamp = std::chrono::system_clock::now().time_since_epoch().count();
    m_directory = (fs::path(parent) / ("epix-diag-" + std::to_string(getpid()) + "-" + std::to_string(stamp))).string();
    if (!fs::create_directory(m_directory)) throw std::runtime_error("Diagnostic directory already exists");
    writeFile(fs::path(m_directory)/".probe", "", 0);
    fs::remove(fs::path(m_directory)/".probe");
    std::cout << "EPIX DIAGNOSTIC prepared: " << m_directory << " frames=" << frames
              << " memory_bytes=" << 2*MaxBytes*frames << "; dump after EndRun" << std::endl;
}

EpixQuadDiagnostic::~EpixQuadDiagnostic()
{
    finish("destructor-before-EndRun");
    join();
}

void EpixQuadDiagnostic::arm()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_started && !m_finished) {
        m_started = true;
        m_accept = true;
    }
}

void EpixQuadDiagnostic::configuration(const void* data, size_t size)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_finished) return;
    if (m_count || size > 16*1024*1024 || !data || !size) { ++m_errors; return; }
    m_config.assign(static_cast<const char*>(data), static_cast<const char*>(data)+size);
}

void EpixQuadDiagnostic::input(uint64_t pulse, uint64_t timestamp, const void* data, size_t size)
{
    if (!m_accept) return;
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_accept || m_finished || m_count == m_samples.size()) return;
    auto& s = m_samples[m_count++];
    s.pulse = pulse;
    s.timestamp = timestamp;
    if (!data || !size || size > MaxBytes) ++m_errors;
    else { std::memcpy(s.input.data(), data, size); s.inputSize = size; }
    if (m_count == m_samples.size()) m_accept = false;
}

void EpixQuadDiagnostic::prewrite(uint64_t pulse, uint64_t timestamp, const void* data, size_t size, bool persist)
{
    if (m_finished || m_paired) return;
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_finished) return;
    for (unsigned i=0; i<m_count; ++i) {
        auto& s = m_samples[i];
        if (s.pulse != pulse || s.timestamp != timestamp) continue;
        if (s.prewriteSize || !data || !size || size > MaxBytes) { ++m_errors; return; }
        std::memcpy(s.prewrite.data(), data, size);
        s.prewriteSize = size;
        s.persist = persist;
        if (++m_pairs == m_samples.size()) m_paired = true;
        return;
    }
}

void EpixQuadDiagnostic::finish(const std::string& reason)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_finished.exchange(true)) return;
    m_accept = false;
    try { m_writer = std::thread([this, reason]() { dump(reason); }); }
    catch (const std::exception& e) {
        std::cerr << "EPIX DIAGNOSTIC dump not started: " << e.what() << std::endl;
    }
}

void EpixQuadDiagnostic::join() { if (m_writer.joinable()) m_writer.join(); }

void EpixQuadDiagnostic::dump(const std::string& reason)
{
    try {
        nlohmann::json rows = nlohmann::json::array();
        writeFile(fs::path(m_directory)/"configure.xtc2", m_config.data(), m_config.size());
        for (unsigned i=0; i<m_count; ++i) {
            const auto& s = m_samples[i];
            std::string input = "input-" + std::to_string(i) + ".bin";
            std::string prewrite = "prewrite-" + std::to_string(i) + ".xtc2";
            writeFile(fs::path(m_directory)/input, s.input.data(), s.inputSize);
            writeFile(fs::path(m_directory)/prewrite, s.prewrite.data(), s.prewriteSize);
            rows.push_back({{"slot",i}, {"pulse_id",s.pulse}, {"timestamp",s.timestamp},
                            {"input",input}, {"input_bytes",s.inputSize},
                            {"prewrite",prewrite}, {"prewrite_bytes",s.prewriteSize},
                            {"submitted_to_file_writer",s.persist}});
        }
        const bool complete = reason == "EndRun" && m_count == m_samples.size() &&
                              m_pairs == m_count && !m_errors && !m_config.empty();
        nlohmann::json manifest = {{"schema_version",1}, {"status",complete ? "complete" : "incomplete"},
            {"finish_reason",reason}, {"requested_frames",m_samples.size()}, {"captured",m_count},
            {"paired",m_pairs}, {"capture_errors",m_errors}, {"samples",rows},
            {"scope","One sample per DRP process; incoming lane-0 DMA and datagram immediately before writer submission"}};
        const auto text = manifest.dump(2) + "\n";
        writeFile(fs::path(m_directory)/"manifest.json.tmp", text.data(), text.size());
        fs::rename(fs::path(m_directory)/"manifest.json.tmp", fs::path(m_directory)/"manifest.json");
        std::cout << "EPIX DIAGNOSTIC saved: " << m_directory << " status="
                  << manifest["status"] << " paired=" << m_pairs << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "EPIX DIAGNOSTIC save failed: " << m_directory << ": " << e.what() << std::endl;
        // No final manifest is published on failure; partial files are not a result.
    }
}
