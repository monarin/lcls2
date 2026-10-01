// Offline test: no detector, driver, EPICS or DAQ process is constructed.
#include "EpixQuadDiagnostic.hh"
#include <nlohmann/json.hpp>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

using Drp::EpixQuadDiagnostic;
namespace fs = std::filesystem;
nlohmann::json manifest(const std::string& dir) {
    std::ifstream in(fs::path(dir)/"manifest.json");
    nlohmann::json value; in >> value; return value;
}
std::vector<char> bytes(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}
int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string parent = argv[1];
    for (unsigned n : {0u, 65u}) {
        bool rejected = false;
        try { EpixQuadDiagnostic d(parent,n); } catch (const std::invalid_argument&) { rejected=true; }
        assert(rejected);
    }
    {
        EpixQuadDiagnostic d(parent,8);
        const std::string config = "configure-copy";
        d.configuration(config.data(), config.size());
        d.input(999,999,"ignored-before-arm",18);
        d.arm();
        std::vector<std::thread> workers;
        for (unsigned t=0; t<4; ++t) workers.emplace_back([&d,t]() {
            for (unsigned i=0; i<2; ++i) {
                unsigned id=t*2+i+1;
                std::vector<char> input(1000, char(id));
                d.input(id,id+1000,input.data(),input.size());
                input.assign(input.size(),char(id+20)); // Simulate recycled DMA.
                d.prewrite(id,id+1000,input.data(),input.size(),true);
                input.assign(input.size(),0); // Simulate recycled pebble.
            }
        });
        for (auto& t : workers) t.join();
        d.arm(); // Must not reset or rearm a full sample.
        d.input(100,100,"over-limit",10);
        assert(fs::is_empty(d.directory())); // No acquisition-time file writes.
        d.finish("EndRun"); d.join();
        d.prewrite(1,1001,"late",4,true);
        d.finish("second-EndRun"); // No replacement/double dump.
        auto m = manifest(d.directory());
        assert(m["status"] == "complete" && m["captured"] == 8 && m["paired"] == 8);
        assert(bytes(fs::path(d.directory())/"configure.xtc2") == std::vector<char>(config.begin(),config.end()));
        for (auto& s : m["samples"]) {
            auto id=s["pulse_id"].get<unsigned>();
            assert(bytes(fs::path(d.directory())/s["input"].get<std::string>()) == std::vector<char>(1000,char(id)));
            assert(bytes(fs::path(d.directory())/s["prewrite"].get<std::string>()) == std::vector<char>(1000,char(id+20)));
        }
    }
    {
        EpixQuadDiagnostic d(parent,2);
        d.configuration("c",1); d.arm();
        d.input(1,11,"input",5);
        d.prewrite(1,12,"wrong timestamp",15,true);
        d.input(2,22,"bad size",EpixQuadDiagnostic::MaxBytes+1);
        d.finish("EndRun"); d.join();
        auto m=manifest(d.directory());
        assert(m["status"] == "incomplete" && m["paired"] == 0 && m["capture_errors"] == 1);
    }
    std::string interrupted;
    {
        EpixQuadDiagnostic d(parent,1); interrupted=d.directory();
        d.configuration("c",1); d.arm();
        d.input(1,11,"input",5); d.prewrite(1,11,"formatted",9,false);
    }
    auto m=manifest(interrupted);
    assert(m["status"] == "incomplete" && m["finish_reason"] == "destructor-before-EndRun");
    assert(m["samples"][0]["submitted_to_file_writer"] == false);
    std::cout << "PASS: bounded allocation, concurrent matching, owned copies, no live I/O, one-shot, errors, shutdown\n";
}
