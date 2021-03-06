#ifndef DRP_H
#define DRP_H

#include <vector>
#include <cstdint>

#include "spscqueue.hh"
#include "pgpdriver.h"

struct EventHeader {
    uint64_t pulseId;
    uint64_t timeStamp;
    uint32_t l1Count;
    uint32_t version;
    unsigned rawSamples:24;
    unsigned channelMask:8;
    uint32_t reserved;
};

struct PGPData
{
    uint64_t pulse_id;
    uint8_t buffer_mask;
    unsigned damaged : 1;
    unsigned counter : 7;
    DmaBuffer* buffers[8];
};

struct Parameters
{
    std::string eb_server_ip;
    unsigned contributor_id;
    std::string output_dir;
};

// Per-Event-Buffer-with-Boundaries-Listed-Explicitly
class Pebble
{
public:
    Pebble() : _stack(_stack_buffer) {}
    void* fex_data() {return reinterpret_cast<void*>(_fex_buffer);}
    PGPData* pgp_data;
    void* malloc(size_t size)
    {
        void* curr_stack = _stack;
        _stack += size;
        return curr_stack;
    }
private:
    uint8_t _fex_buffer[128*1024];
    uint8_t _stack_buffer[128*1024];
    uint8_t* _stack;
};


using PebbleQueue = SPSCQueue<Pebble*>;

struct MemPool
{
    MemPool(int num_workers, int num_entries);
    DmaBufferPool dma;
    std::vector<PGPData> pgp_data;
    PebbleQueue pebble_queue;
    std::vector<PebbleQueue> worker_input_queues;
    std::vector<PebbleQueue> worker_output_queues;
    SPSCQueue<int> collector_queue;
    PebbleQueue output_queue;
    int num_entries;
private:
    std::vector<Pebble> pebble;
};

struct Counters
{
    int64_t total_bytes_received;
    int64_t event_count;
    Counters() : total_bytes_received(0), event_count(0) {}
};

class MyBatchManager;

void pin_thread(const pthread_t& th, int cpu);
void monitor_func(std::atomic<Counters*>& p, MemPool& pool, MyBatchManager& myBatchMan);

#endif // DRP_H
