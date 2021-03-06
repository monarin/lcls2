#ifndef PGPREADER_H
#define PGPREADER_H

#include <vector>
#include "drp.hh"

class MovingAverage
{
public:
    MovingAverage(int n);
    int add_value(int value);
private:
    int64_t index;
    int sum;
    int N;
    std::vector<int> values;
};

class PGPReader
{
public:
    PGPReader(MemPool& pool, int device_id, int lanes_mask, int nworkers);
    PGPData* process_lane(DmaBuffer* buffer);
    void send_to_worker(Pebble* pebble_data);
    void send_all_workers(Pebble* pebble);
    void run();
    std::atomic<Counters*>& get_counters() {return m_pcounter;};
private:
    AxisG2Device m_dev;
    MemPool& m_pool;
    int m_nlanes;
    int m_buffer_mask;
    int m_last_complete;
    uint64_t m_worker;
    int m_nworkers;
    MovingAverage m_avg_queue_size;
    Counters m_c1, m_c2;
    std::atomic<Counters*> m_pcounter;
};

#endif // PGPREADER_H
