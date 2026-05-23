from libc.stdint cimport uint64_t, uint32_t


cdef struct Xtc1ClockTime:
    uint32_t low
    uint32_t high


cdef struct Xtc1TimeStamp:
    uint32_t low
    uint32_t high


cdef struct Xtc1Sequence:
    Xtc1ClockTime clock
    Xtc1TimeStamp stamp
    uint32_t env


cdef struct Xtc1Damage:
    uint32_t value


cdef struct Xtc1Src:
    uint32_t log
    uint32_t phy


cdef struct Xtc1TypeId:
    uint32_t value


cdef struct Xtc1Xtc:
    Xtc1Damage damage
    Xtc1Src src
    Xtc1TypeId contains
    uint32_t extent


cdef struct Xtc1Dgram:
    Xtc1Sequence seq
    Xtc1Xtc xtc
