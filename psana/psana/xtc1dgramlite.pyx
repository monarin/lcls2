from cpython.buffer cimport (
    PyBUF_ANY_CONTIGUOUS,
    PyBUF_SIMPLE,
    PyBuffer_Release,
    PyObject_GetBuffer,
)
from psana.xtc1dgramlite cimport Xtc1Dgram, Xtc1Sequence, Xtc1Xtc


cdef unsigned _xtc1_to_xtc2_service(unsigned raw_service):
    # LCLS1 TransitionId values differ from psana2 for several transitions.
    if raw_service == 4:   # Configure
        return 2
    if raw_service == 6:   # BeginRun
        return 4
    if raw_service == 7:   # EndRun
        return 5
    if raw_service == 8:   # BeginCalibCycle
        return 6           # BeginStep
    if raw_service == 9:   # EndCalibCycle
        return 7           # EndStep
    if raw_service == 10:  # Enable
        return 8
    if raw_service == 11:  # Disable
        return 9
    return raw_service


cdef class Xtc1DgramLite:
    cdef uint64_t _payload
    cdef uint64_t _timestamp
    cdef uint64_t _size
    cdef unsigned _service
    cdef unsigned _raw_service
    cdef uint32_t _env
    cdef uint32_t _damage
    cdef uint32_t _src_log
    cdef uint32_t _src_phy
    cdef uint32_t _contains
    cdef uint32_t _extent

    def __init__(self, view):
        """Create a light-weight XTC1 datagram header view from a buffer object."""
        cdef Xtc1Dgram* d
        cdef char* view_ptr
        cdef Py_buffer buf

        PyObject_GetBuffer(view, &buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
        if buf.len < sizeof(Xtc1Dgram):
            PyBuffer_Release(&buf)
            raise ValueError(
                f"XTC1 datagram header requires {sizeof(Xtc1Dgram)} bytes, "
                f"got {buf.len}"
            )

        view_ptr = <char *>buf.buf
        d = <Xtc1Dgram *>(view_ptr)

        if d.xtc.extent < sizeof(Xtc1Xtc):
            PyBuffer_Release(&buf)
            raise ValueError(
                f"Invalid XTC1 extent {d.xtc.extent}; "
                f"minimum is {sizeof(Xtc1Xtc)}"
            )

        self._payload = d.xtc.extent - sizeof(Xtc1Xtc)
        self._timestamp = <uint64_t>d.seq.clock.high << 32 | d.seq.clock.low
        self._raw_service = (d.seq.stamp.low >> 24) & 0xf
        self._service = _xtc1_to_xtc2_service(self._raw_service)
        self._size = sizeof(Xtc1Sequence) + d.xtc.extent
        self._env = d.seq.env
        self._damage = d.xtc.damage.value
        self._src_log = d.xtc.src.log
        self._src_phy = d.xtc.src.phy
        self._contains = d.xtc.contains.value
        self._extent = d.xtc.extent

        PyBuffer_Release(&buf)

    @property
    def payload(self):
        return self._payload

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def service(self):
        """Transition id mapped onto psana2 TransitionId values."""
        return self._service

    @property
    def raw_service(self):
        """Original LCLS1 TransitionId value from the datagram timestamp."""
        return self._raw_service

    @property
    def size(self):
        return self._size

    @property
    def env(self):
        return self._env

    @property
    def damage(self):
        return self._damage

    @property
    def src_log(self):
        return self._src_log

    @property
    def src_phy(self):
        return self._src_phy

    @property
    def contains(self):
        return self._contains

    @property
    def extent(self):
        return self._extent
