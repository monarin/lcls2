#pragma once

#include "BEBDetector.hh"
#include "psdaq/service/Semaphore.hh"
#include "EpixQuadDiagnostic.hh"
#include <memory>

namespace Drp {

class EpixQuad : public BEBDetector
{
public:
    EpixQuad(Parameters* para, MemPool* pool);
    ~EpixQuad();
    unsigned enable   (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info) override;
    unsigned disable  (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info) override;
    void slowupdate(XtcData::Xtc&, const void* bufEnd) override;
    bool scanEnabled() override;
    void shutdown() override;
    void event(XtcData::Dgram&, const void* bufEnd, PGPEvent*, uint64_t count) override;
    void diagnosticDgram(const XtcData::Dgram&, uint64_t pulseId, bool persist) override;
    void write_image(XtcData::Xtc&, const void* bufEnd, std::vector< XtcData::Array<uint8_t> >&, XtcData::NamesId&);
protected:
    void           _connectionInfo(PyObject*) override;
    unsigned       _configure(XtcData::Xtc&, const void* bufEnd, XtcData::ConfigIter&) override;
    void           _event    (XtcData::Xtc&, const void* bufEnd,
                              uint64_t l1count,
                              std::vector< XtcData::Array<uint8_t> >&) override;
private:
    std::unique_ptr<EpixQuadDiagnostic> m_diagnostic;
    void           _monStreamEnable ();
    void           _monStreamDisable();
protected:
    Pds::Semaphore    m_env_sem;
    bool              m_env_empty;
    XtcData::NamesId  m_evtNamesId[8];
  };

}
