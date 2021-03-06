#ifndef DIGITIZER_H
#define DIGITIZER_H

#include <vector>

#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "drp.hh"

void hsdExample(XtcData::Xtc& parent, std::vector<XtcData::NameIndex>& namesVec, unsigned nameId, Pebble* pebble_data, uint32_t** dma_buffers, std::vector<unsigned>& lanes);
void add_hsd_names(XtcData::Xtc& parent, std::vector<XtcData::NameIndex>& namesVec);

#endif // DIGITIZER_H
