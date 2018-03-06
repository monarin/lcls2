//g++ -Wall -std=c++11 -I /reg/neh/home/yoon82/temp/lcls2 peak.cpp psalg/src/PeakFinderAlgos.cpp psalg/src/LocalExtrema.cpp -o peak

#include <iostream>
#include <vector>
#include <stdlib.h>

#include "psalg/include/PeakFinderAlgos.h"
#include "psalg/include/LocalExtrema.h"
#include "psalg/include/Types.h"

#include "xtcdata/xtc/DescData.hh" // Array

#include <chrono> // timer
typedef std::chrono::high_resolution_clock Clock;

using namespace psalgos;

class Foo {
public:
  Foo(){
    //rank = 1;
  }
  ~Foo(){
    std::cout << "destruct foo" << std::endl;
  }

private:
  //uint32_t rank;
  // vectors are 24 bytes
  std::vector<uint16_t> v;// = {1};
};

int main () {

  Foo *aaa = new Foo();
  //std::cout << "Foo: " << sizeof(aaa) << std::endl;
  delete aaa;

  float *dat = new float[6];
  dat[0] = 9;
  dat[1] = 8;
  dat[2] = 7;
  dat[3] = 6;
  dat[4] = 5;
  dat[5] = 4;
  uint32_t *shape = new uint32_t[2];
  shape[0] = 2;
  shape[1] = 3;
  uint32_t dim = 2;

  XtcData::Array<float> arr1;
  arr1(dat, shape, dim);

  std::cout << "arr1 rank: " << arr1.rank() << std::endl;
  XtcData::Array<float> arr(dat, shape, dim);
  std::cout << "val: " << arr(1,2) << std::endl;
  std::cout << "rank: " << arr.rank() << std::endl;
  std::cout << arr.shape()[0] << std::endl;
  std::cout << arr.data()[2] << std::endl;

  // DRP: you are given PEBBLE, data, mask

  // Step 0: fake data and mask
  unsigned int rows = 185;
  unsigned int cols = 388;
  int16_t *data  = new int16_t[rows*cols];
  int16_t *data1 = new int16_t[rows*cols];
  for(unsigned int i=0; i<rows*cols; i++) {
      data[i]  = rand() % 10;
      data1[i] = data[i];
  }
  data[1900] = 1000; // peak 1
  data[1901] = 900;
  data[1902] = 900;
  data[5900] = 500; // peak 2
  data[5901] = 800;
  data[5902] = 300;
  /*
  data1[1900] = 1000; // peak 1
  data1[1901] = 900;
  data1[1902] = 900;
  data1[5900] = 500; // peak 2
  data1[5901] = 800;
  data1[5902] = 300;*/

  data1[900] = 1000; // peak 1
  data1[901] = 900;
  data1[902] = 900;
  data1[3900] = 500; // peak 2
  data1[3901] = 800;
  data1[3902] = 300;
  uint16_t *mask = new uint16_t[rows*cols];
  for(unsigned int i=0; i<rows*cols; i++) {
      mask[i] = 1;
  }

  //uint8_t *buf = NULL;
  uint8_t *buf = new uint8_t[10240*100000*2]; // PEBBLE fex_stack

  auto t1 = Clock::now();
    
  // Step 1: Init PeakFinderAlgos
  const size_t  seg = 0;
  const unsigned pbits = 1;
  if (pbits) std::cout << "+buf Address stored " << (void *) buf << std::endl;
  
  PeakFinderAlgos *ptr;
  if (!buf) {
      ptr = new PeakFinderAlgos(seg, pbits);
  } else {
      std::cout << "drp peakfinderalgos" << std::endl;
      //ptr = new(buf) PeakFinderAlgos(seg, pbits, buf+sizeof(PeakFinderAlgos)); // TODO: initialize outside the pebble, and reuse
      ptr = new PeakFinderAlgos(seg, pbits, buf);
  }

  auto t2 = Clock::now();

  // Step 2: Set params
  const float npix_min = 2;
  const float npix_max = 30; 
  const float amax_thr = 200;
  const float atot_thr = 600;
  const float son_min = 7;
  ptr->setPeakSelectionPars(npix_min, npix_max, amax_thr, atot_thr, son_min);
  
  auto t3 = Clock::now();

  // Step 3: Peak finder
  const size_t rank = 3;
  const double r0 = 4;
  const double dr = 2;
  const double nsigm = 0;
  ptr->peakFinderV3r3(data, mask, rows, cols, rank, r0, dr, nsigm);

  auto t4 = Clock::now();

  if (pbits) std::cout << "*** BUF *** " << (void*) buf << std::endl; // TODO: update buf to the latest address

  std::cout << "Analyzing fake data, Delta t: " << std::endl
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl
            << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << std::endl
            << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << std::endl
            << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t1).count() 
            << " milliseconds" << std::endl;

  ptr->_printVectorOfPeaks_drp(ptr->vectorOfPeaksSelected_drp());

  //std::vector<std::vector<float> > a = ptr->peaksSelected();
  //std::cout << a[1][1] << std::endl;

  std::cout << "_convPeaksSelected: " << ptr->rows[1] << " " << ptr->cols[1] << std::endl;

  ptr->peakFinderV3r3(data1, mask, rows, cols, rank, r0, dr, nsigm);
  std::cout << "_convPeaksSelected: " << ptr->rows[1] << " " << ptr->cols[1] << std::endl;

  delete ptr;

// DATA
// Peak 1
//Seg:  0 Row:   4 Col: 348 Npix: 24 Imax:  995.7 Itot: 2843.5 CGrav r:   4.0 c: 349.0 Sigma r: 0.29 c: 0.86 Rows[   1:   7] Cols[ 345: 351] B:  4.3 N:  1.8 S/N:314.7
// Peak 2
//Seg:  0 Row:  15 Col:  81 Npix: 23 Imax:  796.0 Itot: 1637.3 CGrav r:  15.0 c:  80.8 Sigma r: 0.35 c: 0.74 Rows[  12:  18] Cols[  78:  83] B:  4.0 N:  2.2 S/N:157.5

// DATA1
// Peak 1
//Seg:  0 Row:   2 Col: 124 Npix: 18 Imax:  994.6 Itot: 2808.5 CGrav r:   2.0 c: 125.0 Sigma r: 0.21 c: 0.85 Rows[   1:   5] Cols[ 121: 127] B:  5.4 N:  2.3 S/N:289.4
// Peak 2
//Seg:  0 Row:  10 Col:  21 Npix: 16 Imax:  795.4 Itot: 1611.8 CGrav r:  10.0 c:  20.9 Sigma r: 0.28 c: 0.71 Rows[   7:  13] Cols[  18:  22] B:  4.6 N:  1.7 S/N:241.4

  return 0;
}
