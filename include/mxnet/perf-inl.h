#pragma once

#include <sys/time.h>
#include <cstdint>

namespace mxnet { 

inline uint64_t get_time(){
  struct timeval t;
  gettimeofday(&t, NULL);
  return static_cast<uint64_t>(t.tv_sec * 1000000 + t.tv_usec);
}

} // namespace mxnet 
