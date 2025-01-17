#pragma once 

#include <cstddef>
#include <cstdint>

#ifndef ALIGNED
#if WIN32
#define ALIGNED(x)      alignas((x))
#else
#define ALIGNED(x)      __attribute__((aligned((x))))
#endif
#endif

namespace ifm {


static const ALIGNED(4) uint8_t ifm_data0[] =
{
0x29, 0x46, 0x0e, 0x50, 0x4a, 0x59, 0x1f, 0x10, 0x0f, 0x53, 0x41, 0x12, 0x5e, 0x27, 0x64, 0x15, 0x61, 0x74, 0x49, 0x08, 0x23, 0x39, 0x55, 0x4a, 0x0f, 0x37, 0x73, 0x1d, 0x4b, 0x4a, 0x66, 0x2c, 
0x4d, 0x3c, 0x74, 0x42, 0x5f, 0x0b, 0x70, 0x47, 0x34, 0x7a, 0x6a, 0x28, 0x0d, 0x22, 0x61, 0x45, 0x28, 0x2a, 0x36, 0x38, 0x72, 0x74, 0x45, 0x28, 0x4a, 0x6a, 0x4c, 0x34, 0x27, 0x35, 0x19, 0x55, 
0x6f, 0x45, 0x30, 0x71, 0x73, 0x44, 0x41, 0x29, 0x3d, 0x57, 0x67, 0x0c, 0x5e, 0x4c, 0x61, 0x27, 0x22, 0x28, 0x12, 0x03, 0x78, 0x10, 0x17, 0x05, 0x58, 0x4c, 0x17, 0x08, 0x2d, 0x49, 0x6d, 0x02, 
0x05, 0x37, 0x5e, 0x12, 0x61, 0x4f, 0x7e, 0x72, 0x22, 0x66, 0x1e, 0x4a, 0x32, 0x30, 0x43, 0x51, 0x6a, 0x1c, 0x51, 0x3d, 0x04, 0x61, 0x34, 0x33, 0x65, 0x5c, 0x2e, 0x3f, 0x7d, 0x5f, 0x56, 0x42, };


static const ALIGNED(4) uint8_t ifm_data1[] =
{
0x20, 0x7b, 0x38, 0x5c, 0x02, 0x0a, 0x59, 0x28, 0x59, 0x09, 0x2f, 0x45, 0x0b, 0x70, 0x31, 0x65, 0x4e, 0x7f, 0x4b, 0x0d, 0x03, 0x48, 0x6b, 0x11, 0x40, 0x24, 0x6e, 0x4e, 0x57, 0x1d, 0x7c, 0x33, 
0x3f, 0x01, 0x16, 0x25, 0x4c, 0x19, 0x62, 0x60, 0x54, 0x36, 0x1c, 0x67, 0x62, 0x46, 0x1f, 0x39, 0x4c, 0x10, 0x0a, 0x4f, 0x15, 0x42, 0x5a, 0x64, 0x50, 0x62, 0x11, 0x68, 0x30, 0x3f, 0x5a, 0x12, };




const uint8_t* get_buffer(int index) {
    switch (index) {
        case 0:
            return ifm_data0;
        case 1:
            return ifm_data1;
        default:
            return nullptr;
    }
}

uint32_t get_num_buffers(void) 
{
	return 2;
}

} /* namespace ifm */