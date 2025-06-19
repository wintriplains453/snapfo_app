#ifndef CONVERT_FP_H
#define CONVERT_FP_H

#include <cstdint>

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif

extern "C" {
EXPORT_API void convert_fp32_to_fp16(float* input, uint16_t* output, int32_t size);
EXPORT_API void convert_fp16_to_fp32(uint16_t* input, float* output, int32_t size);
}

#endif