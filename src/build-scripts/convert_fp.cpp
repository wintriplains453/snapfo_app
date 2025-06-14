#include "convert_fp.h"
#include <cmath>
#include <cstdint>
#include <cstring> // Добавляем для std::memcpy

// Преобразование FP32 в FP16
void convert_fp32_to_fp16(float* input, uint16_t* output, int32_t size) {
    for (int32_t i = 0; i < size; ++i) {
        float f = input[i];
        uint32_t f_bits;
        std::memcpy(&f_bits, &f, sizeof(float));

        uint32_t sign = (f_bits >> 16) & 0x8000; // Извлекаем знак
        int32_t exponent = ((f_bits >> 23) & 0xFF) - 127; // Извлекаем экспоненту
        uint32_t mantissa = f_bits & 0x7FFFFF; // Извлекаем мантиссу

        if (exponent <= -15) {
            // Субнормальные или нули
            output[i] = sign;
        } else if (exponent >= 16) {
            // Бесконечность или NaN
            output[i] = sign | 0x7C00;
        } else {
            // Нормализованные значения
            exponent += 15; // Смещение для FP16
            mantissa >>= 13; // Усекаем мантиссу
            output[i] = sign | (exponent << 10) | mantissa;
        }
    }
}

// Преобразование FP16 в FP32
void convert_fp16_to_fp32(uint16_t* input, float* output, int32_t size) {
    for (int32_t i = 0; i < size; ++i) {
        uint16_t h = input[i];
        uint32_t sign = (h & 0x8000) << 16; // Знак
        int32_t exponent = (h & 0x7C00) >> 10; // Экспонента
        uint32_t mantissa = h & 0x03FF; // Мантисса

        uint32_t f_bits;
        if (exponent == 0) {
            // Субнормальные или нули
            f_bits = sign;
        } else if (exponent == 0x1F) {
            // Бесконечность или NaN
            f_bits = sign | 0x7F800000;
        } else {
            exponent = exponent - 15 + 127; // Смещение для FP32
            mantissa <<= 13; // Расширяем мантиссу
            f_bits = sign | (exponent << 23) | mantissa;
        }

        std::memcpy(&output[i], &f_bits, sizeof(float));
    }
}