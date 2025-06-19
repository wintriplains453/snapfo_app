import 'dart:typed_data';

class Float16Converter {
  // Converts Float32List to Uint16List representing IEEE 754 float16
  static Uint16List toFloat16(Float32List input) {
    final output = Uint16List(input.length);
    final byteData = ByteData(input.length * 4);

    for (int i = 0; i < input.length; i++) {
      final float32 = input[i];
      byteData.setFloat32(i * 4, float32, Endian.little);

      if (float32.isNaN) {
        output[i] = 0x7FFF; // NaN in float16
        continue;
      }
      if (float32.isInfinite) {
        output[i] = float32.isNegative ? 0xFC00 : 0x7C00; // -Inf or +Inf
        continue;
      }

      final bits = byteData.getInt32(i * 4, Endian.little);
      final sign = (bits >> 31) & 0x1;
      final exponent = ((bits >> 23) & 0xFF) - 127 + 15; // Adjust bias
      final mantissa = (bits >> 13) & 0x3FF; // Take 10 bits of mantissa

      if (exponent >= 31) {
        output[i] = (sign << 15) | 0x7C00; // Infinity
      } else if (exponent <= 0) {
        output[i] = (sign << 15); // Zero or denormal
      } else {
        output[i] = (sign << 15) | (exponent << 10) | mantissa;
      }
    }
    return output;
  }

  // Converts Uint16List (float16) to Float32List
  static Float32List fromFloat16(Uint16List input) {
    final output = Float32List(input.length);
    final byteData = ByteData(4);

    for (int i = 0; i < input.length; i++) {
      final half = input[i];
      final sign = (half >> 15) & 0x1;
      final exponent = (half >> 10) & 0x1F;
      final mantissa = half & 0x3FF;

      if (exponent == 0x1F) {
        if (mantissa == 0) {
          output[i] = sign == 1 ? double.negativeInfinity : double.infinity;
        } else {
          output[i] = double.nan;
        }
      } else if (exponent == 0) {
        output[i] = 0.0; // Zero or denormal
      } else {
        final floatBits = (sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13);
        byteData.setInt32(0, floatBits, Endian.little);
        output[i] = byteData.getFloat32(0, Endian.little);
      }
    }
    return output;
  }
}