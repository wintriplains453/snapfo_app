import 'dart:typed_data';
import 'package:image/image.dart' as img;

class ImagePreprocessor {
  static img.Image resizeImage(img.Image input, int size) {
    final w = input.width;
    final h = input.height;
    if (w < h) {
      final newW = size;
      final newH = (h * size / w).round();
      return img.copyResize(input, width: newW, height: newH, interpolation: img.Interpolation.linear);
    } else {
      final newH = size;
      final newW = (w * size / h).round();
      return img.copyResize(input, width: newW, height: newH, interpolation: img.Interpolation.linear);
    }
  }

  static Float32List normalizeAndToTensor({
    required img.Image image,
    double mean = 0.5,
    double std = 0.5,
  }) {
    final w = image.width;
    final h = image.height;
    final c = 3; // RGB

    final Float32List tensor = Float32List(w * h * c);
    int idx = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final pixel = image.getPixel(x, y);
        final r = pixel.r / 255.0;
        final g = pixel.g / 255.0;
        final b = pixel.b / 255.0;

        tensor[idx] = (r - mean) / std;
        tensor[idx + w * h] = (g - mean) / std;
        tensor[idx + 2 * w * h] = (b - mean) / std;
        idx++;
      }
    }
    return tensor;
  }

  /// Overall: decode, resize(1024), normalize => [1,3,H,W]
  static Float32List preprocessImage(Uint8List bytes,
      {int resizeSize = 1024, double mean = 0.5, double std = 0.5}) {
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw Exception("Failed to decode image bytes");
    }
    final resized = resizeImage(decoded, resizeSize);
    return normalizeAndToTensor(image: resized, mean: mean, std: std);
  }
}
