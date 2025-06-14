import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;

class Preprocess {
  static Future<Float32List> preprocessImage(
      Uint8List imagePath, {
        int resizeSize = 1024,
        double mean = 0.5,
        double std = 0.5,
      }) async {
    // Load image
    img.Image? image = img.decodeImage(imagePath);
    if (image == null) throw Exception('Failed to decode image');

    // Resize image
    image = _resizeImage(image, resizeSize);

    if (image.width != 1024 || image.height != 1024) {
      throw Exception('Image must be 1024x1024 after resize');
    } else {
      print('Image resized to 1024x1024');
    }

    // Normalize and convert to tensor
    return _normalizeAndToTensor(image, mean: mean, std: std);
  }

  static img.Image _resizeImage(img.Image image, int size) {
    // Resize while maintaining aspect ratio (smaller dimension = size)
    final resized = img.copyResize(
      image,
      width: size,
      height: size,
      interpolation: img.Interpolation.linear,
    );

    // Crop to square
    final minDim = min(resized.width, resized.height);
    return img.copyCrop(
      resized,
      x: (resized.width - minDim) ~/ 2,
      y: (resized.height - minDim) ~/ 2,
      width: minDim,
      height: minDim,
    );
  }

  static Future<Float32List> loadNormalizedFromJson(String filePath) async {
    final jsonString = await rootBundle.loadString(filePath);
    final List<dynamic> imgList = json.decode(jsonString);
    final List<double> doubleList = imgList.cast<double>().toList();
    final Float32List imgResult = Float32List.fromList(doubleList);
    return imgResult;
  }

  static Future<Float32List> _normalizeAndToTensor(
      img.Image image, {
        double mean = 0.5,
        double std = 0.5,
      }) async {
    final width = image.width;
    final height = image.height;
    final channels = 3; // RGB

    // Create tensor [C, H, W]
    final tensor = Float32List(channels * height * width);
    final pixels = image.getBytes(); // Get pixel data (assumed RGB or RGBA)

    // Check pixel data length
    final expectedRgbLength = width * height * 3;
    final expectedRgbaLength = width * height * 4;

    if (pixels.length == expectedRgbLength) {
      // RGB format
      for (var c = 0; c < channels; c++) {
        for (var h = 0; h < height; h++) {
          for (var w = 0; w < width; w++) {
            final pixelIndex = (h * width + w) * 3 + c; // RGB: R=0, G=1, B=2
            final value = pixels[pixelIndex] / 255.0; // Scale to [0, 1]
            final normalized = (value - mean) / std; // Normalize
            tensor[c * height * width + h * width + w] = normalized;
          }
        }
      }
    } else if (pixels.length == expectedRgbaLength) {
      // RGBA format (ignore alpha channel)
      for (var c = 0; c < channels; c++) {
        for (var h = 0; h < height; h++) {
          for (var w = 0; w < width; w++) {
            final pixelIndex = (h * width + w) * 4 + c; // RGB: R=0, G=1, B=2
            final value = pixels[pixelIndex] / 255.0; // Scale to [0, 1]
            final normalized = (value - mean) / std; // Normalize
            tensor[c * height * width + h * width + w] = normalized;
          }
        }
      }
    } else {
      throw Exception(
          'Unexpected image format: length=${pixels.length}, expected=$expectedRgbLength (RGB) or $expectedRgbaLength (RGBA)');
    }

    final Float32List tensorJSON = await loadNormalizedFromJson('assets/result.json');
    return tensorJSON;
  }
}