import 'dart:typed_data';
import 'package:flutter/src/widgets/framework.dart';
import 'package:image/image.dart' as img;
import 'preprocess.dart';
import 'inference_runner.dart';

class ImageEditor {
  static Future<img.Image> prepareNp(Float32List x, List<int> shape) async {
    final channels = shape[1];
    final height = shape[2];
    final width = shape[3];

    if (x.length != channels * height * width) {
      throw Exception('Invalid input size: expected ${channels * height * width}, got ${x.length}');
    }

    // Диагностика: выводим диапазон значений для каждого канала
    for (var c = 0; c < channels; c++) {
      final start = c * height * width;
      final end = start + height * width;
      final channelData = x.sublist(start, end);
      final min = channelData.reduce((a, b) => a < b ? a : b);
      final max = channelData.reduce((a, b) => a > b ? a : b);
      print('Channel $c (RGB order: ${["R", "G", "B"][c]}): min=$min, max=$max');
    }

    final out = Float32List(height * width * channels);
    for (var h = 0; h < height; h++) {
      for (var w = 0; w < width; w++) {
        // Input tensor is in CHW format, RGB order
        final rIndex = 0 * height * width + h * width + w; // Red
        final gIndex = 1 * height * width + h * width + w; // Green
        final bIndex = 2 * height * width + h * width + w; // Blue
        final dstIndex = (h * width + w) * channels;

        // Normalize from [-1, 1] to [0, 255]
        out[dstIndex] = ((x[rIndex] + 1) / 2).clamp(0, 1) * 255; // R
        out[dstIndex + 1] = ((x[gIndex] + 1) / 2).clamp(0, 1) * 255; // G
        out[dstIndex + 2] = ((x[bIndex] + 1) / 2).clamp(0, 1) * 255; // B
      }
    }

    final image = img.Image(width: width, height: height);
    for (var h = 0; h < height; h++) {
      for (var w = 0; w < width; w++) {
        final index = (h * width + w) * channels;
        image.setPixelRgba(
          w,
          h,
          out[index].round(),     // R
          out[index + 1].round(), // G
          out[index + 2].round(), // B
          255,
        );
      }
    }

    return image;
  }

  static Future<img.Image> edit({
    required Uint8List inputBytes,
    required String editingName,
    required double editingDegree,
    bool align = false,
    bool combinedPreEditor = false,
    required BuildContext context,
  }) async {
    if (combinedPreEditor) {
      throw UnimplementedError('combinedPreEditor is not supported');
    }

    final start = DateTime.now();

    // Заглушка для align
    if (align) {
      throw UnimplementedError('Image alignment not implemented');
    }

    final tensor = await Preprocess.preprocessImage(
      inputBytes,
      resizeSize: 1024,
      mean: 0.5,
      std: 0.5,
    );

    print('Preprocess image: ${DateTime.now().difference(start).inMilliseconds} ms');

    // Запуск инференса
    final inferenceStart = DateTime.now();
    final (invImages, inversionResults) = await InferenceRunner.runOnBatch(tensor);
    final editedImage = await InferenceRunner.runEditingOnBatch(
      resultBatch: inversionResults,
      editingName: editingName,
      editingDegree: editingDegree,
      context: context,
    );

    print('Inference: ${DateTime.now().difference(inferenceStart).inMilliseconds} ms');

    // Постобработка

    final image = await Preprocess.loadNormalizedFromJson('assets/result.json');
    print('Blue channel sample: ${editedImage.sublist(0, 10)}');
    print('Green channel sample: ${editedImage.sublist(1024 * 1024, 1024 * 1024 + 10)}');
    print('Red channel sample: ${editedImage.sublist(2 * 1024 * 1024, 2 * 1024 * 1024 + 10)}');

    final resultImage = await prepareNp(editedImage, [1, 3, 1024, 1024]);

    print('Total: ${DateTime.now().difference(start).inMilliseconds} ms');
    return resultImage;
  }
}