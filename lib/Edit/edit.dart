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

    final out = Float32List(height * width * channels);
    for (var h = 0; h < height; h++) {
      for (var w = 0; w < width; w++) {
        for (var c = 0; c < channels; c++) {
          final srcIndex = c * height * width + h * width + w;
          final dstIndex = (h * width + w) * channels + c;
          out[dstIndex] = (x[srcIndex] + 1) / 2;
          if (out[dstIndex] < 0) out[dstIndex] = 0;
          if (out[dstIndex] > 1) out[dstIndex] = 1;
          out[dstIndex] *= 255;
        }
      }
    }

    final image = img.Image(width: width, height: height);
    for (var h = 0; h < height; h++) {
      for (var w = 0; w < width; w++) {
        final index = (h * width + w) * channels;
        image.setPixelRgba(
          w,
          h,
          out[index].round(),
          out[index + 1].round(),
          out[index + 2].round(),
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
    final resultImage = await prepareNp(editedImage, [1, 3, 1024, 1024]);

    print('Total: ${DateTime.now().difference(start).inMilliseconds} ms');
    return resultImage;
  }
}