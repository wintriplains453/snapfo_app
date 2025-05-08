import 'dart:math';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

class Preprocess {
  static Future<Float32List> preprocessImage(
      Uint8List imagePath, {
        int resizeSize = 1024,
        double mean = 0.5,
        double std = 0.5,
      }) async {
    // Загрузка изображения
    img.Image? image = img.decodeImage(imagePath);
    if (image == null) throw Exception('Не удалось декодировать изображение');

    // Изменение размера
    image = _resizeImage(image, resizeSize);

    if (image.width != 1024 || image.height != 1024) {
      throw Exception('Image must be 1024x1024 after resize');
    } else {
      print('image 1024x1024');
    }

    // Нормализация и преобразование в тензор
    return _normalizeAndToTensor(image, mean, std);
  }

  static img.Image _resizeImage(img.Image image, int size) {
    // Сначала изменяем размер с сохранением пропорций (меньшая сторона = size)
    final resized = img.copyResize(
      image,
      width: size,
      height: size,
      interpolation: img.Interpolation.linear,
    );

    // Затем обрезаем до квадрата
    final minDim = min(resized.width, resized.height);
    return img.copyCrop(
      resized,
      x: (resized.width - minDim) ~/ 2,
      y: (resized.height - minDim) ~/ 2,
      width: minDim,
      height: minDim,
    );
  }

  static Float32List _normalizeAndToTensor(img.Image image, double mean, double std) {
    final width = image.width;
    final height = image.height;
    final channels = 3; // RGB

    // Создаем тензор [1, C, H, W]
    final tensor = Float32List(channels * height * width);
    final pixels = image.getBytes(); // Получаем байты изображения в RGB формате

    // Нормализация и транспонирование (HWC -> CHW)
    for (var c = 0; c < channels; c++) {
      for (var h = 0; h < height; h++) {
        for (var w = 0; w < width; w++) {
          final pixelIndex = (h * width + w) * channels + c;
          final value = pixels[pixelIndex] / 255.0;
          final normalized = (value - mean) / std;
          tensor[c * height * width + h * width + w] = normalized.toDouble();
        }
      }
    }

    return tensor;
  }
}
