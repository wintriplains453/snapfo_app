import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:logger/logger.dart';
import '../onnx_wrapper.dart';
import 'dart:convert';

// Настройка логирования
final logger = Logger(
  printer: PrettyPrinter(),
  output: MultiOutput([
    ConsoleOutput(),
    FileOutput(file: File('styleclip_dart.log')),
  ]),
);

void _logTensorStats(String stage, List<Float32List> tensors, List<String> names) {
  for (int i = 0; i < tensors.length; i++) {
    final tensor = tensors[i];
    final name = names[i];
    bool hasNaN = false;
    bool hasInfinity = false;
    double minVal = double.infinity;
    double maxVal = -double.infinity;
    for (var val in tensor) {
      if (val.isNaN) hasNaN = true;
      if (val.isInfinite) hasInfinity = true;
      if (!val.isNaN && !val.isInfinite) {
        minVal = minVal < val ? minVal : val;
        maxVal = maxVal > val ? maxVal : val;
      }
    }
    print('$stage [$name] stats: hasNaN=$hasNaN, hasInfinity=$hasInfinity, min=$minVal, max=$maxVal, length=${tensor.length}');
  }
}

class StyleClipEditor {
  static const templates = [
    'a bad photo of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a photo of my {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a pixelated photo of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a photo of one {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'a low resolution photo of a {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a pixelated photo of a {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'a dark photo of a {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
  ];

  static const styleSpaceDimensions = [
    512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, // 15 x 512
    256, 256, 256, // 3 x 256
    128, 128, 128, // 3 x 128
    64, 64, 64, // 3 x 64
    32, 32, // 2 x 32
  ];

  static const toRgbIndices = [1, 4, 7, 10, 13, 16, 19, 22, 25];
  static const styleSpaceIndicesWithoutToRgb = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24];

  // Функция для загрузки токенов из JSON
  static Future<List<List<int>>> loadTokensFromJson() async {
    try {
      final jsonString = await rootBundle.loadString('assets/tokens/blonde.json');
      final List<dynamic> jsonData = json.decode(jsonString);
      final tokens = jsonData.map((dynamic row) => (row as List<dynamic>).cast<int>()).toList();
      return tokens;
    } catch (e) {
      throw Exception('Failed to load tokens from JSON: $e');
    }
  }

  static Future<(List<Float32List>, List<Float32List>)> getStyleclipGlobalEdits(
      List<Float32List> startS,
      double factor,
      String editingName,
      context,
      ) async {
    // Парсинг имени редактирования
    final direction = editingName.replaceFirst('styleclip_global_', '');
    final parts = direction.split('_');
    final neutralText = parts[0];
    final targetText = parts[1];
    final disentangleStr = parts[2];
    final disentanglement = double.parse(disentangleStr);

    // Формирование текстовых промптов
    final allSentences = [
      ...templates.map((t) => t.replaceFirst('{}', targetText)),
      ...templates.map((t) => t.replaceFirst('{}', neutralText)),
    ];
    final tokens = await loadTokensFromJson();
    print('Loaded tokens: ${tokens.length} sentences, each with ${tokens[0].length} tokens');

    if (tokens.length != 84 || tokens.any((t) => t.length != 77)) {
      throw Exception('Invalid tokens: expected 84 sentences with 77 tokens each');
    }

    // Запуск ONNX-модели
    final session = CustomSession('clip_text_encoder_compressed');
    final disentangleTensor = CustomTensor(Float32List.fromList([disentanglement]), [1]);
    final flatTokens = tokens.expand((tokenList) => tokenList).toList();
    final tokensTensor = CustomTensor(
      Int32List.fromList(flatTokens),
      [allSentences.length, 77],
      type: 'int32',
    );

    final outputs = await session.runAsync(
      CustomRunOptions(),
      {
        'input.1': tokensTensor,
        'onnx::Less_1': disentangleTensor,
      },
      outputNames: List.generate(26, (i) => 'o${i + 1}'),
    );

    // Проверка размеров выходных тензоров
    for (var i = 0; i < outputs.length; i++) {
      final expectedLength = styleSpaceDimensions[i];
      if (outputs[i]!.length != expectedLength) {
        print('Warning: Output o${i + 1} has length ${outputs[i]!.length}, expected $expectedLength. Using zero-filled tensor.');
        outputs[i] = Float32List(expectedLength);
      }
    }

    // Разделение на StyleSpace и toRGB направления
    final editsSs = <Float32List>[];
    final editsRgb = <Float32List>[];
    for (var i = 0; i < styleSpaceDimensions.length; i++) {
      if (styleSpaceIndicesWithoutToRgb.contains(i)) {
        editsSs.add(Float32List.fromList(outputs[i]!));
      } else if (toRgbIndices.contains(i)) {
        editsRgb.add(Float32List.fromList(outputs[i]!));
      }
    }

    _logTensorStats('editsSs', editsSs, List.generate(editsSs.length, (i) => 'edit_ss_$i'));
    _logTensorStats('editsRgb', editsRgb, List.generate(editsRgb.length, (i) => 'edit_rgb_$i'));

    // Проверка на некорректные значения
    bool hasInvalidValues(Float32List tensor) {
      return tensor.any((val) => val.isNaN || val.isInfinite);
    }
    if (editsSs.any(hasInvalidValues) || editsRgb.any(hasInvalidValues)) {
      throw Exception('Invalid values in editsSs or editsRgb');
    }

    // Применение фактора к StyleSpace направлениям
    final editedSsList = <Float32List>[];
    for (var i = 0; i < styleSpaceIndicesWithoutToRgb.length; i++) {
      final ssIndex = styleSpaceIndicesWithoutToRgb[i];
      if (ssIndex >= startS.length) {
        print('Warning: ssIndex $ssIndex out of range for startS length ${startS.length}');
        continue;
      }
      final orig = startS[ssIndex];
      final delta = editsSs[i];
      final out = Float32List(orig.length);
      if (delta.length == 1) {
        // Broadcasting scalar value
        for (var j = 0; j < orig.length; j++) {
          out[j] = orig[j] + (factor / 1.5) * delta[0];
        }
      } else {
        // Broadcasting delta to match orig length
        for (var j = 0; j < orig.length; j++) {
          final deltaIndex = delta.length > 1 ? (j % delta.length) : 0;
          out[j] = orig[j] + (factor / 1.5) * delta[deltaIndex];
        }
      }
      editedSsList.add(out);
    }

    // Применение фактора к toRGB направлениям
    final editedRgbList = <Float32List>[];
    for (var i = 0; i < toRgbIndices.length; i++) {
      final rgbIndex = toRgbIndices[i];
      if (rgbIndex >= startS.length) {
        print('Warning: rgbIndex $rgbIndex out of range for startS length ${startS.length}');
        continue;
      }
      final orig = startS[rgbIndex];
      final delta = editsRgb[i];
      final out = Float32List(orig.length);
      if (delta.length == 1) {
        // Broadcasting scalar value
        for (var j = 0; j < orig.length; j++) {
          out[j] = orig[j] + (1.0 / 1.5) * delta[0];
        }
      } else {
        // Broadcasting delta to match orig length
        for (var j = 0; j < orig.length; j++) {
          final deltaIndex = delta.length > 1 ? (j % delta.length) : 0;
          out[j] = orig[j] + (1.0 / 1.5) * delta[deltaIndex];
        }
      }
      editedRgbList.add(out);
    }

    print('style_clip success!!!');
    return (editedSsList, editedRgbList);
  }
}