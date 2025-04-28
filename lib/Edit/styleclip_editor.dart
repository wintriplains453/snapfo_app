import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:logger/logger.dart';
import '../onnx_wrapper.dart';
import 'simple_tokenizer.dart';
import 'dart:convert';

// Настройка логирования
final logger = Logger(
  printer: PrettyPrinter(),
  output: MultiOutput([
    ConsoleOutput(),
    FileOutput(file: File('styleclip_dart.log')),
  ]),
);

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

  static const toRgbIndices = [1, 4, 7, 10, 13, 16, 19, 22, 25]; // Индексы toRGB слоев
  static const styleSpaceIndicesWithoutToRgb = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15];

  static Future<(List<Float32List>, List<Float32List>)> getStyleclipGlobalEdits(
      List<Float32List> startS,
      double factor,
      String editingName,
      context,
      ) async {
    logger.i('Starting getStyleclipGlobalEdits: editingName=$editingName, factor=$factor');

    // Парсинг имени редактирования
    final direction = editingName.replaceFirst('styleclip_global_', '');
    final parts = direction.split('_');
    if (parts.length != 3) {
      logger.e('Invalid editing name format: $editingName');
      throw Exception('Invalid editing name format: $editingName');
    }
    final neutralText = parts[0];
    final targetText = parts[1];
    final disentangleStr = parts[2];
    final disentanglement = double.parse(disentangleStr);
    logger.i('Parsed: neutralText=$neutralText, targetText=$targetText, disentanglement=$disentanglement');

    // Формирование текстовых промптов
    final allSentences = [
      ...templates.map((t) => t.replaceFirst('{}', targetText)),
      ...templates.map((t) => t.replaceFirst('{}', neutralText)),
    ];
    logger.i('Generated ${allSentences.length} sentences');

    // Токенизация
    final tokenizer = await SimpleTokenizer.create(context: context);
    final tokens = await tokenize(
      allSentences,
      tokenizer: tokenizer,
      contextLength: 77,
      truncate: true,
      context: context,
    );
    print(tokenizer.bpe("dog"));
    print(tokenizer.encoder.length);
    print(tokenizer.encoder.entries.take(5));
    print(tokenizer.decode([49406, 320, 2103, 335, 606, 531, 539, 320, 1710, 593]));
    print('Tokens shape: ${tokens.length}x${tokens[0].length}');
    print('Sample tokens: ${tokens[0].sublist(0, 10)}');
    logger.i('Tokenized: ${tokens.length} token lists, sample_tokens=${tokens[0].sublist(0, 10)}');

    // Запуск clip_text_encoder_compressed.onnx
    logger.i('Running ONNX model');
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
    logger.i('ONNX model output: ${outputs.length} directions');
    for (var i = 0; i < outputs.length; i++) {
      final output = outputs[i]!;
      final mean = output.fold(0.0, (sum, x) => sum + x) / output.length;
      final variance = output.fold(0.0, (sum, x) => sum + (x - mean) * (x - mean)) / output.length;
      final std = sqrt(variance); // Use sqrt from dart:math
      logger.d('Direction $i: length=${output.length}, mean=$mean, std=$std');
      if (output.any((x) => x.isNaN || x.isInfinite)) {
        logger.w('Direction $i contains NaN or Infinite values');
      }
    }

    // Разделение на StyleSpace и toRGB направления
    final editsSs = styleSpaceIndicesWithoutToRgb.map((i) => outputs[i]!).toList();
    final editsRgb = toRgbIndices.map((i) => outputs[i]!).toList();
    logger.i('StyleSpace directions: ${editsSs.length}, toRGB directions: ${editsRgb.length}');

    // Применение фактора к StyleSpace направлениям
    final editedSsList = <Float32List>[];
    for (var i = 0; i < editsSs.length; i++) {
      final orig = startS[i];
      final delta = editsSs[i];
      final out = Float32List(orig.length);
      for (var j = 0; j < orig.length; j++) {
        out[j] = orig[j] + (factor / 1.5) * delta[j % delta.length];
      }
      final mean = out.fold(0.0, (sum, x) => sum + x) / out.length;
      final variance = out.fold(0.0, (sum, x) => sum + (x - mean) * (x - mean)) / out.length;
      final std = sqrt(variance); // Use sqrt from dart:math
      logger.d('Edited StyleSpace $i: length=${out.length}, mean=$mean, std=$std');
      editedSsList.add(out);
    }

    // Применение фактора 1.0 к toRGB слоям
    final editedRgbList = <Float32List>[];
    for (var i = 0; i < editsRgb.length; i++) {
      final orig = startS[editsSs.length + i];
      final delta = editsRgb[i];
      final out = Float32List(orig.length);
      for (var j = 0; j < orig.length; j++) {
        out[j] = orig[j] + (1.0 / 1.5) * delta[j % delta.length];
      }
      final mean = out.fold(0.0, (sum, x) => sum + x) / out.length;
      final variance = out.fold(0.0, (sum, x) => sum + (x - mean) * (x - mean)) / out.length;
      final std = sqrt(variance); // Use sqrt from dart:math
      logger.d('Edited toRGB $i: length=${out.length}, mean=$mean, std=$std');
      editedRgbList.add(out);
    }

    logger.i('Completed getStyleclipGlobalEdits');
    return (editedSsList, editedRgbList);
  }
}