import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:snapfo_app/onnx_wrapper.dart';
import 'latent_editor.dart';
import 'package:flutter/material.dart';

class ResultBatch {
  final Float32List latents;
  final Float32List fusedFeat;
  final Float32List predictedFeat;
  final Float32List wE4e;
  final Float32List input;

  ResultBatch({
    required this.latents,
    required this.fusedFeat,
    required this.predictedFeat,
    required this.wE4e,
    required this.input,
  });
}

class InferenceRunner {
  // Сессии для всех моделей
  static CustomSession? _interpolateSession;
  static CustomSession? _invertSession;
  static CustomSession? _fuserSession;
  static CustomSession? _e4eEncoderSession;
  static CustomSession? _decoderWithoutNewFeatureSession;
  static CustomSession? _decoderRgbWithoutNewFeatureSession;
  static CustomSession? _encoderSession;
  static CustomSession? _decoderWithNewFeatureSession;
  static CustomSession? _decoderRgbWithNewFeatureSession;

  static final _interfaceganSessions = <String, CustomSession>{};

  /// Утилита для логирования статистики тензора
  static void _logTensorStats(String stage, List<Float32List> tensors, List<String> names) {
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

  /// Инициализация окружения
  static Future<void> initEnv() async {
    try {
      await OnnxWrapper.initEnv();
    } catch (e) {
      print('Error initializing ONNX environment: $e');
      rethrow;
    }
  }

  /// Загрузка всех моделей
  static Future<void> loadModels() async {
    print('[InferenceRunner.loadModels] Starting...');

    // Загружаем все модели
    await Future.wait([
      _loadModel('interpolate', 'assets/models/interpolate.onnx'),
      _loadModel('invert', 'assets/models/invert_compressed.onnx'),
      _loadModel('fuser', 'assets/models/fuser_compressed.onnx'),
      _loadModel('e4e_encoder', 'assets/models/e4e_encoder_compressed.onnx'),
      _loadModel('decoder_without_new_feature', 'assets/models/decoder_without_new_feature.onnx'),
      _loadModel('decoder_rgb_without_new_feature', 'assets/models/decoder_rgb_without_new_feature.onnx'),
      _loadModel('encoder', 'assets/models/encoder_compressed.onnx'),
      _loadModel('decoder_with_new_feature', 'assets/models/decoder_with_new_feature_compressed.onnx'),
      _loadModel('decoder_rgb_with_new_feature', 'assets/models/decoder_rgb_with_new_feature.onnx'),
      _loadModel('interfacegan_age', 'assets/models/interfacegan_age.onnx'),
      _loadModel('decoder_stylespace', 'assets/models/decoder_stylespace.onnx'),
      _loadModel('clip_text_encoder_compressed', 'assets/models/clip_text_encoder_compressed.onnx'),
    ]);

    // Инициализируем сессии
    _interpolateSession = CustomSession('interpolate');
    _invertSession = CustomSession('invert');
    _fuserSession = CustomSession('fuser');
    _e4eEncoderSession = CustomSession('e4e_encoder');
    _decoderWithoutNewFeatureSession = CustomSession('decoder_without_new_feature');
    _decoderRgbWithoutNewFeatureSession = CustomSession('decoder_rgb_without_new_feature');
    _encoderSession = CustomSession('encoder');
    _decoderWithNewFeatureSession = CustomSession('decoder_with_new_feature');
    _decoderRgbWithNewFeatureSession = CustomSession('decoder_rgb_with_new_feature');
    _interfaceganSessions['age'] = CustomSession('interfacegan_age');
    CustomSession('decoder_stylespace');
    CustomSession('clip_text_encoder_compressed');

    print('[loadModels] All models loaded successfully!');

  }

  static final TensorPool _tensorPool = TensorPool();

  static void dispose() {
    _interpolateSession?.dispose();
    _invertSession?.dispose();
    _fuserSession?.dispose();
    _e4eEncoderSession?.dispose();
    _decoderWithoutNewFeatureSession?.dispose();
    _decoderRgbWithoutNewFeatureSession?.dispose();
    _encoderSession?.dispose();
    _decoderWithNewFeatureSession?.dispose();
    _decoderRgbWithNewFeatureSession?.dispose();
    _interfaceganSessions.forEach((_, session) => session.dispose());
    _interfaceganSessions.clear();
    _tensorPool.clear();
    print('[InferenceRunner.dispose]');
  }

  static Future<void> _loadModel(String key, String assetPath) async {
    try {
      await OnnxWrapper.loadModel(key: key, assetPath: assetPath);
      print('Model $key loaded successfully');
    } catch (e) {
      print('Error loading model $key: $e');
      rethrow;
    }
  }

  // Аналог Python: runInterfacegan
  static Future<Float32List> runInterfacegan(
      Float32List latent,
      double degree,
      String editingName,
      ) async {
    final session = _interfaceganSessions[editingName];
    if (session == null) {
      throw Exception('$editingName model not loaded. Call loadModels()');
    }

    final latentTensor = CustomTensor.createTensorWithDataList(latent, [1, latent.length]);
    final degreeTensor = CustomTensor.createTensorWithDataList(
      Float32List.fromList([degree]),
      [1],
    );

    final results = await session.runAsync(
      CustomRunOptions(),
      {
        'latent': latentTensor,
        'degree': degreeTensor,
      },
      outputNames: ['output'],
    );

    if (results.isEmpty || results[0] == null) {
      throw Exception('No output from $editingName');
    }

    return results[0]!;
  }

  // Аналог Python: run_on_batch
  static Future<(Float32List, ResultBatch)> runOnBatch(Float32List inputTensor) async {
    // Шаг 1: Запуск interpolate.onnx
    final xOut = await _runInterpolate(inputTensor);
    final x = xOut[0];
    xOut.clear();

    // Шаг 2: Запуск invert.onnx
    final invertOut = await _runInvert(x);
    final wRecon = invertOut[0];
    final predictedFeat = invertOut[1];
    invertOut.clear();

    // Шаг 3: Запуск decoder_without_new_feature.onnx
    final decoderOut = await _runDecoderWithoutNewFeature(wRecon);
    final wFeat = decoderOut[1];
    decoderOut.clear();

    // Шаг 4: Запуск fuser.onnx
    final fusedFeat = await _runFuser(_concatAlongAxis1(predictedFeat, wFeat));

    // Шаг 5: Запуск e4e_encoder.onnx
    final wE4e = await _runE4eEncoder(x);
    // OnnxWrapper.disposeModel('e4e_encoder');

    // Формируем image
    final imageOut = await _runDecoderWithNewFeature([wRecon, fusedFeat]);
    final image = imageOut[0];

    final resultBatch = ResultBatch(
      latents: wRecon,
      fusedFeat: fusedFeat,
      predictedFeat: predictedFeat,
      wE4e: wE4e,
      input: inputTensor,
    );

    return (image, resultBatch);
  }

  // Аналог Python: run_editing_on_batch
  static Future<Float32List> runEditingOnBatch({
    required ResultBatch resultBatch,
    required String editingName,
    required double editingDegree,
    required BuildContext context,
  }) async {
    return runEditingCore(
      latent: resultBatch.latents,
      wE4e: resultBatch.wE4e,
      fusedFeat: resultBatch.fusedFeat,
      editingName: editingName,
      editingDegree: editingDegree,
      context: context,
    );
  }

  // Аналог Python: run_editing_core
  static Future<Float32List> runEditingCore({
    required Float32List latent,
    required Float32List wE4e,
    required Float32List fusedFeat,
    required String editingName,
    required double editingDegree,
    required BuildContext context,
  }) async {
    // 1) Получаем отредактированные латентные представления
    final editedLatents = await LatentEditor.getEditedLatent(latent, editingName, editingDegree, context);
    final editedWE4e = await LatentEditor.getEditedLatent(wE4e, editingName, editingDegree, context);

    // 2) Проверяем, используется ли stylespace
    final isStylespace = editedLatents is (List<Float32List>, List<Float32List>);

    // 3) Обрабатываем оригинальное w_e4e
    final outOrig = await _runDecoderWithoutNewFeature(wE4e);
    final fsX = outOrig[1];

    // 4) Обрабатываем отредактированное w_e4e
    late List<Float32List> secondOut;
    if (isStylespace) {
      final (arrA, arrB) = editedWE4e as (List<Float32List>, List<Float32List>);

      final selectedInputs = [
        ...arrA.sublist(0, 9), // style_1–9 (должны быть 512)
        ...arrB.sublist(0, 5), // to_rgb_stylespace_1–5 (должны быть 512)
      ];

      secondOut = await _runDecoderRgbWithoutNewFeature(selectedInputs);
    } else {
      secondOut = await _runDecoderWithoutNewFeature(editedWE4e as Float32List);
    }
    final fsY = secondOut[1];
    secondOut.clear();

    // 5) Вычисляем дельту
    final delta = _elementwiseSubtract(fsX, fsY);

    // 6) Получаем отредактированные фичи
    final cat = _concatAlongAxis1(fusedFeat, delta);
    final editedFeatOut = await _runEncoder(cat);
    final editedFeat = editedFeatOut[0];

    // 7) Генерируем финальное изображение
    late List<Float32List> finalOut;
    if (isStylespace) {
      final (arrA, arrB) = editedLatents as (List<Float32List>, List<Float32List>);
      finalOut = await _runDecoderRgbWithNewFeature([...arrA, ...arrB, editedFeat]);
    } else {
      finalOut = await _runDecoderWithNewFeature([editedLatents as Float32List, editedFeat]);
    }
    // editedFeat.clear();
    return finalOut[0];
  }

  // Вспомогательные методы для запуска ONNX-моделей
  static Future<List<Float32List>> _runInterpolate(Float32List input) async {
    if (_interpolateSession == null) throw Exception('Interpolate model not loaded');
    final inputTensor = CustomTensor.createTensorWithDataList(input, [1, 3, 1024, 1024]);
    final results = await _interpolateSession!.runAsync(
      CustomRunOptions(),
      {'x': inputTensor},
      outputNames: ['output'],
    );
    if (results.isEmpty || results[0] == null) throw Exception('No output from interpolate');
    print('_runInterpolate success return!!!!');
    return [results[0]!];
  }

  static Future<List<Float32List>> _runInvert(Float32List input) async {
    if (_invertSession == null) throw Exception('Invert model not loaded');
    if (input.length != 1 * 3 * 256 * 256) {
      throw Exception('Input tensor has incorrect length: ${input.length}, expected ${1 * 3 * 256 * 256}');
    }

    final inputTensor = CustomTensor.createTensorWithDataList(input, [1, 3, 256, 256]);
    final results = await _invertSession!.runAsync(
      CustomRunOptions(),
      {'input.1': inputTensor},
      outputNames: ['w_recon', 'predicted_feat'],
    );
    if (results.length != 2 || results.any((r) => r == null)) throw Exception('Invalid outputs from invert');
    print('_runInvert success return!!!!');
    return results.cast<Float32List>();
  }

  static Future<Float32List> _runFuser(Float32List input) async {
    if (_fuserSession == null) throw Exception('Fuser model not loaded');
    final inputTensor = CustomTensor.createTensorWithDataList(input, [1, 1024, 64, 64]);
    final results = await _fuserSession!.runAsync(
      CustomRunOptions(),
      {'x': inputTensor},
      outputNames: ['fused_feat'],
    );
    if (results.isEmpty || results[0] == null) throw Exception('No output from fuser');
    ('_runFuser', [results[0]!], ['fused_feat']);
    print('_runFuser success return!!!!');
    return results[0]!;
  }

  static Future<Float32List> _runE4eEncoder(Float32List input) async {
    if (_e4eEncoderSession == null) throw Exception('E4eEncoder model not loaded');
    final inputTensor = CustomTensor.createTensorWithDataList(input, [1, 3, 256, 256]);
    final results = await _e4eEncoderSession!.runAsync(
      CustomRunOptions(),
      {'input.1': inputTensor},
      outputNames: ['w_e4e'],
    );
    if (results.isEmpty || results[0] == null) throw Exception('No output from e4e_encoder');
    print('_runE4eEncoder success return!!!!');
    return results[0]!;
  }

  static Future<List<Float32List>> _runDecoderWithoutNewFeature(Float32List input) async {
    if (_decoderWithoutNewFeatureSession == null) throw Exception('Decoder model not loaded');
    final inputTensor = CustomTensor.createTensorWithDataList(input, [1, 18, 512]);
    final results = await _decoderWithoutNewFeatureSession!.runAsync(
      CustomRunOptions(),
      {'latent': inputTensor},
      outputNames: ['image', 'feature'],
    );
    if (results.length != 2 || results.any((r) => r == null)) throw Exception('Invalid outputs from decoder');
    print('_runDecoderWithoutNewFeature success return!!!!');
    return results.cast<Float32List>();
  }

  static Future<List<Float32List>> _runDecoderRgbWithoutNewFeature(List<Float32List> inputs) async {
    if (_decoderRgbWithoutNewFeatureSession == null) throw Exception('Decoder RGB model not loaded');

    // for (int i = 0; i < inputs.length; i++) {
    //   print('inputs[$i] length: ${inputs[i].length}');
    // }

    final inputMap = <String, CustomTensor>{};
    for (int i = 0; i < 9; i++) {
      if (inputs[i].length != 512) {
        throw Exception('style_${i + 1} has incorrect length: ${inputs[i].length}, expected 512');
      }
      inputMap['style_${i + 1}'] = CustomTensor.createTensorWithDataList(inputs[i], [1, 512]);
    }
    for (int i = 0; i < 5; i++) {
      inputMap['to_rgb_stylespace_${i + 1}'] = CustomTensor.createTensorWithDataList(inputs[9 + i], [1, 512]);
    }

    final results = await _decoderRgbWithoutNewFeatureSession!.runAsync(
      CustomRunOptions(),
      inputMap,
      outputNames: ['image', 'feature'],
    );
    if (results.length != 2 || results.any((r) => r == null)) throw Exception('Invalid outputs from RGB decoder');
    print('_runDecoderRgbWithoutNewFeature success return!!!!');
    return results.cast<Float32List>();
  }

  static Future<List<Float32List>> _runEncoder(Float32List input) async {
    if (_encoderSession == null) throw Exception('Encoder model not loaded');
    final inputTensor = CustomTensor.createTensorWithDataList(input, [1, 1024, 64, 64]);
    final results = await _encoderSession!.runAsync(
      CustomRunOptions(),
      {'input.1': inputTensor},
      outputNames: ['edited_feat'],
    );
    if (results.isEmpty || results[0] == null) throw Exception('No output from encoder');
    print('_runEncoder success return!!!!');
    return [results[0]!];
  }

  static Future<List<Float32List>> _runDecoderWithNewFeature(List<Float32List> inputs) async {
    if (_decoderWithNewFeatureSession == null) throw Exception('Decoder with new feature model not loaded');

    final latent = inputs[0];
    final newFeature = inputs[1];

    final latentTensor = CustomTensor.createTensorWithDataList(latent, [1, 18, 512]);
    final newFeatureTensor = CustomTensor.createTensorWithDataList(newFeature, [1, 512, 64, 64]);

    final results = await _decoderWithNewFeatureSession!.runAsync(
      CustomRunOptions(),
      {
        'latent': latentTensor,
        'onnx::ConvTranspose_1': newFeatureTensor,
      },
      outputNames: ['image'],
    );
    if (results.isEmpty || results[0] == null) throw Exception('No output from decoder with new feature');
    print('_runDecoderWithNewFeature success return!!!!');
    return [results[0]!];
  }

  static Future<List<Float32List>> _runDecoderRgbWithNewFeature(List<Float32List> inputs) async {
      if (_decoderRgbWithNewFeatureSession == null) {
        throw Exception("Decoder RGB with new feature model not loaded");
      }

      final expectedLengths = [
        512, 512, 512, 512, 512, 512, 512, 512, 512, 512, // style_1–10
        256, 256, 128, 128, 64, 64, 32, // style_11–17
        512, 512, 512, 512, 512, 256, 128, 64, 32, // to_rgb_stylespace_1–9
        512 * 64 * 64 // new_feature
      ];
      for (var i = 0; i < inputs.length; i++) {
        // print('inputs[$i] length: ${inputs[i].length}, expected: ${expectedLengths[i]}');
        if (inputs[i].length != expectedLengths[i]) {
          throw Exception('Input $i has length ${inputs[i].length}, expected ${expectedLengths[i]}');
        }
      }

      final inputMap = <String, CustomTensor>{};
      for (int i = 0; i < 10; i++) {
        inputMap['style_${i + 1}'] = CustomTensor.createTensorWithDataList(inputs[i], [1, 512]);
      }
      inputMap['style_11'] = CustomTensor.createTensorWithDataList(inputs[10], [1, 256]);
      inputMap['style_12'] = CustomTensor.createTensorWithDataList(inputs[11], [1, 256]);
      inputMap['style_13'] = CustomTensor.createTensorWithDataList(inputs[12], [1, 128]);
      inputMap['style_14'] = CustomTensor.createTensorWithDataList(inputs[13], [1, 128]);
      inputMap['style_15'] = CustomTensor.createTensorWithDataList(inputs[14], [1, 64]);
      inputMap['style_16'] = CustomTensor.createTensorWithDataList(inputs[15], [1, 64]);
      inputMap['style_17'] = CustomTensor.createTensorWithDataList(inputs[16], [1, 32]);
      for (int i = 0; i < 5; i++) {
        inputMap['to_rgb_stylespace_${i + 1}'] = CustomTensor.createTensorWithDataList(inputs[17 + i], [1, 512]);
      }
      inputMap['to_rgb_stylespace_6'] = CustomTensor.createTensorWithDataList(inputs[22], [1, 256]);
      inputMap['to_rgb_stylespace_7'] = CustomTensor.createTensorWithDataList(inputs[23], [1, 128]);
      inputMap['to_rgb_stylespace_8'] = CustomTensor.createTensorWithDataList(inputs[24], [1, 64]);
      inputMap['to_rgb_stylespace_9'] = CustomTensor.createTensorWithDataList(inputs[25], [1, 32]);
      inputMap['new_feature'] = CustomTensor.createTensorWithDataList(inputs[26], [1, 512, 64, 64]);

      // Выполнение модели
      final results = await _decoderRgbWithNewFeatureSession!.runAsync(
        CustomRunOptions(),
        inputMap,
        outputNames: ['image'],
      );

      if (results.isEmpty || results[0] == null) {
        throw Exception('[Isolate] No output from RGB decoder with new feature');
      }

      print('_runDecoderRgbWithNewFeature success');
      return [results[0]!];
  }

  // Утилиты
  static Float32List _elementwiseSubtract(Float32List a, Float32List b) {
    if (a.length != b.length) throw Exception('Arrays length mismatch');
    final key = 'subtract_${a.length}';
    final result = _tensorPool.getBuffer(key, a.length);
    for (int i = 0; i < a.length; i++) {
      result[i] = a[i] - b[i];
    }
    return result;
  }

  static Float32List _concatAlongAxis1(Float32List a, Float32List b) {
    final key = 'concat_${a.length}_${b.length}';
    final result = _tensorPool.getBuffer(key, a.length + b.length);
    result.setAll(0, a);
    result.setAll(a.length, b);
    return result;
  }
}

class TensorPool {
  final Map<String, Float32List> _buffers = {};

  Float32List getBuffer(String key, int size) {
    if (size < 0) {
      throw ArgumentError('Buffer size cannot be negative: $size');
    }
    if (!_buffers.containsKey(key) || _buffers[key]!.length != size) {
      _buffers[key] = Float32List(size);
      print('[TensorPool] Allocated new buffer for key=$key, size=$size');
    } else {
      print('[TensorPool] Reusing buffer for key=$key, size=$size');
    }
    return _buffers[key]!;
  }

  void clear() {
    _buffers.clear();
    print('[TensorPool] Cleared all buffers');
  }

  int get bufferCount => _buffers.length;
  int get totalSize => _buffers.values.fold(0, (sum, buffer) => sum + buffer.length);
}