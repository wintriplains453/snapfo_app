import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter/services.dart';
import 'package:snapfo_app/onnx_wrapper.dart';
// import 'package:onnxruntime/onnxruntime.dart';

// Suppose you have a `LatentEditor.getEditedLatent(...)` function
// that returns either a Float32List or a tuple of two Lists: (List<Float32List>, List<Float32List>)
import 'latent_editor.dart';

// ====================== Вспомогательные классы ======================
class CustomTensor {
  final Float32List data;
  final List<int> shape;

  CustomTensor(this.data, this.shape);

  factory CustomTensor.createTensorWithDataList(Float32List data, List<int> shape) {
    return CustomTensor(data, shape);
  }

  Map<String, dynamic> toMap() {
    return {
      'data': data,
      'shape': shape,
    };
  }
}

class CustomRunOptions {
  void release() {} // Пустой метод для совместимости
}

class CustomSession {
  final String sessionKey;

  CustomSession(this.sessionKey);

  Future<List<Float32List?>> runAsync(
      CustomRunOptions runOptions,
      Map<String, CustomTensor> inputs, {
        List<String> outputNames = const ['output'],
      }) async {
    final inputMap = inputs.map((key, value) => MapEntry(key, value.toMap()));

    final results = await OnnxWrapper.runComplex(
      sessionKey: sessionKey,
      inputs: inputMap,
      outputNames: outputNames,
    );

    return outputNames.map((name) => results[name]).toList();
  }
}
// ====================== Конец вспомогательных классов ======================

class InferenceRunner {
  // Сессии для всех моделей
  static CustomSession? _preEditorSession;
  static CustomSession? _interfaceganAgeSession;
  static CustomSession? _decoderWithoutNewFeatureSession;
  static CustomSession? _decoderRgbWithoutNewFeatureSession;
  static CustomSession? _encoderSession;
  static CustomSession? _decoderWithNewFeatureSession;
  static CustomSession? _decoderRgbWithNewFeatureSession;

  /// Инициализация окружения
  static Future<void> initEnv() async {
    try {
      await OnnxWrapper.initEnv();
    } catch (e) {
      print("Error initializing ONNX environment: $e");
      rethrow;
    }
  }

  /// Загрузка всех моделей
  static Future<void> loadModels() async {
    print("[InferenceRunner.loadModels] Starting...");

    // Загружаем все модели
    await Future.wait([
      _loadModel('interfacegan_age', 'assets/models/interfacegan_age.onnx'),
      _loadModel('pre_editor', 'assets/models/pre_editor.onnx'),
      // _loadModel('decoder_without_new_feature', 'assets/models/decoder_without_new_feature.onnx'),
      // _loadModel('decoder_rgb_without_new_feature', 'assets/models/decoder_rgb_without_new_feature.onnx'),
      // _loadModel('encoder', 'assets/models/encoder.onnx'),
      // _loadModel('decoder_with_new_feature', 'assets/models/decoder_with_new_feature.onnx'),
      // _loadModel('decoder_rgb_with_new_feature', 'assets/models/decoder_rgb_with_new_feature.onnx'),
    ]);

    // Инициализируем сессии
    _preEditorSession = CustomSession('pre_editor');
    _interfaceganAgeSession = CustomSession('interfacegan_age');
    _decoderWithoutNewFeatureSession = CustomSession('decoder_without_new_feature');
    _decoderRgbWithoutNewFeatureSession = CustomSession('decoder_rgb_without_new_feature');
    _encoderSession = CustomSession('encoder');
    _decoderWithNewFeatureSession = CustomSession('decoder_with_new_feature');
    _decoderRgbWithNewFeatureSession = CustomSession('decoder_rgb_with_new_feature');

    print("[loadModels] All models loaded successfully!");
  }

  static Future<void> _loadModel(String key, String assetPath) async {
    try {
      await OnnxWrapper.loadModel(key: key, assetPath: assetPath);
      print("Model $key loaded successfully");
    } catch (e) {
      print("Error loading model $key: $e");
      rethrow;
    }
  }

  // ====================== Основные методы ======================

  // --------------------------------------------------------------------------
  // (A) runPreEditor => returns [image, w_recon, w_e4e, fused_feat]
  // Python: run_pre_editor(x)
  // --------------------------------------------------------------------------
  static Future<List<Float32List>> runPreEditor(Float32List inputTensor) async {
    if (_preEditorSession == null) {
      throw Exception("PreEditor model not loaded. Call loadModels() first.");
    }

    final input = CustomTensor.createTensorWithDataList(
        inputTensor,
        [1, 3, 1024, 1024] // Форма входного тензора
    );

    final results = await _preEditorSession!.runAsync(
      CustomRunOptions(),
      {'input': input},
      outputNames: ['image', 'w_recon', 'w_e4e', 'fused_feat'],
    );

    if (results.length != 4 || results.any((r) => r == null)) {
      throw Exception("Invalid outputs from pre_editor");
    }

    // --------------------------------------------------------------------------
    // (B) Example runInterfacegan => for "age" or similar
    // In Python: run_onnx(interfacegan_age, (latent, degree))

    return results.cast<Float32List>();
  }

  // --------------------------------------------------------------------------
  // (B) Example runInterfacegan => for "age" or similar
  // In Python: run_onnx(interfacegan_age, (latent, degree))
  // You can replicate more editing models as needed.
  // --------------------------------------------------------------------------
  static Future<Float32List> runInterfacegan(Float32List latent, double degree) async {
    if (_interfaceganAgeSession == null) {
      throw Exception("interfacegan_age model not loaded. Call loadModels()");
    }

    final latentTensor = CustomTensor.createTensorWithDataList(latent, [1, latent.length]);
    final degreeTensor = CustomTensor.createTensorWithDataList(
      Float32List.fromList([degree]),
      [1],
    );

    final results = await _interfaceganAgeSession!.runAsync(
      CustomRunOptions(),
      {
        'latent': latentTensor,
        'degree': degreeTensor,
      },
    );

    if (results.isEmpty || results[0] == null) {
      throw Exception("No output from interfacegan_age");
    }

    return results[0]!;
  }

  // --------------------------------------------------------------------------
  // (C) runEditingCore - EXACT replication of your Python function
  //
  //   def run_editing_core(latent, w_e4e, fused_feat, editing_name, editing_degree):
  //       edited_latents = get_edited_latent(latent, editing_name, editing_degree)
  //       edited_w_e4e    = get_edited_latent(w_e4e, editing_name, editing_degree)
  //       is_stylespace = isinstance(edited_latents, tuple)
  //
  //       e4e_inv, fs_x = run_onnx(decoder_without_new_feature, (w_e4e,))
  //       if is_stylespace:
  //           e4e_edit, fs_y = run_onnx(decoder_rgb_without_new_feature, tuple(edited_w_e4e[0] + edited_w_e4e[1]))
  //       else:
  //           e4e_edit, fs_y = run_onnx(decoder_without_new_feature, (edited_w_e4e,))
  //
  //       delta = fs_x - fs_y
  //       edited_feat = run_onnx(encoder, (concat(fused_feat, delta),))[0]
  //
  //       if is_stylespace:
  //           image_edit = run_onnx(decoder_rgb_with_new_feature, tuple(edited_latents[0] + edited_latents[1] + [edited_feat]))
  //       else:
  //           image_edit = run_onnx(decoder_with_new_feature, tuple([edited_latents] + [edited_feat]))
  //
  //       image_edit = image_edit[0]
  //       return image_edit
  // --------------------------------------------------------------------------
  static Future<Float32List> runEditingCore({
    required Float32List latent,
    required Float32List wE4e,
    required Float32List fusedFeat,
    required String editingName,
    required double editingDegree,
  }) async {
    // 1) Получаем отредактированные латентные представления
    final editedLatents = await LatentEditor.getEditedLatent(latent, editingName, editingDegree);
    final editedWE4e = await LatentEditor.getEditedLatent(wE4e, editingName, editingDegree);

    // Проверяем, используется ли stylespace
    final isStylespace = editedLatents is (List<Float32List>, List<Float32List>);

    // 2) Обрабатываем оригинальное изображение
    final outOrig = await _runDecoderWithoutNewFeature(wE4e);
    final fsX = outOrig[1]; // Получаем fs_x

    // 3) Обрабатываем отредактированное изображение
    late List<Float32List> secondOut;
    if (isStylespace) {
      final (arrA, arrB) = editedWE4e as (List<Float32List>, List<Float32List>);
      secondOut = await _runDecoderRgbWithoutNewFeature([...arrA, ...arrB]);
    } else {
      secondOut = await _runDecoderWithoutNewFeature(editedWE4e as Float32List);
    }
    final fsY = secondOut[1]; // Получаем fs_y

    // 4) Вычисляем дельту
    final delta = _elementwiseSubtract(fsX, fsY);

    // 5) Получаем отредактированные фичи
    final cat = _concatAlongAxis1(fusedFeat, delta);
    final eF = await _runEncoder(cat);
    final editedFeat = eF[0];

    // 6) Генерируем финальное изображение
    late List<Float32List> finalOut;
    if (isStylespace) {
      final (arrA, arrB) = editedLatents as (List<Float32List>, List<Float32List>);
      finalOut = await _runDecoderRgbWithNewFeature([...arrA, ...arrB, editedFeat]);
    } else {
      finalOut = await _runDecoderWithNewFeature([editedLatents as Float32List, editedFeat]);
    }

    return finalOut[0]; // Возвращаем отредактированное изображение
  }

  // --------------------------------------------------------------------------
  // Below are helper sub-functions to replicate your run_onnx calls
  // for the various decoders & encoder.
  // Each returns a list of Float32List outputs (like [e4e_inv, fs_x], etc.)
  // --------------------------------------------------------------------------
  // ====================== Вспомогательные методы ======================

  static Future<List<Float32List>> _runDecoderWithoutNewFeature(Float32List input) async {
    if (_decoderWithoutNewFeatureSession == null) {
      throw Exception("Decoder model not loaded");
    }

    final inputTensor = CustomTensor.createTensorWithDataList(input, [1, input.length]);
    final results = await _decoderWithoutNewFeatureSession!.runAsync(
      CustomRunOptions(),
      {'input': inputTensor},
      outputNames: ['e4e_inv', 'fs_x'],
    );

    if (results.length != 2 || results.any((r) => r == null)) {
      throw Exception("Invalid outputs from decoder");
    }

    return results.cast<Float32List>();
  }

  static Future<List<Float32List>> _runDecoderRgbWithoutNewFeature(List<Float32List> inputs) async {
    if (_decoderRgbWithoutNewFeatureSession == null) {
      throw Exception("Decoder RGB model not loaded");
    }

    // Объединяем все входные тензоры в один
    final combined = _flattenManyFloat32Lists(inputs);
    final inputTensor = CustomTensor.createTensorWithDataList(combined, [1, combined.length]);

    final results = await _decoderRgbWithoutNewFeatureSession!.runAsync(
      CustomRunOptions(),
      {'input': inputTensor},
      outputNames: ['e4e_edit', 'fs_y'],
    );

    if (results.length != 2 || results.any((r) => r == null)) {
      throw Exception("Invalid outputs from RGB decoder");
    }

    return results.cast<Float32List>();
  }

  static Future<List<Float32List>> _runEncoder(Float32List input) async {
    if (_encoderSession == null) {
      throw Exception("Encoder model not loaded");
    }

    final inputTensor = CustomTensor.createTensorWithDataList(input, [1, input.length]);
    final results = await _encoderSession!.runAsync(
      CustomRunOptions(),
      {'input': inputTensor},
    );

    if (results.isEmpty || results[0] == null) {
      throw Exception("No output from encoder");
    }

    return [results[0]!];
  }

  static Future<List<Float32List>> _runDecoderWithNewFeature(List<Float32List> inputs) async {
    if (_decoderWithNewFeatureSession == null) {
      throw Exception("Decoder with new feature model not loaded");
    }

    final combined = _flattenManyFloat32Lists(inputs);
    final inputTensor = CustomTensor.createTensorWithDataList(combined, [1, combined.length]);

    final results = await _decoderWithNewFeatureSession!.runAsync(
      CustomRunOptions(),
      {'input': inputTensor},
    );

    if (results.isEmpty || results[0] == null) {
      throw Exception("No output from decoder with new feature");
    }

    return [results[0]!];
  }

  static Future<List<Float32List>> _runDecoderRgbWithNewFeature(List<Float32List> inputs) async {
    if (_decoderRgbWithNewFeatureSession == null) {
      throw Exception("Decoder RGB with new feature model not loaded");
    }

    final combined = _flattenManyFloat32Lists(inputs);
    final inputTensor = CustomTensor.createTensorWithDataList(combined, [1, combined.length]);

    final results = await _decoderRgbWithNewFeatureSession!.runAsync(
      CustomRunOptions(),
      {'input': inputTensor},
    );

    if (results.isEmpty || results[0] == null) {
      throw Exception("No output from RGB decoder with new feature");
    }

    return [results[0]!];
  }

  // ====================== Утилиты ======================

  // --------------------------------------------------------------------------
  // Utility: elementwise subtract
  // --------------------------------------------------------------------------
  static Float32List _elementwiseSubtract(Float32List a, Float32List b) {
    if (a.length != b.length) throw Exception("Arrays length mismatch");
    final result = Float32List(a.length);
    for (int i = 0; i < a.length; i++) {
      result[i] = a[i] - b[i];
    }
    return result;
  }

  // --------------------------------------------------------------------------
  // Utility: concat along axis=1 => [1, X + Y]
  // For shape [1, a.length] + [1, b.length]
  // --------------------------------------------------------------------------
  static Float32List _concatAlongAxis1(Float32List a, Float32List b) {
    final result = Float32List(a.length + b.length);
    result.setAll(0, a);
    result.setAll(a.length, b);
    return result;
  }

  // --------------------------------------------------------------------------
  // Utility: flatten multiple Float32Lists into one big Float32List
  // --------------------------------------------------------------------------
  static Float32List _flattenManyFloat32Lists(List<Float32List> inputs) {
    final totalLength = inputs.fold(0, (sum, list) => sum + list.length);
    final result = Float32List(totalLength);
    int offset = 0;
    for (final list in inputs) {
      result.setAll(offset, list);
      offset += list.length;
    }
    return result;
  }
}
