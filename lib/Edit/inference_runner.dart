import 'dart:math';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'latent_editor.dart';

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
  // Sessions for all models
  static OrtSession? _interpolateSession;
  static OrtSession? _invertSession;
  static OrtSession? _fuserSession;
  static OrtSession? _e4eEncoderSession;
  static OrtSession? _decoderWithoutNewFeatureSession;
  static OrtSession? _decoderRgbWithoutNewFeatureSession;
  static OrtSession? _encoderSession;
  static OrtSession? _decoderWithNewFeatureSession;
  static OrtSession? _decoderRgbWithNewFeatureSession;
  static OrtSession? _decoderStylespaceSession; // Renamed from decoderStylespaceModelBytes

  static final _interfaceganSessions = <String, OrtSession>{};

  static OrtSession? get decoderStylespaceSession => _decoderStylespaceSession;

  /// Logs tensor statistics for debugging
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

  static Float32List createZerosLike(Float32List input) {
    return Float32List(input.length);
  }

  static void logTensor(String name, Float32List tensor) {
    print(name);
    for (var i = 0; i < min(tensor.length, 10); i++) {
      print('${tensor[i]}');
    }
  }

  /// FP32 to FP16 conversion
  static Uint16List convertToFp16(Float32List input) {
    final result = Uint16List(input.length);
    for (int i = 0; i < input.length; i++) {
      final bytes = ByteData(4)..setFloat32(0, input[i], Endian.little);
      final bits = bytes.getInt32(0, Endian.little);
      final sign = (bits >> 31) & 0x1;
      final exp = (bits >> 23) & 0xFF;
      final mantissa = bits & 0x7FFFFF;

      if (exp == 0xFF) {
        result[i] = (sign << 15) | (0x1F << 10) | (mantissa != 0 ? 0x200 : 0);
        continue;
      }

      final newExp = exp - 127 + 15;
      if (newExp >= 0x1F) {
        result[i] = (sign << 15) | (0x1F << 10);
        continue;
      }
      if (newExp <= 0) {
        if (newExp < -10) {
          result[i] = (sign << 15);
          continue;
        }
        final shift = 14 - newExp;
        final newMantissa = (mantissa >> 13) | (1 << 23);
        result[i] = (sign << 15) | (newMantissa >> shift);
        continue;
      }

      final newMantissa = mantissa >> 13;
      result[i] = (sign << 15) | (newExp << 10) | newMantissa;
    }
    return result;
  }

  /// FP16 to FP32 conversion
  static Float32List convertFp16ToFp32(Uint16List input) {
    final result = Float32List(input.length);
    for (int i = 0; i < input.length; i++) {
      final half = input[i];
      final sign = (half >> 15) & 0x1;
      final exp = (half >> 10) & 0x1F;
      final mantissa = half & 0x3FF;

      double value;
      if (exp == 0) {
        value = mantissa == 0 ? 0.0 : pow(2, -14) * (mantissa / 1024.0);
      } else if (exp == 31) {
        value = mantissa == 0 ? double.infinity : double.nan;
      } else {
        value = pow(2, exp - 15) * (1 + mantissa / 1024.0);
      }
      result[i] = (sign == 1 ? -value : value).toDouble();
    }
    return result;
  }

  /// Flatten nested list to Float32List
  static Float32List flattenNestedList(dynamic input) {
    final List<double> flattened = [];

    void flatten(dynamic item) {
      if (item is List) {
        for (var subItem in item) {
          flatten(subItem);
        }
      } else if (item is double) {
        flattened.add(item);
      } else if (item is Float32List) {
        flattened.addAll(item);
      } else {
        throw Exception('Unexpected type in nested list: ${item.runtimeType}');
      }
    }

    flatten(input);
    return Float32List.fromList(flattened);
  }

  /// Flatten nested list to Uint16List for FP16
  static Uint16List flattenList(dynamic input) {
    final List<int> flattened = [];

    void flatten(dynamic item) {
      if (item is List) {
        for (var subItem in item) {
          flatten(subItem);
        }
      } else if (item is double) {
        final tempFloat32 = Float32List(1)..[0] = item;
        final tempUint16 = convertToFp16(tempFloat32);
        flattened.add(tempUint16[0]);
      } else if (item is int) {
        flattened.add(item);
      } else {
        throw Exception('Unexpected type in list: ${item.runtimeType}');
      }
    }

    flatten(input);
    return Uint16List.fromList(flattened);
  }

  /// Initialize ONNX environment
  static Future<void> initEnv() async {
    try {
      OrtEnv.instance.init();
      print('ONNX environment initialized');
    } catch (e) {
      print('Error initializing ONNX environment: $e');
      rethrow;
    }
  }

  /// Load all models
  static Future<void> loadModels() async {
    print('[InferenceRunner.loadModels] Starting...');
    final sessionOptions = OrtSessionOptions();
    try {
      final modelDir = 'assets/models';
      // Load models
      _interpolateSession = OrtSession.fromBuffer(
        (await rootBundle.load('$modelDir/interpolateIR9.onnx')).buffer.asUint8List(),
        sessionOptions,
      );
      print('Model interpolateIR9 loaded successfully');

      _invertSession = OrtSession.fromBuffer(
        (await rootBundle.load('$modelDir/invert_optimized_opt_fp16.onnx')).buffer.asUint8List(),
        sessionOptions,
      );
      print('Model invert_compressedFP16 loaded successfully');

      _fuserSession = OrtSession.fromBuffer(
        (await rootBundle.load('$modelDir/fuser.onnx')).buffer.asUint8List(),
        sessionOptions,
      );
      print('Model fuser loaded successfully');

      _e4eEncoderSession = OrtSession.fromBuffer(
        (await rootBundle.load('$modelDir/e4e_encoder_opt_fp16.onnx')).buffer.asUint8List(),
        sessionOptions,
      );
      print('Model e4e_encoder_compressedFP16 loaded successfully');

      _decoderWithoutNewFeatureSession = OrtSession.fromBuffer(
        (await rootBundle.load('$modelDir/decoder_without_new_feature.onnx')).buffer.asUint8List(),
        sessionOptions,
      );
      print('Model decoder_without_new_feature loaded successfully');

      _decoderRgbWithoutNewFeatureSession = OrtSession.fromBuffer(
        (await rootBundle.load('$modelDir/decoder_rgb_without_new_feature.onnx')).buffer.asUint8List(),
        sessionOptions,
      );
      print('Model decoder_rgb_without_new_feature loaded successfully');

      _encoderSession = OrtSession.fromBuffer(
        (await rootBundle.load('$modelDir/encoder.onnx')).buffer.asUint8List(),
        sessionOptions,
      );
      print('Model encoder loaded successfully');

      _decoderWithNewFeatureSession = OrtSession.fromBuffer(
        (await rootBundle.load('$modelDir/decoder_with_new_feature_opt_fp32.onnx')).buffer.asUint8List(),
        sessionOptions,
      );
      print('Model decoder_with_new_feature loaded successfully');

      _decoderRgbWithNewFeatureSession = OrtSession.fromBuffer(
        (await rootBundle.load('$modelDir/decoder_rgb_with_new_feature.onnx')).buffer.asUint8List(),
        sessionOptions,
      );
      print('Model decoder_rgb_with_new_feature loaded successfully');

      _interfaceganSessions['interfacegan_age'] = OrtSession.fromBuffer(
        (await rootBundle.load('$modelDir/interfacegan_age.onnx')).buffer.asUint8List(),
        sessionOptions,
      );
      print('Model interfacegan_age loaded successfully');

      _decoderStylespaceSession = OrtSession.fromBuffer(
        (await rootBundle.load('$modelDir/decoder_stylespace.onnx')).buffer.asUint8List(),
        sessionOptions,
      );
      print('Model decoder_stylespace loaded successfully');

      print('[loadModels] All models loaded successfully!');
    } catch (e) {
      print('Error loading models: $e');
      rethrow;
    }
  }

  static final TensorPool _tensorPool = TensorPool();

  static void dispose() {
    _interpolateSession?.release();
    _invertSession?.release();
    _fuserSession?.release();
    _e4eEncoderSession?.release();
    _decoderWithoutNewFeatureSession?.release();
    _decoderRgbWithoutNewFeatureSession?.release();
    _encoderSession?.release();
    _decoderWithNewFeatureSession?.release();
    _decoderRgbWithNewFeatureSession?.release();
    _decoderStylespaceSession?.release();
    _interfaceganSessions.forEach((_, session) => session.release());
    _interfaceganSessions.clear();
    _tensorPool.clear();
    print('[InferenceRunner.dispose]');
  }

  // Run InterfaceGAN model
  static Future<Float32List> runInterfacegan(
      Float32List latent,
      double degree,
      String editingName,
      ) async {
    final session = _interfaceganSessions[editingName];
    if (session == null) {
      throw Exception('$editingName model not loaded. Call loadModels()');
    }

    if (latent.length != 1 * 18 * 512) {
      throw Exception('Invalid latent length: ${latent.length}, expected ${1 * 18 * 512}');
    }

    final latentTensor = OrtValueTensor.createTensorWithDataList(latent, [1, 18, 512]);
    final degreeTensor = OrtValueTensor.createTensorWithDataList(
      Float32List.fromList([degree]),
      [1],
    );

    final results = await session.runAsync(
      OrtRunOptions(),
      {
        'start_w': latentTensor,
        'factor': degreeTensor,
      },
    );

    if (results == null || results.isEmpty || results[0] == null) {
      throw Exception('No output from $editingName');
    }

    final output = flattenNestedList(results[0]!.value);
    if (output.length != 1 * 18 * 512) {
      throw Exception('Unexpected output length: ${output.length}, expected ${1 * 18 * 512}');
    }

    latentTensor.release();
    degreeTensor.release();
    results.forEach((e) => e?.release());
    print('_runInterfacegan success return!!!!');
    _logTensorStats('_runInterfacegan', [output], ['edited_latent']);
    return output;
  }

  // Run pipeline on batch
  static Future<(Float32List, ResultBatch)> runOnBatch(Float32List inputTensor) async {
    // Step 1: Run interpolate.onnx
    print('Input shape: ${inputTensor.length}, first 10 values: ${inputTensor.sublist(0, min(10, inputTensor.length))}');
    final xOut = await _runInterpolate(inputTensor);
    final x = xOut[0];
    print('Interpolate output shape: ${x.length}, first 10 values: ${x.sublist(0, min(10, x.length))}');
    xOut.clear();

    // Step 2: Run invert.onnx
    final invertOut = await _runInvert(x);
    final wRecon = invertOut[0];
    final predictedFeat = invertOut[1];
    print('wRecon shape: ${wRecon.length}, first 10 values: ${wRecon.sublist(0, min(10, wRecon.length))}');
    print('predictedFeat shape: ${predictedFeat.length}, first 10 values: ${predictedFeat.sublist(0, min(10, predictedFeat.length))}');
    invertOut.clear();

    // Step 3: Run decoder_without_new_feature.onnx for wRecon
    final decoderOut = await _runDecoderWithoutNewFeature(wRecon);
    final wFeat = decoderOut[1];
    print('wFeat shape: ${wFeat.length}, first 10 values: ${wFeat.sublist(0, min(10, wFeat.length))}');
    decoderOut.clear();

    // Step 4: Run fuser.onnx
    final fusedFeat = await _runFuser(_concatAlongAxis1(predictedFeat, wFeat));
    print('fusedFeat shape: ${fusedFeat.length}, first 10 values: ${fusedFeat.sublist(0, min(10, fusedFeat.length))}');

    // Step 5: Compute zero delta
    final delta = createZerosLike(fusedFeat);
    print('delta shape: ${delta.length}, first 10 values: ${delta.sublist(0, min(10, delta.length))}');

    // Step 6: Run encoder.onnx
    final cat = _concatAlongAxis1(fusedFeat, delta);
    final encoderOut = await _runEncoder(cat);
    final encodedFeat = encoderOut[0];
    print('encodedFeat shape: ${encodedFeat.length}, first 10 values: ${encodedFeat.sublist(0, min(10, encodedFeat.length))}');

    // Step 7: Run decoder_with_new_feature.onnx
    final imageOut = await _runDecoderWithNewFeature([wRecon, encodedFeat]);
    final image = imageOut[0];
    print('image shape: ${image.length}, first 10 values: ${image.sublist(0, min(10, image.length))}');

    // Step 8: Run e4e_encoder.onnx
    final wE4e = await _runE4eEncoder(x);
    print('wE4e shape: ${wE4e.length}, first 10 values: ${wE4e.sublist(0, min(10, wE4e.length))}');

    final resultBatch = ResultBatch(
      latents: wRecon,
      fusedFeat: fusedFeat, // Changed to fusedFeat as in Python
      predictedFeat: predictedFeat,
      wE4e: wE4e,
      input: inputTensor,
    );

    return (image, resultBatch);
  }

  // Run editing on batch
  static Future<Float32List> runEditingOnBatch({
    required ResultBatch resultBatch,
    required String editingName,
    required double editingDegree,
    required BuildContext? context,
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

  // Run editing core
  static Future<Float32List> runEditingCore({
    required Float32List latent,
    required Float32List wE4e,
    required Float32List fusedFeat,
    required String editingName,
    required double editingDegree,
    required BuildContext? context,
  }) async {
    try {
      // Get edited latents
      final editedLatents = await LatentEditor.getEditedLatent(latent, editingName, editingDegree, context);
      final editedWE4e = await LatentEditor.getEditedLatent(wE4e, editingName, editingDegree, context);

      // Check if stylespace is used
      final isStylespace = editedLatents is (List<Float32List>, List<Float32List>);

      // Process original w_e4e
      final outOrig = await _runDecoderWithoutNewFeature(wE4e);
      final fsX = outOrig[1];
      print('fsX shape: ${fsX.length}, first 10 values: ${fsX.sublist(0, min(10, fsX.length))}');
      outOrig.clear();

      // Process edited w_e4e
      late List<Float32List> secondOut;
      if (isStylespace) {
        final (arrA, arrB) = editedWE4e as (List<Float32List>, List<Float32List>);
        final selectedInputs = [...arrA.sublist(0, 9), ...arrB.sublist(0, 5)];
        secondOut = await _runDecoderRgbWithoutNewFeature(selectedInputs);
      } else {
        secondOut = await _runDecoderWithoutNewFeature(editedWE4e as Float32List);
      }
      final fsY = secondOut[1];
      print('fsY shape: ${fsY.length}, first 10 values: ${fsY.sublist(0, min(10, fsY.length))}');
      secondOut.clear();

      // Compute delta
      final delta = _elementwiseSubtract(fsX, fsY);
      print('delta shape: ${delta.length}, first 10 values: ${delta.sublist(0, min(10, delta.length))}');

      // Get edited features
      final cat = _concatAlongAxis1(fusedFeat, delta);
      final editedFeatOut = await _runEncoder(cat);
      final editedFeat = editedFeatOut[0];
      print('editedFeat shape: ${editedFeat.length}, first 10 values: ${editedFeat.sublist(0, min(10, editedFeat.length))}');

      // Generate final image
      late List<Float32List> finalOut;
      if (isStylespace) {
        final (arrA, arrB) = editedLatents as (List<Float32List>, List<Float32List>);
        finalOut = await _runDecoderRgbWithNewFeature([...arrA, ...arrB, editedFeat]);
      } else {
        finalOut = await _runDecoderWithNewFeature([editedLatents as Float32List, editedFeat]);
      }

      print('finalOut shape: ${finalOut[0].length}, first 10 values: ${finalOut[0].sublist(0, min(10, finalOut[0].length))}');
      return finalOut[0];
    } catch (e) {
      print('Error during image editing: $e');
      rethrow;
    }
  }

  // Helper methods for running ONNX models
  static Future<List<Float32List>> _runInterpolate(Float32List input) async {
    if (_interpolateSession == null) throw Exception('Interpolate model not loaded');
    final inputTensor = OrtValueTensor.createTensorWithDataList(input, [1, 3, 1024, 1024]);
    final results = await _interpolateSession!.runAsync(
      OrtRunOptions(),
      {'x': inputTensor},
    );
    if (results == null || results.isEmpty || results[0] == null) throw Exception('No output from interpolate');

    final output = flattenNestedList(results[0]!.value);
    inputTensor.release();
    results.forEach((e) => e?.release());
    print('_runInterpolate success return!!!!');
    return [output];
  }

  static Future<List<Float32List>> _runInvert(Float32List input) async {
    if (_invertSession == null) throw Exception('Invert model not loaded');
    if (input.length != 1 * 3 * 256 * 256) {
      throw Exception('Input tensor has incorrect length: ${input.length}, expected ${1 * 3 * 256 * 256}');
    }

    final inputFp16 = convertToFp16(input);
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      inputFp16,
      [1, 3, 256, 256],
      ONNXTensorElementDataType.float16,
    );

    final results = await _invertSession!.runAsync(
      OrtRunOptions(),
      {'input.1': inputTensor},
    );

    if (results == null || results.length != 2 || results.any((r) => r == null)) {
      throw Exception('Invalid outputs from invert');
    }

    final wReconRaw = flattenList(results[0]!.value);
    final predictedFeatRaw = flattenList(results[1]!.value);
    final wRecon = convertFp16ToFp32(wReconRaw);
    final predictedFeat = convertFp16ToFp32(predictedFeatRaw);

    if (wRecon.length != 1 * 18 * 512) {
      throw Exception('Unexpected w_recon length: ${wRecon.length}, expected ${1 * 18 * 512}');
    }
    if (predictedFeat.length != 1 * 512 * 64 * 64) {
      throw Exception('Unexpected predicted_feat length: ${predictedFeat.length}, expected ${1 * 512 * 64 * 64}');
    }

    _logTensorStats('_runInvert', [wRecon, predictedFeat], ['w_recon', 'predicted_feat']);
    inputTensor.release();
    results.forEach((e) => e?.release());
    print('_runInvert success return!!!!');
    return [wRecon, predictedFeat];
  }

  static Future<Float32List> _runFuser(Float32List input) async {
    if (_fuserSession == null) throw Exception('Fuser model not loaded');
    final inputTensor = OrtValueTensor.createTensorWithDataList(input, [1, 1024, 64, 64]);
    final results = await _fuserSession!.runAsync(
      OrtRunOptions(),
      {'x': inputTensor},
    );
    if (results == null || results.isEmpty || results[0] == null) throw Exception('No output from fuser');

    final output = flattenNestedList(results[0]!.value);
    const expectedLength = 1 * 512 * 64 * 64;
    if (output.length != expectedLength) {
      throw Exception('Unexpected fuser output length: ${output.length}, expected $expectedLength');
    }

    _logTensorStats('_runFuser', [output], ['fused_feat']);
    inputTensor.release();
    results.forEach((e) => e?.release());
    print('_runFuser success return!!!!');
    return output;
  }

  static Future<Float32List> _runE4eEncoder(Float32List input) async {
    if (_e4eEncoderSession == null) throw Exception('E4eEncoder model not loaded');
    if (input.length != 1 * 3 * 256 * 256) {
      throw Exception('Input tensor has incorrect length: ${input.length}, expected ${1 * 3 * 256 * 256}');
    }

    final inputFp16 = convertToFp16(input);
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      inputFp16,
      [1, 3, 256, 256],
      ONNXTensorElementDataType.float16,
    );

    final results = await _e4eEncoderSession!.runAsync(
      OrtRunOptions(),
      {'input.1': inputTensor},
    );

    if (results == null || results.isEmpty || results[0] == null) {
      throw Exception('No output from e4e_encoder');
    }

    final wE4eRaw = flattenList(results[0]!.value);
    final wE4e = convertFp16ToFp32(wE4eRaw);
    if (wE4e.length != 1 * 18 * 512) {
      throw Exception('Unexpected w_e4e length: ${wE4e.length}, expected ${1 * 18 * 512}');
    }

    inputTensor.release();
    results.forEach((e) => e?.release());
    print('_runE4eEncoder success return!!!!');
    return wE4e;
  }

  static Future<List<Float32List>> _runDecoderWithoutNewFeature(Float32List input) async {
    if (_decoderWithoutNewFeatureSession == null) throw Exception('Decoder model not loaded');
    final inputTensor = OrtValueTensor.createTensorWithDataList(input, [1, 18, 512]);
    final results = await _decoderWithoutNewFeatureSession!.runAsync(
      OrtRunOptions(),
      {'latent': inputTensor},
    );

    if (results == null || results.length != 2 || results.any((r) => r == null)) {
      throw Exception('Invalid outputs from decoder');
    }

    final image = flattenNestedList(results[0]!.value);
    final feature = flattenNestedList(results[1]!.value);

    const expectedImageLength = 1 * 3 * 64 * 64;
    const expectedFeatureLength = 1 * 512 * 64 * 64;
    if (image.length != expectedImageLength) {
      throw Exception('Unexpected image output length: ${image.length}, expected $expectedImageLength');
    }
    if (feature.length != expectedFeatureLength) {
      throw Exception('Unexpected feature output length: ${feature.length}, expected $expectedFeatureLength');
    }

    _logTensorStats('_runDecoderWithoutNewFeature', [image, feature], ['image', 'feature']);
    inputTensor.release();
    results.forEach((e) => e?.release());
    print('_runDecoderWithoutNewFeature success return!!!!');
    return [image, feature];
  }

  static Future<List<Float32List>> _runDecoderRgbWithoutNewFeature(List<Float32List> inputs) async {
    if (_decoderRgbWithoutNewFeatureSession == null) throw Exception('Decoder RGB model not loaded');
    final inputMap = <String, OrtValueTensor>{};
    for (int i = 0; i < 9; i++) {
      if (inputs[i].length != 512) {
        throw Exception('style_${i + 1} has incorrect length: ${inputs[i].length}, expected 512');
      }
      inputMap['style_${i + 1}'] = OrtValueTensor.createTensorWithDataList(inputs[i], [1, 512]);
    }
    for (int i = 0; i < 5; i++) {
      inputMap['to_rgb_stylespace_${i + 1}'] = OrtValueTensor.createTensorWithDataList(inputs[9 + i], [1, 512]);
    }

    final results = await _decoderRgbWithoutNewFeatureSession!.runAsync(
      OrtRunOptions(),
      inputMap,
    );
    if (results == null || results.length != 2 || results.any((r) => r == null)) {
      throw Exception('Invalid outputs from RGB decoder');
    }

    final image = flattenNestedList(results[0]!.value);
    final feature = flattenNestedList(results[1]!.value);

    const expectedImageLength = 1 * 3 * 1024 * 1024;
    const expectedFeatureLength = 1 * 512 * 64 * 64;
    if (image.length != expectedImageLength) {
      throw Exception('Unexpected image output length: ${image.length}, expected $expectedImageLength');
    }
    if (feature.length != expectedFeatureLength) {
      throw Exception('Unexpected feature output length: ${feature.length}, expected $expectedFeatureLength');
    }

    _logTensorStats('_runDecoderRgbWithoutNewFeature', [image, feature], ['image', 'feature']);
    inputMap.values.forEach((tensor) => tensor.release());
    results.forEach((e) => e?.release());
    print('_runDecoderRgbWithoutNewFeature success return!!!!');
    return [image, feature];
  }

  static Future<List<Float32List>> _runEncoder(Float32List input) async {
    if (_encoderSession == null) throw Exception('Encoder model not loaded');
    final inputTensor = OrtValueTensor.createTensorWithDataList(input, [1, 1024, 64, 64]);
    final results = await _encoderSession!.runAsync(
      OrtRunOptions(),
      {'input.1': inputTensor},
    );
    if (results == null || results.isEmpty || results[0] == null) throw Exception('No output from encoder');

    final output = flattenNestedList(results[0]!.value);
    const expectedLength = 1 * 512 * 64 * 64;
    if (output.length != expectedLength) {
      throw Exception('Unexpected encoder output length: ${output.length}, expected $expectedLength');
    }

    _logTensorStats('_runEncoder', [output], ['output']);
    inputTensor.release();
    results.forEach((e) => e?.release());
    print('_runEncoder success return!!!!');
    return [output];
  }

  static Future<List<Float32List>> _runDecoderWithNewFeature(List<Float32List> inputs) async {
    if (_decoderWithNewFeatureSession == null) throw Exception('Decoder with new feature model not loaded');
    final latent = inputs[0];
    final newFeature = inputs[1];

    final latentTensor = OrtValueTensor.createTensorWithDataList(latent, [1, 18, 512]);
    final newFeatureTensor = OrtValueTensor.createTensorWithDataList(newFeature, [1, 512, 64, 64]);

    final results = await _decoderWithNewFeatureSession!.runAsync(
      OrtRunOptions(),
      {'latent': latentTensor, 'onnx::ConvTranspose_1': newFeatureTensor},
    );

    if (results == null || results.isEmpty || results[0] == null) {
      throw Exception('No output from decoder with new feature');
    }

    final image = flattenNestedList(results[0]!.value);
    latentTensor.release();
    newFeatureTensor.release();
    results.forEach((e) => e?.release());
    print('_runDecoderWithNewFeature success return!!!!');
    return [image];
  }

  static Future<List<Float32List>> _runDecoderRgbWithNewFeature(List<Float32List> inputs) async {
    if (_decoderRgbWithNewFeatureSession == null) throw Exception('Decoder RGB with new feature model not loaded');

    final expectedLengths = [
      512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 256, 128, 128, 64, 64, 32,
      512, 512, 512, 512, 512, 256, 128, 64, 32, 512 * 64 * 64,
    ];
    for (var i = 0; i < inputs.length; i++) {
      if (inputs[i].length != expectedLengths[i]) {
        throw Exception('inputs[$i] has incorrect length: ${inputs[i].length}, expected ${expectedLengths[i]}');
      }
    }

    final inputMap = <String, OrtValueTensor>{};
    final inputsFp16 = inputs.map((input) => convertToFp16(input)).toList();

    for (int i = 0; i < 10; i++) {
      inputMap['style_${i + 1}'] = OrtValueTensor.createTensorWithDataList(inputsFp16[i], [1, 512], ONNXTensorElementDataType.float16);
    }
    inputMap['style_11'] = OrtValueTensor.createTensorWithDataList(inputsFp16[10], [1, 256], ONNXTensorElementDataType.float16);
    inputMap['style_12'] = OrtValueTensor.createTensorWithDataList(inputsFp16[11], [1, 256], ONNXTensorElementDataType.float16);
    inputMap['style_13'] = OrtValueTensor.createTensorWithDataList(inputsFp16[12], [1, 128], ONNXTensorElementDataType.float16);
    inputMap['style_14'] = OrtValueTensor.createTensorWithDataList(inputsFp16[13], [1, 128], ONNXTensorElementDataType.float16);
    inputMap['style_15'] = OrtValueTensor.createTensorWithDataList(inputsFp16[14], [1, 64], ONNXTensorElementDataType.float16);
    inputMap['style_16'] = OrtValueTensor.createTensorWithDataList(inputsFp16[15], [1, 64], ONNXTensorElementDataType.float16);
    inputMap['style_17'] = OrtValueTensor.createTensorWithDataList(inputsFp16[16], [1, 32], ONNXTensorElementDataType.float16);
    for (int i = 0; i < 5; i++) {
      inputMap['to_rgb_stylespace_${i + 1}'] = OrtValueTensor.createTensorWithDataList(inputsFp16[17 + i], [1, 512], ONNXTensorElementDataType.float16);
    }
    inputMap['to_rgb_stylespace_6'] = OrtValueTensor.createTensorWithDataList(inputsFp16[22], [1, 256], ONNXTensorElementDataType.float16);
    inputMap['to_rgb_stylespace_7'] = OrtValueTensor.createTensorWithDataList(inputsFp16[23], [1, 128], ONNXTensorElementDataType.float16);
    inputMap['to_rgb_stylespace_8'] = OrtValueTensor.createTensorWithDataList(inputsFp16[24], [1, 64], ONNXTensorElementDataType.float16);
    inputMap['to_rgb_stylespace_9'] = OrtValueTensor.createTensorWithDataList(inputsFp16[25], [1, 32], ONNXTensorElementDataType.float16);
    inputMap['new_feature'] = OrtValueTensor.createTensorWithDataList(inputsFp16[26], [1, 512, 64, 64], ONNXTensorElementDataType.float16);

    final results = await _decoderRgbWithNewFeatureSession!.runAsync(
      OrtRunOptions(),
      inputMap,
    );

    if (results == null || results.isEmpty || results[0] == null) {
      throw Exception('No output from RGB decoder with new feature');
    }

    final outputRaw = flattenList(results[0]!.value);
    final output = convertFp16ToFp32(outputRaw);

    inputMap.values.forEach((tensor) => tensor.release());
    results.forEach((e) => e?.release());
    print('_runDecoderRgbWithNewFeature success return!!!!');
    return [output];
  }

  // Utilities
  static Float32List _elementwiseSubtract(Float32List a, Float32List b) {
    if (a.length != b.length) throw Exception('Arrays length mismatch');
    final key = 'subtract_${a.length}_${b.length}';
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
    if (size < 0) throw ArgumentError('Buffer size cannot be negative: $size');
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