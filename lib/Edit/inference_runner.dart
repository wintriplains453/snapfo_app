import 'package:onnxruntime/onnxruntime.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:snapfo_app/libs/convertFP.dart';

class ResultBatch {
  final Float32List latents;
  final Float32List fusedFeat;
  final Float32List predictedFeat;
  final Float32List wE4e;
  final Float32List input;  ResultBatch({
    required this.latents,
    required this.fusedFeat,
    required this.predictedFeat,
    required this.wE4e,
    required this.input,
  });
}class InferenceRunner {
  static OrtSession? _interpolateSession;
  static OrtSession? _invertSession;  static final TensorPool _tensorPool = TensorPool();  static void _logTensorStats(String stage, List<Float32List> tensors, List<String> names) {
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
  }  static Float32List createZerosLike(Float32List input) {
    return Float32List(input.length);
  }  static void logTensor(String name, Float32List tensor) {
    print(name);
    for (var i = 0; i < tensor.length; i++) {
      print('${tensor[i]}');
    }
  }  static List<double> flattenTensor(dynamic tensor) {
    final List<double> flattened = [];

    void flatten(dynamic value) {
      if (value is double) {
        flattened.add(value);
      } else if (value is List) {
        for (var item in value) {
          flatten(item);
        }
      } else {
        throw Exception('Unexpected tensor value type: ${value.runtimeType}');
      }
    }

    flatten(tensor);
    return flattened;

  }  static Future<void> initEnv() async {
    try {
      print('ONNX Runtime environment initialized');
      final providers = OrtEnv.instance.availableProviders();
      print('Available ONNX Runtime providers: $providers');
    } catch (e) {
      print('Error initializing ONNX environment: $e');
      rethrow;
    }
  }  static Future<void> loadModels() async {
    print('[InferenceRunner.loadModels] Starting...');

    try {
      final interpolateModelBytes = await rootBundle.load('assets/models/interpolateIR9.onnx');
      final interpolateSessionOptions = OrtSessionOptions()..appendCPUProvider(CPUFlags.useNone);
      _interpolateSession = OrtSession.fromBuffer(interpolateModelBytes.buffer.asUint8List(), interpolateSessionOptions);
      print('Model interpolateIR9 loaded successfully');

      final invertModelBytes = await rootBundle.load('assets/models/invert_compressed.onnx');
      final invertSessionOptions = OrtSessionOptions()..appendCPUProvider(CPUFlags.useNone);
      _invertSession = OrtSession.fromBuffer(invertModelBytes.buffer.asUint8List(), invertSessionOptions);
      print('Model invert_compressedIR9 loaded successfully');
    } catch (e) {
      print('Error loading models: $e');
      rethrow;
    }

    print('[loadModels] All models loaded successfully!');

  }

  static void dispose() {
    _interpolateSession?.release();
    _invertSession?.release();
    _tensorPool.clear();
    print('[InferenceRunner.dispose]');
  }

  static Future<List<Float32List>> _runInterpolate(Float32List input) async {
    if (_interpolateSession == null) throw Exception('Interpolate model not loaded');
    if (input.length != 1 * 3 * 1024 * 1024) {
      throw Exception('Input tensor has incorrect length: ${input.length}, expected ${1 * 3 * 1024 * 1024}');
    }

    final inputTensor = OrtValueTensor.createTensorWithDataList(input, [1, 3, 1024, 1024]);
    final runOptions = OrtRunOptions();
    final results = await _interpolateSession!.runAsync(runOptions, {'x': inputTensor});

    if (results!.isEmpty || results[0] == null) throw Exception('No output from interpolate');

// Log raw tensor structure for debugging
    print('Raw interpolate output type: ${results[0]!.value.runtimeType}');
// Flatten using robust flattenTensor
    final flatOutput = Float32List.fromList(flattenTensor(results[0]!.value));

    if (flatOutput.length != 1 * 3 * 256 * 256) {
      throw Exception('Unexpected interpolate output length: ${flatOutput.length}, expected ${1 * 3 * 256 * 256}');
    }
    _logTensorStats('_runInterpolate', [flatOutput], ['output']);
    print('_runInterpolate success return!!!!');

    inputTensor.release();
    results.forEach((element) => element?.release());
    runOptions.release();

    return [flatOutput];

  }

  static Future<List<Float32List>> _runInvert(Float32List input) async {
    if (_invertSession == null) throw Exception('Invert model not loaded');
    if (input.length != 1 * 3 * 256 * 256) {
      throw Exception('Input tensor has incorrect length: ${input.length}, expected ${1 * 3 * 256 * 256}');
    }

    // Convert input from FP32 to FP16
    final inputFp16 = convertToFp16(input);
    print("type of input!!!");
    print(inputFp16.runtimeType);

    // Create ONNX tensor with FP16 data
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      inputFp16,
      [1, 3, 256, 256],
      ONNXTensorElementDataType.float16,
    );

    final runOptions = OrtRunOptions();
    final results = await _invertSession!.runAsync(runOptions, {'input.1': inputTensor});

    if (results!.length != 2 || results.any((r) => r == null)) {
      throw Exception('Invalid outputs from invert');
    }

    print('w_recon value type: ${results[0]!.value.runtimeType}');
    print('w_recon value sample: ${results[0]!.value.toString().substring(0, 100)}'); // Первые 100 символов
    print('predicted_feat value type: ${results[1]!.value.runtimeType}');
    print('predicted_feat value sample: ${results[1]!.value.toString().substring(0, 100)}');

    // Extract raw outputs (expected to be FP16)
    final wReconRaw = flattenList(results[0]!.value);
    final predictedFeatRaw = flattenList(results[1]!.value);

    print('Raw w_recon output type: ${wReconRaw.runtimeType}');
    print('Raw predicted_feat output type: ${predictedFeatRaw.runtimeType}');

    // Convert outputs from FP16 to FP32
    final wRecon = convertFp16ToFp32(wReconRaw);
    final predictedFeat = convertFp16ToFp32(predictedFeatRaw);

    // Validate output lengths
    if (wRecon.length != 1 * 18 * 512) {
      throw Exception('Unexpected w_recon length: ${wRecon.length}, expected ${1 * 18 * 512}');
    }
    if (predictedFeat.length != 1 * 512 * 64 * 64) {
      throw Exception('Unexpected predicted_feat length: ${predictedFeat.length}, expected ${1 * 512 * 64 * 64}');
    }

    _logTensorStats('_runInvert', [wRecon, predictedFeat], ['w_recon', 'predicted_feat']);
    print('_runInvert success return!!!!');

    // Release resources
    inputTensor.release();
    results.forEach((element) => element?.release());
    runOptions.release();

    return [wRecon, predictedFeat];
  }

  static Future<(Float32List, ResultBatch)> runOnBatch(Float32List inputTensor) async {
    print('Input shape: ${inputTensor.length}, first 10 values: ${inputTensor.sublist(0, 10)}');
    // final directory = await getApplicationDocumentsDirectory();
    // final inputFile = File('${directory.path}/input_tensor.txt');
    // await inputFile.writeAsString(inputTensor.join('\n'));
    // print('Saved input tensor to ${inputFile.path}');

    final xOut = await _runInterpolate(inputTensor);
    final x = xOut[0];
    print('Interpolate output shape: ${x.length}, first 10 values: ${x.sublist(0, 10)}');

// final interpolateFile = File('${directory.path}/interpolate_output.txt');
// await interpolateFile.writeAsString(x.join('\n'));
// print('Saved interpolate output to ${interpolateFile.path}');

    xOut.clear();

    final invertOut = await _runInvert(x);
    final wRecon = invertOut[0];
    final predictedFeat = invertOut[1];
    print('wRecon shape: ${wRecon.length}, first 10 values: ${wRecon.sublist(0, 10)}');
    print('predictedFeat shape: ${predictedFeat.length}, first 10 values: ${predictedFeat.sublist(0, 10)}');

// final wReconFile = File('${directory.path}/w_recon_output.txt');
// await wReconFile.writeAsString(wRecon.join('\n'));
// print('Saved wRecon output to ${wReconFile.path}');

    final resultBatch = ResultBatch(
      latents: wRecon,
      fusedFeat: Float32List(0),
      predictedFeat: predictedFeat,
      wE4e: Float32List(0),
      input: inputTensor,
    );

    invertOut.clear();

    return (x, resultBatch);

  }  static Future<Float32List> runEditingOnBatch({
    required ResultBatch resultBatch,
    required String editingName,
    required double editingDegree,
    required BuildContext context,
  }) async {
    print('Running editing with $editingName, degree: $editingDegree');
    return resultBatch.input;
  }
}class TensorPool {
  final Map<String, Float32List> _buffers = {};  Float32List getBuffer(String key, int size) {
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
  }  void clear() {
    _buffers.clear();
    print('[TensorPool] Cleared all buffers');
  }  int get bufferCount => _buffers.length;
  int get totalSize => _buffers.values.fold(0, (sum, buffer) => sum + buffer.length);
}

Uint16List flattenList(dynamic input) {
  final List<int> flattened = [];

  void flatten(dynamic item) {
    if (item is List) {
      for (var subItem in item) {
        flatten(subItem);
      }
    } else if (item is int) {
      flattened.add(item);
    } else {
      throw Exception('Unexpected type in list: ${item.runtimeType}');
    }
  }

  flatten(input);
  return Uint16List.fromList(flattened);
}