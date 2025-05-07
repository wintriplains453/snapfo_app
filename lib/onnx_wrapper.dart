import 'dart:typed_data';
import 'package:flutter/services.dart' show MethodChannel, PlatformException, rootBundle;

class OnnxWrapper {
  static const _channel = MethodChannel('com.example.snapfo_app/onnx');

  static Future<void> initEnv() async {
    try {
      await _channel.invokeMethod('initEnv');
    } on PlatformException catch (e) {
      throw Exception('Failed to initialize ONNX environment: ${e.message}');
    }
  }

  static Future<void> loadModel({required String key, String? assetPath, Uint8List? modelData}) async {
    try {
      if (modelData != null) {
        if (modelData.isEmpty) {
          throw Exception('Model data for $key is empty');
        }
        await _channel.invokeMethod('loadModel', {
          'key': key,
          'modelData': modelData,
        });
      } else if (assetPath != null) {
        final modelData = await _loadAsset(assetPath);
        if (modelData.isEmpty) {
          throw Exception('Loaded model data for $key from $assetPath is empty');
        }
        await _channel.invokeMethod('loadModel', {
          'key': key,
          'modelData': modelData,
        });
      } else {
        throw Exception('Either assetPath or modelData must be provided for $key');
      }
      print('Model $key loaded successfully');
    } catch (e) {
      print('Error loading model $key: $e');
      rethrow;
    }
  }

  static Future<void> disposeModel(String key) async {
    try {
      await _channel.invokeMethod('disposeModel', {'key': key});
      print('Model $key disposed successfully');
    } catch (e) {
      print('Error disposing model $key: $e');
      rethrow;
    }
  }

  static Future<Map<String, Float32List>> runComplex({
    required String sessionKey,
    required Map<String, Map<String, dynamic>> inputs,
    required List<String> outputNames,
  }) async {
    try {
      final results = await _channel.invokeMethod<Map<dynamic, dynamic>>(
        'runComplex',
        {
          'sessionKey': sessionKey,
          'inputs': inputs.map((key, value) => MapEntry(key, {
            'data': value['data'],
            'shape': value['shape'],
            'type': value['type'] ?? 'float32',
          })),
          'outputNames': outputNames,
        },
      );

      return results!.map((key, value) {
        return MapEntry(
          key.toString(),
          Float32List.fromList((value as List<dynamic>).cast<double>().map((e) => e.toDouble()).toList()),
        );
      });
    } on PlatformException catch (e) {
      throw Exception('Failed to run complex inference: ${e.message}');
    }
  }

  static Future<Uint8List> _loadAsset(String path) async {
    try {
      final byteData = await rootBundle.load(path);
      return byteData.buffer.asUint8List();
    } catch (e) {
      throw Exception('Failed to load asset $path: ${e.toString()}');
    }
  }
}

class CustomTensor {
  final dynamic data;
  final List<int> shape;
  final String type;

  CustomTensor(this.data, this.shape, {this.type = 'float32'});

  factory CustomTensor.createTensorWithDataList(dynamic data, List<int> shape, {String type = 'float32'}) {
    return CustomTensor(data, shape, type: type);
  }

  Map<String, dynamic> toMap() {
    return {
      'data': data,
      'shape': shape,
      'type': type,
    };
  }
}

class CustomRunOptions {
  void release() {}
}

class CustomSession {
  final String sessionKey;
  bool _isDisposed = false;

  CustomSession(this.sessionKey);

  Future<List<Float32List?>> runAsync(
      CustomRunOptions runOptions,
      Map<String, CustomTensor> inputs, {
        List<String> outputNames = const ['output'],
      }) async {
    if (_isDisposed) {
      throw Exception('Session $sessionKey is already disposed');
    }
    try {
      final inputMap = inputs.map((key, value) => MapEntry(key, {
        'data': value.data,
        'shape': value.shape,
        'type': value.type,
      }));

      final results = await OnnxWrapper.runComplex(
        sessionKey: sessionKey,
        inputs: inputMap,
        outputNames: outputNames,
      );

      return outputNames.map((name) => results[name]).toList();
    } catch (e) {
      throw Exception('Session run failed: ${e.toString()}');
    }
  }

  void dispose() {
    if (!_isDisposed) {
      OnnxWrapper.disposeModel(sessionKey);
      _isDisposed = true;
      print('Session $sessionKey disposed');
    }
  }
}