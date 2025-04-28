import 'dart:typed_data';
import 'package:flutter/services.dart' show MethodChannel, PlatformException, rootBundle;
import 'package:path_provider/path_provider.dart';
import 'dart:io';

class OnnxWrapper {
  static const _channel = MethodChannel('com.example.snapfo_app/onnx');

  static Future<void> initEnv() async {
    try {
      await _channel.invokeMethod('initEnv');
    } on PlatformException catch (e) {
      throw Exception('Failed to initialize ONNX environment: ${e.message}');
    }
  }

  static Future<void> loadModel({required String key, required String assetPath}) async {
    try {
      final modelBytes = await _loadAsset(assetPath);

      if (key == 'pre_editor') {
        final tempDir = await getTemporaryDirectory();
        final modelFile = File('${tempDir.path}/$key.onnx');
        await modelFile.writeAsBytes(modelBytes);

        try {
          final dataBytes = await _loadAsset('assets/models/$key.onnx.data');
          final dataFile = File('${tempDir.path}/$key.onnx.data');
          await dataFile.writeAsBytes(dataBytes);
        } catch (e) {
          print('Файл внешних данных для $key не найден');
        }

        await _channel.invokeMethod('loadModel', {
          'key': key,
          'modelBytes': modelBytes,
          'modelPath': modelFile.path,
        });
      } else {
        await _channel.invokeMethod('loadModel', {
          'key': key,
          'modelBytes': modelBytes,
        });
      }

      print('Model $key loaded successfully from $assetPath');
    } on PlatformException catch (e) {
      throw Exception('Failed to load model $key: ${e.message}');
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

  CustomSession(this.sessionKey);

  Future<List<Float32List?>> runAsync(
      CustomRunOptions runOptions,
      Map<String, CustomTensor> inputs, {
        List<String> outputNames = const ['output'],
      }) async {
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
}