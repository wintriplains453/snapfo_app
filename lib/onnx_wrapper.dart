import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter/services.dart';

/// Класс для работы с ONNX моделями через нативную реализацию на Kotlin
class OnnxWrapper {
  static const _channel = MethodChannel('com.example.snapfo_app/onnx');

  static Future<void> initEnv() async {
    try {
      await _channel.invokeMethod('initEnv');
    } on PlatformException catch (e) {
      throw Exception("Failed to initialize ONNX environment: ${e.message}");
    }
  }

  static Future<void> loadModel({required String key, required String assetPath,}) async {
    try {
      final modelBytes = await _loadAsset(assetPath);
      await _channel.invokeMethod('loadModel', {
        'key': key,
        'modelBytes': modelBytes,
      });
      print('Model $key loaded successfully from $assetPath');
    } on PlatformException catch (e) {
      throw Exception("Failed to load model $key: ${e.message}");
    }
  }

  static Future<Map<String, List<double>>> runInference({required String sessionKey, required Map<String, dynamic> inputs, required List<String> outputNames,}) async {
    try {
      // Преобразуем входные данные в правильный формат
      final processedInputs = inputs.map((key, value) {
        if (key == "start_w") {
          // Для 3D тензора start_w
          return MapEntry(key, List.generate(1, (_) =>
              List.generate(16, (_) =>
                  List.filled(32, value as double))));
        } else if (key == "factor") {
          // Для 1D параметра factor
          return MapEntry(key, [value as double]);
        } else {
          throw ArgumentError('Unsupported input key: $key');
        }
      });

      final results = await _channel.invokeMethod<Map<dynamic, dynamic>>(
        'runInference',
        {
          'sessionKey': sessionKey,
          'inputs': processedInputs,
          'outputNames': outputNames,
        },
      );

      return results!.map<String, List<double>>((key, value) {
        final dynamicList = value as List<dynamic>;
        final doubleList = dynamicList.cast<double>().toList();
        return MapEntry(key.toString(), doubleList);
      });
    } on PlatformException catch (e) {
      throw Exception("Inference failed: ${e.message}");
    }
  }

  /// Вспомогательный метод для загрузки бинарных данных из assets
  static Future<Uint8List> _loadAsset(String path) async {
    try {
      final byteData = await rootBundle.load(path);
      return byteData.buffer.asUint8List();
    } catch (e) {
      throw Exception("Failed to load asset $path: ${e.toString()}");
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
          'inputs': inputs,
          'outputNames': outputNames,
        },
      );

      return results!.map((key, value) {
        return MapEntry(
          key.toString(),
          Float32List.fromList((value as List).cast<double>().map((e) => e.toDouble()).toList()),
        );
      });
    } on PlatformException catch (e) {
      throw Exception("Failed to run complex inference: ${e.message}");
    }
  }
}


///Новый класс
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

///Новый класс
class CustomRunOptions {
  void release() {} // Пустой метод для совместимости
}

///Новый класс
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
      }));

      final results = await OnnxWrapper.runComplex(
        sessionKey: sessionKey,
        inputs: inputMap,
        outputNames: outputNames,
      );

      return outputNames.map((name) => results[name]).toList();
    } catch (e) {
      throw Exception("Session run failed: ${e.toString()}");
    }
  }
}