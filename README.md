# Как это работает

Убрана библиотека import ```'package:onnxruntime/onnxruntime.dart';``` и вместо неё в файле app/build.gradle.rt подключен ```implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")```, она собирает onnxruntime версии 1.20.0 


## Сборка onnxruntime

создается C++ файл с библиотекой в которой находятся готовые методы, каждый C++ файл скомпилирован и вложен в нужную архитектуру, например arm64-v8a

```
onnxruntime-android-1.20.0.aar
├── jni/
│   ├── arm64-v8a/
│   │   └── libonnxruntime.so
│   ├── armeabi-v7a/
│   └── x86_64/
├── classes.jar
└── AndroidManifest.xml
```

## Доступ к onnxruntime

Доступ к onnxruntime происходит через каналы платформы файла MainActivity по пути app/src/main/kotlin/com/example/snapfo_app/MainActivity.kt

Есть ещё файл OnnxNativeHelper.kt - файл в который вынесена логика из MainActivity для удобства, всё написано на языке котлин

В dart папки lib есть файл onnx_wrapper в котором находится несколько классов, основной это OnnxWrapper для обращения onnx через _channel (название канала)

```typescript
class OnnxWrapper {
  static const _channel = MethodChannel('com.example.snapfo_app/onnx'); // <- название канала onnx (такое же название лежит в MainActivity)

  static Future<void> initEnv() async { // <- всё должно происходить асинхронно (метод initEnv)
    try {
      await _channel.invokeMethod('initEnv'); // <- название метода из MainActivity при образении через _channel
    } on PlatformException catch (e) {
      throw Exception("Failed to initialize ONNX environment: ${e.message}");
    }
  }
```

в файле inference_runner используется всё что есть в onnx_wrapper который возможно при необходимости дополнять методами и связывать с OnnxNativeHelper.kt



