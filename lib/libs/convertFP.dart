import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'dart:io';

typedef ConvertFp32ToFp16C = Void Function(Pointer<Float>, Pointer<Uint16>, Int32);
typedef ConvertFp32ToFp16Dart = void Function(Pointer<Float>, Pointer<Uint16>, int);

typedef ConvertFp16ToFp32C = Void Function(Pointer<Uint16>, Pointer<Float>, Int32);
typedef ConvertFp16ToFp32Dart = void Function(Pointer<Uint16>, Pointer<Float>, int);

final DynamicLibrary nativeLib = Platform.isAndroid
    ? DynamicLibrary.open('libconvert_fp.so')
    : DynamicLibrary.process();

final convertFp32ToFp16 = nativeLib
    .lookup<NativeFunction<ConvertFp32ToFp16C>>('convert_fp32_to_fp16')
    .asFunction<ConvertFp32ToFp16Dart>();

final convertFp16ToFp32Native = nativeLib // Переименовали переменную
    .lookup<NativeFunction<ConvertFp16ToFp32C>>('convert_fp16_to_fp32')
    .asFunction<ConvertFp16ToFp32Dart>();

Uint16List convertToFp16(Float32List input) {
  final fp32Ptr = malloc<Float>(input.length);
  final fp16Ptr = malloc<Uint16>(input.length);

  for (int i = 0; i < input.length; i++) {
    fp32Ptr[i] = input[i];
  }

  convertFp32ToFp16(fp32Ptr, fp16Ptr, input.length);

  final result = Uint16List.fromList(fp16Ptr.asTypedList(input.length));

  malloc.free(fp32Ptr);
  malloc.free(fp16Ptr);

  return result;
}

Float32List convertFp16ToFp32(Uint16List input) {
  final fp16Ptr = malloc<Uint16>(input.length);
  final fp32Ptr = malloc<Float>(input.length);

  for (int i = 0; i < input.length; i++) {
    fp16Ptr[i] = input[i];
  }

  convertFp16ToFp32Native(fp16Ptr, fp32Ptr, input.length); // Используем переименованную переменную

  final result = Float32List.fromList(fp32Ptr.asTypedList(input.length));

  malloc.free(fp16Ptr);
  malloc.free(fp32Ptr);

  return result;
}