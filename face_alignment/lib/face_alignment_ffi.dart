import 'dart:ffi';
import 'package:ffi/ffi.dart';

class FaceAlignmentBindings {
  final DynamicLibrary _lib;

  FaceAlignmentBindings(this._lib);

  // Define the align_face_ffi function
  late final align_face_ffi = _lib.lookupFunction<
      Pointer<Utf8> Function(Pointer<Double>, Int32, Pointer<Utf8>),
      Pointer<Utf8> Function(Pointer<Double>, int, Pointer<Utf8>)>(
    'align_face_ffi',
  );

  // Define the free_string function
  late final free_string = _lib.lookupFunction<
      Void Function(Pointer<Utf8>),
      void Function(Pointer<Utf8>)>(
    'free_string',
  );
}

// Extension to convert Pointer<Utf8> to Dart String
extension Utf8PointerExtension on Pointer<Utf8> {
  String toDartString() {
    if (this == nullptr) return '';
    return Utf8PointerExtension(this).toDartString(); // Correctly convert Pointer<Utf8> to String
  }
}