import 'dart:ffi';
import 'package:ffi/ffi.dart';

class FaceAlignmentBindings {
  final DynamicLibrary _lib;

  FaceAlignmentBindings(this._lib);

  late final align_face_ffi = _lib.lookupFunction<
      Pointer<Utf8> Function(Pointer<Double>, Int32, Pointer<Utf8>),
      Pointer<Utf8> Function(Pointer<Double>, int, Pointer<Utf8>)>(
    'align_face_ffi',
  );

  late final free_string = _lib.lookupFunction<
      Void Function(Pointer<Utf8>),
      void Function(Pointer<Utf8>)>(
    'free_string',
  );
}