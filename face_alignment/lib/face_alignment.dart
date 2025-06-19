import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:path_provider/path_provider.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'face_alignment_ffi.dart';

class FaceAlignment {
  static final DynamicLibrary _lib = Platform.isAndroid
      ? DynamicLibrary.open('libface_alignment.so')
      : DynamicLibrary.process();

  static Future<String> alignFace(String imagePath) async {
    final imageFile = File(imagePath);
    if (!await imageFile.exists()) {
      throw Exception('Image file does not exist at $imagePath');
    }

    final faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        enableLandmarks: true,
        performanceMode: FaceDetectorMode.accurate,
      ),
    );

    try {
      final inputImage = InputImage.fromFile(imageFile);
      final faces = await faceDetector.processImage(inputImage);
      if (faces.isEmpty) {
        throw Exception("No faces found in image");
      }

      final largestFace = faces.reduce((a, b) =>
      (a.boundingBox.width * a.boundingBox.height >
          b.boundingBox.width * b.boundingBox.height)
          ? a
          : b);

      final landmarks = largestFace.landmarks;
      final requiredLandmarks = [
        FaceLandmarkType.leftEye,
        FaceLandmarkType.rightEye,
        FaceLandmarkType.noseBase,
        FaceLandmarkType.leftMouth,
        FaceLandmarkType.rightMouth,
      ];

      // Log all available landmarks for debugging
      print('Available landmarks:');
      landmarks.forEach((type, landmark) {
        print('Landmark $type: ${landmark?.position}');
      });

      // Collect required landmarks, checking for nulls
      final mappedLandmarks = <double>[];
      for (var type in requiredLandmarks) {
        final landmark = landmarks[type];
        if (landmark == null) {
          throw Exception('Missing required landmark: $type');
        }
        mappedLandmarks.add(landmark.position.x.toDouble());
        mappedLandmarks.add(landmark.position.y.toDouble());
      }

      // Log the mapped landmarks to verify
      print('Mapped landmarks: $mappedLandmarks');

      final landmarksPtr = malloc.allocate<Double>(mappedLandmarks.length * sizeOf<Double>());
      for (int i = 0; i < mappedLandmarks.length; i++) {
        landmarksPtr[i] = mappedLandmarks[i];
      }

      final faceAlignment = FaceAlignmentBindings(_lib);
      final cImagePath = imagePath.toNativeUtf8();
      final resultPtr = faceAlignment.align_face_ffi(landmarksPtr, 5, cImagePath);
      final result = resultPtr.toDartString();
      faceAlignment.free_string(resultPtr);
      malloc.free(cImagePath);
      malloc.free(landmarksPtr);

      if (result.startsWith('Error: ')) {
        throw Exception(result);
      }

      return result;
    } catch (e) {
      rethrow;
    } finally {
      await faceDetector.close();
    }
  }

  static Future<String> alignFaceAsync(String imagePath) async {
    final tempDir = await getTemporaryDirectory();
    final tempPath = '${tempDir.path}/${DateTime.now().millisecondsSinceEpoch}.jpg';
    final sourceFile = File(imagePath);
    if (!await sourceFile.exists()) {
      throw Exception('Source image does not exist at $imagePath');
    }
    await sourceFile.copy(tempPath);
    final tempFile = File(tempPath);
    if (!await tempFile.exists()) {
      throw Exception('Failed to copy image to $tempPath');
    }
    print('Copied image to: $tempPath');
    try {
      final result = await alignFace(tempPath);
      return result;
    } finally {
      if (await tempFile.exists()) {
        await tempFile.delete();
        print('Deleted temporary file: $tempPath');
      }
    }
  }
}