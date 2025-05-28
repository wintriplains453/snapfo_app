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
    // Initialize FaceDetector with options
    final faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        enableLandmarks: true, // Enable landmark detection
        performanceMode: FaceDetectorMode.accurate, // Use accurate mode
      ),
    );

    // Load image and convert to InputImage
    final inputImage = InputImage.fromFilePath(imagePath);

    // Detect faces
    final faces = await faceDetector.processImage(inputImage);
    if (faces.isEmpty) {
      await faceDetector.close();
      throw Exception("No faces found");
    }

    // Select the largest face by bounding box area
    final largestFace = faces.reduce((a, b) =>
    (a.boundingBox.width * a.boundingBox.height >
        b.boundingBox.width * b.boundingBox.height)
        ? a
        : b);

    // Get landmarks (Google ML Kit provides limited landmarks)
    final landmarks = largestFace.landmarks;

    // Map Google ML Kit landmarks to dlib 68-point model
    final mappedLandmarks = <double>[];
    const dlibIndices = [
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, // Chin
      17, 18, 19, 20, 21, // Left eyebrow
      22, 23, 24, 25, 26, // Right eyebrow
      27, 28, 29, 30, // Nose
      31, 32, 33, 34, 35, // Nostrils
      36, 37, 38, 39, 40, 41, // Left eye
      42, 43, 44, 45, 46, 47, // Right eye
      48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, // Mouth outer
      60, 61, 62, 63, 64, 65, 66, 67 // Mouth inner
    ];

    // Helper function to get landmark coordinates or default to (0, 0)
    double getX(FaceLandmark? landmark) => landmark?.position.x.toDouble() ?? 0.0;
    double getY(FaceLandmark? landmark) => landmark?.position.y.toDouble() ?? 0.0;

    // Map available ML Kit landmarks to dlib indices
    final landmarkMap = {
      // Based on FaceLandmarkType from google_mlkit_face_detection
      36: landmarks[FaceLandmarkType.leftEye], // Left eye
      39: landmarks[FaceLandmarkType.leftEye], // Approximate other eye points
      42: landmarks[FaceLandmarkType.rightEye], // Right eye
      45: landmarks[FaceLandmarkType.rightEye], // Approximate other eye points
      30: landmarks[FaceLandmarkType.noseBase], // Nose tip
      48: landmarks[FaceLandmarkType.leftMouth], // Mouth left
      54: landmarks[FaceLandmarkType.rightMouth], // Mouth right
    };

    // Fill 68 landmarks (136 coordinates)
    for (int i = 0; i < 68; i++) {
      if (landmarkMap.containsKey(i)) {
        mappedLandmarks.add(getX(landmarkMap[i]));
        mappedLandmarks.add(getY(landmarkMap[i]));
      } else {
        // Fill missing landmarks with zeros
        mappedLandmarks.add(0.0);
        mappedLandmarks.add(0.0);
      }
    }

    // Allocate memory for landmarks
    final landmarksPtr = malloc.allocate<Double>(mappedLandmarks.length * sizeOf<Double>());
    for (int i = 0; i < mappedLandmarks.length; i++) {
      landmarksPtr[i] = mappedLandmarks[i];
    }

    // Call C++ function
    final faceAlignment = FaceAlignmentBindings(_lib);
    final cImagePath = imagePath.toNativeUtf8();
    final resultPtr = faceAlignment.align_face_ffi(landmarksPtr, 68, cImagePath);
    final result = Utf8PointerExtension(resultPtr).toDartString();
    faceAlignment.free_string(resultPtr);
    malloc.free(cImagePath);
    malloc.free(landmarksPtr);


    // Clean up face detector
    await faceDetector.close();

    return result;
  }

  static Future<String> alignFaceAsync(String imagePath) async {
    final tempDir = await getTemporaryDirectory();
    final tempPath = '${tempDir.path}/${imagePath.split('/').last}';
    await File(imagePath).copy(tempPath);
    return alignFace(tempPath);
  }
}