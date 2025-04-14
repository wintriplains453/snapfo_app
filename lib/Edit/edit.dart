import 'dart:typed_data';
import 'preprocess.dart';
import 'inference_runner.dart';
import 'package:image/image.dart' as img;

class ImageEditor {
  /// Equivalent to Python's `edit(...)`.
  /// 1) Preprocess the image
  /// 2) run_pre_editor => (image, w_recon, w_e4e, fused_feat)
  /// 3) run_editing_core => final image
  /// 4) post-process => convert to PNG bytes
  static Future<Uint8List> edit({
    required Uint8List inputBytes,
    required String editingName,
    required double editingDegree,
  }) async {
    // 1) Preprocess
    final preprocessed = ImagePreprocessor.preprocessImage(inputBytes, resizeSize: 1024);
    // 2) Run pre_editor
    final preEditorOutputs = await InferenceRunner.runPreEditor(preprocessed);
    if (preEditorOutputs.length < 4) {
      throw Exception("PreEditor model returned <4 outputs.");
    }
    // For example:
    // 0 => 'image'
    // 1 => 'w_recon'
    // 2 => 'w_e4e'
    // 3 => 'fused_feat'
    final imageTensor = preEditorOutputs[0];
    final wRecon = preEditorOutputs[1];
    final wE4e = preEditorOutputs[2];
    final fusedFeat = preEditorOutputs[3];

    // 3) Run editing core
    // For demonstration, we only handle "age" editing
    // If "smile" or "rotation", you'd do something similar or choose a model
    final finalTensor = await InferenceRunner.runEditingCore(
      latent: wRecon,
      wE4e: wE4e,
      fusedFeat: fusedFeat,
      editingName: editingName,
      editingDegree: editingDegree,
    );

    // 4) Post-process
    // The returned finalTensor might be shape [1, 3, H, W] in [-1..1].
    // Let's assume H=W=1024 for demonstration.
    // Then the length is 3*1024*1024 = 3,145,728 floats.
    const h = 1024;
    const w = 1024;
    final outputBytes = convertCHWFloat32ToImageBytes(
      finalTensor,
      height: h,
      width: w,
      isRangeNeg1To1: true,
    );

    return outputBytes;
  }
}

/// replicate Python's 'prepare_np'
Uint8List convertCHWFloat32ToImageBytes(
    Float32List tensor, {
      required int height,
      required int width,
      bool isRangeNeg1To1 = true,
    }) {
  final outputImage = img.Image(width: width, height: height);

  final cStride = width * height;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      final i = y * width + x;
      double r = tensor[i];
      double g = tensor[i + cStride];
      double b = tensor[i + 2 * cStride];

      if (isRangeNeg1To1) {
        r = (r + 1.0) / 2.0;
        g = (g + 1.0) / 2.0;
        b = (b + 1.0) / 2.0;
      }
      r = r.clamp(0.0, 1.0);
      g = g.clamp(0.0, 1.0);
      b = b.clamp(0.0, 1.0);

      final R = (r * 255).round();
      final G = (g * 255).round();
      final B = (b * 255).round();

      outputImage.setPixelRgba(x, y, R, G, B, 255);
    }
  }

  final pngBytes = img.encodePng(outputImage);
  return Uint8List.fromList(pngBytes);
}
