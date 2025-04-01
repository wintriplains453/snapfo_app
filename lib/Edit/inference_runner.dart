import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:onnxruntime/onnxruntime.dart';

// Suppose you have a `LatentEditor.getEditedLatent(...)` function
// that returns either a Float32List or a tuple of two Lists: (List<Float32List>, List<Float32List>)
import 'latent_editor.dart';

class InferenceRunner {
  static OrtSession? _preEditorSession;
  static OrtSession? _interfaceganAgeSession;
  static OrtSession? _decoderWithoutNewFeatureSession;
  static OrtSession? _decoderWithNewFeatureSession;
  static OrtSession? _encoderSession;

  // Additional: for the is_stylespace branch
  static OrtSession? _decoderRgbWithoutNewFeatureSession;
  static OrtSession? _decoderRgbWithNewFeatureSession;

  /// Initialize the ORT env once
  static void initEnv() {
    try {
      OrtEnv.instance.init();
    } catch (_) {}
  }

  /// Load all relevant models
  /// (You must list these .onnx files in your pubspec.yaml assets section)
  static Future<void> loadModels() async {
    print("[InferenceRunner.loadModels] Starting...");

    // interfacegan_age (or other editing model)
    _interfaceganAgeSession ??= await _loadSession('assets/models/interfacegan_age.onnx');
    print("[loadModels] interfacegan_age.onnx loaded.");
    // pre_editor
    _preEditorSession ??= await _loadSession('assets/models/pre_editor.onnx');
    print("[loadModels] pre_editor.onnx loaded.");

    // decoder_without_new_feature
    _decoderWithoutNewFeatureSession ??=
    await _loadSession('assets/models/decoder_without_new_feature.onnx');

    // decoder_with_new_feature
    _decoderWithNewFeatureSession ??=
    await _loadSession('assets/models/decoder_with_new_feature.onnx');

    // encoder
    _encoderSession ??= await _loadSession('assets/models/encoder.onnx');

    // decoder_rgb_without_new_feature
    _decoderRgbWithoutNewFeatureSession ??=
    await _loadSession('assets/models/decoder_rgb_without_new_feature.onnx');

    // decoder_rgb_with_new_feature
    _decoderRgbWithNewFeatureSession ??=
    await _loadSession('assets/models/decoder_rgb_with_new_feature.onnx');

    print("[loadModels] All model loads completed!");
  }

  /// Helper to read an ONNX file from assets and create a session
  static Future<OrtSession> _loadSession(String path) async {
    print("[_loadSession] Attempting to load session from $path");
    final sessionOptions = OrtSessionOptions();
    final rawFile = await rootBundle.load(path);
    final bytes = rawFile.buffer.asUint8List();
    return OrtSession.fromBuffer(bytes, sessionOptions);
  }

  /// Dispose all loaded sessions + ORT environment
  static void dispose() {
    _preEditorSession?.release();
    _preEditorSession = null;

    _interfaceganAgeSession?.release();
    _interfaceganAgeSession = null;

    _decoderWithoutNewFeatureSession?.release();
    _decoderWithoutNewFeatureSession = null;

    _decoderWithNewFeatureSession?.release();
    _decoderWithNewFeatureSession = null;

    _encoderSession?.release();
    _encoderSession = null;

    _decoderRgbWithoutNewFeatureSession?.release();
    _decoderRgbWithoutNewFeatureSession = null;

    _decoderRgbWithNewFeatureSession?.release();
    _decoderRgbWithNewFeatureSession = null;

    try {
      OrtEnv.instance.release();
    } catch (_) {}
  }

  // --------------------------------------------------------------------------
  // (A) runPreEditor => returns [image, w_recon, w_e4e, fused_feat]
  // Python: run_pre_editor(x)
  // --------------------------------------------------------------------------
  static Future<List<Float32List>> runPreEditor(Float32List inputTensor) async {
    if (_preEditorSession == null) {
      throw Exception("PreEditor model not loaded. Call loadModels() first.");
    }

    // shape e.g. [1,3,1024,1024]
    final shape = [1, 3, 1024, 1024];
    final inputOrt = OrtValueTensor.createTensorWithDataList(inputTensor, shape);

    final runOptions = OrtRunOptions();
    final outputs =
    await _preEditorSession!.runAsync(runOptions, {'input': inputOrt});

    inputOrt.release();
    runOptions.release();

    if (outputs == null || outputs.isEmpty) {
      throw Exception("No outputs from pre_editor");
    }

    // We expect 4 outputs: image, w_recon, w_e4e, fused_feat
    final resultList = <Float32List>[];
    for (final ortVal in outputs) {
      if (ortVal != null) {
        resultList.add(ortVal.value as Float32List);
        ortVal.release();
      }
    }
    return resultList;
  }

  // --------------------------------------------------------------------------
  // (B) Example runInterfacegan => for "age" or similar
  // In Python: run_onnx(interfacegan_age, (latent, degree))
  // You can replicate more editing models as needed.
  // --------------------------------------------------------------------------
  static Future<Float32List> runInterfacegan(Float32List latent, double degree) async {
    if (_interfaceganAgeSession == null) {
      throw Exception("interfacegan_age onnx not loaded. Call loadModels()");
    }
    final shape = [1, latent.length];
    final latentTensor = OrtValueTensor.createTensorWithDataList(latent, shape);
    final degreeTensor = OrtValueTensor.createTensorWithDataList(
      Float32List.fromList([degree]),
      [1],
    );

    final runOptions = OrtRunOptions();
    final outs = await _interfaceganAgeSession!.runAsync(
      runOptions,
      {
        'latent': latentTensor,
        'degree': degreeTensor,
      },
    );

    latentTensor.release();
    degreeTensor.release();
    runOptions.release();

    if (outs == null || outs.isEmpty || outs[0] == null) {
      throw Exception("No output from interfacegan_age");
    }
    final data = outs[0]!.value as Float32List;
    outs[0]!.release();
    return data;
  }

  // --------------------------------------------------------------------------
  // (C) runEditingCore - EXACT replication of your Python function
  //
  //   def run_editing_core(latent, w_e4e, fused_feat, editing_name, editing_degree):
  //       edited_latents = get_edited_latent(latent, editing_name, editing_degree)
  //       edited_w_e4e    = get_edited_latent(w_e4e, editing_name, editing_degree)
  //       is_stylespace = isinstance(edited_latents, tuple)
  //
  //       e4e_inv, fs_x = run_onnx(decoder_without_new_feature, (w_e4e,))
  //       if is_stylespace:
  //           e4e_edit, fs_y = run_onnx(decoder_rgb_without_new_feature, tuple(edited_w_e4e[0] + edited_w_e4e[1]))
  //       else:
  //           e4e_edit, fs_y = run_onnx(decoder_without_new_feature, (edited_w_e4e,))
  //
  //       delta = fs_x - fs_y
  //       edited_feat = run_onnx(encoder, (concat(fused_feat, delta),))[0]
  //
  //       if is_stylespace:
  //           image_edit = run_onnx(decoder_rgb_with_new_feature, tuple(edited_latents[0] + edited_latents[1] + [edited_feat]))
  //       else:
  //           image_edit = run_onnx(decoder_with_new_feature, tuple([edited_latents] + [edited_feat]))
  //
  //       image_edit = image_edit[0]
  //       return image_edit
  // --------------------------------------------------------------------------
  static Future<Float32List> runEditingCore({
    required Float32List latent,
    required Float32List wE4e,
    required Float32List fusedFeat,
    required String editingName,
    required double editingDegree,
  }) async {
    // 1) Get the edited latents
    final editedLatents = await LatentEditor.getEditedLatent(latent, editingName, editingDegree);
    final editedWE4e = await LatentEditor.getEditedLatent(wE4e, editingName, editingDegree);

    // Check if style-space
    final bool isStylespace =
    editedLatents is (List<Float32List>, List<Float32List>);

    // 2) e4e_inv, fs_x = decoder_without_new_feature(w_e4e)
    final outOrig = await _runDecoderWithoutNewFeature(wE4e);
    final e4eInv = outOrig[0];
    final fsX = outOrig[1];

    // 3) If styleclip => e4e_edit, fs_y = decoder_rgb_without_new_feature,
    //    else => e4e_edit, fs_y = decoder_without_new_feature
    late List<Float32List> secondOut;
    if (isStylespace) {
      // editedWE4e is a tuple (List<Float32List>, List<Float32List>)
      // We must flatten them into a single list => used as "tuple(...)"
      final (List<Float32List> arrA, List<Float32List> arrB) =
      editedWE4e as (List<Float32List>, List<Float32List>);
      final combined = <Float32List>[...arrA, ...arrB];
      secondOut = await _runDecoderRgbWithoutNewFeature(combined);
    } else {
      // If not stylespace, we assume it's a single Float32List
      secondOut = await _runDecoderWithoutNewFeature(editedWE4e as Float32List);
    }
    final e4eEdit = secondOut[0];
    final fsY = secondOut[1];

    // 4) delta = fs_x - fs_y
    final delta = _elementwiseSubtract(fsX, fsY);

    // 5) edited_feat = run_onnx(encoder, (concat(fused_feat, delta), ))[0]
    final cat = _concatAlongAxis1(fusedFeat, delta);
    final eF = await _runEncoder(cat);
    final editedFeat = eF[0];

    // 6) final decode => either "decoder_rgb_with_new_feature" or "decoder_with_new_feature"
    late List<Float32List> finalOut;
    if (isStylespace) {
      final (List<Float32List> arrA, List<Float32List> arrB) =
      editedLatents as (List<Float32List>, List<Float32List>);
      final combinedInput = <Float32List>[...arrA, ...arrB, editedFeat];
      finalOut = await _runDecoderRgbWithNewFeature(combinedInput);
    } else {
      finalOut = await _runDecoderWithNewFeature(
        [(editedLatents as Float32List), editedFeat],
      );
    }
    final imageEdit = finalOut[0]; // image_edit

    return imageEdit;
  }

  // --------------------------------------------------------------------------
  // Below are helper sub-functions to replicate your run_onnx calls
  // for the various decoders & encoder.
  // Each returns a list of Float32List outputs (like [e4e_inv, fs_x], etc.)
  // --------------------------------------------------------------------------

  static Future<List<Float32List>> _runDecoderWithoutNewFeature(
      Float32List inputLatent,
      ) async {
    if (_decoderWithoutNewFeatureSession == null) {
      throw Exception("decoder_without_new_feature not loaded.");
    }
    // shape might be [1, 512] or something
    final shape = [1, inputLatent.length];
    final inTensor = OrtValueTensor.createTensorWithDataList(inputLatent, shape);

    final runOptions = OrtRunOptions();
    final outs = await _decoderWithoutNewFeatureSession!.runAsync(
      runOptions,
      {'input': inTensor},
    );
    inTensor.release();
    runOptions.release();

    if (outs == null || outs.isEmpty) {
      throw Exception("decoder_without_new_feature returned no outputs");
    }

    final result = <Float32List>[];
    for (final o in outs) {
      if (o != null) {
        result.add(o.value as Float32List);
        o.release();
      }
    }
    // e.g. [e4e_inv, fs_x]
    return result;
  }

  static Future<List<Float32List>> _runDecoderRgbWithoutNewFeature(
      List<Float32List> multiLatents,
      ) async {
    if (_decoderRgbWithoutNewFeatureSession == null) {
      throw Exception("decoder_rgb_without_new_feature not loaded.");
    }
    // In Python: run_onnx(..., tuple(edited_w_e4e[0] + edited_w_e4e[1]))
    // So we flatten multiple Float32List into a single input array to pass.
    // But OnnxRuntime in Dart expects a map of input names => OrtValue.
    //
    // If your actual model has multiple inputs (like input0, input1, etc.),
    // you'd do that. If it expects a single input with shape, you might need
    // to flatten them.
    //
    // For demonstration, let's assume the model has a single input named "input"
    // and expects the latents concatenated. We'll do a naive flatten.

    final combined = _flattenManyFloat32Lists(multiLatents);
    // shape might be [1, sumOfLengths]
    final shape = [1, combined.length];
    final inTensor = OrtValueTensor.createTensorWithDataList(combined, shape);

    final runOptions = OrtRunOptions();
    final outs = await _decoderRgbWithoutNewFeatureSession!.runAsync(
      runOptions,
      {'input': inTensor},
    );
    inTensor.release();
    runOptions.release();

    if (outs == null || outs.isEmpty) {
      throw Exception("decoder_rgb_without_new_feature returned no outputs");
    }

    final result = <Float32List>[];
    for (final o in outs) {
      if (o != null) {
        result.add(o.value as Float32List);
        o.release();
      }
    }
    // e.g. [e4e_edit, fs_y]
    return result;
  }

  static Future<List<Float32List>> _runEncoder(Float32List cat) async {
    if (_encoderSession == null) {
      throw Exception("encoder.onnx not loaded.");
    }
    // shape might be [1, cat.length]
    final shape = [1, cat.length];
    final inTensor = OrtValueTensor.createTensorWithDataList(cat, shape);

    final runOptions = OrtRunOptions();
    final outs = await _encoderSession!.runAsync(runOptions, {'input': inTensor});
    inTensor.release();
    runOptions.release();

    if (outs == null || outs.isEmpty) {
      throw Exception("encoder returned no output");
    }

    final result = <Float32List>[];
    for (final o in outs) {
      if (o != null) {
        result.add(o.value as Float32List);
        o.release();
      }
    }
    // e.g. [edited_feat]
    return result;
  }

  static Future<List<Float32List>> _runDecoderWithNewFeature(
      List<Float32List> latents,
      ) async {
    if (_decoderWithNewFeatureSession == null) {
      throw Exception("decoder_with_new_feature not loaded.");
    }
    // In Python: run_onnx(decoder_with_new_feature, tuple([edited_latents] + [edited_feat]))
    // That is effectively 2 arrays concatenated. You might do the same approach
    // as above or if your ONNX has multiple inputs, supply them.
    final combined = _flattenManyFloat32Lists(latents);
    final shape = [1, combined.length];
    final inTensor = OrtValueTensor.createTensorWithDataList(combined, shape);

    final runOptions = OrtRunOptions();
    final outs = await _decoderWithNewFeatureSession!.runAsync(
      runOptions,
      {'input': inTensor},
    );
    inTensor.release();
    runOptions.release();

    if (outs == null || outs.isEmpty) {
      throw Exception("decoder_with_new_feature returned no outputs");
    }

    final result = <Float32List>[];
    for (final o in outs) {
      if (o != null) {
        result.add(o.value as Float32List);
        o.release();
      }
    }
    // e.g. [image_edit]
    return result;
  }

  static Future<List<Float32List>> _runDecoderRgbWithNewFeature(
      List<Float32List> latents,
      ) async {
    if (_decoderRgbWithNewFeatureSession == null) {
      throw Exception("decoder_rgb_with_new_feature not loaded.");
    }
    // same flatten approach as above
    final combined = _flattenManyFloat32Lists(latents);
    final shape = [1, combined.length];
    final inTensor = OrtValueTensor.createTensorWithDataList(combined, shape);

    final runOptions = OrtRunOptions();
    final outs = await _decoderRgbWithNewFeatureSession!.runAsync(
      runOptions,
      {'input': inTensor},
    );
    inTensor.release();
    runOptions.release();

    if (outs == null || outs.isEmpty) {
      throw Exception("decoder_rgb_with_new_feature returned no outputs");
    }

    final result = <Float32List>[];
    for (final o in outs) {
      if (o != null) {
        result.add(o.value as Float32List);
        o.release();
      }
    }
    // e.g. [image_edit]
    return result;
  }

  // --------------------------------------------------------------------------
  // Utility: flatten multiple Float32Lists into one big Float32List
  // --------------------------------------------------------------------------
  static Float32List _flattenManyFloat32Lists(List<Float32List> inputs) {
    int totalLen = 0;
    for (var x in inputs) {
      totalLen += x.length;
    }
    final out = Float32List(totalLen);
    int offset = 0;
    for (var arr in inputs) {
      out.setAll(offset, arr);
      offset += arr.length;
    }
    return out;
  }

  // --------------------------------------------------------------------------
  // Utility: elementwise subtract
  // --------------------------------------------------------------------------
  static Float32List _elementwiseSubtract(Float32List a, Float32List b) {
    if (a.length != b.length) {
      throw Exception("Cannot subtract arrays of different length");
    }
    final out = Float32List(a.length);
    for (int i = 0; i < a.length; i++) {
      out[i] = a[i] - b[i];
    }
    return out;
  }

  // --------------------------------------------------------------------------
  // Utility: concat along axis=1 => [1, X + Y]
  // For shape [1, a.length] + [1, b.length]
  // --------------------------------------------------------------------------
  static Float32List _concatAlongAxis1(Float32List a, Float32List b) {
    final out = Float32List(a.length + b.length);
    out.setAll(0, a);
    out.setAll(a.length, b);
    return out;
  }
}
