import 'dart:math';
import 'dart:typed_data';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:flutter/material.dart';
import 'styleclip_editor.dart';
import 'inference_runner.dart';

class LatentEditor {
  static final interfaceganDirections = {
    'age': 'interfacegan_age',
  };

  static const styleSpaceDimensions = [
    512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, // 15 x 512
    256, 256, 256, // 3 x 256
    128, 128, 128, // 3 x 128
    64, 64, 64, // 3 x 64
    32, 32, // 2 x 32
  ];

  /// Logs tensor statistics for debugging
  static void _logTensorStats(String stage, List<Float32List> tensors, List<String> names) {
    for (int i = 0; i < tensors.length; i++) {
      final tensor = tensors[i];
      final name = names[i];
      bool hasNaN = false;
      bool hasInfinity = false;
      double minVal = double.infinity;
      double maxVal = -double.infinity;
      for (var val in tensor) {
        if (val.isNaN) hasNaN = true;
        if (val.isInfinite) hasInfinity = true;
        if (!val.isNaN && !val.isInfinite) {
          minVal = minVal < val ? minVal : val;
          maxVal = maxVal > val ? maxVal : val;
        }
      }
      print('$stage [$name] stats: length=${tensor.length}, hasNaN=$hasNaN, hasInfinity=$hasInfinity, min=$minVal, max=$maxVal');
    }
  }

  static Future<dynamic> getEditedLatent(
      Float32List originalLatent,
      String editingName,
      double editingDegree,
      BuildContext? context,
      ) async {
    print('[LatentEditor.getEditedLatent] Starting for editingName=$editingName, degree=$editingDegree');
    print('[LatentEditor] originalLatent length=${originalLatent.length}, expected=9216');
    if (originalLatent.length != 9216) {
      throw Exception('Invalid latent length: ${originalLatent.length}, expected 9216');
    }

    if (interfaceganDirections.containsKey(editingName)) {
      // InterfaceGAN editing (e.g., age)
      final sessionKey = interfaceganDirections[editingName]!;
      print('[LatentEditor] Running InterfaceGAN for $sessionKey');
      final editedLatent = await InferenceRunner.runInterfacegan(originalLatent, editingDegree, sessionKey);
      print('[LatentEditor] InterfaceGAN success, output length=${editedLatent.length}');
      _logTensorStats('InterfaceGAN', [editedLatent], ['edited_latent']);
      return editedLatent;
    } else if (editingName.startsWith('styleclip_global_')) {
      // StyleClip global editing
      print('[LatentEditor] Running StyleClip for $editingName');
      final session = InferenceRunner.decoderStylespaceSession;
      if (session == null) throw Exception('decoder_stylespace model not loaded');

      // Input is FP32
      final latentTensor = OrtValueTensor.createTensorWithDataList(originalLatent, [1, 18, 512]);

      // Define output names (17 styles + 9 RGB)
      final outputModelNames = [
        ...List.generate(17, (i) => 'style_out_$i'),
        ...List.generate(9, (i) => 'rgb_out_$i'),
      ];

      // Run decoder_stylespace
      final outputs = await session.runAsync(
        OrtRunOptions(),
        {'w': latentTensor},
      );
      latentTensor.release();

      print('decoder_stylespace success return!!!!');
      if (outputs == null || outputs.length != 26 || outputs.any((output) => output == null)) {
        outputs?.forEach((e) => e?.release());
        throw Exception('Invalid outputs from decoder_stylespace: ${outputs?.length ?? 'null'} outputs, expected 26');
      }

      // Process FP32 outputs
      final stylespaceLatent = List<Float32List>.generate(26, (i) {
        final output = InferenceRunner.flattenNestedList(outputs[i]!.value);
        final expectedLength = styleSpaceDimensions[i];
        if (output.length != expectedLength) {
          print('Warning: Output $i has length ${output.length}, expected $expectedLength. Using zero-filled tensor.');
          return Float32List(expectedLength)..fillRange(0, expectedLength, 0.0);
        }
        return output;
      });

      outputs.forEach((e) => e?.release());

      // Log stylespace outputs
      _logTensorStats('decoder_stylespace', stylespaceLatent, outputModelNames);

      // Call StyleClip editing
      final (editedSsList, editedRgbList) = await StyleClipEditor.getStyleclipGlobalEdits(
        stylespaceLatent,
        editingDegree,
        editingName,
        context,
      );

      // Correct editedSsList (17 styles)
      final correctedSsList = List<Float32List>.filled(17, Float32List(0));
      for (var index = 0; index < 17; index++) {
        final expectedLength = styleSpaceDimensions[index];
        if (StyleClipEditor.styleSpaceIndicesWithoutToRgb.contains(index)) {
          final ssIndex = StyleClipEditor.styleSpaceIndicesWithoutToRgb.indexOf(index);
          if (ssIndex < editedSsList.length && editedSsList[ssIndex].length == expectedLength) {
            correctedSsList[index] = Float32List.fromList(editedSsList[ssIndex]);
          } else {
            print('[LatentEditor] Warning: editedSsList[$ssIndex] has length ${ssIndex < editedSsList.length ? editedSsList[ssIndex].length : 'N/A'}, expected $expectedLength. Using fallback.');
            correctedSsList[index] = Float32List(expectedLength)..fillRange(0, expectedLength, 0.0);
          }
        } else {
          correctedSsList[index] = Float32List(expectedLength)..fillRange(0, expectedLength, 0.0);
        }
        print('correctedSsList o${index + 1} has length ${correctedSsList[index].length}');
      }

      // Correct editedRgbList (9 RGB)
      final correctedRgbList = List<Float32List>.filled(9, Float32List(0));
      for (var index = 0; index < 9; index++) {
        final expectedLength = styleSpaceDimensions[StyleClipEditor.toRgbIndices[index]];
        if (index < editedRgbList.length && editedRgbList[index].length == expectedLength) {
          correctedRgbList[index] = Float32List.fromList(editedRgbList[index]);
        } else {
          print('[LatentEditor] Warning: editedRgbList[$index] has length ${index < editedRgbList.length ? editedRgbList[index].length : 'N/A'}, expected $expectedLength. Using fallback.');
          correctedRgbList[index] = Float32List(expectedLength)..fillRange(0, expectedLength, 0.0);
        }
        print('correctedRgbList o${index + 1} has length ${correctedRgbList[index].length}');
      }

      print('editedSsList length: ${correctedSsList.length}, expected: 17');
      print('editedRgbList length: ${correctedRgbList.length}, expected: 9');

      // Return stylespace tuple
      return (correctedSsList, correctedRgbList);
    } else {
      throw Exception('Edit name $editingName is not available');
    }
  }
}