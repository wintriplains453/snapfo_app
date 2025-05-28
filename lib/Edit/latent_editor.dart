import 'dart:math';
import 'dart:typed_data';
import '../onnx_wrapper.dart';
import 'styleclip_editor.dart';
import 'package:flutter/material.dart';

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
      BuildContext context,
      ) async {
    print('[LatentEditor.getEditedLatent] Starting for editingName=$editingName, degree=$editingDegree');
    print('[LatentEditor] originalLatent length=${originalLatent.length}, expected=9216');

    if (interfaceganDirections.containsKey(editingName)) {
      final sessionKey = interfaceganDirections[editingName]!;
      print('[LatentEditor] Running InterfaceGAN for $sessionKey');
      final session = CustomSession(sessionKey);
      final latentTensor = CustomTensor.createTensorWithDataList(originalLatent, [1, 18, 512]);
      final degreeTensor = CustomTensor.createTensorWithDataList(Float32List.fromList([editingDegree]), [1]);

      final outputs = await session.runAsync(
        CustomRunOptions(),
        {'start_w': latentTensor, 'factor': degreeTensor},
        outputNames: ['edited_age'],
      );
      if (outputs.isEmpty || outputs[0] == null) {
        throw Exception('No output from $sessionKey');
      }
      print('[LatentEditor] InterfaceGAN success, output length=${outputs[0]!.length}');
      return outputs[0]!;
    } else if (editingName.startsWith('styleclip_global_')) {
      print('[LatentEditor] Running StyleClip for $editingName');
      final session = CustomSession('decoder_stylespace');
      final latentTensor = CustomTensor.createTensorWithDataList(originalLatent, [1, 18, 512]);

      final outputModelNames = [
      ...List.generate(17, (i) => 'style_out_$i'),
      ...List.generate(9, (i) => 'rgb_out_$i'),
      ];

      final outputs = await session.runAsync(
      CustomRunOptions(),
      {'w': latentTensor},
      outputNames: outputModelNames,
      );

      print('decoder_stylespace success return!!!!');
      if (outputs.any((output) => output == null)) {
       throw Exception('Invalid outputs from decoder_stylespace');
      }

      // Преобразуем выходы в список Float32List
      final stylespaceLatent = List<Float32List>.generate(26, (i) {
        final expectedLength = styleSpaceDimensions[i];
        if (outputs[i] == null || outputs[i]!.length != expectedLength) {
          print('Warning: Output $i has length ${outputs[i]?.length ?? 'N/A'}, expected $expectedLength. Using zero-filled tensor.');
          return Float32List(expectedLength)..fillRange(0, expectedLength, 0.0);
        }
        return outputs[i]!;
      });

      for (var i = 0; i < stylespaceLatent.length; i++) {
        if (stylespaceLatent[i].length != styleSpaceDimensions[i]) {
          print('Error: stylespaceLatent[$i] has length ${stylespaceLatent[i].length}, expected ${styleSpaceDimensions[i]}');
        }
      }

      // Вызов StyleClip редактирования
      final (editedSsList, editedRgbList) = await StyleClipEditor.getStyleclipGlobalEdits(
      stylespaceLatent,
      editingDegree,
      editingName,
      context,
      );

      // Проверяем и корректируем размеры editedSsList
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
        print('correctedSsList o${index + 1} has length ${correctedSsList[index]!.length}');
      }

      // Проверяем и корректируем размеры editedRgbList
      final correctedRgbList = List<Float32List>.filled(9, Float32List(0));
      for (var index = 0; index < 9; index++) {
        final expectedLength = styleSpaceDimensions[StyleClipEditor.toRgbIndices[index]];
        if (index < editedRgbList.length && editedRgbList[index].length == expectedLength) {
          correctedRgbList[index] = Float32List.fromList(editedRgbList[index]);
        } else {
          print('[LatentEditor] Warning: editedRgbList[$index] has length ${index < editedRgbList.length ? editedRgbList[index].length : 'N/A'}, expected $expectedLength. Using fallback.');
          correctedRgbList[index] = Float32List(expectedLength)..fillRange(0, expectedLength, 0.0);
        }
        print('correctedRgbList o${index + 1} has length ${correctedRgbList[index]!.length}');
      }

      print('editedSsList length: ${correctedSsList.length}, expected: 17');
      print('editedSsList length: ${correctedSsList.length}, expected: 17');
      print('editedRgbList length: ${correctedRgbList.length}, expected: 9');

      // Возвращаем кортеж для StyleSpace
      return (correctedSsList, correctedRgbList);
    } else {
      throw Exception('Edit name $editingName is not available');
    }
  }
}