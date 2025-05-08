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
    512, 512, 512, 512, 512, 512, 512, 512, 512, 512, // 10 x 512 (style_out_0–9)
    256, 256, // 2 x 256 (style_out_10–11)
    128, 128, // 2 x 128 (style_out_12–13)
    64, 64, // 2 x 64 (style_out_14–15)
    32, // 1 x 32 (style_out_16)
    512, // 1 x 512 (rgb_out_0)
    512, 512, 512, // 3 x 512 (rgb_out_1–3)
    512, // 1 x 512 (rgb_out_4)
    256, // 1 x 256 (rgb_out_5)
    128, // 1 x 128 (rgb_out_6)
    64, // 1 x 64 (rgb_out_7)
    32, // 1 x 32 (rgb_out_8)
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

    if (interfaceganDirections.containsKey(editingName)) {
      final sessionKey = interfaceganDirections[editingName]!;
      print('[LatentEditor] Running InterfaceGAN for $sessionKey');
      final session = CustomSession(sessionKey);
      final latentTensor = CustomTensor.createTensorWithDataList(originalLatent, [1, originalLatent.length]);
      final degreeTensor = CustomTensor.createTensorWithDataList(Float32List.fromList([editingDegree]), [1]);

      final outputs = await session.runAsync(
        CustomRunOptions(),
        {'latent': latentTensor, 'degree': degreeTensor},
        outputNames: ['output'],
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
    final stylespaceLatent = outputs.cast<Float32List>();

    // Вызов StyleClip редактирования
    final (editedSsList, editedRgbList) = await StyleClipEditor.getStyleclipGlobalEdits(
    stylespaceLatent,
    editingDegree,
    editingName,
    context,
    );

    // Проверяем и корректируем размеры editedSsList
    final correctedSsList = List<Float32List>.generate(17, (index) {
      final expectedLength = styleSpaceDimensions[index];
      if (editedSsList[index].length != expectedLength) {
        print('[LatentEditor] Warning: editedSsList[$index] has length ${editedSsList[index].length}, expected $expectedLength. Using fallback.');
        return Float32List(expectedLength)..fillRange(0, expectedLength, 0.0);
      }
    return Float32List.fromList(editedSsList[index]);
    });

    // Проверяем и корректируем размеры editedRgbList
    final correctedRgbList = List<Float32List>.generate(9, (index) {
      final expectedLength = styleSpaceDimensions[17 + index];
      if (editedRgbList[index].length != expectedLength) {
        print('[LatentEditor] Warning: editedRgbList[$index] has length ${editedRgbList[index].length}, expected $expectedLength. Using fallback.');
        return Float32List(expectedLength)..fillRange(0, expectedLength, 0.0);
      }
      return Float32List.fromList(editedRgbList[index]);
    });

    print('editedSsList length: ${correctedSsList.length}, expected: 17');
    print('editedRgbList length: ${correctedRgbList.length}, expected: 9');

    // Возвращаем кортеж для StyleSpace
    return (correctedSsList, correctedRgbList);
    } else {
    throw Exception('Edit name $editingName is not available');
    }
  }
}