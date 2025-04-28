import 'dart:typed_data';
import '../onnx_wrapper.dart';
import 'styleclip_editor.dart';
import 'package:flutter/material.dart';

class LatentEditor {
  static final interfaceganDirections = {
    'age': 'interfacegan_age',
  };

  static Future<dynamic> getEditedLatent(
      Float32List originalLatent,
      String editingName,
      double editingDegree,
      BuildContext context,
      ) async {
    if (interfaceganDirections.containsKey(editingName)) {
      final sessionKey = interfaceganDirections[editingName]!;
      final session = CustomSession(sessionKey);
      // Предполагаем, что форма тензора [1, originalLatent.length]
      final latentTensor = CustomTensor.createTensorWithDataList(
        originalLatent,
        [1, originalLatent.length],
      );
      final degreeTensor = CustomTensor.createTensorWithDataList(
        Float32List.fromList([editingDegree]),
        [1],
      );
      final outputs = await session.runAsync(
        CustomRunOptions(),
        {'latent': latentTensor, 'degree': degreeTensor},
        outputNames: ['output'],
      );
      if (outputs.isEmpty || outputs[0] == null) {
        throw Exception('No output from $sessionKey');
      }
      return outputs[0]!;
    } else if (editingName.startsWith('styleclip_global_')) {
      final session = CustomSession('decoder_stylespace');
      final latentTensor = CustomTensor.createTensorWithDataList(originalLatent, [1, 18, 512],);

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

      // Возвращаем кортеж для StyleSpace
      return (editedSsList, editedRgbList);
    } else {
      throw Exception('Edit name $editingName is not available');
    }
  }
}