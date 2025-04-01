import 'dart:typed_data';
import 'styleclip_editor.dart';
import 'inference_runner.dart';

class LatentEditor {
  static Future<Object> getEditedLatent(
      Object originalLatent,
      String editingName,
      double editingDegree,
      ) async {
    if (editingName == 'age' || editingName == 'smile' || editingName == 'rotation') {
      final typedLatent = originalLatent as Float32List;
      final edited = await InferenceRunner.runInterfacegan(typedLatent, editingDegree);
      return edited; // single Float32List
    } else if (editingName.startsWith('styleclip_global_')) {
      final styleSpaceLatent = <Float32List>[originalLatent as Float32List];
      final (ss, rgb) = StyleclipEditor.getStyleclipGlobalEdits(
        styleSpaceLatent,
        editingDegree,
        editingName,
      );
      return (ss, rgb);
    } else {
      throw Exception("Edit name $editingName is not implemented.");
    }
  }
}
