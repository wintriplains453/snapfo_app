import 'runners/simple_runner.dart';

void main() {
  final runner = SimpleRunner(editorCkptPth: "pretrained_models/sfe_editor_light.pt");

  // Выполнение редактирования
  final String editedImagePath = "notebook/images/gosling.jpg";
  final String neutralPrompt = "face";
  final String targetPrompt = "hair_face";
  final double disentanglement = 0.18;

  runner.edit(
    origImgPth: "notebook/images/smith.jpg",
    editingName: "age",
    editedPower: 5,
    savePth: "editing_res/smith/smith.jpg",
    align: true,
    saveInversion: true,
  );
}