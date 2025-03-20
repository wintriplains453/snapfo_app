import 'dart:io';
import 'dart:ui';
import 'package:image/image.dart';
import 'runners/fse_inference_runner.dart';
// import 'package:your_package/transforms_registry.dart';
// import 'package:your_package/other_dependencies.dart';

class SimpleRunner {
  final FSEInferenceRunner inferenceRunner;

  SimpleRunner({required String editorCkptPth, String simpleConfigPth = "configs/simple_inference.yaml"}) :
    inferenceRunner = FSEInferenceRunner(_loadConfig(simpleConfigPth, editorCkptPth)) {
    inferenceRunner.setup();
    inferenceRunner.method.eval();
    inferenceRunner.method.decoder = inferenceRunner.method.decoder.toFloat();
  }

  static Config _loadConfig(String simpleConfigPth, String editorCkptPth) {
    // Загрузите конфигурацию из файла YAML
    var config = OmegaConf.load(simpleConfigPth);
    config.model.checkpointPath = editorCkptPth;
    config.methodsArgs.fseFull = {};
    return config;
  }

  Future<Image> edit({
    required String origImgPth,
    required String editingName,
    required double editedPower,
    required String savePth,
    bool align = false,
    bool useMask = false,
    double maskThreshold = 0.995,
    String? maskPath,
    bool saveE4E = false,
    bool saveInversion = false,
  }) async {
    final saveDir = Directory(savePth).parent;
    if (!await saveDir.exists()) {
      await saveDir.create(recursive: true);
    }

    String alignedImagePth = origImgPth;

    if (align) {
      final alignedImage = await runAlignment(origImgPth);
      final saveAlignPth = '${saveDir.path}/${basenameWithoutExtension(savePth)}_aligned.jpg';
      print('Save aligned image to $saveAlignPth');
      alignedImage.save(saveAlignPth);
      alignedImagePth = saveAlignPth;
    }

    Image? mask;
    if (useMask && maskPath == null) {
      print('Preparing mask');
      maskPath = await extractMask(alignedImagePth, saveDir.path, trash: maskThreshold);
      print('Done');
    }

    if (useMask && maskPath != null) {
      print('Use mask from $maskPath');
      mask = decodeImage(File(maskPath).readAsBytesSync());
    }

    final origImg = decodeImage(File(alignedImagePth).readAsBytesSync());
    final transformDict = transformsRegistry['face_1024']().getTransforms();
    final transformedImg = transformDict['test'](origImg);

    final invImages = await inferenceRunner.runOnBatch(transformedImg);
    final editedImage = await inferenceRunner.runEditingOnBatch(
      methodResBatch: invImages,
      editingName: editingName,
      editingDegrees: [editedPower],
      mask: mask,
      returnE4E: saveE4E,
    );

    // if (saveInversion) {
    //   final saveInvPth = '${saveDir.path}/${basenameWithoutExtension(savePth)}_inversion.jpg';
    //   final invImage = tensorToImage(invImages[0]);
    //   invImage.save(saveInvPth);
    // }

    if (saveE4E) {
      // Обработка e4e
      // ... (аналогично)
    }

    final finalEditedImage = tensorToImage(editedImage[0]);
    finalEditedImage.save(savePth);

    // if (align) {
    //   final unalignedPath = '${saveDir.path}/${basenameWithoutExtension(savePth)}_unaligned.jpg';
    //   await unalign(finalEditedImage, origImgPth, unalignedPath);
    // }

    return finalEditedImage;
  }

  void availableEditings() {
    final editsTypes = <String>[];
    for (var field in inferenceRunner.latentEditor.runtimeType.toString().split('_')) {
      if (field.contains('directions')) {
        editsTypes.add(field);
      }
    }

    print('This code handles the following editing directions for following methods:');
    final availableDirections = {};
    for (var editType in editsTypes) {
      print('$editType:');
      final editTypeDirections = inferenceRunner.latentEditor[editType]?.keys;
