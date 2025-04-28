import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:opencv_dart/opencv.dart' as cv;
import '../onnx_wrapper.dart';

import 'Edit/inference_runner.dart';
import 'Edit/edit.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  bool _isLoadingModels = true;
  List<Uint8List> images = [];

  @override
  void initState() {
    super.initState();
    _initOnnx();
  }

  Future<void> _initOnnx() async {
    InferenceRunner.initEnv();
    print("[initEnv] initEnv() completed!");
    InferenceRunner.loadModels();
    print("[_initOnnx] loadModels() completed!");
    setState(() => _isLoadingModels = false);
    print("[_initOnnx] Done, set _isLoadingModels=false");
  }

  Future<void> pickAndEdit() async {
    if (_isLoadingModels) {
      print("Models still loading...");
      return;
    }
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;

    final path = picked.path;
    final mat = cv.imread(path);
    print("cv.imread => width: ${mat.cols}, height: ${mat.rows}");
    final bytes = await picked.readAsBytes();

    // Передаём context из State в ImageEditor.edit
    final edited = await ImageEditor.edit(
      inputBytes: bytes,
      editingName: 'styleclip_global_face with hair_face with blonde hair_0.2',
      editingDegree: 16.0,
      align: false,
      combinedPreEditor: false,
      context: context,
    );

    setState(() {
      images = [bytes, edited as Uint8List];
    });
  }

  @override
  void dispose() {
    // Clean up
    // InferenceRunner.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoadingModels) {
      return MaterialApp(
        home: Scaffold(
          appBar: AppBar(title: const Text("Loading Models...")),
          body: const Center(child: CircularProgressIndicator()),
        ),
      );
    }

    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text("ONNX Editing Example")),
        body: Column(
          children: [
            ElevatedButton(
              onPressed: pickAndEdit,
              child: const Text("Pick an Image & Edit (Age)"),
            ),
            Expanded(
              child: ListView.builder(
                itemCount: images.length,
                itemBuilder: (ctx, idx) => Card(
                  child: Image.memory(images[idx]),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}