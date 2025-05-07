import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:opencv_dart/opencv.dart' as cv;
import 'Edit/edit.dart';
import 'Edit/inference_runner.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  bool _isLoadingModels = true;
  List<Uint8List> images = [];
  final picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _initOnnx();
  }

  Future<void> _initOnnx() async {
    try {
      await InferenceRunner.initEnv();
      print("[initEnv] initEnv() completed!");
      await InferenceRunner.loadModels();
      print("[_initOnnx] loadModels() completed!");
      setState(() => _isLoadingModels = false);
      print("[_initOnnx] Done, set _isLoadingModels=false");
    } catch (e) {
      print("Error initializing ONNX: $e");
    }
  }

  Future<dynamic> _convertImageToBytes(dynamic image) async {
    try {
      print("Converting image of type: ${image.runtimeType}");
      if (image is Uint8List) return image;
      if (image is cv.Mat) return cv.imencode('.jpg', image);

      if (image is Image) {
        final completer = Completer<ui.Image>();
        final imageStream = image.image.resolve(ImageConfiguration.empty);
        imageStream.addListener(ImageStreamListener((info, _) {
          completer.complete(info.image);
        }));

        final uiImage = await completer.future;
        final byteData = await uiImage.toByteData(format: ui.ImageByteFormat.png);
        return byteData?.buffer.asUint8List();
      }

      print("Unsupported image type: ${image.runtimeType}");
      return null;
    } catch (e) {
      print("Error converting image: $e");
      return null;
    }
  }

  Future<void> pickAndEdit() async {
    if (_isLoadingModels) {
      print("Models still loading...");
      return;
    }

    try {
      final picked = await picker.pickImage(source: ImageSource.gallery);
      if (picked == null) return;

      final bytes = await picked.readAsBytes();
      final mat = cv.imread(picked.path);
      print("cv.imread => width: ${mat.cols}, height: ${mat.rows}");

      final edited = await ImageEditor.edit(
        inputBytes: bytes,
        editingName: 'styleclip_global_face with hair_face with blonde hair_0.2',
        editingDegree: 16.0,
        align: false,
        combinedPreEditor: false,
        context: context,
      );

      print("Edited image type: ${edited.runtimeType}");
      final editedBytes = await _convertImageToBytes(edited);
      if (editedBytes == null) {
        throw Exception('Failed to convert edited image to bytes');
      }

      print('Image ready!!');
      setState(() {
        images = [bytes, editedBytes];
        // dispose();
      });
    } catch (e) {
      print('Error during image editing: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: ${e.toString()}')),
      );
    }
  }

  @override
  void dispose() {
    InferenceRunner.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("ONNX Editing Example")),
      body: _isLoadingModels
          ? const Center(child: CircularProgressIndicator())
          : Column(
        children: [
          ElevatedButton(
            onPressed: pickAndEdit,
            child: const Text("Pick an Image & Edit"),
          ),
          Expanded(
            child: images.isEmpty
                ? const Center(child: Text("No images selected"))
                : ListView.builder(
              itemCount: images.length,
              itemBuilder: (ctx, idx) => Padding(
                padding: const EdgeInsets.all(8.0),
                child: Image.memory(images[idx]),
              ),
            ),
          ),
        ],
      ),
    );
  }
}