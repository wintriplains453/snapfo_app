import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:opencv_dart/opencv.dart' as cv;
import 'Edit/edit.dart';
import 'package:image/image.dart' as img;
import 'Edit/inference_runner.dart';
import 'package:face_alignment/face_alignment.dart';

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
  bool _isProcessing = false; // Track processing state
  List<Uint8List> images = []; // Store original, aligned, edited images
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
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error loading models: $e')),
        );
      }
    }
  }

  Future<Uint8List?> _convertImageToBytes(dynamic image) async {
    try {
      print("Converting image of type: ${image.runtimeType}");
      if (image is Uint8List) {
        return Future.value(image);
      }
      if (image is cv.Mat) {
        final (success, encoded) = cv.imencode('.jpg', image);
        if (!success) {
          throw Exception("Failed to encode cv.Mat to JPEG");
        }
        return Future.value(encoded);
      }

      if (image is ui.Image) {
        final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
        if (byteData == null) {
          throw Exception("Failed to convert ui.Image to byte data");
        }
        return Future.value(byteData.buffer.asUint8List());
      }

      if (image is img.Image) {
        return Future.value(img.encodeJpg(image, quality: 90));
      }

      if (image is Image) {
        if (image.image is MemoryImage) {
          return Future.value((image.image as MemoryImage).bytes);
        } else if (image.image is FileImage) {
          final file = (image.image as FileImage).file;
          return Future.value(await file.readAsBytes());
        } else {
          final completer = Completer<ui.Image>();
          final imageStream = image.image.resolve(ImageConfiguration.empty);
          imageStream.addListener(ImageStreamListener((info, _) {
            completer.complete(info.image);
          }));

          final uiImage = await completer.future;
          final byteData = await uiImage.toByteData(format: ui.ImageByteFormat.png);
          return Future.value(byteData?.buffer.asUint8List());
        }
      }

      print("Unsupported image type: ${image.runtimeType}");
      return Future.value(null);
    } catch (e) {
      print("Error converting image: $e");
      return Future.value(null);
    }
  }

  Future<void> pickAlignAndEdit() async {
    if (_isLoadingModels) {
      print("Models still loading...");
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please wait for models to load')),
      );
      return;
    }

    if (_isProcessing) {
      print("Processing in progress...");
      return;
    }

    setState(() => _isProcessing = true);

    try {
      final picked = await picker.pickImage(source: ImageSource.gallery);
      if (picked == null) {
        setState(() => _isProcessing = false);
        return;
      }

      final originalBytes = await picked.readAsBytes();
      final mat = cv.imread(picked.path);
      print("cv.imread => width: ${mat.cols}, height: ${mat.rows}");

      // Align face
      print("Starting face alignment...");
      final alignedPath = await FaceAlignment.alignFaceAsync(picked.path);
      final alignedFile = File(alignedPath);
      final alignedBytes = await alignedFile.readAsBytes();
      print("Face alignment completed, aligned image size: ${alignedBytes.length}");

      // Edit aligned image
      print("Starting ONNX editing...");
      final edited = await ImageEditor.edit(
        inputBytes: alignedBytes,
        editingName: 'age',
        editingDegree: 10.0,
        align: false, // Already aligned
        combinedPreEditor: false,
        context: context,
      );

      print("Edited image type: ${edited.runtimeType}");
      final editedBytes = await _convertImageToBytes(edited);
      if (editedBytes == null) {
        throw Exception('Failed to convert edited image to bytes');
      }

      print('Image processing completed!!');
      setState(() {
        images = [originalBytes, alignedBytes, editedBytes];
        _isProcessing = false;
      });
    } catch (e) {
      print('Error during image processing: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: ${e.toString()}')),
        );
      }
      setState(() => _isProcessing = false);
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
      appBar: AppBar(title: const Text("ONNX Editing with Face Alignment")),
      body: _isLoadingModels
          ? const Center(child: CircularProgressIndicator())
          : Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: ElevatedButton(
              onPressed: _isProcessing ? null : pickAlignAndEdit,
              child: _isProcessing
                  ? const CircularProgressIndicator(color: Colors.white)
                  : const Text("Pick, Align & Edit Image"),
            ),
          ),
          Expanded(
            child: images.isEmpty
                ? const Center(child: Text("No images selected"))
                : ListView.builder(
              itemCount: images.length,
              itemBuilder: (ctx, idx) => Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    Text(
                      idx == 0
                          ? "Original"
                          : idx == 1
                          ? "Aligned"
                          : "Edited",
                      style: const TextStyle(
                          fontWeight: FontWeight.bold),
                    ),
                    Image.memory(
                      images[idx],
                      height: 200,
                      fit: BoxFit.contain,
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// FaceAlignmentDemo remains unchanged
class FaceAlignmentDemo extends StatefulWidget {
  @override
  _FaceAlignmentDemoState createState() => _FaceAlignmentDemoState();
}

class _FaceAlignmentDemoState extends State<FaceAlignmentDemo> {
  File? _image;
  File? _alignedImage;
  final picker = ImagePicker();

  Future<void> _pickImage() async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
      try {
        final alignedPath = await FaceAlignment.alignFaceAsync(pickedFile.path);
        setState(() {
          _alignedImage = File(alignedPath);
        });
      } catch (e) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Face Alignment Demo')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _image == null
                ? Text('No image selected.')
                : Image.file(_image!, height: 200),
            SizedBox(height: 20),
            _alignedImage == null
                ? Text('No aligned image.')
                : Image.file(_alignedImage!, height: 200),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _pickImage,
              child: Text('Pick and Align Image'),
            ),
          ],
        ),
      ),
    );
  }
}