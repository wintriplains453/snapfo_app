import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'dart:async';

void main() {
  runApp(
    MaterialApp(
      home: ImagePickerExample(),
    ),
  );
}

class ImagePickerExample extends StatefulWidget {
  @override
  _ImagePickerExampleState createState() => _ImagePickerExampleState();
}

class _ImagePickerExampleState extends State<ImagePickerExample> {
  String imagePath = '';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey,
      appBar: AppBar(
        title: Text('CONV'),
        backgroundColor: Colors.grey[900],
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () async {
                final ImagePicker picker = ImagePicker();
                final XFile? image = await picker.pickImage(source: ImageSource.gallery);
                if (image != null) {
                  await saveImageToDirectory(image.path);
                  setState(() {
                    imagePath = image.path;
                  });
                }
              },
              child: const Text('Выбрать изображение'),
            ),
            if (imagePath.isNotEmpty)
              Padding(
                padding: const EdgeInsets.all(16.0),
                child: Image.file(
                  File(imagePath),
                  width: 300,
                  height: 300,
                  fit: BoxFit.cover,
                ),
              ),
          ],
        ),
      ),
    );
  }

  Future<void> saveImageToDirectory(String imagePath) async {
    final directory = await getApplicationDocumentsDirectory();
    final newDirectory = Directory('${directory.path}/styleGan/notebook');

    if (!await newDirectory.exists()) {
      await newDirectory.create(recursive: true);
    }

    final imageFile = File(imagePath);
    final newImagePath = '${newDirectory.path}/${imageFile.uri.pathSegments.last}';
    await imageFile.copy(newImagePath);

    print('Image saved to $newImagePath');
  }
}
