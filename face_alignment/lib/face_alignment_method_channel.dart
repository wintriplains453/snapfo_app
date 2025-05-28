import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'face_alignment_platform_interface.dart';

/// An implementation of [FaceAlignmentPlatform] that uses method channels.
class MethodChannelFaceAlignment extends FaceAlignmentPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('face_alignment');

  @override
  Future<String?> getPlatformVersion() async {
    final version = await methodChannel.invokeMethod<String>('getPlatformVersion');
    return version;
  }
}
