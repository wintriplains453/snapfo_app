import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'face_alignment_method_channel.dart';

abstract class FaceAlignmentPlatform extends PlatformInterface {
  /// Constructs a FaceAlignmentPlatform.
  FaceAlignmentPlatform() : super(token: _token);

  static final Object _token = Object();

  static FaceAlignmentPlatform _instance = MethodChannelFaceAlignment();

  /// The default instance of [FaceAlignmentPlatform] to use.
  ///
  /// Defaults to [MethodChannelFaceAlignment].
  static FaceAlignmentPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [FaceAlignmentPlatform] when
  /// they register themselves.
  static set instance(FaceAlignmentPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }
}
