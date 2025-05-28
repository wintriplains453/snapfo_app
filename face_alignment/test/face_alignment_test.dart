import 'package:flutter_test/flutter_test.dart';
import 'package:face_alignment/face_alignment.dart';
import 'package:face_alignment/face_alignment_platform_interface.dart';
import 'package:face_alignment/face_alignment_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockFaceAlignmentPlatform
    with MockPlatformInterfaceMixin
    implements FaceAlignmentPlatform {

  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  final FaceAlignmentPlatform initialPlatform = FaceAlignmentPlatform.instance;

  test('$MethodChannelFaceAlignment is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelFaceAlignment>());
  });

  test('getPlatformVersion', () async {
    FaceAlignment faceAlignmentPlugin = FaceAlignment();
    MockFaceAlignmentPlatform fakePlatform = MockFaceAlignmentPlatform();
    FaceAlignmentPlatform.instance = fakePlatform;

    expect(await faceAlignmentPlugin.getPlatformVersion(), '42');
  });
}
