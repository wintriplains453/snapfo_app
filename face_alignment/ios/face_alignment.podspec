#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint face_alignment.podspec` to validate before publishing.
#
Pod::Spec.new do |s| s.name = 'face_alignment' s.version = '1.0.0' s.summary = 'Face alignment plugin for Flutter' s.description = <<-DESC A Flutter plugin for face alignment using OpenCV and flutter_mediapipe. DESC s.homepage = 'https://example.com' s.license = { :file => '../LICENSE' } s.author = { 'Your Name' => 'your.email@example.com' } s.source = { :path => '.' } s.source_files = 'Classes/**/*.{h,m,mm,cpp}' s.dependency 'Flutter' s.platform = :ios, '11.0'

s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' } s.swift_version = '5.0'

s.dependency 'OpenCV', '~> 4.5.0' end


