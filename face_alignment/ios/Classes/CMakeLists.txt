Specify the minimum CMake version required

cmake_minimum_required(VERSION 3.10)

Define the project

project(snapfo_app LANGUAGES CXX)

Set C++ standard to C++17

set(CMAKE_CXX_STANDARD 17) set(CMAKE_CXX_STANDARD_REQUIRED ON) set(CMAKE_CXX_EXTENSIONS OFF)

Find OpenCV package

find_package(OpenCV REQUIRED) if(NOT OpenCV_FOUND) message(FATAL_ERROR "OpenCV not found. Please ensure it is installed and available.") endif()

Find Eigen3 package

find_package(Eigen3 REQUIRED) if(NOT Eigen3_FOUND) message(FATAL_ERROR "Eigen3 not found. Please ensure it is installed and available.") endif()

Define the shared library

add_library(face_alignment SHARED face_alignment.cpp )

Specify include directories for the target

target_include_directories(face_alignment PUBLIC ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} )

Link libraries to the target

target_link_libraries(face_alignment PRIVATE ${OpenCV_LIBS} Eigen3::Eigen )

Set compiler options for optimization and warnings

target_compile_options(face_alignment PRIVATE -Wall -Wextra -O2 # Optimize for performance )

For Flutter Android integration, set output directory

if(ANDROID) set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined") set_target_properties(face_alignment PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../android/libs/${ANDROID_ABI}" ) endif()

For iOS, set architecture and other properties

if(IOS) set_target_properties(face_alignment PROPERTIES FRAMEWORK FALSE MACOSX_FRAMEWORK_IDENTIFIER com.example.snapfo_app XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "" ) endif()