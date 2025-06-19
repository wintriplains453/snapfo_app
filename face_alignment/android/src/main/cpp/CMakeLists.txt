cmake_minimum_required(VERSION 3.10...3.31)

project(snapfo_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Specify OpenCV path
set(OpenCV_DIR "E:/opencvAndroid/OpenCV-android-sdk/sdk/native/jni")
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found at ${OpenCV_DIR}. Please ensure it is installed and the path is correct.")
endif()

set(GLM_INCLUDE_DIR "E:/GLM/gglm/glm")
if(NOT EXISTS "${GLM_INCLUDE_DIR}/glm.hpp")
    message(FATAL_ERROR "GLM not found at ${GLM_INCLUDE_DIR}/glm.hpp")
endif()

add_library(face_alignment SHARED face_alignment.cpp)

target_include_directories(face_alignment PUBLIC
        ${OpenCV_DIR}
        ${GLM_INCLUDE_DIR}
)

target_link_libraries(face_alignment PRIVATE
        ${OpenCV_LIBS}
)

target_compile_options(face_alignment PRIVATE
        -Wall -Wextra -O2
)

if(ANDROID)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
    set_target_properties(face_alignment PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../android/libs/${ANDROID_ABI}"
    )
endif()

if(IOS)
    set_target_properties(face_alignment PROPERTIES
            FRAMEWORK FALSE
            MACOSX_FRAMEWORK_IDENTIFIER com.example.snapfo_app
            XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY ""
    )
endif()