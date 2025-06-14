cmake_minimum_required(VERSION 3.10)
project(snapfo_app LANGUAGES CXX)

# Устанавливаем стандарт C++
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Создаем разделяемую библиотеку
add_library(convert_fp SHARED
        convert_fp.cpp
)

# Указываем директории для заголовочных файлов
target_include_directories(convert_fp PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Устанавливаем параметры для экспорта функций
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

# Устанавливаем имя выходного файла
set_target_properties(convert_fp PROPERTIES
        OUTPUT_NAME "convert_fp"
)