cmake_minimum_required(VERSION 3.11)
project(mlc)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add the library 'la' from the source file in the 'lib' folder
add_library(tensor lib/tensor.c)
add_library(la lib/la.c)

add_executable(main main.c)
target_link_libraries(main PUBLIC tensor la m)
target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib)
