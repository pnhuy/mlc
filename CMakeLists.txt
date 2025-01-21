cmake_minimum_required(VERSION 3.11)
project(mlc)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(utils lib/utils.c)
add_library(tensor lib/tensor.c)
add_library(la lib/la.c)
add_library(linear_models lib/linear_models.c)

target_link_libraries(tensor PUBLIC utils m)
target_link_libraries(la PUBLIC tensor utils m)
target_link_libraries(linear_models PUBLIC la tensor utils m)

add_executable(test_tensor test/test_tensor.c)
target_link_libraries(test_tensor PUBLIC la)
target_include_directories(test_tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_executable(test_la test/test_la.c)
target_link_libraries(test_la la)
target_include_directories(test_la PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_executable(test_linear_models test/test_linear_models.c)
target_link_libraries(test_linear_models la linear_models)
target_include_directories(test_linear_models PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_executable(main main.c)
target_link_libraries(main PUBLIC la linear_models)
target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib)
