cmake_minimum_required(VERSION 3.11)
project(mlc)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add the library 'la' from the source file in the 'lib' folder
add_library(la lib/la.c)

# Add the executable 'main' from the source file in the root folder
add_executable(main main.c)

# Link the 'la' library to the 'main' executable
target_link_libraries(main PUBLIC la)

# Specify include directories for the 'la' library
target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib)
