cmake_minimum_required(VERSION 3.17)

set(SPIDREXAMPLE "SpidrExample")
PROJECT(${SPIDREXAMPLE})

add_executable(${SPIDREXAMPLE} SpidrExample.cpp)

# add Spidr lib
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ ${CMAKE_CURRENT_BINARY_DIR}/SpidrLib)
target_link_libraries(${SPIDREXAMPLE} PRIVATE SpidrLib)

# On Windows and Linux these come from glad in the HDILib which is linked against in the SpidrLib
# Prevent glfw including the OpenGL headers. It's important to define this AFTER linking against SpidrLib
add_definitions(-DGLFW_INCLUDE_NONE)

# Set C++17 language standard
target_compile_features(${SPIDREXAMPLE} PRIVATE cxx_std_17)
