cmake_minimum_required(VERSION 3.17)

set(SPIDRWRAPPER "spidr")
project(${SPIDRWRAPPER})

# works with newer pip versions or option --use-feature=in-tree-build for older
set(SPIDRLIB_DIR ${PROJECT_SOURCE_DIR}/..)
message(${SPIDRLIB_DIR})

# add python wrapper lib
add_subdirectory(pybind11)
pybind11_add_module(${SPIDRWRAPPER}
	src/SpidrBind.cpp
	src/SpidrWrapper.h
	src/SpidrWrapper.cpp)

# add Spidr lib
add_subdirectory(${SPIDRLIB_DIR} ./SpidrLib)
target_link_libraries(${SPIDRWRAPPER} PUBLIC SpidrLib)

# On Windows and Linux these come from glad in the HDILib which is linked against in the SpidrLib
# Prevent glfw including the OpenGL headers. It's important to define this AFTER linking against SpidrLib
add_definitions(-DGLFW_INCLUDE_NONE)

# Set C++17 language standard
target_compile_features(${SPIDRWRAPPER} PUBLIC cxx_std_17)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(${SPIDRWRAPPER} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})