cmake_minimum_required(VERSION 3.21.0)

option(CREATE_EXAMPLE "Creates an example executable using the Spidr library" ON)
option(CREATE_PYTHON_WRAPPER "Creates a python wrapper of the Spidr library" ON)
option(USE_AVX "Use AVX instructions sets if available" ON)
option(USE_OPENMP "Use OpenMP - by default ON" ON)

# setup project
set(SPIDRLIB "SpidrLib")
PROJECT(${SPIDRLIB} C CXX)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /permissive- /Zc:__cplusplus")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif(MSVC)

# -----------------------------------------------------------------------------
# Library dependencies
# -----------------------------------------------------------------------------

# find external packages
find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED)

if(UNIX AND NOT APPLE)
   find_package(TBB REQUIRED)
endif()

set(HDILib_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILibSlim")
message(STATUS "Build HDILibSlim from submodule at ${HDILib_DIR}")
set(INSTALL_HDILibSlim OFF)
add_subdirectory(${HDILib_DIR})

if(UNIX)
    set_target_properties(hdiutils PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(hdidata PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(hdidimensionalityreduction PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------
# set souce files of this project
set(HEADERS 
	include/FeatureExtraction.h
	include/DistanceCalculation.h
	include/SpidrAnalysis.h
	include/TsneComputation.h
	include/SpidrAnalysisParameters.h
	include/KNNUtils.h
	include/FeatureUtils.h
	include/EvalUtils.h
	include/KNNDists.h
)

set(LIB
	src/FeatureExtraction.cpp
	src/DistanceCalculation.cpp
	src/SpidrAnalysis.cpp
	src/TsneComputation.cpp
	src/SpidrAnalysisParameters.cpp
)

set(UTILS
	src/KNNUtils.cpp
	src/FeatureUtils.cpp
	src/EvalUtils.cpp
)

set(SOURCES ${HEADERS} ${LIB} ${UTILS})

source_group(Headers FILES ${HEADERS})
source_group(Lib FILES ${LIB})
source_group(Utils FILES ${UTILS})

# -----------------------------------------------------------------------------
# CMake Target
# -----------------------------------------------------------------------------
add_library(${SPIDRLIB} STATIC ${SOURCES})

# -----------------------------------------------------------------------------
# Target properties
# -----------------------------------------------------------------------------
# Request C++17
target_compile_features(${SPIDRLIB} PRIVATE cxx_std_17)

if (MSVC)
    target_compile_options(${SPIDRLIB} PRIVATE /bigobj)
endif()

if(UNIX)
    set_target_properties(${SPIDRLIB} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Hardware accelerations: AVX
check_and_link_AVX(${SPIDRLIB} ${USE_AVX})

if(MSVC)
    SET( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:mainCRTStartup" )
endif()

# Use glfw (https://github.com/glfw/glfw) as OpenGL library for t-SNE computation with the HDILib
set(GLFW_BUILD_DOCS OFF CACHE BOOL "GLFW lib only" )
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "disable GLFX examples")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "disable GLFX tests") 
set(GLFW_INSTALL OFF CACHE BOOL "GLFW lib only" )
add_subdirectory("external/glfw")
add_dependencies(${SPIDRLIB} glfw)

# Prevent glfw including the OpenGL headers
# On Windows and Linux these come from glad in the HDILib
target_compile_definitions(${SPIDRLIB} PUBLIC -DGLFW_INCLUDE_NONE)

# -----------------------------------------------------------------------------
# Target include directories
# -----------------------------------------------------------------------------
target_include_directories(${SPIDRLIB} PUBLIC "include")

# Include external header-only libraries
target_include_directories(${SPIDRLIB} PUBLIC "${HDILib_DIR}")
target_include_directories(${SPIDRLIB} PUBLIC "external/spdlog/include")
target_include_directories(${SPIDRLIB} PUBLIC "external/eigen/")
target_include_directories(${SPIDRLIB} PUBLIC "external/hnswlib")
target_include_directories(${SPIDRLIB} PUBLIC "external/glfw/include")

# -----------------------------------------------------------------------------
# Target library linking
# -----------------------------------------------------------------------------
target_link_libraries(${SPIDRLIB} PUBLIC glfw)
target_link_libraries(${SPIDRLIB} PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries(${SPIDRLIB} PUBLIC hdidimensionalityreduction hdiutils hdidata)

if(UNIX AND NOT APPLE)
   target_link_libraries(${SPIDRLIB} PRIVATE TBB::tbb)
endif()

# -----------------------------------------------------------------------------
# Target dependency management
# -----------------------------------------------------------------------------
add_dependencies(${SPIDRLIB} hdidimensionalityreduction hdiutils hdidata)

# -----------------------------------------------------------------------------
# Other targets
# -----------------------------------------------------------------------------
# Example project
if(CREATE_EXAMPLE)
	message(STATUS "Creating example executable")
	set(SPIDREXAMPLE "SpidrExample")
	add_executable(${SPIDREXAMPLE} example/SpidrExample.cpp)
	target_link_libraries(${SPIDREXAMPLE} PUBLIC SpidrLib)
	target_compile_features(${SPIDREXAMPLE} PUBLIC cxx_std_17)
	target_compile_definitions(${SPIDREXAMPLE} PUBLIC -DGLFW_INCLUDE_NONE)
endif()

if(CREATE_PYTHON_WRAPPER)
	message(STATUS "Creating python wrapper")
	set(SPIDRWRAPPER "spidr")
	
	add_subdirectory(python_wrapper/pybind11)
	
	pybind11_add_module(${SPIDRWRAPPER}
		python_wrapper/src/SpidrBind.cpp
		python_wrapper/src/SpidrWrapper.h
		python_wrapper/src/SpidrWrapper.cpp)
	
	target_link_libraries(${SPIDRWRAPPER} PUBLIC SpidrLib)
	target_compile_features(${SPIDRWRAPPER} PUBLIC cxx_std_17)
	target_compile_definitions(${SPIDRWRAPPER} PUBLIC -DGLFW_INCLUDE_NONE)
	
	# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
	# define (VERSION_INFO) here.
	target_compile_definitions(${SPIDRWRAPPER} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
endif()
