cmake_minimum_required(VERSION 3.21.0)

option(BUILD_HDILIB "Build the HDILib from submodule - otherwise provide the variable HDILIB_ROOT" ON)

# setup vcpkg cmake toolchain
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CRT_LINKAGE dynamic)
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/external/vcpkg/scripts/buildsystems/vcpkg.cmake"
  CACHE STRING "Vcpkg toolchain file")

# setup project
set(SPIDRLIB "SpidrLib")
PROJECT(${SPIDRLIB} C CXX)

# -----------------------------------------------------------------------------
# Libraries
# -----------------------------------------------------------------------------

# find external packages
find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 3.4 CONFIG REQUIRED NO_MODULE)
find_package(lz4 CONFIG REQUIRED)
find_package(flann CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(spdlog 1.9.2 CONFIG REQUIRED)

# Set include path, which vcpkg does not automatically provide (only *_FOUND and *_DIR)
#find_path(flann_INCLUDE_DIR NAMES flann.hpp PATH_SUFFIXES flann)
set(flann_INCLUDE_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include")

if(BUILD_HDILIB)
	message(STATUS "Build HDILib from submodule")	
	set(INSTALL_PREBUILT_DEPENDENCIES ON)
	
	# use the HDILib submodule
	set(HDILib_Sub "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib")
	add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib" ${CMAKE_CURRENT_BINARY_DIR}/HDILib)
	
	# manage some HDILib things for hdidimensionalityreduction hdiutils hdidata
	target_include_directories(hdiutils PUBLIC $<BUILD_INTERFACE:${HDILib_Sub}>  
    $<INSTALL_INTERFACE:>)
	
	target_include_directories(hdidata PUBLIC $<BUILD_INTERFACE:${HDILib_Sub}>  
    $<INSTALL_INTERFACE:>)

	target_include_directories(hdidimensionalityreduction PUBLIC $<BUILD_INTERFACE:${HDILib_Sub}>  
    $<INSTALL_INTERFACE:>)

	set(HDILib_FOUND TRUE)

else()
	message(STATUS "Using prebuilt external HDILib")	
	if(NOT DEFINED HDILIB_ROOT OR NOT DEFINED FLANN_ROOT OR NOT DEFINED LZ4_ROOT)
		message(FATAL_ERROR "Define HDILIB_ROOT, FLANN_ROOT, LZ4_ROOT if you are not using pre-build external libraries")	
	endif()
	
	find_package(HDILib COMPONENTS hdiutils hdidata hdidimensionalityreduction PATHS ${HDILIB_ROOT} CONFIG REQUIRED)
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
    target_compile_options(${SPIDRLIB} PUBLIC /bigobj)
endif()

# Hardware accelations: SSE and AVX
set(USE_AVX OFF CACHE BOOL "Don't use AVX by default to support old hardware" )
if(${USE_AVX})
	add_definitions(-DUSE_AVX)
	if( MSVC )
		ADD_DEFINITIONS(/arch:AVX)
	endif()
	message(STATUS "Using AVX instruction set extensions")
endif()

add_definitions(-DUSE_SSE)

if( MSVC )
    SET( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:mainCRTStartup" )
endif()

# Use glfw (https://github.com/glfw/glfw) as OpenGL library for t-SNE computation with the HDILib
set(GLFW_BUILD_DOCS OFF CACHE BOOL "GLFW lib only" )
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "disable GLFX examples")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "disable GLFX tests") 
set(GLFW_INSTALL OFF CACHE BOOL "GLFW lib only" )

# Prevent glfw including the OpenGL headers
# On Windows and Linux these come from glad in the HDILib
add_definitions(-DGLFW_INCLUDE_NONE)

# -----------------------------------------------------------------------------
# Target include directories
# -----------------------------------------------------------------------------
target_include_directories(${SPIDRLIB} PUBLIC "include")

# Include external header-only libraries
target_include_directories(${SPIDRLIB} PUBLIC "external/hnswlib")

# -----------------------------------------------------------------------------
# Target library linking
# -----------------------------------------------------------------------------
target_link_libraries(${SPIDRLIB} PUBLIC glfw)
target_link_libraries(${SPIDRLIB} PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(${SPIDRLIB} PUBLIC Eigen3::Eigen)
target_link_libraries(${SPIDRLIB} PUBLIC spdlog::spdlog_header_only)
target_link_libraries(${SPIDRLIB} PUBLIC lz4::lz4)
target_link_libraries(${SPIDRLIB} PUBLIC flann::flann_cpp)
target_link_libraries(${SPIDRLIB} PUBLIC hdidimensionalityreduction hdiutils hdidata ${CMAKE_DL_LIBS})

# -----------------------------------------------------------------------------
# Target dependency management
# -----------------------------------------------------------------------------
if(BUILD_HDILIB)
	add_dependencies(${SPIDRLIB} hdidimensionalityreduction hdiutils hdidata)
endif()
