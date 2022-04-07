cmake_minimum_required(VERSION 3.17)

set(SPIDRLIB "SpidrLib")
PROJECT(${SPIDRLIB})

# Set the directory where CMake include files are to be found
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(InstallArtifactoryPackage)
include(CMakeHDILib)

# -----------------------------------------------------------------------------
# Libraries
# -----------------------------------------------------------------------------
set(HDILib_VERSION latest)
set(flann_VERSION 1.8.5)
set(lz4_VERSION 1.9.3)
if(APPLE)
    set(flann_VERSION 1.9.1)
endif()

# find external packages
find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED)

# install external packages: HDILib
# by default, a pre-build library is downloaded
if(NOT USE_ARTIFACTORY_LIBS)
    set(USE_ARTIFACTORY_LIBS TRUE CACHE BOOL "Use the prebuilt libraries from artifactory")
endif()

set(LIBRARY_INSTALL_DIR ${PROJECT_BINARY_DIR})

if (USE_ARTIFACTORY_LIBS AND NOT ARTIFACTORY_LIBS_INSTALLED)
    # Both HDILib and flann are available prebuilt in the lkeb-artifactory as combined Debug/Release packages
    # lz4 is also available in the lkb-artifactory in separate Debug and |Release packages
    install_artifactory_package(HDILib ${HDILib_VERSION} biovault TRUE) 
    install_artifactory_package(flann ${flann_VERSION} lkeb TRUE)
    install_artifactory_package(lz4 ${lz4_VERSION} _ FALSE)

    message(STATUS "module path ${CMAKE_MODULE_PATH}")
    message(STATUS "Call find_package HDILib search ${CMAKE_SOURCE_DIR}/HDILib/lib/cmake/HDILib")
    message(STATUS "Run find_package HDILib")

    set(ARTIFACTORY_LIBS_INSTALLED TRUE CACHE BOOL "Use the prebuilt libraries from artifactory" FORCE)
endif()

if(USE_ARTIFACTORY_LIBS AND ARTIFACTORY_LIBS_INSTALLED)
	message(STATUS "Using PREBUILT external libraries")	
    # If the HDILib_VERSION is not a x.y.z version number 
    # omit the version argument from find package
    find_package(HDILib COMPONENTS hdidimensionalityreduction hdiutils hdidata PATHS "${LIBRARY_INSTALL_DIR}/HDILib" CONFIG REQUIRED)
else()
	message(STATUS "Using NON prebuilt external libraries")	
	if(NOT DEFINED HDILIB_ROOT OR NOT DEFINED FLANN_ROOT OR NOT DEFINED LZ4_ROOT)
		message(FATAL_ERROR "Define HDILIB_ROOT, FLANN_ROOT, LZ4_ROOT if you are not using pre-build external libraries")	
	endif()
	
	find_package(HDILib COMPONENTS hdiutils hdidata hdidimensionalityreduction PATHS ${HDILIB_ROOT} CONFIG REQUIRED)
endif()

if(HDILib_FOUND)
	message(STATUS "HDILib found")	
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
add_subdirectory("external/glfw")

# Prevent glfw including the OpenGL headers
# On Windows and Linux these come from glad in the HDILib
add_definitions(-DGLFW_INCLUDE_NONE)

# -----------------------------------------------------------------------------
# Target include directories
# -----------------------------------------------------------------------------
target_include_directories(${SPIDRLIB} PUBLIC "include")

target_include_directories(${SPIDRLIB} PUBLIC "${CMAKE_SOURCE_DIR}/external/glfw/include")

# Include external header-only libraries
target_include_directories(${SPIDRLIB} PUBLIC "external/spdlog/include")
target_include_directories(${SPIDRLIB} PUBLIC "external/hnswlib")
target_include_directories(${SPIDRLIB} PUBLIC "external/eigen/")

#target_include_directories(${SPIDRLIB} PUBLIC ${HDILib_INCLUDE_DIR})

set_HDILib_project_includes(${SPIDRLIB} PUBLIC)
set_flann_project_includes(${SPIDRLIB} PUBLIC)
set_lz4_project_includes(${SPIDRLIB} PUBLIC)

# -----------------------------------------------------------------------------
# Target library linking
# -----------------------------------------------------------------------------
target_link_libraries(${SPIDRLIB} general glfw)
target_link_libraries(${SPIDRLIB} general OpenMP::OpenMP_CXX)

# you need to link against flann::flann even though it is not used in SPIDRLIB since otherwise the HDILib will be confused
set_flann_project_link_libraries(${SPIDRLIB}) 
set_HDILib_project_link_libraries(${SPIDRLIB})
set_lz4_project_link_libraries(${SPIDRLIB}) 
