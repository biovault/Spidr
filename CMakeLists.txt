cmake_minimum_required(VERSION 3.17)

set(SPIDRLIB "SpidrLib")

PROJECT(${SPIDRLIB})

# find external packages
find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED)
if(DEFINED ENV{HDILIBSLIM_ROOT})
    set(HDILIBSLIM_PATH $ENV{HDILIBSLIM_ROOT} CACHE PATH  "Path to HDILib_slim")
endif()

find_package(HDILibSlim COMPONENTS hdiutils hdidata hdidimensionalityreduction PATHS ${HDILIBSLIM_PATH} CONFIG REQUIRED)

if(HDILibSlim_FOUND)
	message(STATUS "HDILibSlim found at ${HDILIBSLIM_PATH} with includes at ${HDILibSlim_INCLUDE_DIR}")	# make sure to manually set HDILIBSLIM_ROOT to the install dir of the HDILibSlim (fork of the HDILib with fewer dependencies)
endif()

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

add_library(${SPIDRLIB} STATIC ${SOURCES})

# Request C++17
target_compile_features(${SPIDRLIB} PRIVATE cxx_std_17)

if (MSVC)
    target_compile_options(${SPIDRLIB} PUBLIC /bigobj)
endif()

target_include_directories(${SPIDRLIB} PUBLIC "include")

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

target_include_directories(${SPIDRLIB} PUBLIC "${CMAKE_SOURCE_DIR}/external/glfw/include")
target_link_libraries(${SPIDRLIB} PUBLIC glfw)

# Link OpenMP for easy parallelization
target_link_libraries(${SPIDRLIB} PUBLIC OpenMP::OpenMP_CXX)

# Include external header-only libraries
target_include_directories(${SPIDRLIB} PUBLIC "external/spdlog/include")
target_include_directories(${SPIDRLIB} PUBLIC "external/hnswlib")
target_include_directories(${SPIDRLIB} PUBLIC "external/eigen/")

# Link against the HDILibSlim, https://github.com/alxvth/HDILibSlim/ (a slimmed-down version of https://github.com/biovault/HDILib)
# make sure to have it built and HDILIBSlim_ROOT set
target_include_directories(${SPIDRLIB} PUBLIC ${HDILibSlim_INCLUDE_DIR})
target_link_libraries(${SPIDRLIB} PUBLIC HDISlim::hdidimensionalityreduction HDISlim::hdiutils HDISlim::hdidata OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS}) # no need to link against flann::flann since it is not used in SPIDRLIB
