cmake_minimum_required(VERSION 3.21.0)

option(BUILD_HDILIB "Build the HDILib from submodule - otherwise provide the variable HDILIB_ROOT" ON)
option(BUILD_HDILIBSLIM "Build the HDILibSlim from submodule - otherwise provide the variable HDILIBSLIM_ROOT" ON)
option(USE_HDILIBSLIM "Use a pre-build slim HDILib version, provide the variable HDILIBSLIM_ROOT" OFF)
option(CREATE_EXAMPLE "Creates an example executable using the Spidr library" ON)
option(CREATE_PYTHON_WRAPPER "Creates a python wrapper of the Spidr library" ON)
option(USE_AVX "Use AVX instructions sets if available" ON)

if(NOT (BUILD_HDILIBSLIM OR USE_HDILIBSLIM))
	# setup vcpkg cmake toolchain
	set(VCPKG_LIBRARY_LINKAGE static)
	set(VCPKG_CRT_LINKAGE dynamic)
	set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/external/vcpkg/scripts/buildsystems/vcpkg.cmake"
	  CACHE STRING "Vcpkg toolchain file")
endif()

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

if(NOT (BUILD_HDILIBSLIM OR USE_HDILIBSLIM))
	find_package(lz4 CONFIG REQUIRED)
	find_package(flann CONFIG REQUIRED)

	# Set include path, which vcpkg does not automatically provide (only *_FOUND and *_DIR)
	#find_path(flann_INCLUDE_DIR NAMES flann.hpp PATH_SUFFIXES flann)
	set(flann_INCLUDE_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include")
endif()

if(BUILD_HDILIB)
	message(STATUS "Build HDILib from submodule")
	
	# use the HDILib submodule
	set(INSTALL_PREBUILT_DEPENDENCIES ON)
	set(HDILib_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib")
	add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib" ${CMAKE_CURRENT_BINARY_DIR}/HDILib)
	
	# manage some HDILib things for hdidimensionalityreduction hdiutils hdidata
	target_include_directories(hdiutils PUBLIC $<BUILD_INTERFACE:${HDILib_INCLUDE_DIR}>  
    $<INSTALL_INTERFACE:>)
	target_include_directories(hdidata PUBLIC $<BUILD_INTERFACE:${HDILib_INCLUDE_DIR}>  
    $<INSTALL_INTERFACE:>)
	target_include_directories(hdidimensionalityreduction PUBLIC $<BUILD_INTERFACE:${HDILib_INCLUDE_DIR}>  
    $<INSTALL_INTERFACE:>)
    
    target_link_libraries(hdidimensionalityreduction PRIVATE lz4::lz4)
    if(UNIX)
        set_target_properties(hdiutils PROPERTIES POSITION_INDEPENDENT_CODE ON)
        set_target_properties(hdidata PROPERTIES POSITION_INDEPENDENT_CODE ON)
        set_target_properties(hdidimensionalityreduction PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
elseif(BUILD_HDILIBSLIM)
	message(STATUS "Build HDILibSlim from submodule")

	set(HDILib_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib")
	add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external/HDILibSlim" ${CMAKE_CURRENT_BINARY_DIR}/HDILibSlim)
	
    if(UNIX)
        set_target_properties(hdiutils PROPERTIES POSITION_INDEPENDENT_CODE ON)
        set_target_properties(hdidata PROPERTIES POSITION_INDEPENDENT_CODE ON)
        set_target_properties(hdidimensionalityreduction PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
else()
	if(USE_HDILIBSLIM)
		message(STATUS "Using prebuilt external HDILibSlim, https://github.com/alxvth/HDILibSlim")	
		if(NOT DEFINED HDILIBSLIM_ROOT)
			message(FATAL_ERROR "Define HDILIBSLIM_ROOT")	
		endif()
		find_package(HDILibSlim COMPONENTS hdiutils hdidata hdidimensionalityreduction PATHS ${HDILIBSLIM_ROOT} CONFIG REQUIRED)
		
		set(HDILib_INCLUDE_DIR ${HDILibSlim_INCLUDE_DIR})
		set(HDI_PREFIX HDISlim)
		
	else()
		message(STATUS "Using prebuilt external HDILib")	
		if(NOT DEFINED HDILIB_ROOT OR NOT DEFINED FLANN_ROOT OR NOT DEFINED LZ4_ROOT)
			message(FATAL_ERROR "Define HDILIB_ROOT, FLANN_ROOT, LZ4_ROOT if you are using pre-build external libraries")	
		endif()
		
		find_package(HDILib COMPONENTS hdiutils hdidata hdidimensionalityreduction PATHS ${HDILIB_ROOT} CONFIG REQUIRED)
		
		set(HDILib_INCLUDE_DIR ${HDILib_INCLUDE_DIR})
		set(HDI_PREFIX HDI)
	endif()
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

# Hardware accelations: SSE and AVX
set(AXV_CompileOption $<IF:$<CXX_COMPILER_ID:MSVC>,/arch:AVX,-DUSE_AVX>)
set(AXV2_CompileOption $<IF:$<CXX_COMPILER_ID:MSVC>,/arch:AVX2,-DUSE_AVX2>)
set(SSE2_CompileOption $<IF:$<CXX_COMPILER_ID:MSVC>,/arch:SSE2,-DUSE_SSE2>)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(${AXV_CompileOption} COMPILER_OPT_AVX_SUPPORTED)
check_cxx_compiler_flag(${AXV2_CompileOption} COMPILER_OPT_AVX2_SUPPORTED)

if(${USE_AVX} AND ${COMPILER_OPT_AVX2_SUPPORTED})
	MESSAGE( STATUS "Use AXV2")
	target_compile_options(${SPIDRLIB} PRIVATE ${AXV2_CompileOption})
elseif(${USE_AVX} AND ${COMPILER_OPT_AVX_SUPPORTED})
	MESSAGE( STATUS "Use AXV")
	target_compile_options(${SPIDRLIB} PRIVATE ${AXV_CompileOption})
else()
	MESSAGE( STATUS "Use SSE2")
	target_compile_options(${SPIDRLIB} PRIVATE ${SSE2_CompileOption})
endif()

if( MSVC )
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
target_include_directories(${SPIDRLIB} PUBLIC "external/hnswlib")
target_include_directories(${SPIDRLIB} PUBLIC "external/eigen/")
target_include_directories(${SPIDRLIB} PUBLIC "external/spdlog/include")
target_include_directories(${SPIDRLIB} PUBLIC "external/glfw/include")
target_include_directories(${SPIDRLIB} PUBLIC ${HDILib_INCLUDE_DIR})

# -----------------------------------------------------------------------------
# Target library linking
# -----------------------------------------------------------------------------
target_link_libraries(${SPIDRLIB} PRIVATE glfw)

if(OpenMP_CXX_FOUND)
	target_link_libraries(${SPIDRLIB} PUBLIC OpenMP::OpenMP_CXX)
endif()

if(BUILD_HDILIB OR BUILD_HDILIBSLIM)
	target_link_libraries(${SPIDRLIB} PUBLIC hdidimensionalityreduction hdiutils hdidata ${CMAKE_DL_LIBS})
else()
	target_link_libraries(${SPIDRLIB} PUBLIC ${HDI_PREFIX}::hdidimensionalityreduction ${HDI_PREFIX}::hdiutils ${HDI_PREFIX}::hdidata ${CMAKE_DL_LIBS})
endif()

if(NOT (BUILD_HDILIBSLIM OR USE_HDILIBSLIM))
	target_link_libraries(${SPIDRLIB} PUBLIC lz4::lz4)
	target_link_libraries(${SPIDRLIB} PUBLIC flann::flann_cpp)
endif()

# -----------------------------------------------------------------------------
# Target dependency management
# -----------------------------------------------------------------------------
if(BUILD_HDILIB)
	add_dependencies(${SPIDRLIB} hdidimensionalityreduction hdiutils hdidata)
endif()

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


