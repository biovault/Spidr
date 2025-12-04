cmake_minimum_required(VERSION 3.24)

option(CREATE_EXAMPLE "Creates an example executable using the Spidr library" ON)
option(CREATE_PYTHON_WRAPPER "Creates a python wrapper of the Spidr library" ON)
option(SPIDR_USE_AVX "Use AVX instructions sets if available" ON)

# setup project
set(SPIDRLIB "SpidrLib")
PROJECT(${SPIDRLIB} C CXX)

# -----------------------------------------------------------------------------
# CMake Options
# -----------------------------------------------------------------------------

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /permissive- /Zc:__cplusplus")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MD")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif(MSVC)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(CMakeCheckSetAVXSpidr)
include(CMakeFetchUtils)
include(get_cpm)

# -----------------------------------------------------------------------------
# Library dependencies
# -----------------------------------------------------------------------------

# find external packages
find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED)

if(UNIX AND NOT APPLE)
   find_package(TBB REQUIRED)
endif()

find_package(Eigen3 CONFIG)
if(NOT eigen_SOURCE_DIR AND NOT TARGET Eigen3::Eigen AND NOT Eigen3_FOUND)
    set(BUILD_TESTING OFF CACHE BOOL "Enable testing for Eigen" FORCE)
    set(EIGEN_BUILD_TESTING  OFF CACHE BOOL "Enable creation of Eigen tests." FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "Enable creation of Eigen documentation" FORCE)
    set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "Toggles the building of the Eigen demos" FORCE)
    fetch_content_url(eigen "https://gitlab.com/libeigen/eigen/-/archive/5.0.1.tar.gz")
else()
    message(STATUS "Spidr: Use external Eigen")
endif()

if(TARGET hdislimdimensionalityreduction AND TARGET hdislimutils AND TARGET hdislimdata)
    if(NOT HDILibSlim_LINK_LIBS)
        set(HDILibSlim_LINK_LIBS hdislimdimensionalityreduction hdislimutils hdislimdata)
    endif()
    
    if(NOT HDILibSlim_INCLUDE_DIR AND HDILibSlim_SOURCE_DIR)
        set(HDILibSlim_INCLUDE_DIR ${HDILibSlim_SOURCE_DIR})
    endif()
else()
    set(HDILibSlim_INSTALL OFF)
    set(HDILibSlim_ENABLE_AVX ${SPIDR_USE_AVX})
    set(HDILibSlim_ENABLE_PID ON)
    set(HDILibSlim_OPTIMIZATION_LEVEL 2)

    CPMAddPackage(
     NAME              HDILibSlim
     GIT_REPOSITORY    https://github.com/alxvth/HDILibSlim.git
     GIT_TAG           8eb510528dc4169079fa97cb324c13b54d060ec4
     EXCLUDE_FROM_ALL  YES
     OPTIONS "HDILibSlim_INSTALL ${HDILibSlim_INSTALL}"
             "HDILibSlim_ENABLE_AVX ${HDILibSlim_ENABLE_AVX}"
             "HDILibSlim_ENABLE_PID ${HDILibSlim_ENABLE_PID}"
             "HDILibSlim_OPTIMIZATION_LEVEL ${HDILibSlim_OPTIMIZATION_LEVEL}"
    )
    
    set(HDILibSlim_INCLUDE_DIR ${HDILibSlim_SOURCE_DIR})
    set(HDILibSlim_LINK_LIBS hdislimdimensionalityreduction hdislimutils hdislimdata)
    
    message(STATUS "HDILibSlim_SOURCE_DIR: ${HDILibSlim_SOURCE_DIR}")

    set_target_properties(hdislimdata hdislimdimensionalityreduction hdislimutils
        PROPERTIES 
        FOLDER HDILibSlim
    )
endif()

find_package(glfw3 CONFIG)

if(glfw3_FOUND)
    message(STATUS "glfw3_FOUND")
endif()

if(NOT glfw3_FOUND)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "GLFW lib only" )
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "disable GLFX examples")
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "disable GLFX tests") 
    set(GLFW_INSTALL OFF CACHE BOOL "GLFW lib only" )
    set(SPIDR_LOCAL_GLFW3 ON)
    message(STATUS "Use local glfw copy")
    add_subdirectory("external/glfw")
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
spidr_check_and_set_AVX(${SPIDRLIB} ${SPIDR_USE_AVX})

# -----------------------------------------------------------------------------
# Target include directories
# -----------------------------------------------------------------------------
target_include_directories(${SPIDRLIB} PUBLIC "include")

# Include external header-only libraries
target_include_directories(${SPIDRLIB} PUBLIC "${HDILibSlim_SOURCE_DIR}")
target_include_directories(${SPIDRLIB} PUBLIC "external/spdlog/include")
target_include_directories(${SPIDRLIB} PUBLIC "external/hnswlib")

if(SPIDR_LOCAL_GLFW3)
    target_include_directories(${SPIDRLIB} PUBLIC "external/glfw/include")
endif()

# -----------------------------------------------------------------------------
# Target library linking
# -----------------------------------------------------------------------------
target_link_libraries(${SPIDRLIB} PUBLIC glfw)
target_link_libraries(${SPIDRLIB} PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(${SPIDRLIB} PUBLIC ${OPENGL_LIBRARIES})
target_link_libraries(${SPIDRLIB} PUBLIC ${HDILibSlim_LINK_LIBS})
target_link_libraries(${SPIDRLIB} PUBLIC Eigen3::Eigen)

if(UNIX AND NOT APPLE)
   target_link_libraries(${SPIDRLIB} PRIVATE TBB::tbb)
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
