function(get_major_version full_version major_version)
    if (WIN32)
        set(MSVC_80_VERSION 8)
        set(MSVC_90_VERSION 9)
        set(MSVC_100_VERSION 10)
        set(MSVC_110_VERSION 11)
        set(MSVC_120_VERSION 12)
        set(MSVC_140_VERSION 14)
        set(MSVC_141_VERSION 15)
        set(MSVC_142_VERSION 16)
        set(majver "${MSVC_${MSVC_TOOLSET_VERSION}_VERSION}")
    else()
        string(REPLACE "." ";" verlist ${full_version})
        list(GET verlist 0 majver)
    endif()
    set(compiler_version ${majver} PARENT_SCOPE)
endfunction()


macro(get_settings)

    set(compiler_name UNSUPPORTED)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(compiler_name gcc)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
        set(compiler_name apple-clang)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        set(compiler_name "Visual Studio")
    endif()

    set(compiler_version UNSUPPORTED)
    get_major_version("${CMAKE_CXX_COMPILER_VERSION}" "${compiler_version}")
    #message("Using compiler major version: -${compiler_version}-")

    set(os_name UNSUPPORTED)
    if (APPLE)
        set(os_name Macos)
    elseif (UNIX AND NOT APPLE)
        set(os_name Linux)
    elseif (WIN32)
        set(os_name Windows)
    endif()


endmacro(get_settings)

function(find_artifactory_package)
    file(REMOVE aql_out.txt)
    set(CURL_COMMAND)
    list(APPEND CURL_COMMAND curl -k -u conan-user:XQlM?4KxtCPOp@0t -X POST -H "content-type: text/plain" --data @${CMAKE_BINARY_DIR}/aql.json https://lkeb-artifactory.lumc.nl:443/artifactory/api/search/aql)
    execute_process(COMMAND ${CURL_COMMAND} RESULT_VARIABLE results OUTPUT_FILE ${PROJECT_SOURCE_DIR}/aql_out.txt)

    set(path_match "\"path\"")
    file(STRINGS ${PROJECT_SOURCE_DIR}/aql_out.txt result_file REGEX "${path_match}")
    #message("result_file ${result_file}")
    list(LENGTH result_file res_length)

    message(STATUS "result length ${res_length}")
    foreach(line ${result_file})
        message(STATUS "Artifactory package found: ${line}\n")
    endforeach()

    if (res_length LESS 1)
        message(FATAL_ERROR "No matching artifactory packages found. Please check the query.")
    endif()

    if (res_length GREATER 1)
        message(FATAL_ERROR "Too many matching artifactory packages found. Contact the HDPS group for more info and supply the aql_out.txt and aql.json files")
    endif()

    list(GET result_file 0 path_line)
    #message("Retrieved path ${path_line}")
    string(REGEX MATCH "[^ \"]*/0" package_id "${path_line}")
    #message("package id ${package_id}")
    set(package_url "https://lkeb-artifactory.lumc.nl/artifactory/conan/${package_id}/conan_package.tgz" PARENT_SCOPE)
endfunction()

# Artifactory requires the following information to find a package
# The file conaninfo.txt is used for the search as it is saved
# with the conan properties visible to the AQL query
#
# conan.settings.compiler.version windows:15 linux:9 clang:10
# conan.settings.compiler  - gcc, Visual Studio, apple-clang
# Not used for HDILib conan.settings.build_type Release Debug
# conan.settings.os Linux, Windows, Macos
# conan.package.version latest or 0.1.0 or 0.2.0, or 0.3.0
# conan.package.name
# conan.package.user     lkeb
# conan.package.channel  stable
# filename conaninfo.txt
#

macro(get_artifactory_package
        package_name package_version package_builder
        compiler_name compiler_version os_name is_combined_package)
    if(is_combined_package)
        # retrieve the single combined package from lkeb-artifactory
        file(REMOVE ${CMAKE_BINARY_DIR}/aql.json)
        configure_file(${PROJECT_SOURCE_DIR}/cmake/aql_multi.json.in ${CMAKE_BINARY_DIR}/aql.json)
        find_artifactory_package()
        message(STATUS "package url ${package_url} - name ${package_name}")
        file(DOWNLOAD ${package_url} "${PROJECT_SOURCE_DIR}/${package_name}.tgz")
        execute_process(COMMAND cmake -E make_directory "${LIBRARY_INSTALL_DIR}/${package_name}")
        execute_process(COMMAND cmake -E tar xvf "${PROJECT_SOURCE_DIR}/${package_name}.tgz" WORKING_DIRECTORY "${LIBRARY_INSTALL_DIR}/${package_name}")
    else()
        # retrieve the separate Release and Debug packages  from lkeb-artifactory
        set(option_shared "False") # hardcoded - TODO make parameter
        file(REMOVE ${CMAKE_BINARY_DIR}/aql.json)
        if (os_name STREQUAL "Windows")
            set(build_runtime_type "\"@conan.settings.compiler.runtime\":{\"$eq\" : \"MD\"}")
        else()
            set(build_runtime_type "\"@conan.settings.build_type\":{\"$eq\" : \"Release\"}")
        endif()
        configure_file(${PROJECT_SOURCE_DIR}/cmake/aql_single.json.in ${CMAKE_BINARY_DIR}/aql.json)
        set(AQL_FILE "")
        #file(READ ${CMAKE_BINARY_DIR}/aql.json AQL_FILE)
        #message(status "AQL_FILE: ${AQL_FILE}")
        find_artifactory_package()
        message(STATUS "package url ${package_url} - name ${package_name}")
        file(DOWNLOAD ${package_url} "${PROJECT_SOURCE_DIR}/${package_name}_Release.tgz")
        execute_process(COMMAND cmake -E make_directory "${LIBRARY_INSTALL_DIR}/${package_name}/Release")
        execute_process(COMMAND cmake -E tar xvf "${PROJECT_SOURCE_DIR}/${package_name}_Release.tgz" WORKING_DIRECTORY "${LIBRARY_INSTALL_DIR}/${package_name}/Release")

        file(REMOVE ${CMAKE_BINARY_DIR}/aql.json)
        if (os_name STREQUAL "Windows")
            set(build_runtime_type "\"@conan.settings.compiler.runtime\":{\"$eq\" : \"MDd\"}")
        else()
            set(build_runtime_type "\"@conan.settings.build_type\":{\"$eq\" : \"Debug\"}")
        endif()
        configure_file(${PROJECT_SOURCE_DIR}/cmake/aql_single.json.in ${CMAKE_BINARY_DIR}/aql.json)
        find_artifactory_package()
        message(STATUS "package url ${package_url} - name ${package_name}")
        file(DOWNLOAD ${package_url} "${PROJECT_SOURCE_DIR}/${package_name}_Debug.tgz")
        execute_process(COMMAND cmake -E make_directory "${LIBRARY_INSTALL_DIR}/${package_name}/Debug")
        execute_process(COMMAND cmake -E tar xvf "${PROJECT_SOURCE_DIR}/${package_name}_Debug.tgz" WORKING_DIRECTORY "${LIBRARY_INSTALL_DIR}/${package_name}/Debug")
    endif()
endmacro()

# Install a package version from the artifactory.
# The specific version will be based on the current OS, compiler
# and build settings
# package_name : a package in the lkeb-artifactory - e.g. HDILib or flann
# package_version: the version to be installed
# package_builder: the artifactory submitter (usually biovault or lkeb)
# is_combined_package: either TRUE or FALSE:
#           TRUE - the package contains both debug and release
#           FALSE - the package contains either debug or release 

function(install_artifactory_package package_name package_version package_builder is_combined_package)
    cmake_parse_arguments(PARSE_ARGV 0 IAPARG 
        "COMBINED_PACKAGE"
        "PACKAGE_NAME;PACKAGE_VERSION;PACKAGE_BUILDER;"
        "" ${ARGN})

    message(STATUS "Installing package * ${package_name} * from lkeb-artifactory.lumc.nl")
    get_settings()
    set(package_name ${package_name})
    set(package_version ${package_version})
    set(package_builder ${package_builder})
    set(is_combined_package ${is_combined_package})

    get_artifactory_package("${package_name}" "${package_version}" "${package_builder}"
        "${compiler_name}" "${compiler_version}"
        "${os_name}" "${is_combined_package}")
    # add the HDILib to the module path
    set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/${package_name}" ${CMAKE_MODULE_PATH} PARENT_SCOPE)
endfunction()