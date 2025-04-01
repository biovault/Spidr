# -----------------------------------------------------------------------------
# Fetch generic repository
# -----------------------------------------------------------------------------
# usage example: 
#  fetch_content(faiss "https://github.com/facebookresearch/faiss" "v1.9.0")
#  fetch_content_src(faiss "https://github.com/facebookresearch/faiss" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss" "v1.9.0")
# sets faiss_SOURCE_DIR automatically as ${FETCH_REPO_SOURCE}
# FETCH_REPO_TAG can be a tag or a commit hash

function(fetch_content FETCH_REPO_NAME FETCH_REPO_LINK FETCH_REPO_TAG)

    include(FetchContent)

    message(STATUS "Cloning ${FETCH_REPO_NAME} from ${FETCH_REPO_LINK} at tag/commit ${FETCH_REPO_TAG}")

    FetchContent_Declare(
        ${FETCH_REPO_NAME}
        GIT_REPOSITORY ${FETCH_REPO_LINK}
        GIT_TAG ${FETCH_REPO_TAG}
    )

    FetchContent_MakeAvailable(${FETCH_REPO_NAME})

    set(${FETCH_REPO_NAME}_SOURCE_DIR "${${FETCH_REPO_NAME}_SOURCE_DIR}" PARENT_SCOPE)
    message(STATUS "Cloned ${FETCH_REPO_NAME} to ${FETCH_REPO_NAME}_SOURCE_DIR: ${${FETCH_REPO_NAME}_SOURCE_DIR}")

endfunction()

function(fetch_content_url FETCH_REPO_NAME FETCH_URL)

    include(FetchContent)

    message(STATUS "Cloning ${FETCH_REPO_NAME} from URL ${FETCH_URL}")

    FetchContent_Declare(
        ${FETCH_REPO_NAME}
        URL ${FETCH_URL}
    )

    FetchContent_MakeAvailable(${FETCH_REPO_NAME})

    set(${FETCH_REPO_NAME}_SOURCE_DIR "${${FETCH_REPO_NAME}_SOURCE_DIR}" PARENT_SCOPE)
    message(STATUS "Cloned ${FETCH_REPO_NAME} to ${FETCH_REPO_NAME}_SOURCE_DIR: ${${FETCH_REPO_NAME}_SOURCE_DIR}")

endfunction()

function(fetch_content_src FETCH_REPO_NAME FETCH_REPO_LINK FETCH_REPO_SOURCE FETCH_REPO_TAG)

    include(FetchContent)

    message(STATUS "Cloning ${FETCH_REPO_NAME} from ${FETCH_REPO_LINK} at tag/commit ${FETCH_REPO_TAG}")

    FetchContent_Declare(
        ${FETCH_REPO_NAME}
        GIT_REPOSITORY ${FETCH_REPO_LINK}
        GIT_TAG ${FETCH_REPO_TAG}
        SOURCE_DIR ${FETCH_REPO_SOURCE}
    )

    FetchContent_MakeAvailable(${FETCH_REPO_NAME})

    set(${FETCH_REPO_NAME}_SOURCE_DIR "${${FETCH_REPO_NAME}_SOURCE_DIR}" PARENT_SCOPE)
    message(STATUS "Cloned ${FETCH_REPO_NAME} to ${FETCH_REPO_NAME}_SOURCE_DIR: ${${FETCH_REPO_NAME}_SOURCE_DIR}")

endfunction()

function(fetch_content_src_url FETCH_REPO_NAME FETCH_URL FETCH_REPO_SOURCE)

    include(FetchContent)

    message(STATUS "Cloning ${FETCH_REPO_NAME} from URL ${FETCH_URL}")

    FetchContent_Declare(
        ${FETCH_REPO_NAME}
        URL ${FETCH_URL}
        SOURCE_DIR ${FETCH_REPO_SOURCE}
    )

    FetchContent_MakeAvailable(${FETCH_REPO_NAME})

    set(${FETCH_REPO_NAME}_SOURCE_DIR "${${FETCH_REPO_NAME}_SOURCE_DIR}" PARENT_SCOPE)
    message(STATUS "Cloned ${FETCH_REPO_NAME} to ${FETCH_REPO_NAME}_SOURCE_DIR: ${${FETCH_REPO_NAME}_SOURCE_DIR}")

endfunction()
