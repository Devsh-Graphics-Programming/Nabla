# A configure-time and build-time script, that produces source with current git commit info
# required variables: GIT_EXECUTABLE, WORKING_DIRECTORY, INPUT_FILE, OUTPUT_FILE

execute_process( COMMAND ${GIT_EXECUTABLE} log -1 --format=%H
    WORKING_DIRECTORY ${WORKING_DIRECTORY}
    OUTPUT_VARIABLE NBL_GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process( COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
    WORKING_DIRECTORY ${WORKING_DIRECTORY}
    OUTPUT_VARIABLE NBL_GIT_COMMIT_SHORT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# since CMake 3.10.3: "The generated file is modified and its timestamp updated on subsequent cmake runs only if its content is changed."
# we can spam this each run/build
configure_file("${INPUT_FILE}" "${OUTPUT_FILE}")
