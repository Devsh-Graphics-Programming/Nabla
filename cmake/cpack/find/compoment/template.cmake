list(TRANSFORM @_NBL_PROXY_@ PREPEND "${CMAKE_CURRENT_LIST_DIR}/")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(@_NBL_PACKAGE_@ DEFAULT_MSG @_NBL_PROXY_@)