list(TRANSFORM @_NBL_PROXY_@ PREPEND "${CMAKE_CURRENT_LIST_DIR}/")

set(@_NBL_COMPOMENT_D_@ "${CMAKE_CURRENT_LIST_DIR}/@_NBL_COMPOMENT_D_V_@")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(@_NBL_PACKAGE_@ DEFAULT_MSG @_NBL_PROXY_@ @_NBL_COMPOMENT_D_@)