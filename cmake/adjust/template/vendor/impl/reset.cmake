# reset profile vars, for sanity

foreach(LANG CXX C)
    unset(NBL_${LANG}_COMPILE_OPTIONS)
    unset(NBL_${LANG}_RELEASE_COMPILE_OPTIONS)
    unset(NBL_${LANG}_RELWITHDEBINFO_COMPILE_OPTIONS)
    unset(NBL_${LANG}_DEBUG_COMPILE_OPTIONS)
endforeach()