# init profiles vars by resetting required lists

foreach(LANG CXX C)
    foreach(WHAT COMPILE LINK DEFINITIONS)
        set(NBL_${LANG}_${WHAT}_OPTIONS "")
        foreach(CONFIG RELEASE RELWITHDEBINFO DEBUG)
            set(NBL_${LANG}_${CONFIG}_${WHAT}_OPTIONS "")
        endforeach()
    endforeach()
endforeach()