if (IRR_BUILD_CEGUI)
    add_subdirectory(CEGUI)
    set(IRR_EXT_CEGUI_INCLUDE_DIRS
        ${IRR_EXT_CEGUI_INCLUDE_DIRS}
        PARENT_SCOPE
    )
    set(IRR_EXT_CEGUI_LIB
        ${IRR_EXT_CEGUI_LIB}
        PARENT_SCOPE
    )
endif()
