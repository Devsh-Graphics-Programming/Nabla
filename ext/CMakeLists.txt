# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

if (NBL_BUILD_CEGUI)
    set(_OLD_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
    set(CMAKE_PREFIX_PATH ${CEGUI_DEPS_INSTALL_PATH})
    #message(FATAL_ERROR "${CEGUI_DEPS_INSTALL_PATH}") # CEGUI_DEPS_INSTALL_PATH is empty forward it around somehow
    add_subdirectory(CEGUI)
    set(NBL_EXT_CEGUI_INCLUDE_DIRS
        ${NBL_EXT_CEGUI_INCLUDE_DIRS}
        PARENT_SCOPE
    )
    set(NBL_EXT_CEGUI_LIB
        ${NBL_EXT_CEGUI_LIB}
        PARENT_SCOPE
    )
    set(CMAKE_PREFIX_PATH ${_OLD_CMAKE_PREFIX_PATH})
endif()

if (NBL_BUILD_BULLET)
    add_subdirectory(Bullet)
    set(NBL_EXT_BULLET_INCLUDE_DIRS
        ${NBL_EXT_BULLET_INCLUDE_DIRS}
        PARENT_SCOPE
    )
    set(NBL_EXT_BULLET_LIB
        ${NBL_EXT_BULLET_LIB}
        PARENT_SCOPE
    )
endif()

if (NBL_BUILD_MITSUBA_LOADER)
    add_subdirectory(MitsubaLoader)
    set(NBL_EXT_MITSUBA_LOADER_INCLUDE_DIRS
        ${NBL_EXT_MITSUBA_LOADER_INCLUDE_DIRS}
        PARENT_SCOPE
    )
    set(NBL_EXT_MITSUBA_LOADER_LIB
        ${NBL_EXT_MITSUBA_LOADER_LIB}
        PARENT_SCOPE
    )
endif()

if (NBL_BUILD_RADEON_RAYS)
    add_subdirectory(RadeonRays)
    set(NBL_EXT_RADEON_RAYS_INCLUDE_DIRS
        ${NBL_EXT_RADEON_RAYS_INCLUDE_DIRS}
        PARENT_SCOPE
    )
    set(NBL_EXT_RADEON_RAYS_LIB
        ${NBL_EXT_RADEON_RAYS_LIB}
        PARENT_SCOPE
    )
endif()

if (NBL_BUILD_OPTIX)
    add_subdirectory(OptiX)
    set(NBL_EXT_OPTIX_INCLUDE_DIRS
        ${NBL_EXT_OPTIX_INCLUDE_DIRS}
        PARENT_SCOPE
    )
    set(NBL_EXT_OPTIX_LIB
        ${NBL_EXT_OPTIX_LIB}
        PARENT_SCOPE
    )
endif()