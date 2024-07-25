# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

start_tracking_variables_for_propagation_to_parent()

# TODO: all of those options bellow should be defined here
option(NBL_BUILD_TEXT_RENDERING "Enable Nabla Text Rendering extension building and integration?" OFF)

# TODO: also all variables bellow should be killed from build system since we have logical
# targets which properties (like include search directories or outputs) can be queried 
# with generator expressions, no more hardcoded parent scope vars without
# proper configuration handle! The only thing each extension should 
# pass as an internal cache variable is it's target name.

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

if (NBL_BUILD_IMGUI)
    add_subdirectory(ImGui)
    set(NBL_EXT_IMGUI_UI_INCLUDE_DIRS
        ${NBL_EXT_IMGUI_UI_INCLUDE_DIRS}
        PARENT_SCOPE
    )
    set(NBL_EXT_IMGUI_UI_LIB
        ${NBL_EXT_IMGUI_UI_LIB}
        PARENT_SCOPE
    )
endif()

if(NBL_BUILD_TEXT_RENDERING)
	add_subdirectory(TextRendering)
endif()

propagate_changed_variables_to_parent_scope()