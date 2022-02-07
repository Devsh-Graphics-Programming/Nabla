// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_E_DRIVER_TYPES_H_INCLUDED__
#define __NBL_E_DRIVER_TYPES_H_INCLUDED__

namespace nbl
{
namespace video
{
//! An enum for all types of drivers the Irrlicht Engine supports.
enum E_DRIVER_TYPE  // future rename to E_API
{
    //! Null driver, useful for applications to run the engine without visualisation.
    /** The null device is able to load textures, but does not
            render and display any graphics. */
    EDT_NULL,  // depr

    //! OpenGL device, available on most platforms.
    /** Performs hardware accelerated rendering of 3D and 2D
            primitives. */
    EDT_OPENGL,

    //! Vulkan device, not available yet
    //EDT_VULKAN,

    //! No driver, just for counting the elements
    EDT_COUNT
};

}  // end namespace video
}  // end namespace nbl

#endif
