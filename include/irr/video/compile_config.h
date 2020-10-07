// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_COMPILE_CONFIG_H_INCLUDED__
#define __NBL_VIDEO_COMPILE_CONFIG_H_INCLUDED__

#define NEW_MESHES

#include <stdio.h> // TODO: Although included elsewhere this is required at least for mingw

//! Uncomment this line to compile with the SDL device
//#define _NBL_COMPILE_WITH_SDL_DEVICE_
#ifdef NO_NBL_COMPILE_WITH_SDL_DEVICE_
#undef _NBL_COMPILE_WITH_SDL_DEVICE_
#endif

#define NEW_SHADERS 1

#if defined(_NBL_SERVER_)
#   define NO_NBL_COMPILE_WITH_VULKAN_
#   define NO_NBL_COMPILE_WITH_OPENGL_
#endif

#ifdef NO_NBL_COMPILE_WITH_OPENGL_
#   undef _NBL_COMPILE_WITH_OPENGL_
#endif

#endif
