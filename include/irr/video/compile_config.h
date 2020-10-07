// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_COMPILE_CONFIG_H_INCLUDED__
#define __NBL_VIDEO_COMPILE_CONFIG_H_INCLUDED__

#if defined(_NBL_SERVER_)
#   define NO_NBL_COMPILE_WITH_VULKAN_
#   define NO_NBL_COMPILE_WITH_OPENGL_
#endif

#ifdef NO_NBL_COMPILE_WITH_OPENGL_
#   undef _NBL_COMPILE_WITH_OPENGL_
#endif

#endif
