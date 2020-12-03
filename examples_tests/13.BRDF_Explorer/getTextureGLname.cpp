// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <COpenGL2DTexture.h>

uint32_t getTextureGLname(nbl::video::IVirtualTexture* _texture)
{
    if (!_texture)
        return 0u;
    return static_cast<nbl::video::COpenGL2DTexture*>(_texture)->getOpenGLName();
}