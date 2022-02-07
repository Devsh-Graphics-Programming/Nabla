// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_OPENGL_IMAGE_VIEW_H_INCLUDED__
#define __NBL_VIDEO_C_OPENGL_IMAGE_VIEW_H_INCLUDED__

#include "nbl/video/IGPUImageView.h"
#include "nbl/video/COpenGLImage.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_
namespace nbl
{
namespace video
{
class COpenGLImageView final : public IGPUImageView
{
protected:
    virtual ~COpenGLImageView()
    {
        if(name)
            glDeleteTextures(1u, &name);
    }

    GLuint name;
    GLenum target;
    GLenum internalFormat;

public:
    _NBL_STATIC_INLINE_CONSTEXPR GLenum ViewTypeToGLenumTarget[IGPUImageView::ET_COUNT] = {
        GL_TEXTURE_1D, GL_TEXTURE_2D, GL_TEXTURE_3D, GL_TEXTURE_CUBE_MAP, GL_TEXTURE_1D_ARRAY, GL_TEXTURE_2D_ARRAY, GL_TEXTURE_CUBE_MAP_ARRAY};
    _NBL_STATIC_INLINE_CONSTEXPR GLenum ComponentMappingToGLenumSwizzle[IGPUImageView::SComponentMapping::ES_COUNT] = {GL_INVALID_ENUM, GL_ZERO, GL_ONE, GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA};

    COpenGLImageView(SCreationParams&& _params)
        : IGPUImageView(std::move(_params)), name(0u), target(GL_INVALID_ENUM), internalFormat(GL_INVALID_ENUM)
    {
        target = ViewTypeToGLenumTarget[params.viewType];
        internalFormat = getSizedOpenGLFormatFromOurFormat(params.format);
        assert(internalFormat != GL_INVALID_ENUM);
        //COpenGLExtensionHandler::extGlCreateTextures(target, 1, &name);
        glGenTextures(1, &name);
        COpenGLExtensionHandler::extGlTextureView(name, target, static_cast<COpenGLImage*>(params.image.get())->getOpenGLName(), internalFormat,
            params.subresourceRange.baseMipLevel, params.subresourceRange.levelCount,
            params.subresourceRange.baseArrayLayer, params.subresourceRange.layerCount);

        GLuint swizzle[4u] = {GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA};
        for(auto i = 0u; i < 4u; i++)
        {
            auto currentMapping = (&params.components.r)[i];
            if(currentMapping == IGPUImageView::SComponentMapping::ES_IDENTITY)
                continue;
            swizzle[i] = ComponentMappingToGLenumSwizzle[currentMapping];
        }
        COpenGLExtensionHandler::extGlTextureParameterIuiv(name, target, GL_TEXTURE_SWIZZLE_RGBA, swizzle);
    }

    void regenerateMipMapLevels() override
    {
        if(params.subresourceRange.levelCount <= 1u)
            return;

        COpenGLExtensionHandler::extGlGenerateTextureMipmap(name, target);
    }

    inline GLuint getOpenGLName() const { return name; }
    inline GLenum getOpenGLTextureType() const { return target; }
    inline GLenum getInternalFormat() const { return internalFormat; }
};

}
}
#endif

#endif