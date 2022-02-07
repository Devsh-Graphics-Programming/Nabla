// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_OPENGL_SAMPLER_H_INCLUDED__
#define __NBL_VIDEO_C_OPENGL_SAMPLER_H_INCLUDED__

#include "nbl/video/IGPUSampler.h"
#include "COpenGLExtensionHandler.h"
#include <algorithm>

#ifdef _NBL_COMPILE_WITH_OPENGL_
namespace nbl
{
namespace video
{
class COpenGLSampler : public IGPUSampler
{
    //! Get native wrap mode value
    inline static GLenum getTextureWrapMode(uint8_t clamp)
    {
        using namespace asset;
        GLenum mode = GL_REPEAT;
        switch(clamp)
        {
            case ETC_REPEAT:
                mode = GL_REPEAT;
                break;
            case ETC_CLAMP_TO_EDGE:
                mode = GL_CLAMP_TO_EDGE;
                break;
            case ETC_CLAMP_TO_BORDER:
                mode = GL_CLAMP_TO_BORDER;
                break;
            case ETC_MIRROR:
                mode = GL_MIRRORED_REPEAT;
                break;
            case ETC_MIRROR_CLAMP_TO_EDGE:
                if(COpenGLExtensionHandler::Version >= 440 || COpenGLExtensionHandler::FeatureAvailable[COpenGLExtensionHandler::NBL_EXT_texture_mirror_clamp] || COpenGLExtensionHandler::FeatureAvailable[COpenGLExtensionHandler::NBL_ATI_texture_mirror_once])
                    mode = GL_MIRROR_CLAMP_TO_EDGE;
                else
                    mode = GL_CLAMP;
                break;
            case ETC_MIRROR_CLAMP_TO_BORDER:
                if(COpenGLExtensionHandler::FeatureAvailable[COpenGLExtensionHandler::NBL_EXT_texture_mirror_clamp])
                    mode = GL_MIRROR_CLAMP_TO_BORDER_EXT;
                else
                    mode = GL_CLAMP;
                break;
        }
        return mode;
    }

protected:
    ~COpenGLSampler()
    {
        COpenGLExtensionHandler::extGlDeleteSamplers(1, &m_GLname);
    }

public:
    COpenGLSampler(const asset::ISampler::SParams& _params)
        : IGPUSampler(_params)
    {
        using gl = COpenGLExtensionHandler;

        gl::extGlCreateSamplers(1, &m_GLname);  //TODO before we were using GlGenSamplers for some reason..

        constexpr GLenum minFilterMap[2][2]{
            {GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST_MIPMAP_LINEAR},
            {GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR_MIPMAP_LINEAR}};

        gl::extGlSamplerParameteri(m_GLname, GL_TEXTURE_MIN_FILTER, minFilterMap[m_params.MinFilter][m_params.MipmapMode]);
        gl::extGlSamplerParameteri(m_GLname, GL_TEXTURE_MAG_FILTER, m_params.MaxFilter == ETF_NEAREST ? GL_NEAREST : GL_LINEAR);

        if(m_params.AnisotropicFilter)
            gl::extGlSamplerParameteri(m_GLname, GL_TEXTURE_MAX_ANISOTROPY_EXT, std::min(1u << m_params.AnisotropicFilter, uint32_t(gl::MaxAnisotropy)));

        gl::extGlSamplerParameteri(m_GLname, GL_TEXTURE_WRAP_S, getTextureWrapMode(m_params.TextureWrapU));
        gl::extGlSamplerParameteri(m_GLname, GL_TEXTURE_WRAP_T, getTextureWrapMode(m_params.TextureWrapV));
        gl::extGlSamplerParameteri(m_GLname, GL_TEXTURE_WRAP_R, getTextureWrapMode(m_params.TextureWrapW));

        gl::extGlSamplerParameterf(m_GLname, GL_TEXTURE_LOD_BIAS, m_params.LodBias);
        gl::extGlSamplerParameterf(m_GLname, GL_TEXTURE_MIN_LOD, m_params.MinLod);
        gl::extGlSamplerParameterf(m_GLname, GL_TEXTURE_MAX_LOD, m_params.MaxLod);

        if(m_params.CompareEnable)
        {
            gl::extGlSamplerParameteri(m_GLname, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);

            constexpr GLenum compareFuncMap[8]{
                GL_NEVER,
                GL_LESS,
                GL_EQUAL,
                GL_LEQUAL,
                GL_GREATER,
                GL_NOTEQUAL,
                GL_GEQUAL,
                GL_ALWAYS};
            gl::extGlSamplerParameteri(m_GLname, GL_TEXTURE_COMPARE_FUNC, compareFuncMap[m_params.CompareFunc]);
        }

        constexpr GLfloat borderColorMap[3][4]{
            {0.f, 0.f, 0.f, 0.f},
            {0.f, 0.f, 0.f, 1.f},
            {1.f, 1.f, 1.f, 1.f}};
        if(m_params.BorderColor / 2u)
            gl::extGlSamplerParameterfv(m_GLname, GL_TEXTURE_BORDER_COLOR, borderColorMap[m_params.BorderColor / 2u]);
    }

    GLuint getOpenGLName() const { return m_GLname; }

private:
    GLuint m_GLname;
};

}
}
#endif

#endif