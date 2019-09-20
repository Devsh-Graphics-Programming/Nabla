#ifndef __IRR_C_OPENGL_DESCRIPTOR_SET_H_INCLUDED__
#define __IRR_C_OPENGL_DESCRIPTOR_SET_H_INCLUDED__

#include "irr/video/IGPUDescriptorSet.h"
#include "irr/macros.h"
#include "COpenGLExtensionHandler.h"
#include "COpenGLBuffer.h"
#include "COpenGL1DTexture.h"
#include "COpenGL1DTextureArray.h"
#include "COpenGL2DTexture.h"
#include "COpenGL2DTextureArray.h"
#include "COpenGLCubemapTexture.h"
#include "COpenGLCubemapArrayTexture.h"
#include "COpenGL3DTexture.h"
#include "COpenGLBufferView.h"
#include "COpenGLTextureView.h"

namespace irr {
namespace video
{

class COpenGLDescriptorSet : public IGPUDescriptorSet
{
public:
    struct SMultibindParams
    {
        struct SMultibindBuffers
        {
            const GLuint* buffers;
            const GLintptr* offsets;
            const GLsizeiptr* sizes;
        };
        struct SMultibindTextures
        {
            const GLuint* textures;
            const GLenum* targets;//for when ARB_multi_bind isn't there
        };
        struct SMultibindTextureImages
        {
            const GLuint* textures;
            const GLenum* formats;
        };
        
        SMultibindBuffers ubos;
        SMultibindBuffers ssbos;
        SMultibindTextures textures;
        SMultibindTextureImages textureImages;
    };

public:
    COpenGLDescriptorSet(core::smart_refctd_dynamic_array<SWriteDescriptorSet>&& _descriptors) : IGPUDescriptorSet(std::move(_descriptors))
    {
        size_t uboCount = 0ull;
        size_t ssboCount = 0ull;
        size_t textureCount = 0ull;
        size_t imageCount = 0ull;
        for (const auto& desc : (*m_descriptors))
        {
            switch (desc.descriptorType)
            {
            case asset::EDT_UNIFORM_BUFFER:
                uboCount += desc.info->size();
                break;
            case asset::EDT_STORAGE_BUFFER:
                ssboCount += desc.info->size();
                break;
            case asset::EDT_UNIFORM_TEXEL_BUFFER: //GL_TEXTURE_BUFFER
                _IRR_FALLTHROUGH;
            case asset::EDT_COMBINED_IMAGE_SAMPLER:
                textureCount += desc.info->size();
                break;
            case asset::EDT_STORAGE_IMAGE:
                _IRR_FALLTHROUGH;
            case asset::EDT_STORAGE_TEXEL_BUFFER:
                imageCount += desc.info->size();
            default: break;
            }
        }

        m_names = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<GLuint>>(uboCount+ssboCount+textureCount+imageCount);
        m_offsets = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<GLintptr>>(uboCount+ssboCount);
        m_sizes = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<GLuint>>(uboCount+ssboCount);
        m_extraEnums = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<GLuint>>(textureCount+imageCount);

        const size_t uboNamesOffset = 0ull;
        const size_t ssboNamesOffset = uboCount;
        const size_t texNamesOffset = uboCount+ssboCount;
        const size_t imagNamesOffset = texNamesOffset + textureCount;
        const size_t uboBufOffset = 0ull;
        const size_t ssboBufOffset = uboCount;
        const size_t enums_texTargetsOffset = 0ull;
        const size_t enums_imagFormatsOffset = textureCount;

        auto setBufMultibindParams = [this](SMultibindParams::SMultibindBuffers& _params, size_t _namesOffset, size_t _offset) {
            _params.buffers = (*m_names).data() + _namesOffset;
            _params.offsets = (*m_offsets).data() + _offset;
            _params.sizes = (*m_sizes).data() + _offset;
        };
        setBufMultibindParams(m_multibindParams.ubos, uboNamesOffset, uboBufOffset);
        setBufMultibindParams(m_multibindParams.ssbos, ssboNamesOffset, ssboBufOffset);
        m_multibindParams.textures.textures = (*m_names).data() + texNamesOffset;
        m_multibindParams.textures.targets = (*m_extraEnums).data() + enums_texTargetsOffset;
        m_multibindParams.textureImages.formats = (*m_extraEnums).data() + enums_imagFormatsOffset;

        auto castToCOpenGLTexture = [](asset::IDescriptor* _tex) -> COpenGLTexture* {
            ITexture* tex = static_cast<ITexture*>(_tex);
            switch (tex->getTextureType())
            {
            case ITexture::ETT_1D:
                return static_cast<COpenGL1DTexture*>(tex);
            case ITexture::ETT_1D_ARRAY:
                return static_cast<COpenGL1DTextureArray*>(tex);
            case ITexture::ETT_2D:
                return static_cast<COpenGL2DTexture*>(tex);
            case ITexture::ETT_2D_ARRAY:
                return static_cast<COpenGL2DTextureArray*>(tex);
            case ITexture::ETT_3D:
                return static_cast<COpenGL3DTexture*>(tex);
            case ITexture::ETT_CUBE_MAP:
                return static_cast<COpenGLCubemapTexture*>(tex);
            case ITexture::ETT_CUBE_MAP_ARRAY:
                return static_cast<COpenGLCubemapArrayTexture*>(tex);
            default:
                return nullptr;
            }
        };
        size_t u=0ull, s=0ull, t=0ull, i=0ull;
        for (const auto& desc : (*m_descriptors))
        {
            if (desc.descriptorType==asset::EDT_UNIFORM_BUFFER)
                for (const auto& info : (*desc.info)) {
                    (*m_names)[uboNamesOffset + u] = static_cast<COpenGLBuffer*>(info.desc.get())->getOpenGLName();
                    (*m_offsets)[uboBufOffset + u] = info.buffer.offset;
                    (*m_sizes)[uboBufOffset + u] = info.buffer.size;
                    ++u;
                }
            else if (desc.descriptorType==asset::EDT_STORAGE_BUFFER)
                for (const auto& info : (*desc.info)) {
                    (*m_names)[ssboNamesOffset + s] = static_cast<COpenGLBuffer*>(info.desc.get())->getOpenGLName();
                    (*m_offsets)[ssboBufOffset + s] = info.buffer.offset;
                    (*m_sizes)[ssboBufOffset + s] = info.buffer.size;
                    ++s;
                }
            else if (desc.descriptorType==asset::EDT_COMBINED_IMAGE_SAMPLER)
                for (const auto& info : (*desc.info)) {
                    (*m_names)[texNamesOffset + t] = castToCOpenGLTexture(info.desc.get())->getOpenGLName();
                    (*m_extraEnums)[enums_texTargetsOffset + t] = castToCOpenGLTexture(info.desc.get())->getOpenGLTextureType();
                    ++t;
                }
            else if (desc.descriptorType==asset::EDT_UNIFORM_TEXEL_BUFFER)
                for (const auto& info : (*desc.info)) {
                    (*m_names)[texNamesOffset + t] = static_cast<COpenGLBufferView*>(info.desc.get())->getOpenGLName();
                    (*m_extraEnums)[enums_texTargetsOffset + t] = GL_TEXTURE_BUFFER;
                    ++t;
                }
            else if (desc.descriptorType==asset::EDT_STORAGE_IMAGE)
                for (const auto& info : (*desc.info)) {
                    (*m_names)[imagNamesOffset + i] = static_cast<COpenGLTextureView*>(info.desc.get())->getOpenGLName();
                    (*m_extraEnums)[enums_imagFormatsOffset + i] = static_cast<COpenGLTextureView*>(info.desc.get())->getInternalFormat();
                    ++i;
                }
            else if (desc.descriptorType==asset::EDT_STORAGE_TEXEL_BUFFER)
                for (const auto& info : (*desc.info)) {
                    //TODO not sure. should STORAGE_TEXEL_BUFFER come from COpenGLBufferView?
                    (*m_names)[imagNamesOffset + i] = static_cast<COpenGLBufferView*>(info.desc.get())->getOpenGLName();
                    (*m_extraEnums)[enums_imagFormatsOffset + i] = static_cast<COpenGLBufferView*>(info.desc.get())->getInternalFormat();
                    ++i;
                }
        }
    }

    const SMultibindParams& getMultibindParams() const { return m_multibindParams; }

private:
    SMultibindParams m_multibindParams;
    core::smart_refctd_dynamic_array<GLuint> m_names;
    core::smart_refctd_dynamic_array<GLintptr> m_offsets;
    core::smart_refctd_dynamic_array<GLsizeiptr> m_sizes;
    core::smart_refctd_dynamic_array<GLenum> m_extraEnums;
};

}}

#endif