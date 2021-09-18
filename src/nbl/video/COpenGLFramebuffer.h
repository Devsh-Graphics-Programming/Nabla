#ifndef __NBL_C_OPENGL_FRAMEBUFFER_H_INCLUDED__
#define __NBL_C_OPENGL_FRAMEBUFFER_H_INCLUDED__

#include <array>
#include "nbl/video/IGPUFramebuffer.h"
#include "nbl/video/IOpenGL_FunctionTable.h"
#include "nbl/video/COpenGLImageView.h"

namespace nbl {
namespace video
{

class IOpenGL_LogicalDevice;

class COpenGLFramebuffer final : public IGPUFramebuffer
{
public:
#include "nbl/nblpack.h"
    struct hash_element_t
    {
        using id_t = uint64_t; 
        id_t id;// reinterpret_cast of pointer
        uint32_t mip;
        uint32_t baseLayer;
        uint32_t layerCount;

        inline bool operator==(const hash_element_t& rhs) const { return memcmp(this, &rhs, sizeof(hash_element_t)) == 0; }
        inline bool operator!=(const hash_element_t& rhs) const { return !operator==(rhs); }
    } PACK_STRUCT;
#include "nbl/nblunpack.h"
    using hash_t = std::array<hash_element_t, IGPURenderpass::SCreationParams::MaxColorAttachments+1u>;

private:
    using base_t = IGPUFramebuffer;

    IOpenGL_LogicalDevice* m_device;

    static hash_t getHashImage(const IGPUImage* img, uint32_t ix, uint32_t mip = 0u, int32_t layer = -1)
    {
        hash_t hash;
        memset(hash.data(), 0, sizeof(hash_t::value_type) * hash.size());
        hash[ix].id = reinterpret_cast<hash_t::value_type::id_t>(img);
        hash[ix].baseLayer = layer < 0 ? 0u : static_cast<uint32_t>(layer);
        hash[ix].layerCount = layer < 0 ? img->getCreationParameters().arrayLayers : 1u;
        hash[ix].mip = mip;
        return hash;
    }

public:
    COpenGLFramebuffer(core::smart_refctd_ptr<IOpenGL_LogicalDevice>&& dev, SCreationParams&& params);

    ~COpenGLFramebuffer();

    static hash_t getHashColorImage(const IGPUImage* img, uint32_t mip = 0u, int32_t layer = -1)
    {
        return getHashImage(img, 0u, mip, layer);
    }
    static GLuint getNameColorImage(IOpenGL_FunctionTable* gl, const IGPUImage* img, uint32_t mip = 0u, int32_t layer = -1)
    {
        GLuint fbo;
        gl->extGlCreateFramebuffers(1u, &fbo);
        auto* glimg = static_cast<const COpenGLImage*>(img);
        const GLenum textarget = glimg->getOpenGLTarget();
        if (layer < 0)
            gl->extGlNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, glimg->getOpenGLName(), mip, textarget);
        else
            gl->extGlNamedFramebufferTextureLayer(fbo, GL_COLOR_ATTACHMENT0, glimg->getOpenGLName(), glimg->getOpenGLTarget(), mip, layer);
        GLenum drawbuf = GL_COLOR_ATTACHMENT0;
        gl->extGlNamedFramebufferDrawBuffers(fbo, 1, &drawbuf);

        GLenum status = gl->extGlCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER);
        assert(status == GL_FRAMEBUFFER_COMPLETE);
        if (status != GL_FRAMEBUFFER_COMPLETE)
        {
            gl->glFramebuffer.pglDeleteFramebuffers(1, &fbo);
            return 0u;
        }

        return fbo;
    }
    static hash_t getHashDepthStencilImage(const IGPUImage* img, uint32_t mip = 0u, int32_t layer = -1)
    {
        return getHashImage(img, IGPURenderpass::SCreationParams::MaxColorAttachments, mip, layer);
    }
    static GLuint getNameDepthStencilImage(IOpenGL_FunctionTable* gl, const IGPUImage* img, uint32_t mip = 0u, int32_t layer = -1)
    {
        GLuint fbo;
        auto* glimg = static_cast<const COpenGLImage*>(img);
        const GLenum textarget = glimg->getOpenGLTarget();
        auto format = glimg->getCreationParameters().format;
        GLenum attpoint = GL_INVALID_ENUM;
        if (asset::isDepthOnlyFormat(format))
            attpoint = GL_DEPTH_ATTACHMENT;
        else if (asset::isStencilOnlyFormat(format))
            attpoint = GL_STENCIL_ATTACHMENT;
        else if (asset::isDepthOrStencilFormat(format))
            attpoint = GL_DEPTH_STENCIL_ATTACHMENT;
        else return 0u;
        gl->extGlCreateFramebuffers(1u, &fbo);
        if (layer < 0)
            gl->extGlNamedFramebufferTexture(fbo, attpoint, glimg->getOpenGLName(), mip, textarget);
        else
            gl->extGlNamedFramebufferTextureLayer(fbo, attpoint, glimg->getOpenGLName(), glimg->getOpenGLTarget(), mip, layer);

        GLenum status = gl->extGlCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER);
        assert(status == GL_FRAMEBUFFER_COMPLETE);
        if (status != GL_FRAMEBUFFER_COMPLETE)
        {
            gl->glFramebuffer.pglDeleteFramebuffers(1, &fbo);
            return 0u;
        }

        return fbo;
    }

    hash_t getHashValue() const
    {
        const auto& sub = m_params.renderpass->getSubpasses().begin()[0];
        const auto* attachments = m_params.attachments;

        hash_t hash;
        memset(hash.data(), 0, sizeof(hash_t::value_type)*hash.size());
        for (uint32_t i = 0u; i < sub.colorAttachmentCount; ++i)
        {
            uint32_t a = sub.colorAttachments[i].attachment;
            if (a == IGPURenderpass::ATTACHMENT_UNUSED)
                continue;

            auto& att = attachments[a];
            static_assert(sizeof(hash_t::value_type::id_t)==sizeof(void*), "Bad reinterpret_cast!");
            hash[i].id = reinterpret_cast<hash_t::value_type::id_t>(att.get());
            hash[i].baseLayer = 0u;
            hash[i].layerCount = att->getCreationParameters().subresourceRange.layerCount;
            hash[i].mip = 0u;
        }
        if (sub.depthStencilAttachment && sub.depthStencilAttachment->attachment != IGPURenderpass::ATTACHMENT_UNUSED)
        {
            auto& att = attachments[sub.depthStencilAttachment->attachment];
            hash[IGPURenderpass::SCreationParams::MaxColorAttachments].id = reinterpret_cast<hash_t::value_type::id_t>(att.get());
            hash[IGPURenderpass::SCreationParams::MaxColorAttachments].baseLayer = 0u;
            hash[IGPURenderpass::SCreationParams::MaxColorAttachments].layerCount = att->getCreationParameters().subresourceRange.layerCount;
            hash[IGPURenderpass::SCreationParams::MaxColorAttachments].mip = 0u;
        }

        return hash;
    }

    GLuint createGLFBO(IOpenGL_FunctionTable* gl) const
    {
        const auto& sub = m_params.renderpass->getSubpasses().begin()[0];
        const auto* descriptions = m_params.renderpass->getAttachments().begin();
        const auto* attachments = m_params.attachments;

        GLuint fbo = 0u;
        gl->extGlCreateFramebuffers(1u, &fbo);
        if (!fbo)
            return 0u;

        GLenum drawbuffers[IGPURenderpass::SCreationParams::MaxColorAttachments] = { 0 }; // GL_NONE
        for (uint32_t i = 0u; i < sub.colorAttachmentCount; ++i)
        {
            const uint32_t a = sub.colorAttachments[i].attachment;
            if (a != IGPURenderpass::ATTACHMENT_UNUSED)
            {
                const auto& att = attachments[a];
                const auto& d = descriptions[a];

                auto* glatt = static_cast<COpenGLImageView*>(att.get());
                const GLuint glname = glatt->getOpenGLName();
                //gl->glTexture.pglBindTexture(GL_TEXTURE_2D, glname); // what was it for???
                const GLenum textarget = COpenGLImageView::ViewTypeToGLenumTarget[glatt->getCreationParameters().viewType];

                gl->extGlNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0 + i, glname, 0, textarget);
                //gl->glTexture.pglBindTexture(GL_TEXTURE_2D, 0);
            }

            drawbuffers[i] = (a != IGPURenderpass::ATTACHMENT_UNUSED) ?  (GL_COLOR_ATTACHMENT0 + i) : GL_NONE;
        }
        if (sub.depthStencilAttachment)
        {
            const auto& att = attachments[sub.depthStencilAttachment->attachment];
            auto* glatt = static_cast<COpenGLImageView*>(att.get());
            const GLuint glname = glatt->getOpenGLName();
            const GLenum textarget = COpenGLImageView::ViewTypeToGLenumTarget[glatt->getCreationParameters().viewType];

            const asset::E_FORMAT format = att->getCreationParameters().format;

            GLenum attpoint = GL_INVALID_ENUM;
            if (asset::isDepthOnlyFormat(format))
                attpoint = GL_DEPTH_ATTACHMENT;
            else if (asset::isStencilOnlyFormat(format))
                attpoint = GL_STENCIL_ATTACHMENT;
            else if (asset::isDepthOrStencilFormat(format))
                attpoint = GL_DEPTH_STENCIL_ATTACHMENT;
            else {
                gl->glFramebuffer.pglDeleteFramebuffers(1, &fbo);
                return 0u;
            }

            gl->extGlNamedFramebufferTexture(fbo, attpoint, glname, 0, textarget);
        }

        gl->extGlNamedFramebufferDrawBuffers(fbo, sub.colorAttachmentCount, drawbuffers);

        GLenum status = gl->extGlCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER);
        assert(status == GL_FRAMEBUFFER_COMPLETE);
        if (status != GL_FRAMEBUFFER_COMPLETE)
        {
            gl->glFramebuffer.pglDeleteFramebuffers(1, &fbo);
            return 0u;
        }
        gl->glGeneral.pglFlush();

        const auto& dbgname = getObjectDebugName();
        if (dbgname.size())
        {
            gl->extGlObjectLabel(GL_FRAMEBUFFER, fbo, dbgname.size(), dbgname.c_str());
        }

        return fbo;
    }
};

}
}

#endif
