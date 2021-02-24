#ifndef __NBL_C_OPENGL_FRAMEBUFFER_H_INCLUDED__
#define __NBL_C_OPENGL_FRAMEBUFFER_H_INCLUDED__

#include "nbl/video/IGPUFramebuffer.h"
#include "nbl/video/IOpenGL_FunctionTable.h"
#include "nbl/video/COpenGLImageView.h"

namespace nbl {
namespace video
{

class COpenGLFramebuffer final : public IGPUFramebuffer
{
    using base_t = IGPUFramebuffer;

public:
    using base_t::base_t;

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
            const auto& att = attachments[a];
            const auto& d = descriptions[a];

            auto* glatt = static_cast<COpenGLImageView*>(att.get());
            const GLuint glname = glatt->getOpenGLName();
            const GLenum textarget = COpenGLImageView::ViewTypeToGLenumTarget[glatt->getCreationParameters().viewType];

            gl->extGlNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0 + i, glname, 0, textarget);
            drawbuffers[i] = GL_COLOR_ATTACHMENT0 + i;
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
        if (status != GL_FRAMEBUFFER_COMPLETE)
        {
            gl->glFramebuffer.pglDeleteFramebuffers(1, &fbo);
            return 0u;
        }

        return fbo;
    }
};

}
}

#endif
