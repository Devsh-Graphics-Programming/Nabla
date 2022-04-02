// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_OPEN_GL_FEATURE_MAP_H_INCLUDED__
#define __NBL_C_OPEN_GL_FEATURE_MAP_H_INCLUDED__

#include "nbl/core/core.h"
#include "nbl/system/compile_config.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_

#include "COpenGLStateManager.h"

namespace nbl
{
namespace video
{


//extra extension name that is reported as supported when irrbaw app is running in renderdoc
_NBL_STATIC_INLINE_CONSTEXPR const char* RUNNING_IN_RENDERDOC_EXTENSION_NAME = "GL_NBL_RUNNING_IN_RENDERDOC";



	// deferred initialization
	void initExtensions(bool stencilBuffer);

	static void loadFunctions();

	bool isDeviceCompatibile(core::vector<std::string>* failedExtensions=NULL);

	//! queries the features of the driver, returns true if feature is available
	inline bool queryOpenGLFeature(EOpenGLFeatures feature) const
	{
		return FeatureAvailable[feature];
	}

	//! show all features with availablity
	void dump(std::string* outStr=NULL, bool onlyAvailable=false) const;

	void dumpFramebufferFormats() const;

	// Some variables for properties
	bool StencilBuffer;
	bool TextureCompressionExtension;

	// Some non-boolean properties
	//!
	static int32_t reqUBOAlignment;
	//!
	static int32_t reqSSBOAlignment;
	//!
	static int32_t reqTBOAlignment;
    //!
    static uint64_t maxUBOSize;
    //!
    static uint64_t maxSSBOSize;
    //!
    static uint64_t maxTBOSizeInTexels;
    //!
    static uint64_t maxBufferSize;
    //!
    static uint32_t maxUBOBindings;
    //!
    static uint32_t maxSSBOBindings;
    //! For vertex and fragment shaders
    //! If both the vertex shader and the fragment processing stage access the same texture image unit, then that counts as using two texture image units against this limit.
    static uint32_t maxTextureBindings;
    //! For compute shader
    static uint32_t maxTextureBindingsCompute;
    //!
    static uint32_t maxImageBindings;
	//!
	static int32_t minMemoryMapAlignment;
    //!
    static int32_t MaxComputeWGSize[3];
	//!
	static uint32_t MaxArrayTextureLayers;
	//! Maxmimum texture layers supported by the engine
	static uint8_t MaxTextureUnits;
	//! Maximal Anisotropy
	static uint8_t MaxAnisotropy;
	//! Number of user clipplanes
	static uint8_t MaxUserClipPlanes;
	//! Number of rendertargets available as MRTs
	static uint8_t MaxMultipleRenderTargets;
	//! Optimal number of indices per meshbuffer
	static uint32_t MaxIndices;
	//! Optimal number of vertices per meshbuffer
	static uint32_t MaxVertices;
	//! Maximal vertices handled by geometry shaders
	static uint32_t MaxGeometryVerticesOut;
	//! Maximal LOD Bias
	static float MaxTextureLODBias;
	//!
	static uint32_t MaxVertexStreams;
	//!
	static uint32_t MaxXFormFeedbackComponents;
	//!
	static uint32_t MaxGPUWaitTimeout;
	//! Gives the upper and lower bound on warp/wavefront/SIMD-lane size
	static uint32_t InvocationSubGroupSize[2];

    //TODO should be later changed to SPIR-V extensions enum like it is with OpenGL extensions
    //(however it does not have any implications on API)
    static GLuint SPIR_VextensionsCount;
    static core::smart_refctd_dynamic_array<const GLubyte*> SPIR_Vextensions;

	//! Minimal and maximal supported thickness for lines without smoothing
	GLfloat DimAliasedLine[2];
	//! Minimal and maximal supported thickness for points without smoothing
	GLfloat DimAliasedPoint[2];
	//! Minimal and maximal supported thickness for lines with smoothing
	GLfloat DimSmoothedLine[2];
	//! Minimal and maximal supported thickness for points with smoothing
	GLfloat DimSmoothedPoint[2];

	//! OpenGL version as Integer: 100*Major+Minor, i.e. 2.1 becomes 201
	static uint16_t Version;
	//! GLSL version as Integer: 100*Major+Minor
	static uint16_t ShaderLanguageVersion;

	static bool IsIntelGPU;
	static bool needsDSAFramebufferHack;


	//
	static void extGlTextureBarrier();

	// generic vsync setting method for several extensions
	static void extGlSwapInterval(int interval);

	// the global feature array
	static bool FeatureAvailable[NBL_OpenGL_Feature_Count];

    //
    static PFNGLTEXTUREBARRIERPROC pGlTextureBarrier;
    static PFNGLTEXTUREBARRIERNVPROC pGlTextureBarrierNV;

    // os specific stuff for swapchain
    #if defined(WGL_EXT_swap_control)
    static PFNWGLSWAPINTERVALEXTPROC pWglSwapIntervalEXT;
    #endif
    #if defined(GLX_SGI_swap_control)
    static PFNGLXSWAPINTERVALSGIPROC pGlxSwapIntervalSGI;
    #endif
    #if defined(GLX_EXT_swap_control)
    static PFNGLXSWAPINTERVALEXTPROC pGlxSwapIntervalEXT;
    #endif
    #if defined(GLX_MESA_swap_control)
    static PFNGLXSWAPINTERVALMESAPROC pGlxSwapIntervalMESA;
    #endif






inline void COpenGLExtensionHandler::extGlBindImageTexture(GLuint index, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format)
{
    if (pGlBindImageTexture)
        pGlBindImageTexture(index,texture,level,layered,layer,access,format);
}


inline void COpenGLExtensionHandler::extGlBindImageTextures(GLuint first, GLsizei count, const GLuint* textures, const GLenum* formats)
{
    if (pGlBindImageTextures && !IsIntelGPU) // Intel is a special boy, as always
        pGlBindImageTextures(first, count, textures);
    else
    {
        for (GLsizei i=0; i<count; i++)
        {
            if (!textures || textures[i] == 0u)
                extGlBindImageTexture(first+i, 0u, 0u, GL_FALSE, 0, GL_READ_WRITE, GL_R8);
            else
                extGlBindImageTexture(first+i, textures[i], 0, GL_TRUE, 0, GL_READ_WRITE, formats[i]);
        }
    }
}



inline void COpenGLExtensionHandler::extGlCreateFramebuffers(GLsizei n, GLuint *framebuffers)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlCreateFramebuffers(n, framebuffers);
            return;
        }
    }

    pGlGenFramebuffers(n, framebuffers);
}

inline GLenum COpenGLExtensionHandler::extGlCheckNamedFramebufferStatus(GLuint framebuffer, GLenum target)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
            return pGlCheckNamedFramebufferStatus(framebuffer,target);
        else if (FeatureAvailable[NBL_EXT_direct_state_access])
            return pGlCheckNamedFramebufferStatusEXT(framebuffer,target);
    }

    GLenum retval;
    GLuint bound;
    glGetIntegerv(target==GL_READ_FRAMEBUFFER ? GL_READ_FRAMEBUFFER_BINDING:GL_DRAW_FRAMEBUFFER_BINDING,reinterpret_cast<GLint*>(&bound));

    if (bound!=framebuffer)
        pGlBindFramebuffer(target,framebuffer);
    retval = pGlCheckFramebufferStatus(target);
    if (bound!=framebuffer)
        pGlBindFramebuffer(target,bound);

    return retval;
}

inline void COpenGLExtensionHandler::extGlNamedFramebufferTexture(GLuint framebuffer, GLenum attachment, GLuint texture, GLint level)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlNamedFramebufferTexture(framebuffer, attachment, texture, level);
            return;
        }
        else if (FeatureAvailable[NBL_EXT_direct_state_access])
        {
            pGlNamedFramebufferTextureEXT(framebuffer, attachment, texture, level);
            return;
        }
    }

    GLuint bound;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING,reinterpret_cast<GLint*>(&bound));

    if (bound!=framebuffer)
        pGlBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
    pGlFramebufferTexture(GL_FRAMEBUFFER,attachment,texture,level);
    if (bound!=framebuffer)
        pGlBindFramebuffer(GL_FRAMEBUFFER,bound);
}

inline void COpenGLExtensionHandler::extGlNamedFramebufferTextureLayer(GLuint framebuffer, GLenum attachment, GLuint texture, GLenum textureType, GLint level, GLint layer)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlNamedFramebufferTextureLayer(framebuffer, attachment, texture, level, layer);
            return;
        }
    }

	if (textureType!=GL_TEXTURE_CUBE_MAP)
	{
		if (!needsDSAFramebufferHack && FeatureAvailable[NBL_EXT_direct_state_access])
		{
            pGlNamedFramebufferTextureLayerEXT(framebuffer, attachment, texture, level, layer);
		}
		else
		{
			GLuint bound;
			glGetIntegerv(GL_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(&bound));

			if (bound != framebuffer)
				pGlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
			pGlFramebufferTextureLayer(GL_FRAMEBUFFER, attachment, texture, level, layer);
			if (bound != framebuffer)
				pGlBindFramebuffer(GL_FRAMEBUFFER, bound);
		}
	}
	else
	{
		constexpr GLenum CubeMapFaceToCubeMapFaceGLenum[IGPUImageView::ECMF_COUNT] = {
			GL_TEXTURE_CUBE_MAP_POSITIVE_X,GL_TEXTURE_CUBE_MAP_NEGATIVE_X,GL_TEXTURE_CUBE_MAP_POSITIVE_Y,GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,GL_TEXTURE_CUBE_MAP_POSITIVE_Z,GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
		};

		if (!needsDSAFramebufferHack && FeatureAvailable[NBL_EXT_direct_state_access])
		{
            pGlNamedFramebufferTexture2DEXT(framebuffer, attachment, CubeMapFaceToCubeMapFaceGLenum[layer], texture, level);
		}
		else
		{
			GLuint bound;
			glGetIntegerv(GL_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(&bound));

			if (bound != framebuffer)
				pGlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
			pGlFramebufferTexture2D(GL_FRAMEBUFFER, attachment, CubeMapFaceToCubeMapFaceGLenum[layer], texture, level);
			if (bound != framebuffer)
				pGlBindFramebuffer(GL_FRAMEBUFFER, bound);
		}
	}
}

inline void COpenGLExtensionHandler::extGlBlitNamedFramebuffer(GLuint readFramebuffer, GLuint drawFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlBlitNamedFramebuffer(readFramebuffer, drawFramebuffer, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
            return;
        }
    }

    GLint boundReadFBO = -1;
    GLint boundDrawFBO = -1;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING,&boundReadFBO);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&boundDrawFBO);

    if (static_cast<GLint>(readFramebuffer)!=boundReadFBO)
        extGlBindFramebuffer(GL_READ_FRAMEBUFFER,readFramebuffer);
    if (static_cast<GLint>(drawFramebuffer)!=boundDrawFBO)
        extGlBindFramebuffer(GL_DRAW_FRAMEBUFFER,drawFramebuffer);

    pGlBlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);

    if (static_cast<GLint>(readFramebuffer)!=boundReadFBO)
        extGlBindFramebuffer(GL_READ_FRAMEBUFFER,boundReadFBO);
    if (static_cast<GLint>(drawFramebuffer)!=boundDrawFBO)
        extGlBindFramebuffer(GL_DRAW_FRAMEBUFFER,boundDrawFBO);
}

inline void COpenGLExtensionHandler::extGlNamedFramebufferReadBuffer(GLuint framebuffer, GLenum mode)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlNamedFramebufferReadBuffer(framebuffer, mode);
            return;
        }
        else if (FeatureAvailable[NBL_EXT_direct_state_access])
        {
            pGlFramebufferReadBufferEXT(framebuffer, mode);
            return;
        }
    }

    GLint boundFBO;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING,&boundFBO);

    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        pGlBindFramebuffer(GL_READ_FRAMEBUFFER,framebuffer);
    glReadBuffer(mode);
    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        pGlBindFramebuffer(GL_READ_FRAMEBUFFER,boundFBO);
}

inline void COpenGLExtensionHandler::extGlNamedFramebufferDrawBuffer(GLuint framebuffer, GLenum buf)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlNamedFramebufferDrawBuffer(framebuffer, buf);
            return;
        }
        else if (FeatureAvailable[NBL_EXT_direct_state_access])
        {
            pGlFramebufferDrawBufferEXT(framebuffer, buf);
            return;
        }
    }

    GLint boundFBO;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&boundFBO);

    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        pGlBindFramebuffer(GL_DRAW_FRAMEBUFFER,framebuffer);
    glDrawBuffer(buf);
    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        pGlBindFramebuffer(GL_DRAW_FRAMEBUFFER,boundFBO);
}

inline void COpenGLExtensionHandler::extGlNamedFramebufferDrawBuffers(GLuint framebuffer, GLsizei n, const GLenum *bufs)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlNamedFramebufferDrawBuffers(framebuffer, n, bufs);
            return;
        }
        else if (FeatureAvailable[NBL_EXT_direct_state_access])
        {
            pGlFramebufferDrawBuffersEXT(framebuffer, n, bufs);
            return;
        }
    }

    GLint boundFBO;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&boundFBO);

    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        pGlBindFramebuffer(GL_DRAW_FRAMEBUFFER,framebuffer);
    pGlDrawBuffers(n,bufs);
    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        pGlBindFramebuffer(GL_DRAW_FRAMEBUFFER,boundFBO);
}

inline void COpenGLExtensionHandler::extGlClearNamedFramebufferiv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLint* value)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlClearNamedFramebufferiv(framebuffer, buffer, drawbuffer, value);
            return;
        }
    }

    GLint boundFBO = -1;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&boundFBO);
    if (boundFBO<0)
        return;

    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        extGlBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
    pGlClearBufferiv(buffer, drawbuffer, value);
    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        extGlBindFramebuffer(GL_FRAMEBUFFER,boundFBO);
}

inline void COpenGLExtensionHandler::extGlClearNamedFramebufferuiv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLuint* value)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlClearNamedFramebufferuiv(framebuffer, buffer, drawbuffer, value);
            return;
        }
    }

    GLint boundFBO = -1;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&boundFBO);
    if (boundFBO<0)
        return;

    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        extGlBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
    pGlClearBufferuiv(buffer, drawbuffer, value);
    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        extGlBindFramebuffer(GL_FRAMEBUFFER,boundFBO);
}

inline void COpenGLExtensionHandler::extGlClearNamedFramebufferfv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLfloat* value)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlClearNamedFramebufferfv(framebuffer, buffer, drawbuffer, value);
            return;
        }
    }

    GLint boundFBO = -1;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&boundFBO);
    if (boundFBO<0)
        return;

    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        extGlBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
    pGlClearBufferfv(buffer, drawbuffer, value);
    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        extGlBindFramebuffer(GL_FRAMEBUFFER,boundFBO);
}

inline void COpenGLExtensionHandler::extGlClearNamedFramebufferfi(GLuint framebuffer, GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlClearNamedFramebufferfi(framebuffer, buffer, drawbuffer, depth, stencil);
            return;
        }
    }

    GLint boundFBO = -1;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&boundFBO);
    if (boundFBO<0)
        return;
    extGlBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
    pGlClearBufferfi(buffer, drawbuffer, depth, stencil);
    extGlBindFramebuffer(GL_FRAMEBUFFER,boundFBO);
}


inline void COpenGLExtensionHandler::extGlVertexArrayAttribFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlVertexArrayAttribFormat)
            pGlVertexArrayAttribFormat(vaobj,attribindex,size,type,normalized,relativeoffset);
    }
    else if (!IsIntelGPU&&FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlVertexArrayVertexAttribFormatEXT)
            pGlVertexArrayVertexAttribFormatEXT(vaobj,attribindex,size,type,normalized,relativeoffset);
    }
    else if (pGlVertexAttribFormat&&pGlBindVertexArray)
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
        pGlBindVertexArray(vaobj);
        pGlVertexAttribFormat(attribindex,size,type,normalized,relativeoffset);
        pGlBindVertexArray(restoreVertexArray);
    }
}

inline void COpenGLExtensionHandler::extGlVertexArrayAttribIFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlVertexArrayAttribIFormat)
            pGlVertexArrayAttribIFormat(vaobj,attribindex,size,type,relativeoffset);
    }
    else if (!IsIntelGPU&&FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlVertexArrayVertexAttribIFormatEXT)
            pGlVertexArrayVertexAttribIFormatEXT(vaobj,attribindex,size,type,relativeoffset);
    }
    else if (pGlVertexAttribIFormat&&pGlBindVertexArray)
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
        pGlBindVertexArray(vaobj);
        pGlVertexAttribIFormat(attribindex,size,type,relativeoffset);
        pGlBindVertexArray(restoreVertexArray);
    }
}

inline void COpenGLExtensionHandler::extGlVertexArrayAttribLFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlVertexArrayAttribLFormat)
            pGlVertexArrayAttribLFormat(vaobj,attribindex,size,type,relativeoffset);
    }
    else if (!IsIntelGPU&&FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlVertexArrayVertexAttribLFormatEXT)
            pGlVertexArrayVertexAttribLFormatEXT(vaobj,attribindex,size,type,relativeoffset);
    }
    else if (pGlVertexAttribLFormat&&pGlBindVertexArray)
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
        pGlBindVertexArray(vaobj);
        pGlVertexAttribLFormat(attribindex,size,type,relativeoffset);
        pGlBindVertexArray(restoreVertexArray);
    }
}

inline void COpenGLExtensionHandler::extGlCreateQueries(GLenum target, GLsizei n, GLuint *ids)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlCreateQueries)
            pGlCreateQueries(target, n, ids);
    }
    else
    {
        if (pGlGenQueries)
            pGlGenQueries(n, ids);
    }
}
inline void COpenGLExtensionHandler::extGlGetQueryObjectuiv(GLuint id, GLenum pname, GLuint *params)
{
	if (pGlGetQueryObjectuiv)
		pGlGetQueryObjectuiv(id, pname, params);
}

inline void COpenGLExtensionHandler::extGlGetQueryObjectui64v(GLuint id, GLenum pname, GLuint64 *params)
{
	if (pGlGetQueryObjectui64v)
		pGlGetQueryObjectui64v(id, pname, params);
}

inline void COpenGLExtensionHandler::extGlGetQueryBufferObjectuiv(GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
{
    if (Version<440 && !FeatureAvailable[NBL_ARB_query_buffer_object])
    {
#ifdef _DEBuG
        os::Printer::log("GL_ARB_query_buffer_object unsupported!\n");
#endif // _DEBuG
        return;
    }

    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlGetQueryBufferObjectuiv)
            pGlGetQueryBufferObjectuiv(id, buffer, pname, offset);
    }
    else
    {
        GLint restoreQueryBuffer;
        glGetIntegerv(GL_QUERY_BUFFER_BINDING, &restoreQueryBuffer);
        pGlBindBuffer(GL_QUERY_BUFFER,id);
        if (pGlGetQueryObjectuiv)
            pGlGetQueryObjectuiv(id, pname, reinterpret_cast<GLuint*>(offset));
        pGlBindBuffer(GL_QUERY_BUFFER,restoreQueryBuffer);
    }
}

inline void COpenGLExtensionHandler::extGlGetQueryBufferObjectui64v(GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
{
    if (Version<440 && !FeatureAvailable[NBL_ARB_query_buffer_object])
    {
#ifdef _DEBuG
        os::Printer::log("GL_ARB_query_buffer_object unsupported!\n");
#endif // _DEBuG
        return;
    }

    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlGetQueryBufferObjectui64v)
            pGlGetQueryBufferObjectui64v(id, buffer, pname, offset);
    }
    else
    {
        GLint restoreQueryBuffer;
        glGetIntegerv(GL_QUERY_BUFFER_BINDING, &restoreQueryBuffer);
        pGlBindBuffer(GL_QUERY_BUFFER,id);
        if (pGlGetQueryObjectui64v)
            pGlGetQueryObjectui64v(id, pname, reinterpret_cast<GLuint64*>(offset));
        pGlBindBuffer(GL_QUERY_BUFFER,restoreQueryBuffer);
    }
}

inline void COpenGLExtensionHandler::extGlQueryCounter(GLuint id, GLenum target)
{
	if (pGlQueryCounter)
		pGlQueryCounter(id, target);
}


inline void COpenGLExtensionHandler::extGlTextureBarrier()
{
	if (FeatureAvailable[NBL_ARB_texture_barrier])
		pGlTextureBarrier();
	else if (FeatureAvailable[NBL_NV_texture_barrier])
		pGlTextureBarrierNV();
#ifdef _NBL_DEBUG
    else
        os::Printer::log("EDF_TEXTURE_BARRIER Not Available!\n",ELL_ERROR);
#endif // _NBL_DEBUG
}


inline void COpenGLExtensionHandler::extGlSwapInterval(int interval)
{
	// we have wglext, so try to use that
#if defined(_NBL_WINDOWS_API_) && defined(_NBL_COMPILE_WITH_WINDOWS_DEVICE_)
#ifdef WGL_EXT_swap_control
	if (pWglSwapIntervalEXT)
		pWglSwapIntervalEXT(interval);
#endif
#endif
#ifdef _NBL_COMPILE_WITH_X11_DEVICE_
	//TODO: Check GLX_EXT_swap_control and GLX_MESA_swap_control
#ifdef GLX_SGI_swap_control
	// does not work with interval==0
	if (interval && pGlxSwapIntervalSGI)
		pGlxSwapIntervalSGI(interval);
#elif defined(GLX_EXT_swap_control)
	Display *dpy = glXGetCurrentDisplay();
	GLXDrawable drawable = glXGetCurrentDrawable();

	if (pGlxSwapIntervalEXT)
		pGlxSwapIntervalEXT(dpy, drawable, interval);
#elif defined(GLX_MESA_swap_control)
	if (pGlxSwapIntervalMESA)
		pGlxSwapIntervalMESA(interval);
#endif
#endif
}


inline void COpenGLExtensionHandler::extGlGetInternalformativ(GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint* params)
{
        if (pGlGetInternalformativ)
            pGlGetInternalformativ(target, internalformat, pname, bufSize, params);
    
}

inline void COpenGLExtensionHandler::extGlGetInternalformati64v(GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint64* params)
{
        if (pGlGetInternalformati64v)
            pGlGetInternalformati64v(target, internalformat, pname, bufSize, params);
    
}

}
}

#endif

#endif

