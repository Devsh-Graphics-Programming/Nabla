#ifndef __NBL_C_OPENGL_FUNCTION_TABLE_H_INCLUDED__
#define __NBL_C_OPENGL_FUNCTION_TABLE_H_INCLUDED__

#include "nbl/video/COpenGLFeatureMap.h"
#include "nbl/video/IOpenGL_FunctionTable.h"

namespace nbl::video
{

class COpenGLFunctionTable final : public IOpenGL_FunctionTable
{
	using base_t = IOpenGL_FunctionTable;

public:
	using features_t = COpenGLFeatureMap;
	constexpr static inline auto EGL_API_TYPE = EGL_OPENGL_API;

	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GL4frameBuffer, OpenGLFunctionLoader
		, glBlitNamedFramebuffer
		, glCreateFramebuffers
		, glCheckNamedFramebufferStatus
		, glCheckNamedFramebufferStatusEXT
		, glFramebufferTexture
		, glNamedFramebufferTexture
		, glNamedFramebufferTextureEXT
		, glNamedFramebufferTextureLayer
		, glNamedFramebufferTextureLayerEXT
		, glFramebufferTexture1D
		, glFramebufferTexture3D
		, glNamedFramebufferTexture2DEXT
		, glNamedFramebufferDrawBuffers
		, glFramebufferDrawBuffersEXT
		, glDrawBuffer
		, glNamedFramebufferDrawBuffer
		, glFramebufferDrawBufferEXT
		, glNamedFramebufferReadBuffer
		, glFramebufferReadBufferEXT
		, glClearNamedFramebufferiv
		, glClearNamedFramebufferuiv
		, glClearNamedFramebufferfv
		, glClearNamedFramebufferfi
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GL4buffer, OpenGLFunctionLoader
		, glBindBuffersBase
		, glBindBuffersRange
		, glCreateBuffers
		, glBufferStorage
		, glNamedBufferStorage
		, glNamedBufferStorageEXT
		, glNamedBufferSubData
		, glNamedBufferSubDataEXT
		, glGetBufferSubData
		, glGetNamedBufferSubData
		, glGetNamedBufferSubDataEXT
		, glMapBuffer
		, glMapNamedBuffer
		, glMapNamedBufferEXT
		, glMapNamedBufferRange
		, glMapNamedBufferRangeEXT
		, glFlushMappedNamedBufferRange
		, glFlushMappedNamedBufferRangeEXT
		, glUnmapNamedBuffer
		, glUnmapNamedBufferEXT
		, glClearBufferData
		, glClearNamedBufferData
		, glClearNamedBufferDataEXT
		, glClearBufferSubData
		, glClearNamedBufferSubData
		, glClearNamedBufferSubDataEXT
		, glCopyNamedBufferSubData
		, glNamedCopyBufferSubDataEXT
		, glGetNamedBufferParameteri64v
		, glGetNamedBufferParameteriv
		, glGetNamedBufferParameterivEXT
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GL4texture, OpenGLFunctionLoader
		, glBindTextures
		, glCreateTextures
		, glTexStorage1D
		, glTexStorage3DMultisample
		, glTextureStorage1D
		, glTextureStorage2D
		, glTextureStorage3D
		, glTextureStorage2DMultisample
		, glTextureStorage3DMultisample
		, glTextureBuffer
		, glTextureBufferRange
		, glTextureView
		, glTextureStorage1DEXT
		, glTextureStorage2DEXT
		, glTextureStorage3DEXT
		, glTextureBufferEXT
		, glTextureBufferRangeEXT
		, glTextureStorage2DMultisampleEXT
		, glTextureStorage3DMultisampleEXT
		, glGetTextureImage
		, glGetTextureImageEXT
		, glGetCompressedTextureImage
		, glGetCompressedTextureImageEXT
		, glGetCompressedTexImage
		, glMultiTexSubImage1DEXT
		, glMultiTexSubImage2DEXT
		, glMultiTexSubImage3DEXT
		, glTexSubImage1D
		, glTextureSubImage1D
		, glTextureSubImage2D
		, glTextureSubImage3D
		, glTextureSubImage1DEXT
		, glTextureSubImage2DEXT
		, glTextureSubImage3DEXT
		, glCompressedTexSubImage1D
		, glCompressedTextureSubImage1D
		, glCompressedTextureSubImage2D
		, glCompressedTextureSubImage3D
		, glCompressedTextureSubImage1DEXT
		, glCompressedTextureSubImage2DEXT
		, glCompressedTextureSubImage3DEXT
		, glCopyImageSubData
		, glTextureParameterIuiv
		, glTextureParameterIuivEXT
		, glTexParameterIuiv
		, glGenerateTextureMipmap
		, glGenerateTextureMipmapEXT
		, glClampColor
		, glCreateSamplers
		, glBindSamplers
		, glBindImageTextures
		, glGetTextureHandleARB
		, glGetTextureSamplerHandleARB
		, glMakeTextureHandleResidentARB
		, glMakeTextureHandleNonResidentARB
		, glGetImageHandleARB
		, glMakeImageHandleResidentARB
		, glMakeImageHandleNonResidentARB
		, glIsTextureHandleResidentARB
		, glIsImageHandleResidentARB
		, glGetTextureHandleNV
		, glGetTextureSamplerHandleNV
		, glMakeTextureHandleNonResidentNV
		, glGetImageHandleNV
		, glMakeImageHandleResidentNV
		, glMakeImageHandleNonResidentNV
		, glIsTextureHandleResidentNV
		, glIsImageHandleResidentNV
		, glMakeTextureHandleResidentNV
		, glGetTexImage
		, glTextureParameteriv
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GL4shader, OpenGLFunctionLoader
		, glCreateProgramPipelines
		, glPatchParameterfv
		, glPatchParameteri
		, glPrimitiveRestartIndex
		, glProvokingVertex
		, glLogicOp
		, glPolygonMode
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GL4fragment, OpenGLFunctionLoader
		, glPointParameterf
		, glPointParameterfv
		, glBlendEquationEXT
		, glDepthRangeIndexed
		, glViewportIndexedfv
		, glScissorIndexedv
		, glMinSampleShading
		, glBlendEquationSeparatei
		, glBlendFuncSeparatei
		, glColorMaski
		, glBlendFuncIndexedAMD
		, glBlendFunciARB
		, glBlendEquationIndexedAMD
		, glBlendEquationiARB
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GL4vertex, OpenGLFunctionLoader
		, glCreateVertexArrays
		, glVertexArrayElementBuffer
		, glVertexArrayVertexBuffer
		, glVertexArrayBindVertexBufferEXT
		, glVertexArrayAttribBinding
		, glVertexArrayVertexAttribBindingEXT
		, glEnableVertexArrayAttrib
		, glEnableVertexArrayAttribEXT
		, glDisableVertexArrayAttrib
		, glDisableVertexArrayAttribEXT
		, glVertexAttribLFormat
		, glVertexArrayAttribFormat
		, glVertexArrayAttribIFormat
		, glVertexArrayAttribLFormat
		, glVertexArrayVertexAttribFormatEXT
		, glVertexArrayVertexAttribIFormatEXT
		, glVertexArrayVertexAttribLFormatEXT
		, glVertexArrayBindingDivisor
		, glVertexArrayVertexBindingDivisorEXT
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GL4drawing, OpenGLFunctionLoader
		, glDrawArraysInstancedBaseInstance
		, glDrawElementsInstancedBaseInstance
		, glDrawElementsInstancedBaseVertex
		, glDrawElementsInstancedBaseVertexBaseInstance
		, glDrawTransformFeedback
		, glDrawTransformFeedbackInstanced
		, glDrawTransformFeedbackStream
		, glDrawTransformFeedbackStreamInstanced
		, glMultiDrawArraysIndirect
		, glMultiDrawElementsIndirect
		, glMultiDrawArraysIndirectCount
		, glMultiDrawElementsIndirectCount
		, glMultiDrawArraysIndirectCountARB
		, glMultiDrawElementsIndirectCountARB
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GL4query, OpenGLFunctionLoader
		, glCreateQueries
		, glGetQueryBufferObjectuiv
		, glGetQueryBufferObjectui64v
		, glBeginConditionalRender
		, glEndConditionalRender
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GL4general, OpenGLFunctionLoader
		, glGetFloati_v
		, glGetInternalformati64v
		, glClipControl
		, glDepthRangeArrayv
		, glDepthRange
		, glViewportArrayv
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GL4_sync, OpenGLFunctionLoader
		, glTextureBarrier
		, glTextureBarrierNV
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GL4debug, OpenGLFunctionLoader
		, glDebugMessageControl
		, glDebugMessageControlARB
		, glDebugMessageCallback
		, glDebugMessageCallbackARB
		, glObjectLabel
		, glGetObjectLabel
	);

	GL4frameBuffer gl4Framebuffer;
	GL4buffer gl4Buffer;
	GL4texture gl4Texture;
	GL4shader gl4Shader;
	GL4fragment gl4Fragment;
	GL4vertex gl4Vertex;
	GL4drawing gl4Drawing;
	GL4query gl4Query;
	GL4general gl4General;
	GL4_sync gl4Sync;
	GL4debug gl4Debug;


    COpenGLFunctionTable(const egl::CEGL* _egl, const COpenGLFeatureMap* _features, system::logger_opt_smart_ptr&& logger) :
		IOpenGL_FunctionTable(_egl,_features, std::move(logger)),
		gl4Framebuffer(_egl),
		gl4Buffer(_egl),
		gl4Texture(_egl),
		gl4Shader(_egl),
		gl4Fragment(_egl),
		gl4Vertex(_egl),
		gl4Drawing(_egl),
		gl4Query(_egl),
		gl4General(_egl),
		gl4Sync(_egl),
		gl4Debug(_egl)
    {

    }

	bool isGLES() const override { return false; }

	void extGlDebugMessageControl(GLenum source, GLenum type, GLenum severity, GLsizei count, const GLuint* ids, GLboolean enabled) override
	{
		if (gl4Debug.pglDebugMessageControl)
			gl4Debug.pglDebugMessageControl(source, type, severity, count, ids, enabled);
		else if (gl4Debug.pglDebugMessageControlARB)
			gl4Debug.pglDebugMessageControlARB(source, type, severity, count, ids, enabled);
	}
	void extGlDebugMessageCallback(GLDebugCallbackType callback, const void* userParam) override
	{
		if (gl4Debug.pglDebugMessageCallback)
			gl4Debug.pglDebugMessageCallback(callback, userParam);
		else if (gl4Debug.pglDebugMessageCallbackARB)
			gl4Debug.pglDebugMessageCallbackARB(callback, userParam);
	}

	void extGlBindTextures(const GLuint& first, const GLsizei& count, const GLuint* textures, const GLenum* targets) override
	{
		const GLenum supportedTargets[] = { GL_TEXTURE_1D,GL_TEXTURE_2D, // GL 1.x
									GL_TEXTURE_3D,GL_TEXTURE_RECTANGLE,GL_TEXTURE_CUBE_MAP, // GL 2.x
									GL_TEXTURE_1D_ARRAY,GL_TEXTURE_2D_ARRAY,GL_TEXTURE_BUFFER, // GL 3.x
									GL_TEXTURE_CUBE_MAP_ARRAY,GL_TEXTURE_2D_MULTISAMPLE,GL_TEXTURE_2D_MULTISAMPLE_ARRAY }; // GL 4.x

		if (features->Version >= 440 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_multi_bind])
			gl4Texture.pglBindTextures(first, count, textures);
		else
		{
			int32_t activeTex = 0;
			glGeneral.pglGetIntegerv(GL_ACTIVE_TEXTURE, &activeTex);

			for (GLsizei i = 0; i < count; i++)
			{
				GLuint texture = textures ? textures[i] : 0;

				GLuint unit = first + i;
				glTexture.pglActiveTexture(GL_TEXTURE0 + unit);

				if (texture)
					glTexture.pglBindTexture(targets[i], texture);
				else
				{
					for (size_t j = 0; j < sizeof(supportedTargets) / sizeof(GLenum); j++)
						glTexture.pglBindTexture(supportedTargets[j], 0);
				}
			}

			glTexture.pglActiveTexture(activeTex);
		}
	}

	void extGlCreateTextures(GLenum target, GLsizei n, GLuint* textures) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			gl4Texture.pglCreateTextures(target, n, textures);
		else
			base_t::extGlCreateTextures(target, n, textures);
	}

	void extGlTextureView(GLuint texture, GLenum target, GLuint origtexture, GLenum internalformat, GLuint minlevel, GLuint numlevels, GLuint minlayer, GLuint numlayers) override
	{
		if (gl4Texture.pglTextureView)
			gl4Texture.pglTextureView(texture, target, origtexture, internalformat, minlevel, numlevels, minlayer, numlayers);
	}

	void extGlTextureBuffer(GLuint texture, GLenum internalformat, GLuint buffer) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			gl4Texture.pglTextureBuffer(texture, internalformat, buffer);
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
			gl4Texture.pglTextureBufferEXT(texture, GL_TEXTURE_BUFFER, internalformat, buffer);
		else
		{
			GLint bound;
			glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_BUFFER, &bound);
			glTexture.pglBindTexture(GL_TEXTURE_BUFFER, texture);
			glTexture.pglTexBuffer(GL_TEXTURE_BUFFER, internalformat, buffer);
			glTexture.pglBindTexture(GL_TEXTURE_BUFFER, bound);
		}
	}

	void extGlTextureBufferRange(GLuint texture, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizei length) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{

			if (gl4Texture.pglTextureBufferRange)
				gl4Texture.pglTextureBufferRange(texture, internalformat, buffer, offset, length);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Texture.pglTextureBufferRangeEXT)
				gl4Texture.pglTextureBufferRangeEXT(texture, GL_TEXTURE_BUFFER, internalformat, buffer, offset, length);
		}
		else
		{
			GLint bound;
			glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_BUFFER, &bound);
			glTexture.pglBindTexture(GL_TEXTURE_BUFFER, texture);
			if (glTexture.pglTexBufferRange)
				glTexture.pglTexBufferRange(GL_TEXTURE_BUFFER, internalformat, buffer, offset, length);
			glTexture.pglBindTexture(GL_TEXTURE_BUFFER, bound);
		}
	}

	void extGlTextureStorage2D(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Texture.pglTextureStorage2D)
				gl4Texture.pglTextureStorage2D(texture, levels, internalformat, width, height);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Texture.pglTextureStorage2DEXT)
				gl4Texture.pglTextureStorage2DEXT(texture, target, levels, internalformat, width, height);
		}
		else if (glTexture.pglTexStorage2D)
		{
			GLint bound;
			switch (target)
			{
			case GL_TEXTURE_1D_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_1D_ARRAY, &bound);
				break;
			case GL_TEXTURE_2D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
				break;
			case GL_TEXTURE_RECTANGLE:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE, &bound);
				break;
			default:
				m_logger.log("DevSH would like to ask you what are you doing!!??\n", system::ILogger::ELL_ERROR);
				return;
			}
			glTexture.pglBindTexture(target, texture);
			glTexture.pglTexStorage2D(target, levels, internalformat, width, height);
			glTexture.pglBindTexture(target, bound);
		}
	}

	void extGlTextureStorage3D(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			gl4Texture.pglTextureStorage3D(texture, levels, internalformat, width, height, depth);
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
			gl4Texture.pglTextureStorage3DEXT(texture, target, levels, internalformat, width, height, depth);
		else
		{
			base_t::extGlTextureStorage3D(texture, target, levels, internalformat, width, height, depth);
		}
	}

	void extGlTextureStorage2DMultisample(GLuint texture, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			gl4Texture.pglTextureStorage2DMultisample(texture, samples, internalformat, width, height, fixedsamplelocations);
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
			gl4Texture.pglTextureStorage2DMultisampleEXT(texture, target, samples, internalformat, width, height, fixedsamplelocations);
		else
		{
			base_t::extGlTextureStorage2DMultisample(texture, target, samples, internalformat, width, height, fixedsamplelocations);
		}
	}

	void extGlTextureStorage3DMultisample(GLuint texture, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			gl4Texture.pglTextureStorage3DMultisample(texture, samples, internalformat, width, height, depth, fixedsamplelocations);
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
			gl4Texture.pglTextureStorage3DMultisampleEXT(texture, target, samples, internalformat, width, height, depth, fixedsamplelocations);
		else
		{
			base_t::extGlTextureStorage3DMultisample(texture, target, samples, internalformat, width, height, depth, fixedsamplelocations);
		}
	}

	void extGlTextureSubImage2D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void* pixels) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			gl4Texture.pglTextureSubImage2D(texture, level, xoffset, yoffset, width, height, format, type, pixels);
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
			gl4Texture.pglTextureSubImage2DEXT(texture, target, level, xoffset, yoffset, width, height, format, type, pixels);
		else
		{
			GLint bound;
			switch (target)
			{
			case GL_TEXTURE_1D_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_1D_ARRAY, &bound);
				break;
			case GL_TEXTURE_2D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D, &bound);
				break;
			case GL_TEXTURE_2D_MULTISAMPLE:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
				break;
			case GL_TEXTURE_RECTANGLE:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE, &bound);
				break;
			default:
				m_logger.log("DevSH would like to ask you what are you doing!!??\n", system::ILogger::ELL_ERROR);
				return;
			}
			glTexture.pglBindTexture(target, texture);
			glTexture.pglTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixels);
			glTexture.pglBindTexture(target, bound);
		}
	}

	void extGlTextureSubImage3D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void* pixels) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			gl4Texture.pglTextureSubImage3D(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
			gl4Texture.pglTextureSubImage3DEXT(texture, target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
		else
		{
			base_t::extGlTextureSubImage3D(texture, target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
		}
	}

	void extGlCompressedTextureSubImage2D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void* data) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Texture.pglCompressedTextureSubImage2D)
				gl4Texture.pglCompressedTextureSubImage2D(texture, level, xoffset, yoffset, width, height, format, imageSize, data);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Texture.pglCompressedTextureSubImage2DEXT)
				gl4Texture.pglCompressedTextureSubImage2DEXT(texture, target, level, xoffset, yoffset, width, height, format, imageSize, data);
		}
		else if (glTexture.pglCompressedTexSubImage2D)
		{
			GLint bound;
			switch (target)
			{
			case GL_TEXTURE_1D_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_1D_ARRAY, &bound);
				break;
			case GL_TEXTURE_2D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
				break;
			default:
				m_logger.log("DevSH would like to ask you what are you doing!!??\n", system::ILogger::ELL_ERROR);
				return;
			}
			glTexture.pglBindTexture(target, texture);
			glTexture.pglCompressedTexSubImage2D(target, level, xoffset, yoffset, width, height, format, imageSize, data);
			glTexture.pglBindTexture(target, bound);
		}
	}

	void extGlCompressedTextureSubImage3D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void* data) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Texture.pglCompressedTextureSubImage3D)
				gl4Texture.pglCompressedTextureSubImage3D(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Texture.pglCompressedTextureSubImage3DEXT)
				gl4Texture.pglCompressedTextureSubImage3DEXT(texture, target, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
		}
		else if (glTexture.pglCompressedTexSubImage3D)
		{
			base_t::extGlCompressedTextureSubImage3D(texture, target, level, xoffset, yoffset, zoffset, height, width, depth, format, imageSize, data);
		}
	}

	void extGlGenerateTextureMipmap(GLuint texture, GLenum target) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Texture.pglGenerateTextureMipmap)
				gl4Texture.pglGenerateTextureMipmap(texture);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Texture.pglGenerateTextureMipmapEXT)
				gl4Texture.pglGenerateTextureMipmapEXT(texture, target);
		}
		else
		{
			GLint bound;
			switch (target)
			{
			case GL_TEXTURE_1D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_1D, &bound);
				break;
			case GL_TEXTURE_1D_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_1D_ARRAY, &bound);
				break;
			case GL_TEXTURE_2D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D, &bound);
				break;
			case GL_TEXTURE_2D_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &bound);
				break;
			case GL_TEXTURE_2D_MULTISAMPLE:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE, &bound);
				break;
			case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY, &bound);
				break;
			case GL_TEXTURE_3D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_3D, &bound);
				break;
			case GL_TEXTURE_BUFFER:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_BUFFER, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP_ARRAY, &bound);
				break;
			case GL_TEXTURE_RECTANGLE:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE, &bound);
				break;
			default:
				m_logger.log("DevSH would like to ask you what are you doing!!??\n", system::ILogger::ELL_ERROR);
				return;
			}
			glTexture.pglBindTexture(target, texture);
			glTexture.pglGenerateMipmap(target);
			glTexture.pglBindTexture(target, bound);
		}
	}

	void extGlCreateSamplers(GLsizei n, GLuint* samplers) override
	{
		if (gl4Texture.pglCreateSamplers)
			gl4Texture.pglCreateSamplers(n, samplers);
		else
			base_t::extGlCreateSamplers(n, samplers);
	}

	void extGlBindSamplers(const GLuint& first, const GLsizei& count, const GLuint* samplers) override
	{
		if (features->Version >= 440 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_multi_bind])
		{
			if (gl4Texture.pglBindSamplers)
				gl4Texture.pglBindSamplers(first, count, samplers);
		}
		else
		{
			base_t::extGlBindSamplers(first, count, samplers);
		}
	}

	void extGlBindImageTextures(GLuint first, GLsizei count, const GLuint* textures, const GLenum* formats) override
	{
		// TODO: File a bug report with NVidia about this breaking on a mobile GTX 1050 4GB with driver 471
		//if (gl4Texture.pglBindImageTextures)
		//	gl4Texture.pglBindImageTextures(first, count, textures);
		//else
			base_t::extGlBindImageTextures(first, count, textures, formats);
	}

	void extGlCreateFramebuffers(GLsizei n, GLuint* framebuffers) override
	{
		if (gl4Framebuffer.pglCreateFramebuffers)
			gl4Framebuffer.pglCreateFramebuffers(n, framebuffers);
		else
			base_t::extGlCreateFramebuffers(n, framebuffers);
	}

	GLenum extGlCheckNamedFramebufferStatus(GLuint framebuffer, GLenum target) override
	{
		if (!features->needsDSAFramebufferHack)
		{
			if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
				return gl4Framebuffer.pglCheckNamedFramebufferStatus(framebuffer, target);
			else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
				return gl4Framebuffer.pglCheckNamedFramebufferStatusEXT(framebuffer, target);
		}
		return base_t::extGlCheckNamedFramebufferStatus(framebuffer, target);
	}

	void extGlNamedFramebufferTexture(GLuint framebuffer, GLenum attachment, GLuint texture, GLint level, GLenum textureType) override
	{
		if (!features->needsDSAFramebufferHack)
		{
			if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			{
				gl4Framebuffer.pglNamedFramebufferTexture(framebuffer, attachment, texture, level);
				return;
			}
			else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
			{
				gl4Framebuffer.pglNamedFramebufferTextureEXT(framebuffer, attachment, texture, level);
				return;
			}
		}
		else base_t::extGlNamedFramebufferTexture(framebuffer, attachment, texture, level, textureType);
	}

	void extGlNamedFramebufferTextureLayer(GLuint framebuffer, GLenum attachment, GLuint texture, GLenum textureType, GLint level, GLint layer) override
	{
		if (!features->needsDSAFramebufferHack)
		{
			if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			{
				gl4Framebuffer.pglNamedFramebufferTextureLayer(framebuffer, attachment, texture, level, layer);
				return;
			}
		}

		if (textureType != GL_TEXTURE_CUBE_MAP)
		{
			if (!features->needsDSAFramebufferHack && features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
			{
				gl4Framebuffer.pglNamedFramebufferTextureLayerEXT(framebuffer, attachment, texture, level, layer);
			}
			else
			{
				GLuint bound;
				glGeneral.pglGetIntegerv(GL_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(&bound));

				if (bound != framebuffer)
					glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
				glFramebuffer.pglFramebufferTextureLayer(GL_FRAMEBUFFER, attachment, texture, level, layer);
				if (bound != framebuffer)
					glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, bound);
			}
		}
		else
		{
			constexpr GLenum CubeMapFaceToCubeMapFaceGLenum[] = {
				GL_TEXTURE_CUBE_MAP_POSITIVE_X,GL_TEXTURE_CUBE_MAP_NEGATIVE_X,GL_TEXTURE_CUBE_MAP_POSITIVE_Y,GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,GL_TEXTURE_CUBE_MAP_POSITIVE_Z,GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
			};

			if (!features->needsDSAFramebufferHack && features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
			{
				gl4Framebuffer.pglNamedFramebufferTexture2DEXT(framebuffer, attachment, CubeMapFaceToCubeMapFaceGLenum[layer], texture, level);
			}
			else
			{
				GLuint bound;
				glGeneral.pglGetIntegerv(GL_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(&bound));

				if (bound != framebuffer)
					glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
				glFramebuffer.pglFramebufferTexture2D(GL_FRAMEBUFFER, attachment, CubeMapFaceToCubeMapFaceGLenum[layer], texture, level);
				if (bound != framebuffer)
					glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, bound);
			}
		}
	}

	void extGlBlitNamedFramebuffer(GLuint readFramebuffer, GLuint drawFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter) override
	{
		if (!features->needsDSAFramebufferHack)
		{
			if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			{
				gl4Framebuffer.pglBlitNamedFramebuffer(readFramebuffer, drawFramebuffer, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
				return;
			}
		}

		base_t::extGlBlitNamedFramebuffer(readFramebuffer, drawFramebuffer, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
	}

	void extGlNamedFramebufferReadBuffer(GLuint framebuffer, GLenum mode) override
	{
		if (!features->needsDSAFramebufferHack)
		{
			if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			{
				gl4Framebuffer.pglNamedFramebufferReadBuffer(framebuffer, mode);
				return;
			}
			else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
			{
				gl4Framebuffer.pglFramebufferReadBufferEXT(framebuffer, mode);
				return;
			}
		}

		base_t::extGlNamedFramebufferReadBuffer(framebuffer, mode);
	}

	void extGlNamedFramebufferDrawBuffer(GLuint framebuffer, GLenum buf) override
	{
		if (!features->needsDSAFramebufferHack)
		{
			if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			{
				gl4Framebuffer.pglNamedFramebufferDrawBuffer(framebuffer, buf);
				return;
			}
			else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
			{
				gl4Framebuffer.pglFramebufferDrawBufferEXT(framebuffer, buf);
				return;
			}
		}

		GLint boundFBO;
		glGeneral.pglGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &boundFBO);

		if (static_cast<GLuint>(boundFBO) != framebuffer)
			glFramebuffer.pglBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
		gl4Framebuffer.pglDrawBuffer(buf);
		if (static_cast<GLuint>(boundFBO) != framebuffer)
			glFramebuffer.pglBindFramebuffer(GL_DRAW_FRAMEBUFFER, boundFBO);
	}

	void extGlNamedFramebufferDrawBuffers(GLuint framebuffer, GLsizei n, const GLenum* bufs) override
	{
		if (!features->needsDSAFramebufferHack)
		{
			if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			{
				gl4Framebuffer.pglNamedFramebufferDrawBuffers(framebuffer, n, bufs);
				return;
			}
			else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
			{
				gl4Framebuffer.pglFramebufferDrawBuffersEXT(framebuffer, n, bufs);
				return;
			}
		}

		base_t::extGlNamedFramebufferDrawBuffers(framebuffer, n, bufs);
	}

	void extGlClearNamedFramebufferiv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLint* value) override
	{
		if (!features->needsDSAFramebufferHack)
		{
			if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			{
				gl4Framebuffer.pglClearNamedFramebufferiv(framebuffer, buffer, drawbuffer, value);
				return;
			}
		}

		base_t::extGlClearNamedFramebufferiv(framebuffer, buffer, drawbuffer, value);
	}

	void extGlClearNamedFramebufferuiv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLuint* value) override
	{
		if (!features->needsDSAFramebufferHack)
		{
			if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			{
				gl4Framebuffer.pglClearNamedFramebufferuiv(framebuffer, buffer, drawbuffer, value);
				return;
			}
		}

		base_t::extGlClearNamedFramebufferuiv(framebuffer, buffer, drawbuffer, value);
	}

	void extGlClearNamedFramebufferfv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLfloat* value) override
	{
		if (!features->needsDSAFramebufferHack)
		{
			if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			{
				gl4Framebuffer.pglClearNamedFramebufferfv(framebuffer, buffer, drawbuffer, value);
				return;
			}
		}

		base_t::extGlClearNamedFramebufferfv(framebuffer, buffer, drawbuffer, value);
	}

	void extGlClearNamedFramebufferfi(GLuint framebuffer, GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil) override
	{
		if (!features->needsDSAFramebufferHack)
		{
			if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			{
				gl4Framebuffer.pglClearNamedFramebufferfi(framebuffer, buffer, drawbuffer, depth, stencil);
				return;
			}
		}

		base_t::extGlClearNamedFramebufferfi(framebuffer, buffer, drawbuffer, depth, stencil);
	}

	void extGlCreateBuffers(GLsizei n, GLuint* buffers) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Buffer.pglCreateBuffers)
				gl4Buffer.pglCreateBuffers(n, buffers);
			else if (buffers)
				memset(buffers, 0, n * sizeof(GLuint));
		}
		else base_t::extGlCreateBuffers(n, buffers);
	}

	void extGlBindBuffersBase(const GLenum& target, const GLuint& first, const GLsizei& count, const GLuint* buffers) override
	{
		if (features->Version >= 440 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_multi_bind])
		{
			if (gl4Buffer.pglBindBuffersBase)
				gl4Buffer.pglBindBuffersBase(target, first, count, buffers);
		}
		else base_t::extGlBindBuffersBase(target, first, count, buffers);
	}

	void extGlBindBuffersRange(const GLenum& target, const GLuint& first, const GLsizei& count, const GLuint* buffers, const GLintptr* offsets, const GLsizeiptr* sizes) override
	{
		if (features->Version >= 440 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_multi_bind])
		{
			if (gl4Buffer.pglBindBuffersRange)
				gl4Buffer.pglBindBuffersRange(target, first, count, buffers, offsets, sizes);
		}
		else base_t::extGlBindBuffersRange(target, first, count, buffers, offsets, sizes);
	}

	void extGlNamedBufferStorage(GLuint buffer, GLsizeiptr size, const void* data, GLbitfield flags) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Buffer.pglNamedBufferStorage)
				gl4Buffer.pglNamedBufferStorage(buffer, size, data, flags);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Buffer.pglNamedBufferStorageEXT)
				gl4Buffer.pglNamedBufferStorageEXT(buffer, size, data, flags);
		}
		/*
		else if (gl4Buffer.pglBufferStorage && glBuffer.pglBindBuffer)
		{
			GLint bound;
			glGeneral.pglGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bound);
			glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, buffer);
			gl4Buffer.pglBufferStorage(GL_ARRAY_BUFFER, size, data, flags);
			glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, bound);
		}
		*/
	}

	void extGlNamedBufferSubData(GLuint buffer, GLintptr offset, GLsizeiptr size, const void* data) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Buffer.pglNamedBufferSubData)
				gl4Buffer.pglNamedBufferSubData(buffer, offset, size, data);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Buffer.pglNamedBufferSubDataEXT)
				gl4Buffer.pglNamedBufferSubDataEXT(buffer, offset, size, data);
		}
		//else base_t::extGlNamedBufferSubData(buffer, offset, size, data);
	}

	void* extGlMapNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length, GLbitfield access) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Buffer.pglMapNamedBufferRange)
				return gl4Buffer.pglMapNamedBufferRange(buffer, offset, length, access);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Buffer.pglMapNamedBufferRangeEXT)
				return gl4Buffer.pglMapNamedBufferRangeEXT(buffer, offset, length, access);
		}
		return base_t::extGlMapNamedBufferRange(buffer, offset, length, access);
	}

	void extGlFlushMappedNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Buffer.pglFlushMappedNamedBufferRange)
				gl4Buffer.pglFlushMappedNamedBufferRange(buffer, offset, length);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Buffer.pglFlushMappedNamedBufferRangeEXT)
				gl4Buffer.pglFlushMappedNamedBufferRangeEXT(buffer, offset, length);
		}
		//else base_t::extGlFlushMappedNamedBufferRange(buffer, offset, length);
	}

	GLboolean extGlUnmapNamedBuffer(GLuint buffer) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Buffer.pglUnmapNamedBuffer)
				return gl4Buffer.pglUnmapNamedBuffer(buffer);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Buffer.pglUnmapNamedBufferEXT)
				return gl4Buffer.pglUnmapNamedBufferEXT(buffer);
		}
		return GL_FALSE;
	}

	void extGlClearNamedBufferData(GLuint buffer, GLenum internalformat, GLenum format, GLenum type, const void* data) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Buffer.pglClearNamedBufferData)
				gl4Buffer.pglClearNamedBufferData(buffer, internalformat, format, type, data);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Buffer.pglClearNamedBufferDataEXT)
				gl4Buffer.pglClearNamedBufferDataEXT(buffer, internalformat, format, type, data);
		}
		/*
		else if (gl4Buffer.pglClearBufferData && glBuffer.pglBindBuffer)
		{
			GLint bound;
			glGeneral.pglGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bound);
			glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, buffer);
			gl4Buffer.pglClearBufferData(GL_ARRAY_BUFFER, internalformat, format, type, data);
			glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, bound);
		}
		*/
	}

	void extGlClearNamedBufferSubData(GLuint buffer, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type, const void* data) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Buffer.pglClearNamedBufferSubData)
				gl4Buffer.pglClearNamedBufferSubData(buffer, internalformat, offset, size, format, type, data);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Buffer.pglClearNamedBufferSubDataEXT)
				gl4Buffer.pglClearNamedBufferSubDataEXT(buffer, internalformat, offset, size, format, type, data);
		}
		/*
		else if (gl4Buffer.pglClearBufferSubData && glBuffer.pglBindBuffer)
		{
			GLint bound;
			glGeneral.pglGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bound);
			glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, buffer);
			gl4Buffer.pglClearBufferSubData(GL_ARRAY_BUFFER, internalformat, offset, size, format, type, data);
			glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, bound);
		}
		*/
	}

	void extGlCopyNamedBufferSubData(GLuint readBuffer, GLuint writeBuffer, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Buffer.pglCopyNamedBufferSubData)
				gl4Buffer.pglCopyNamedBufferSubData(readBuffer, writeBuffer, readOffset, writeOffset, size);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Buffer.pglNamedCopyBufferSubDataEXT)
				gl4Buffer.pglNamedCopyBufferSubDataEXT(readBuffer, writeBuffer, readOffset, writeOffset, size);
		}
		/*
		else
		{
			base_t::extGlCopyNamedBufferSubData(readBuffer, writeBuffer, readOffset, writeOffset, size);
		}
		*/
	}

	void extGlGetNamedBufferParameteriv(const GLuint& buffer, const GLenum& value, GLint* data) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Buffer.pglGetNamedBufferParameteriv)
				gl4Buffer.pglGetNamedBufferParameteriv(buffer, value, data);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Buffer.pglGetNamedBufferParameterivEXT)
				gl4Buffer.pglGetNamedBufferParameterivEXT(buffer, value, data);
		}
		/*
		else
		{
			base_t::extGlGetNamedBufferParameteriv(buffer, value, data);
		}
		*/
	}

	void extGlGetNamedBufferParameteri64v(const GLuint& buffer, const GLenum& value, GLint64* data) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Buffer.pglGetNamedBufferParameteri64v)
				gl4Buffer.pglGetNamedBufferParameteri64v(buffer, value, data);
		}
		/*
		else
		{
			base_t::extGlGetNamedBufferParameteri64v(buffer, value, data);
		}
		*/
	}

	void extGlCreateVertexArrays(GLsizei n, GLuint* arrays) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Vertex.pglCreateVertexArrays)
				gl4Vertex.pglCreateVertexArrays(n, arrays);
		}
		else
		{
			base_t::extGlCreateVertexArrays(n, arrays);
		}
	}

	void extGlVertexArrayElementBuffer(GLuint vaobj, GLuint buffer) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Vertex.pglVertexArrayElementBuffer)
				gl4Vertex.pglVertexArrayElementBuffer(vaobj, buffer);
		}
		else
		{
			base_t::extGlVertexArrayElementBuffer(vaobj, buffer);
		}
	}

	void extGlVertexArrayVertexBuffer(GLuint vaobj, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Vertex.pglVertexArrayVertexBuffer)
				gl4Vertex.pglVertexArrayVertexBuffer(vaobj, bindingindex, buffer, offset, stride);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Vertex.pglVertexArrayBindVertexBufferEXT)
				gl4Vertex.pglVertexArrayBindVertexBufferEXT(vaobj, bindingindex, buffer, offset, stride);
		}
		else
		{
			base_t::extGlVertexArrayVertexBuffer(vaobj, bindingindex, buffer, offset, stride);
		}
	}

	void extGlVertexArrayAttribBinding(GLuint vaobj, GLuint attribindex, GLuint bindingindex) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Vertex.pglVertexArrayAttribBinding)
				gl4Vertex.pglVertexArrayAttribBinding(vaobj, attribindex, bindingindex);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Vertex.pglVertexArrayVertexAttribBindingEXT)
				gl4Vertex.pglVertexArrayVertexAttribBindingEXT(vaobj, attribindex, bindingindex);
		}
		else
		{
			base_t::extGlVertexArrayAttribBinding(vaobj, attribindex, bindingindex);
		}
	}

	void extGlEnableVertexArrayAttrib(GLuint vaobj, GLuint index) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Vertex.pglEnableVertexArrayAttrib)
				gl4Vertex.pglEnableVertexArrayAttrib(vaobj, index);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Vertex.pglEnableVertexArrayAttribEXT)
				gl4Vertex.pglEnableVertexArrayAttribEXT(vaobj, index);
		}
		else
		{
			base_t::extGlEnableVertexArrayAttrib(vaobj, index);
		}
	}

	void extGlDisableVertexArrayAttrib(GLuint vaobj, GLuint index) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Vertex.pglDisableVertexArrayAttrib)
				gl4Vertex.pglDisableVertexArrayAttrib(vaobj, index);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Vertex.pglDisableVertexArrayAttribEXT)
				gl4Vertex.pglDisableVertexArrayAttribEXT(vaobj, index);
		}
		else
		{
			base_t::extGlDisableVertexArrayAttrib(vaobj, index);
		}
	}

	void extGlVertexArrayAttribFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Vertex.pglVertexArrayAttribFormat)
				gl4Vertex.pglVertexArrayAttribFormat(vaobj, attribindex, size, type, normalized, relativeoffset);
		}
		else if (!features->isIntelGPU && features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Vertex.pglVertexArrayVertexAttribFormatEXT)
				gl4Vertex.pglVertexArrayVertexAttribFormatEXT(vaobj, attribindex, size, type, normalized, relativeoffset);
		}
		else
		{
			base_t::extGlVertexArrayAttribFormat(vaobj, attribindex, size, type, normalized, relativeoffset);
		}
	}

	void extGlVertexArrayAttribIFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Vertex.pglVertexArrayAttribIFormat)
				gl4Vertex.pglVertexArrayAttribIFormat(vaobj, attribindex, size, type, relativeoffset);
		}
		else if (!features->isIntelGPU && features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Vertex.pglVertexArrayVertexAttribIFormatEXT)
				gl4Vertex.pglVertexArrayVertexAttribIFormatEXT(vaobj, attribindex, size, type, relativeoffset);
		}
		else
		{
			base_t::extGlVertexArrayAttribIFormat(vaobj, attribindex, size, type, relativeoffset);
		}
	}

	void extGlVertexArrayBindingDivisor(GLuint vaobj, GLuint bindingindex, GLuint divisor) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Vertex.pglVertexArrayBindingDivisor)
				gl4Vertex.pglVertexArrayBindingDivisor(vaobj, bindingindex, divisor);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Vertex.pglVertexArrayVertexBindingDivisorEXT)
				gl4Vertex.pglVertexArrayVertexBindingDivisorEXT(vaobj, bindingindex, divisor);
		}
		else
		{
			base_t::extGlVertexArrayBindingDivisor(vaobj, bindingindex, divisor);
		}
	}

	void extGlCreateQueries(GLenum target, GLsizei n, GLuint* ids) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Query.pglCreateQueries)
				gl4Query.pglCreateQueries(target, n, ids);
		}
		else
		{
			base_t::extGlCreateQueries(target, n, ids);
		}
	}

	void extGlTextureParameteriv(GLuint texture, GLenum target, GLenum pname, const GLint* params) override
	{
		if (gl4Texture.pglTextureParameteriv)
		{
			gl4Texture.pglTextureParameteriv(texture, pname, params);

			return;
		}

		GLint bound;
		switch (target)
		{
		case GL_TEXTURE_1D:
			glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_1D, &bound);
			break;
		case GL_TEXTURE_1D_ARRAY:
			glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_1D_ARRAY, &bound);
			break;
		case GL_TEXTURE_2D:
			glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D, &bound);
			break;
		case GL_TEXTURE_CUBE_MAP:
			glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
			break;
		case GL_TEXTURE_2D_ARRAY:
			glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &bound);
			break;
		case GL_TEXTURE_3D:
			glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_3D, &bound);
			break;
		default:
			m_logger.log("DevSH would like to ask you what are you doing!!??\n", system::ILogger::ELL_ERROR);
			return;
		}
		glTexture.pglBindTexture(target, texture);
		glTexture.pglTexParameteriv(target, pname, params);
		glTexture.pglBindTexture(target, bound);
	}

	////////////////
	// GL-exclusive functions
	////////////////

	void extGlTextureStorage1D(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width)
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Texture.pglTextureStorage1D)
				gl4Texture.pglTextureStorage1D(texture, levels, internalformat, width);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Texture.pglTextureStorage1DEXT)
				gl4Texture.pglTextureStorage1DEXT(texture, target, levels, internalformat, width);
		}
		else if (gl4Texture.pglTexStorage1D)
		{
			GLint bound;
			switch (target)
			{
			case GL_TEXTURE_1D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_1D, &bound);
				break;
			default:
				m_logger.log("DevSH would like to ask you what are you doing!!??\n", system::ILogger::ELL_ERROR);
				return;
			}
			glTexture.pglBindTexture(target, texture);
			gl4Texture.pglTexStorage1D(target, levels, internalformat, width);
			glTexture.pglBindTexture(target, bound);
		}
	}

	void extGlGetTextureImage(GLuint texture, GLenum target, GLint level, GLenum format, GLenum type, GLsizei bufSizeHint, void* pixels)
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			gl4Texture.pglGetTextureImage(texture, level, format, type, bufSizeHint, pixels);
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
			gl4Texture.pglGetTextureImageEXT(texture, target, level, format, type, pixels);
		else
		{
			GLint bound = 0;
			switch (target)
			{
			case GL_TEXTURE_1D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_1D, &bound);
				break;
			case GL_TEXTURE_1D_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_1D_ARRAY, &bound);
				break;
			case GL_TEXTURE_2D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
			case GL_TEXTURE_CUBE_MAP:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
				break;
			case GL_TEXTURE_RECTANGLE:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE, &bound);
				break;
			case GL_TEXTURE_2D_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP_ARRAY, &bound);
				break;
			case GL_TEXTURE_3D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_3D, &bound);
				break;
			default:
				break;
			}
			glTexture.pglBindTexture(target, texture);
			gl4Texture.pglGetTexImage(target, level, format, type, pixels);
			glTexture.pglBindTexture(target, bound);
		}
	}
	void extGlGetCompressedTextureImage(GLuint texture, GLenum target, GLint level, GLsizei bufSizeHint, void* pixels)
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			gl4Texture.pglGetCompressedTextureImage(texture, level, bufSizeHint, pixels);
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
			gl4Texture.pglGetCompressedTextureImageEXT(texture, target, level, pixels);
		else
		{
			GLint bound = 0;
			switch (target)
			{
			case GL_TEXTURE_1D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_1D, &bound);
				break;
			case GL_TEXTURE_1D_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_1D_ARRAY, &bound);
				break;
			case GL_TEXTURE_2D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
			case GL_TEXTURE_CUBE_MAP:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
				break;
			case GL_TEXTURE_RECTANGLE:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE, &bound);
				break;
			case GL_TEXTURE_2D_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP_ARRAY, &bound);
				break;
			case GL_TEXTURE_3D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_3D, &bound);
				break;
			default:
				break;
			}
			glTexture.pglBindTexture(target, texture);
			gl4Texture.pglGetCompressedTexImage(target, level, pixels);
			glTexture.pglBindTexture(target, bound);
		}
	}
	void extGlTextureSubImage1D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void* pixels)
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			gl4Texture.pglTextureSubImage1D(texture, level, xoffset, width, format, type, pixels);
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
			gl4Texture.pglTextureSubImage1DEXT(texture, target, level, xoffset, width, format, type, pixels);
		else
		{
			GLint bound;
			switch (target)
			{
			case GL_TEXTURE_1D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_1D, &bound);
				break;
			default:
				m_logger.log("DevSH would like to ask you what are you doing!!??\n", system::ILogger::ELL_ERROR);
				return;
			}
			glTexture.pglBindTexture(target, texture);
			gl4Texture.pglTexSubImage1D(target, level, xoffset, width, format, type, pixels);
			glTexture.pglBindTexture(target, bound);
		}
	}
	void extGlCompressedTextureSubImage1D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void* data)
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Texture.pglCompressedTextureSubImage1D)
				gl4Texture.pglCompressedTextureSubImage1D(texture, level, xoffset, width, format, imageSize, data);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Texture.pglCompressedTextureSubImage1DEXT)
				gl4Texture.pglCompressedTextureSubImage1DEXT(texture, target, level, xoffset, width, format, imageSize, data);
		}
		else if (gl4Texture.pglCompressedTexSubImage1D)
		{
			GLint bound;
			switch (target)
			{
			case GL_TEXTURE_1D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_1D, &bound);
				break;
			default:
				m_logger.log("DevSH would like to ask you what are you doing!!??\n", system::ILogger::ELL_ERROR);
				return;
			}
			glTexture.pglBindTexture(target, texture);
			gl4Texture.pglCompressedTexSubImage1D(target, level, xoffset, width, format, imageSize, data);
			glTexture.pglBindTexture(target, bound);
		}
	}
	void extGlTextureParameterIuiv(GLuint texture, GLenum target, GLenum pname, const GLuint* params)
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
			gl4Texture.pglTextureParameterIuiv(texture, pname, params);
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
			gl4Texture.pglTextureParameterIuivEXT(texture, target, pname, params);
		else
		{
			GLint bound;
			switch (target)
			{
			case GL_TEXTURE_1D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_1D, &bound);
				break;
			case GL_TEXTURE_1D_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_1D_ARRAY, &bound);
				break;
			case GL_TEXTURE_2D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D, &bound);
				break;
			case GL_TEXTURE_2D_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &bound);
				break;
			case GL_TEXTURE_2D_MULTISAMPLE:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE, &bound);
				break;
			case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY, &bound);
				break;
			case GL_TEXTURE_3D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_3D, &bound);
				break;
			case GL_TEXTURE_BUFFER:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_BUFFER, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP_ARRAY, &bound);
				break;
			case GL_TEXTURE_RECTANGLE:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE, &bound);
				break;
			default:
				m_logger.log("DevSH would like to ask you what are you doing!!??\n", system::ILogger::ELL_ERROR);
				return;
			}
			glTexture.pglBindTexture(target, texture);
			gl4Texture.pglTexParameterIuiv(target, pname, params);
			glTexture.pglBindTexture(target, bound);
		}
	}
	GLuint64 extGlGetTextureHandle(GLuint texture)
	{
		if (gl4Texture.pglGetTextureHandleARB)
			return gl4Texture.pglGetTextureHandleARB(texture);
		else if (gl4Texture.pglGetTextureHandleNV)
			return gl4Texture.pglGetTextureHandleNV(texture);
		return 0ull;
	}
	GLuint64 extGlGetTextureSamplerHandle(GLuint texture, GLuint sampler)
	{
		if (gl4Texture.pglGetTextureSamplerHandleARB)
			return gl4Texture.pglGetTextureSamplerHandleARB(texture, sampler);
		else if (gl4Texture.pglGetTextureSamplerHandleNV)
			return gl4Texture.pglGetTextureSamplerHandleNV(texture, sampler);
		return 0ull;
	}
	void extGlMakeTextureHandleResident(GLuint64 handle)
	{
		if (gl4Texture.pglMakeTextureHandleResidentARB)
			return gl4Texture.pglMakeTextureHandleResidentARB(handle);
		else if (gl4Texture.pglMakeTextureHandleResidentNV)
			return gl4Texture.pglMakeTextureHandleResidentNV(handle);
	}
	void extGlMakeTextureHandleNonResident(GLuint64 handle)
	{
		if (gl4Texture.pglMakeTextureHandleNonResidentARB)
			return gl4Texture.pglMakeTextureHandleNonResidentARB(handle);
		else if (gl4Texture.pglMakeTextureHandleNonResidentNV)
			return gl4Texture.pglMakeTextureHandleNonResidentNV(handle);
	}
	GLuint64 extGlGetImageHandle(GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum format)
	{
		if (gl4Texture.pglGetImageHandleARB)
			return gl4Texture.pglGetImageHandleARB(texture, level, layered, layer, format);
		else if (gl4Texture.pglGetImageHandleNV)
			return gl4Texture.pglGetImageHandleNV(texture, level, layered, layer, format);
		return 0ull;
	}
	void extGlMakeImageHandleResident(GLuint64 handle, GLenum access)
	{
		if (gl4Texture.pglMakeImageHandleResidentARB)
			return gl4Texture.pglMakeImageHandleResidentARB(handle, access);
		else if (gl4Texture.pglMakeImageHandleResidentNV)
			return gl4Texture.pglMakeImageHandleResidentNV(handle, access);
	}
	void extGlMakeImageHandleNonResident(GLuint64 handle)
	{
		if (gl4Texture.pglMakeImageHandleNonResidentARB)
			return gl4Texture.pglMakeImageHandleNonResidentARB(handle);
		else if (gl4Texture.pglMakeImageHandleNonResidentNV)
			return gl4Texture.pglMakeImageHandleNonResidentNV(handle);
	}
	GLboolean extGlIsTextureHandleResident(GLuint64 handle)
	{
		if (gl4Texture.pglIsTextureHandleResidentARB)
			return gl4Texture.pglIsTextureHandleResidentARB(handle);
		else if (gl4Texture.pglIsTextureHandleResidentNV)
			return gl4Texture.pglIsTextureHandleResidentNV(handle);
		return false;
	}
	GLboolean extGlIsImageHandleResident(GLuint64 handle)
	{
		if (gl4Texture.pglIsTextureHandleResidentARB)
			return gl4Texture.pglIsTextureHandleResidentARB(handle);
		else if (gl4Texture.pglIsTextureHandleResidentNV)
			return gl4Texture.pglIsTextureHandleResidentNV(handle);
		return false;
	}
	void extGlGetNamedBufferSubData(GLuint buffer, GLintptr offset, GLsizeiptr size, void* data)
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Buffer.pglGetNamedBufferSubData)
				gl4Buffer.pglGetNamedBufferSubData(buffer, offset, size, data);
		}
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Buffer.pglGetNamedBufferSubDataEXT)
				gl4Buffer.pglGetNamedBufferSubDataEXT(buffer, offset, size, data);
		}
		/*
		else if (gl4Buffer.pglGetBufferSubData && glBuffer.pglBindBuffer)
		{
			GLint bound;
			glGeneral.pglGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bound);
			glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, buffer);
			gl4Buffer.pglGetBufferSubData(GL_ARRAY_BUFFER, offset, size, data);
			glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, bound);
		}
		*/
	}
	void extGlVertexArrayAttribLFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset) override
	{
		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Vertex.pglVertexArrayAttribLFormat)
				gl4Vertex.pglVertexArrayAttribLFormat(vaobj, attribindex, size, type, relativeoffset);
		}
		else if (!features->isIntelGPU && features->FeatureAvailable[features->EOpenGLFeatures::NBL_EXT_direct_state_access])
		{
			if (gl4Vertex.pglVertexArrayVertexAttribLFormatEXT)
				gl4Vertex.pglVertexArrayVertexAttribLFormatEXT(vaobj, attribindex, size, type, relativeoffset);
		}
		else if (gl4Vertex.pglVertexAttribLFormat && glVertex.pglBindVertexArray)
		{
			// Save the previous bound vertex array
			GLint restoreVertexArray;
			glGeneral.pglGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
			glVertex.pglBindVertexArray(vaobj);
			gl4Vertex.pglVertexAttribLFormat(attribindex, size, type, relativeoffset);
			glVertex.pglBindVertexArray(restoreVertexArray);
		}
	}
	void extGlGetQueryBufferObjectuiv(GLuint id, GLuint buffer, GLenum pname, GLintptr offset) override
	{
		if (features->Version < 440 && !features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_query_buffer_object])
		{
#ifdef _NBL_DEBUG
			m_logger.log("GL_ARB_query_buffer_object unsupported!\n", system::ILogger::ELL_ERROR);
#endif // _NBL_DEBUG
				return;
		}

		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Query.pglGetQueryBufferObjectuiv)
				gl4Query.pglGetQueryBufferObjectuiv(id, buffer, pname, offset);
		}
		else
		{
			base_t::extGlGetQueryBufferObjectuiv(id, buffer, pname, offset);
		}
	}
	void extGlGetQueryBufferObjectui64v(GLuint id, GLuint buffer, GLenum pname, GLintptr offset) override
	{
		if (features->Version < 440 && !features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_query_buffer_object])
		{
#ifdef _NBL_DEBUG
			m_logger.log("GL_ARB_query_buffer_object unsupported!\n", system::ILogger::ELL_ERROR);
#endif // _NBL_DEBUG
				return;
		}

		if (features->Version >= 450 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_direct_state_access])
		{
			if (gl4Query.pglGetQueryBufferObjectui64v)
				gl4Query.pglGetQueryBufferObjectui64v(id, buffer, pname, offset);
		}
		else
		{
			base_t::extGlGetQueryBufferObjectui64v(id, buffer, pname, offset);
		}
	}
	void extGlTextureBarrier()
	{
		if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_texture_barrier])
			gl4Sync.pglTextureBarrier();
		else if (features->FeatureAvailable[features->EOpenGLFeatures::NBL_NV_texture_barrier])
			gl4Sync.pglTextureBarrierNV();
#ifdef _NBL_DEBUG
		else
			m_logger.log("EDF_TEXTURE_BARRIER Not Available!\n", system::ILogger::ELL_ERROR);
#endif // _NBL_DEBUG
	}
	void extGlGetInternalformati64v(GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint64* params)
	{
		if (features->Version >= 460 || features->FeatureAvailable[features->EOpenGLFeatures::NBL_ARB_internalformat_query])
		{
			if (gl4General.pglGetInternalformati64v)
				gl4General.pglGetInternalformati64v(target, internalformat, pname, bufSize, params);
		}
	}
	void extGlMinSampleShading(GLfloat value) override
	{
		if (gl4Fragment.pglMinSampleShading)
			gl4Fragment.pglMinSampleShading(value);
	}
	void extGlCopyImageSubData(
		GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ,
		GLuint dstName, GLenum dstTarget, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ,
		GLsizei srcWidth, GLsizei srcHeight, GLsizei srcDepth
	) override
	{
		glTexture.pglCopyImageSubData(srcName, srcTarget, srcLevel, srcX, srcY, srcZ, dstName, dstTarget, dstLevel, dstX, dstY, dstZ, srcWidth, srcHeight, srcDepth);
	}

	void extGlDrawArraysInstancedBaseInstance(GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance) override
	{
		gl4Drawing.pglDrawArraysInstancedBaseInstance(mode, first, count, instancecount, baseinstance);
	}

	void extGlDrawElementsInstancedBaseInstance(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount, GLint baseinstance) override
	{
		gl4Drawing.pglDrawElementsInstancedBaseInstance(mode, count, type, indices, instancecount, baseinstance);
	}

	void extGlDrawElementsInstancedBaseVertex(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount, GLint basevertex) override
	{
		gl4Drawing.pglDrawElementsInstancedBaseVertex(mode, count, type, indices, instancecount, basevertex);
	}

	void extGlDrawElementsInstancedBaseVertexBaseInstance(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance) override
	{
		gl4Drawing.pglDrawElementsInstancedBaseVertexBaseInstance(mode, count, type, indices, instancecount, basevertex, baseinstance);
	}

	void extGlMultiDrawArraysIndirect(GLenum mode, const void* indirect, GLsizei drawcount, GLsizei stride) override
	{
		gl4Drawing.pglMultiDrawArraysIndirect(mode, indirect, drawcount, stride);
	}
	void extGlMultiDrawElementsIndirect(GLenum mode, GLenum type, const void* indirect, GLsizei drawcount, GLsizei stride) override
	{
		gl4Drawing.pglMultiDrawElementsIndirect(mode, type, indirect, drawcount, stride);
	}

	void extGlMultiDrawArraysIndirectCount(GLenum mode, const void* indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride) override
	{
		if (features->Version>=460)
			gl4Drawing.pglMultiDrawArraysIndirectCount(mode, indirect, drawcount, maxdrawcount, stride);
		else if (features->isFeatureAvailable(COpenGLFeatureMap::NBL_ARB_indirect_parameters))
			gl4Drawing.pglMultiDrawArraysIndirectCountARB(mode, indirect, drawcount, maxdrawcount, stride);
#ifdef _NBL_DEBUG
		else
		{
			m_logger.log("MDI not supported!", system::ILogger::ELL_ERROR);
		}
#endif
	}
	void extGlMultiDrawElementsIndirectCount(GLenum mode, GLenum type, const void* indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride) override
	{
		if (features->Version>=460 && gl4Drawing.pglMultiDrawElementsIndirectCount) // Intel is a very special boy...
			gl4Drawing.pglMultiDrawElementsIndirectCount(mode, type, indirect, drawcount, maxdrawcount, stride);
		else if (features->isFeatureAvailable(COpenGLFeatureMap::NBL_ARB_indirect_parameters))
			gl4Drawing.pglMultiDrawElementsIndirectCountARB(mode, type, indirect, drawcount, maxdrawcount, stride);
#ifdef _NBL_DEBUG
		else
		{
			m_logger.log("MDI not supported!", system::ILogger::ELL_ERROR);
		}
#endif
	}

	void extGlViewportArrayv(GLuint first, GLsizei count, const GLfloat* v) override
	{
		gl4General.pglViewportArrayv(first, count, v);
	}
	void extGlDepthRangeArrayv(GLuint first, GLsizei count, const GLdouble* v) override
	{
		gl4General.pglDepthRangeArrayv(first, count, v);
	}
	void extGlClipControl(GLenum origin, GLenum depth) override
	{
		gl4General.pglClipControl(origin, depth);
	}

	void extGlLogicOp(GLenum opcode) override
	{
		gl4Shader.pglLogicOp(opcode);
	}

	void extGlPolygonMode(GLenum face, GLenum mode) override
	{
		gl4Shader.pglPolygonMode(face, mode);
	}

	void extGlObjectLabel(GLenum identifier, GLuint name, GLsizei length, const char* label) override
	{
		gl4Debug.pglObjectLabel(identifier, name, length, label);
	}

	void extGlGetObjectLabel(GLenum identifier, GLuint name, GLsizei bufsize, GLsizei* length, GLchar* label) override
	{
		gl4Debug.pglGetObjectLabel(identifier, name, bufsize, length, label);
	}
};

}
#endif
