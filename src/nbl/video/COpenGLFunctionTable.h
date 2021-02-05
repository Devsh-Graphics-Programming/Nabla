#ifndef __NBL_C_OPENGL_FUNCTION_TABLE_H_INCLUDED__
#define __NBL_C_OPENGL_FUNCTION_TABLE_H_INCLUDED__

#include "nbl/video/COpenGLFeatureMap.h"
#include "nbl/video/COpenGL_FunctionTableBase.h"
#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#endif
#include "GL/glext.h"
#include "GL/gl.h"

namespace nbl {
namespace video
{

class COpenGLFunctiontable : public COpenGL_FunctionTableBase
{
	using base_t = COpenGL_FunctionTableBase;

public:
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GL4sync, OpenGLFunctionLoader
		, glTextureBarrier
		, glTextureBarrierNV
	);
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
		, glGetTextureSubImage
		, glGetCompressedTextureSubImage
		, glGetTextureImage
		, glGetTextureImageEXT
		, glGetCompressedTextureImage
		, glGetCompressedTextureImageEXT
		, glGetCompressedTexImage
		, glMultiTexSubImage1DEXT
		, glMultiTexSubImage2DEXT
		, glMultiTexSubImage3DEXT
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
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GL4shader, OpenGLFunctionLoader
		, glCreateProgramPipelines
		, glPatchParameterfv
		, glPatchParameteri
		, glPrimitiveRestartIndex
		, glProvokingVertex
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
		, glBeginQueryIndexed
		, glEndQueryIndexed
		, glGetQueryObjectui64v
		, glGetQueryBufferObjectuiv
		, glGetQueryBufferObjectui64v
		, glQueryCounter
		, glBeginConditionalRender
		, glEndConditionalRender
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GL4general, OpenGLFunctionLoader
		, glGetFloati_v
		, glGetInternalformati64v
		, glClipControl
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

    COpenGLFunctiontable(const egl::CEGL* _egl, const COpenGLFeatureMap* _features) :
        COpenGL_FunctionTableBase(_egl),
		gl4Framebuffer(_egl),
		gl4Buffer(_egl),
		gl4Texture(_egl),
		gl4Shader(_egl),
		gl4Fragment(_egl),
		gl4Vertex(_egl),
		gl4Drawing(_egl),
		gl4Query(_egl),
		gl4General(_egl),
        features(_features)
    {

    }

	inline void extGlTextureStorage1D(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width);
	inline void extGlGetTextureSubImage(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLsizei bufSize, void* pixels);
	inline void extGlGetTextureImage(GLuint texture, GLenum target, GLint level, GLenum format, GLenum type, GLsizei bufSizeHint, void* pixels);
	inline void extGlGetCompressedTextureImage(GLuint texture, GLenum target, GLint level, GLsizei bufSizeHint, void* pixels);
	inline void extGlTextureSubImage1D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void* pixels);
	inline void extGlCompressedTextureSubImage1D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void* data);
	inline void extGlTextureParameterIuiv(GLuint texture, GLenum target, GLenum pname, const GLuint* params);
	inline GLuint64 extGlGetTextureHandle(GLuint texture);
	inline GLuint64 extGlGetTextureSamplerHandle(GLuint texture, GLuint sampler);
	inline void extGlMakeTextureHandleResident(GLuint64 handle);
	inline void extGlMakeTextureHandleNonResident(GLuint64 handle);
	inline GLuint64 extGlGetImageHandle(GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum format);
	inline void extGlMakeImageHandleResident(GLuint64 handle, GLenum access);
	inline void extGlMakeImageHandleNonResident(GLuint64 handle);
	inline GLboolean extGlIsTextureHandleResident(GLuint64 handle);
	inline GLboolean extGlIsImageHandleResident(GLuint64 handle);
	inline void extGlGetNamedBufferSubData(GLuint buffer, GLintptr offset, GLsizeiptr size, void* data);
	inline void extGlVertexArrayAttribLFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset);
	inline void extGlGetQueryBufferObjectuiv(GLuint id, GLuint buffer, GLenum pname, GLintptr offset);
	inline void extGlGetQueryBufferObjectui64v(GLuint id, GLuint buffer, GLenum pname, GLintptr offset);
	inline void extGlTextureBarrier();
	inline void extGlGetInternalformati64v(GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint64* params);

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
					glBindTexture(targets[i], texture);
				else
				{
					for (size_t j = 0; j < sizeof(supportedTargets) / sizeof(GLenum); j++)
						glBindTexture(supportedTargets[j], 0);
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
				os::Printer::log("DevSH would like to ask you what are you doing!!??\n", ELL_ERROR);
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
				os::Printer::log("DevSH would like to ask you what are you doing!!??\n", ELL_ERROR);
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
				os::Printer::log("DevSH would like to ask you what are you doing!!??\n", ELL_ERROR);
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
				os::Printer::log("DevSH would like to ask you what are you doing!!??\n", ELL_ERROR);
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
		if (gl4Texture.pglBindImageTextures)
			gl4Texture.pglBindImageTextures(first, count, textures);
		else
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
		else
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
		else if (gl4Buffer.pglBufferStorage && glBuffer.pglBindBuffer)
		{
			GLint bound;
			glGeneral.pglGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bound);
			glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, buffer);
			gl4Buffer.pglBufferStorage(GL_ARRAY_BUFFER, size, data, flags);
			glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, bound);
		}
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
		else base_t::extGlNamedBufferSubData(buffer, offset, size, data);
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
		else return base_t::extGlMapNamedBufferRange(buffer, offset, length, access);
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
		else base_t::extGlFlushMappedNamedBufferRange(buffer, offset, length);
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
		else return base_t::extGlUnmapNamedBuffer(buffer);
	}

private:
    const COpenGLFeatureMap* features;
};

}
}

#undef GL_GLEXT_PROTOTYPES
#endif
