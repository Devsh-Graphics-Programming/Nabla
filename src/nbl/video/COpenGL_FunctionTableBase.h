#ifndef __NBL_C_OPEN_GL_FUNCTION_TABLE_BASE_H_INCLUDED__
#define __NBL_C_OPEN_GL_FUNCTION_TABLE_BASE_H_INCLUDED__

#include "os.h" // Printer::log

#include "nbl/core/string/UniqueStringLiteralType.h"
#include "nbl/system/DynamicFunctionCaller.h"
#include "nbl/video/CEGL.h"
#include "GLES3/gl32.h"

namespace nbl {
	namespace video {

		// This class contains pointers to functions common in GL 4.6 and GLES 3.2
		// And implements (at least a common part) extGl* methods which can be implemented with those pointers
		class COpenGL_FunctionTableBase
		{
		public:

			class OpenGLFunctionLoader final : public system::FuncPtrLoader
			{
				egl::CEGL* egl;

			public:
				explicit OpenGLFunctionLoader(egl::CEGL* _egl) : egl(_egl) {}

				inline bool isLibraryLoaded() override final
				{
					return true;
				}

				inline void* loadFuncPtr(const char* funcname) override final
				{
					return static_cast<void*>(egl->call.peglGetProcAddress(funcname));
				}
			};

			NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLsync, OpenGLFunctionLoader
				, glFenceSync
				, glDeleteSync
				, glClientWaitSync
				, glWaitSync
				, glMemoryBarrier
			);

			NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLframeBuffer, OpenGLFunctionLoader
				, glBlitFramebuffer
				, glDeleteFramebuffers
				, glGenFramebuffers
				, glBindFramebuffer
				, glCheckFramebufferStatus
				, glFramebufferTexture
				, glFramebufferTextureLayer
				, glFramebufferTexture2D
				, glClearBufferiv
				, glClearBufferuiv
				, glClearBufferfv
				, glClearBufferfi
				, glDrawBuffers
				, glReadBuffer
			);
			NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLbuffer, OpenGLFunctionLoader
				, glBindBufferBase
				, glBindBufferRange
				, glGenBuffers
				, glBindBuffer
				, glDeleteBuffers
				, glBufferSubData
				, glMapBufferRange
				, glFlushMappedBufferRange
				, glUnmapBuffer
				, glCopyBufferSubData
				, glIsBuffer
				, glGetBufferParameteri64v
				, glGetBufferParameteriv
			);
			NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLtexture, OpenGLFunctionLoader
				, glActiveTexture
				, glTexStorage2D
				, glTexStorage3D
				, glTexStorage2DMultisample
				, glTexStorage3DMultisample
				, glTexSubImage3D
				, glCompressedTexSubImage2D
				, glCompressedTexSubImage3D
				, glGenerateMipmap
				, glGenSamplers
				, glDeleteSamplers
				, glBindSampler
				, glSamplerParameteri
				, glSamplerParameterf
				, glSamplerParameterfv
				, glBindImageTexture
				, glBindTexture
				, glTexSubImage2D
				, glTexBuffer
				, glTexBufferRange
			);
			NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLshader, OpenGLFunctionLoader
				, glCreateShader
				, glCreateShaderProgramv
				//, glCreateProgramPipelines
				, glDeleteProgramPipelines
				, glUseProgramStages
				, glShaderSource
				, glCompileShader
				, glCreateProgram
				, glAttachShader
				, glTransformFeedbackVaryings
				, glLinkProgram
				, glUseProgram
				, glDeleteProgram
				, glDeleteShader
				, glGetAttachedShaders
				, glGetShaderInfoLog
				, glGetProgramInfoLog
				, glGetShaderiv
				, glGetProgramiv
				, glGetUniformLocation
				, glProgramUniform1fv
				, glProgramUniform2fv
				, glProgramUniform3fv
				, glProgramUniform4fv
				, glProgramUniform1iv
				, glProgramUniform2iv
				, glProgramUniform3iv
				, glProgramUniform4iv
				, glProgramUniform1uiv
				, glProgramUniform2uiv
				, glProgramUniform3uiv
				, glProgramUniform4uiv
				, glProgramUniformMatrix2fv
				, glProgramUniformMatrix3fv
				, glProgramUniformMatrix4fv
				, glProgramUniformMatrix2x3fv
				, glProgramUniformMatrix3x2fv
				, glProgramUniformMatrix4x2fv
				, glProgramUniformMatrix2x4fv
				, glProgramUniformMatrix3x4fv
				, glProgramUniformMatrix4x3fv
				, glGetActiveUniform
				, glBindProgramPipeline
				, glGetProgramBinary
				, glProgramBinary
				, glProgramParameteri
				, glDepthMask
				, glPixelStorei
				, glPolygonOffset
				//, glPointSize
				, glLineWidth
				, glDepthFunc
				, glHint

			);
			NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLfragment, OpenGLFunctionLoader
				, glBlendEquation
				, glBlendColor
				, glSampleCoverage
				, glSampleMaski
				, glStencilFuncSeparate
				, glStencilOpSeparate
				, glStencilMaskSeparate
				, glBlendFuncSeparate
			);
			NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLvertex, OpenGLFunctionLoader
				, glGenVertexArrays
				, glDeleteVertexArrays
				, glBindVertexArray
				, glBindVertexBuffer
				, glVertexAttribBinding
				, glEnableVertexAttribArray
				, glDisableVertexAttribArray
				, glVertexAttribFormat
				, glVertexAttribIFormat
				, glVertexBindingDivisor
			);
			NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLdrawing, OpenGLFunctionLoader
				, glDrawArrays
				, glDrawElements
				, glDrawArraysInstanced
				, glDrawElementsInstanced
				, glDrawArraysIndirect
				, glDrawElementsIndirect
			);
			NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLtransformFeedback, OpenGLFunctionLoader
				, glGenTransformFeedbacks
				, glDeleteTransformFeedbacks
				, glBindTransformFeedback
				, glBeginTransformFeedback
				, glPauseTransformFeedback
				, glResumeTransformFeedback
				, glEndTransformFeedback
			);
			NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLquery, OpenGLFunctionLoader
				//, glCreateQueries
				, glGenQueries
				, glDeleteQueries
				, glIsQuery
				, glBeginQuery
				, glEndQuery
				, glGetQueryiv
				, glGetQueryObjectuiv
			);
			NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLgeneral, OpenGLFunctionLoader
				, glEnablei
				, glDisablei
				, glEnable
				, glDisable
				, glIsEnabledi
				//glGet functions
				, glGetIntegerv
				, glGetBooleanv
				//, glGetDoublev
				, glGetFloatv
				, glGetString
				//, glGetFloati_v
				, glGetInteger64v
				, glGetIntegeri_v
				, glGetBooleani_v
				, glGetStringi

				, glGetInternalformativ
				//, glGetInternalformati64v
				//, glLogicOp
				, glFlush
				//, glClipControl
				, glFinish

				, glBlendEquationi
				, glBlendEquationSeparatei
				, glBlendFunci
				, glBlendFuncSeparatei
				, glColorMaski
			);
			NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLcompute, OpenGLFunctionLoader
				, glDispatchCompute
				, glDispatchComputeIndirect
			);

			GLsync glSync;
			GLframeBuffer glFramebuffer;
			GLbuffer glBuffer;
			GLtexture glTexture;
			GLshader glShader;
			GLfragment glFragment;
			GLvertex glVertex;
			GLdrawing glDrawing;
			GLtransformFeedback glTransformFeedback;
			GLquery glQuery;
			//GLdebug glDebug;
			GLgeneral glGeneral;
			GLcompute glCompute;

			const egl::CEGL* m_egl;

			virtual void extGlDebugMessageControl(GLenum source, GLenum type, GLenum severity, GLsizei count, const GLuint* ids, GLboolean enabled) = 0;
			using GLDebugCallbackType = GLDEBUGPROC;
			virtual void extGlDebugMessageCallback(GLDebugCallbackType callback, const void* userParam) = 0;

			virtual void extGlBindTextures(const GLuint& first, const GLsizei& count, const GLuint* textures, const GLenum* targets) = 0;
			virtual inline void extGlCreateTextures(GLenum target, GLsizei n, GLuint* textures);
			virtual void extGlTextureView(GLuint texture, GLenum target, GLuint origtexture, GLenum internalformat, GLuint minlevel, GLuint numlevels, GLuint minlayer, GLuint numlayers) = 0;
			virtual void extGlTextureBuffer(GLuint texture, GLenum internalformat, GLuint buffer) = 0;
			virtual void extGlTextureBufferRange(GLuint texture, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizei length) = 0;
			virtual void extGlTextureStorage2D(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height) = 0;
			virtual inline void extGlTextureStorage3D(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth);
			virtual inline void extGlTextureStorage2DMultisample(GLuint texture, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations);
			virtual inline void extGlTextureStorage3DMultisample(GLuint texture, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations);
			virtual void extGlTextureSubImage2D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void* pixels) = 0;
			virtual inline void extGlTextureSubImage3D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void* pixels);
			virtual void extGlCompressedTextureSubImage2D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void* data) = 0;
			virtual void extGlCompressedTextureSubImage3D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void* data);
			virtual void extGlGenerateTextureMipmap(GLuint texture, GLenum target) = 0;
			virtual inline void extGlCreateSamplers(GLsizei n, GLuint* samplers);
			virtual inline void extGlBindSamplers(const GLuint& first, const GLsizei& count, const GLuint* samplers);
			virtual inline void extGlBindImageTextures(GLuint first, GLsizei count, const GLuint* textures, const GLenum* formats);
			virtual inline void extGlCreateFramebuffers(GLsizei n, GLuint* framebuffers);
			virtual inline GLenum extGlCheckNamedFramebufferStatus(GLuint framebuffer, GLenum target);
			virtual void extGlNamedFramebufferTexture(GLuint framebuffer, GLenum attachment, GLuint texture, GLint level, GLenum textureType) = 0;
			virtual void extGlNamedFramebufferTextureLayer(GLuint framebuffer, GLenum attachment, GLuint texture, GLenum textureType, GLint level, GLint layer) = 0;
			virtual inline void extGlBlitNamedFramebuffer(GLuint readFramebuffer, GLuint drawFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter) = 0;
			//inline void extGlActiveStencilFace(GLenum face);
			virtual inline void extGlNamedFramebufferReadBuffer(GLuint framebuffer, GLenum mode);
			virtual void extGlNamedFramebufferDrawBuffer(GLuint framebuffer, GLenum buf) = 0;
			virtual inline void extGlNamedFramebufferDrawBuffers(GLuint framebuffer, GLsizei n, const GLenum* bufs);
			virtual inline void extGlClearNamedFramebufferiv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLint* value);
			virtual inline void extGlClearNamedFramebufferuiv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLuint* value);
			virtual inline void extGlClearNamedFramebufferfv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLfloat* value);
			virtual inline void extGlClearNamedFramebufferfi(GLuint framebuffer, GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil);
			virtual inline void extGlCreateBuffers(GLsizei n, GLuint* buffers);
			virtual inline void extGlBindBuffersBase(const GLenum& target, const GLuint& first, const GLsizei& count, const GLuint* buffers);
			virtual inline void extGlBindBuffersRange(const GLenum& target, const GLuint& first, const GLsizei& count, const GLuint* buffers, const GLintptr* offsets, const GLsizeiptr* sizes);
			virtual void extGlNamedBufferStorage(GLuint buffer, GLsizeiptr size, const void* data, GLbitfield flags) = 0;
			virtual inline void extGlNamedBufferSubData(GLuint buffer, GLintptr offset, GLsizeiptr size, const void* data);
			virtual inline void* extGlMapNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length, GLbitfield access);
			virtual inline void extGlFlushMappedNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length);
			virtual inline GLboolean extGlUnmapNamedBuffer(GLuint buffer);
			// TODO left to do below:
			virtual void extGlClearNamedBufferData(GLuint buffer, GLenum internalformat, GLenum format, GLenum type, const void* data) = 0;
			virtual void extGlClearNamedBufferSubData(GLuint buffer, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type, const void* data) = 0;
			virtual inline void extGlCopyNamedBufferSubData(GLuint readBuffer, GLuint writeBuffer, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size);
			virtual inline GLboolean extGlIsBuffer(GLuint buffer);
			virtual inline void extGlGetNamedBufferParameteriv(const GLuint& buffer, const GLenum& value, GLint* data);
			virtual inline void extGlGetNamedBufferParameteri64v(const GLuint& buffer, const GLenum& value, GLint64* data);
			virtual inline void extGlCreateVertexArrays(GLsizei n, GLuint* arrays);
			virtual inline void extGlVertexArrayElementBuffer(GLuint vaobj, GLuint buffer);
			virtual inline void extGlVertexArrayVertexBuffer(GLuint vaobj, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride);
			virtual inline void extGlVertexArrayAttribBinding(GLuint vaobj, GLuint attribindex, GLuint bindingindex);
			virtual inline void extGlEnableVertexArrayAttrib(GLuint vaobj, GLuint index);
			virtual inline void extGlDisableVertexArrayAttrib(GLuint vaobj, GLuint index);
			virtual inline void extGlVertexArrayAttribFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset);
			virtual inline void extGlVertexArrayAttribIFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset);
			virtual inline void extGlVertexArrayBindingDivisor(GLuint vaobj, GLuint bindingindex, GLuint divisor);
			virtual inline void extGlCreateQueries(GLenum target, GLsizei n, GLuint* ids);
			virtual inline void extGlGetInternalformativ(GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint* params);
			virtual inline void extGlEnablei(GLenum target, GLuint index)
			{
				if (glGeneral.pglEnablei)
					glGeneral.pglEnablei(target, index);
			}
			virtual inline void extGlDisablei(GLenum target, GLuint index)
			{
				if (glGeneral.pglDisablei)
					glGeneral.pglDisablei(target, index);
			}
			virtual inline void extGlBlendEquationi(GLuint buf, GLenum mode)
			{
				if (glGeneral.pglBlendEquationi)
					glGeneral.pglBlendEquationi(buf, mode);
			}
			virtual inline void extGlBlendEquationSeparatei(GLuint buf, GLenum modeRGB, GLenum modeAlpha)
			{
				if (glGeneral.pglBlendEquationSeparatei)
					glGeneral.pglBlendEquationSeparatei(buf, modeRGB, modeAlpha);
			}
			virtual inline void extGlBlendFunci(GLuint buf, GLenum src, GLenum dst)
			{
				if (glGeneral.pglBlendFunci)
					glGeneral.pglBlendFunci(buf, src, dst);
			}
			virtual inline void extGlBlendFuncSeparatei(GLuint buf, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha)
			{
				if (glGeneral.pglBlendFuncSeparatei)
					glGeneral.pglBlendFuncSeparatei(buf, srcRGB, dstRGB, srcAlpha, dstAlpha);
			}
			virtual inline void extGlColorMaski(GLuint buf, GLboolean r, GLboolean g, GLboolean b, GLboolean a)
			{
				if (glGeneral.pglColorMaski)
					glGeneral.pglColorMaski(buf, r, g, b, a);
			}
			virtual inline GLboolean extGlIsEnabledi(GLenum target, GLuint index)
			{
				if (glGeneral.pglIsEnabledi)
					return glGeneral.pglIsEnabledi(target, index);
				return GL_FALSE;
			}
			virtual void extGlMinSampleShading(GLfloat value) = 0;
			virtual inline void extGlSwapInterval(int interval);

			// constructor
			COpenGL_FunctionTableBase(const egl::CEGL* _egl) :
				glSync(_egl),
				glFramebuffer(_egl),
				glBuffer(_egl),
				glTexture(_egl),
				glShader(_egl),
				glFragment(_egl),
				glVertex(_egl),
				glDrawing(_egl),
				glTransformFeedback(_egl),
				glQuery(_egl),
				//glDebug(_egl),
				glGeneral(_egl),
				glCompute(_egl),
				m_egl(_egl)
			{

			}
		};	// end of class COpenGL_FunctionTableBase

		void COpenGL_FunctionTableBase::extGlCreateTextures(GLenum target, GLsizei n, GLuint* textures)
		{
			glGenTextures(n, textures);
		}

		inline void COpenGL_FunctionTableBase::extGlTextureStorage3D(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth)
		{
			// TODO impl in GL
			GLint bound;
			switch (target)
			{
			case GL_TEXTURE_2D_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &bound);
				break;
			case GL_TEXTURE_3D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_3D, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP_ARRAY, &bound);
				break;
			default:
				os::Printer::log("DevSH would like to ask you what are you doing!!??\n", ELL_ERROR);
				return;
			}
			glTexture.pglBindTexture(target, texture);
			glTexture.pglTexStorage3D(target, levels, internalformat, width, height, depth);
			glTexture.pglBindTexture(target, bound);
		}
		inline void COpenGL_FunctionTableBase::extGlTextureStorage2DMultisample(GLuint texture, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations)
		{
			// TODO impl in GL
			GLint bound;
			if (target != GL_TEXTURE_2D_MULTISAMPLE)
			{
				os::Printer::log("DevSH would like to ask you what are you doing!!??\n", ELL_ERROR);
				return;
			}
			else
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE, &bound);
			glTexture.pglBindTexture(GL_TEXTURE_2D_MULTISAMPLE, texture);
			glTexture.pglTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, internalformat, width, height, fixedsamplelocations);
			glTexture.pglBindTexture(GL_TEXTURE_2D_MULTISAMPLE, bound);
		}
		inline void COpenGL_FunctionTableBase::extGlTextureStorage3DMultisample(GLuint texture, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations)
		{
			GLint bound;
			if (target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
			{
				os::Printer::log("DevSH would like to ask you what are you doing!!??\n", ELL_ERROR);
				return;
			}
			else
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY, &bound);
			glTexture.pglBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, texture);
			glTexture.pglTexStorage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, samples, internalformat, width, height, depth, fixedsamplelocations);
			glTexture.pglBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, bound);
		}
		inline void COpenGL_FunctionTableBase::extGlTextureSubImage3D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void* pixels)
		{
			// TODO impl in GL
			GLint bound;
			switch (target)
			{
			case GL_TEXTURE_2D_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &bound);
				break;
			case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY, &bound);
				break;
			case GL_TEXTURE_3D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_3D, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP_ARRAY, &bound);
				break;
			default:
				os::Printer::log("DevSH would like to ask you what are you doing!!??\n", ELL_ERROR);
				return;
			}
			glTexture.pglBindTexture(target, texture);
			glTexture.pglTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
			glTexture.pglBindTexture(target, bound);
		}

		inline void COpenGL_FunctionTableBase::extGlCompressedTextureSubImage3D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void* data)
		{
			GLint bound;
			switch (target)
			{
			case GL_TEXTURE_2D_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &bound);
				break;
			case GL_TEXTURE_3D:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_3D, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP_ARRAY, &bound);
				break;
			default:
				os::Printer::log("DevSH would like to ask you what are you doing!!??\n", ELL_ERROR);
				return;
			}
			glTexture.pglBindTexture(target, texture);
			glTexture.pglCompressedTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
			glTexture.pglBindTexture(target, bound);
		}

		inline void COpenGL_FunctionTableBase::extGlCreateSamplers(GLsizei n, GLuint* samplers)
		{
			// TODO impl in GL
			if (glTexture.pglGenSamplers)
				glTexture.pglGenSamplers(n, samplers);
			else memset(samplers, 0, 4 * n);
		}
		inline void COpenGL_FunctionTableBase::extGlBindSamplers(const GLuint& first, const GLsizei& count, const GLuint* samplers)
		{
			// TODO impl in GL
			for (GLsizei i = 0; i < count; i++)
			{
				GLuint unit = first + i;
				if (samplers)
				{
					if (glTexture.pglBindSampler)
						glTexture.pglBindSampler(unit, samplers[i]);
				}
				else
				{
					if (glTexture.pglBindSampler)
						glTexture.pglBindSampler(unit, 0);
				}
			}
		}
		inline void COpenGL_FunctionTableBase::extGlBindImageTextures(GLuint first, GLsizei count, const GLuint* textures, const GLenum* formats)
		{
			// TODO impl in GL
			for (GLsizei i = 0; i < count; i++)
			{
				if (!textures || textures[i] == 0u)
					glTexture.pglBindImageTexture(first + i, 0u, 0u, GL_FALSE, 0, GL_READ_WRITE, GL_R8);
				else
					glTexture.pglBindImageTexture(first + i, textures[i], 0, GL_TRUE, 0, GL_READ_WRITE, formats[i]);
			}
		}
		inline void COpenGL_FunctionTableBase::extGlCreateFramebuffers(GLsizei n, GLuint* framebuffers)
		{
			// TODO impl in GL
			glFramebuffer.pglGenFramebuffers(n, framebuffers);
		}
		inline GLenum COpenGL_FunctionTableBase::extGlCheckNamedFramebufferStatus(GLuint framebuffer, GLenum target)
		{
			// TODO impl in GL
			GLenum retval;
			GLuint bound;
			glGeneral.pglGetIntegerv(target == GL_READ_FRAMEBUFFER ? GL_READ_FRAMEBUFFER_BINDING : GL_DRAW_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(&bound));

			if (bound != framebuffer)
				glFramebuffer.pglBindFramebuffer(target, framebuffer);
			retval = glFramebuffer.pglCheckFramebufferStatus(target);
			if (bound != framebuffer)
				glFramebuffer.pglBindFramebuffer(target, bound);

			return retval;
		}
		inline void COpenGL_FunctionTableBase::extGlNamedFramebufferTexture(GLuint framebuffer, GLenum attachment, GLuint texture, GLint level, GLenum textureType)
		{
			GLuint bound;
			glGeneral.pglGetIntegerv(GL_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(&bound));

			if (bound != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
			glFramebuffer.pglFramebufferTexture(GL_FRAMEBUFFER, attachment, texture, level);
			if (bound != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, bound);
		}
		inline void COpenGL_FunctionTableBase::extGlBlitNamedFramebuffer(GLuint readFramebuffer, GLuint drawFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter)
		{
			// TODO impl in GL

			GLint boundReadFBO = -1;
			GLint boundDrawFBO = -1;
			glGeneral.pglGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &boundReadFBO);
			glGeneral.pglGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &boundDrawFBO);

			if (static_cast<GLint>(readFramebuffer) != boundReadFBO)
				glFramebuffer.pglBindFramebuffer(GL_READ_FRAMEBUFFER, readFramebuffer);
			if (static_cast<GLint>(drawFramebuffer) != boundDrawFBO)
				glFramebuffer.pglBindFramebuffer(GL_DRAW_FRAMEBUFFER, drawFramebuffer);

			glFramebuffer.pglBlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);

			if (static_cast<GLint>(readFramebuffer) != boundReadFBO)
				glFramebuffer.pglBindFramebuffer(GL_READ_FRAMEBUFFER, boundReadFBO);
			if (static_cast<GLint>(drawFramebuffer) != boundDrawFBO)
				glFramebuffer.pglBindFramebuffer(GL_DRAW_FRAMEBUFFER, boundDrawFBO);
		}
		inline void COpenGL_FunctionTableBase::extGlNamedFramebufferReadBuffer(GLuint framebuffer, GLenum mode)
		{
			// TODO impl in GL

			GLint boundFBO;
			glGeneral.pglGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &boundFBO);

			if (static_cast<GLuint>(boundFBO) != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
			glFramebuffer.pglReadBuffer(mode);
			if (static_cast<GLuint>(boundFBO) != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_READ_FRAMEBUFFER, boundFBO);
		}
		inline void COpenGL_FunctionTableBase::extGlNamedFramebufferDrawBuffers(GLuint framebuffer, GLsizei n, const GLenum* bufs)
		{
			// TODO impl in GL

			GLint boundFBO;
			glGeneral.pglGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &boundFBO);

			if (static_cast<GLuint>(boundFBO) != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
			glFramebuffer.pglDrawBuffers(n, bufs);
			if (static_cast<GLuint>(boundFBO) != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_DRAW_FRAMEBUFFER, boundFBO);
		}
		inline void COpenGL_FunctionTableBase::extGlClearNamedFramebufferiv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLint* value)
		{
			// TODO impl in GL

			GLint boundFBO = -1;
			glGeneral.pglGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &boundFBO);
			if (boundFBO < 0)
				return;

			if (static_cast<GLuint>(boundFBO) != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
			glFramebuffer.pglClearBufferiv(buffer, drawbuffer, value);
			if (static_cast<GLuint>(boundFBO) != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, boundFBO);
		}
		inline void COpenGL_FunctionTableBase::extGlClearNamedFramebufferuiv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLuint* value)
		{
			// TODO impl in GL

			GLint boundFBO = -1;
			glGeneral.pglGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &boundFBO);
			if (boundFBO < 0)
				return;

			if (static_cast<GLuint>(boundFBO) != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
			glFramebuffer.pglClearBufferuiv(buffer, drawbuffer, value);
			if (static_cast<GLuint>(boundFBO) != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, boundFBO);
		}
		inline void COpenGL_FunctionTableBase::extGlClearNamedFramebufferfv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLfloat* value)
		{
			// TODO impl in GL

			GLint boundFBO = -1;
			glGeneral.pglGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &boundFBO);
			if (boundFBO < 0)
				return;

			if (static_cast<GLuint>(boundFBO) != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
			glFramebuffer.pglClearBufferfv(buffer, drawbuffer, value);
			if (static_cast<GLuint>(boundFBO) != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, boundFBO);
		}
		inline void COpenGL_FunctionTableBase::extGlClearNamedFramebufferfi(GLuint framebuffer, GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil)
		{
			// TODO impl in GL

			GLint boundFBO = -1;
			glGeneral.pglGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &boundFBO);
			if (boundFBO < 0)
				return;
			glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
			glFramebuffer.pglClearBufferfi(buffer, drawbuffer, depth, stencil);
			glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, boundFBO);
		}
		inline void COpenGL_FunctionTableBase::extGlCreateBuffers(GLsizei n, GLuint* buffers)
		{
			// TODO impl in GL

			if (glBuffer.pglGenBuffers)
				glBuffer.pglGenBuffers(n, buffers);
			else if (buffers)
				memset(buffers, 0, n * sizeof(GLuint));
		}
		inline void COpenGL_FunctionTableBase::extGlBindBuffersBase(const GLenum& target, const GLuint& first, const GLsizei& count, const GLuint* buffers)
		{
			// TODO impl in GL
			for (GLsizei i = 0; i < count; i++)
			{
				if (glBuffer.pglBindBufferBase)
					glBuffer.pglBindBufferBase(target, first + i, buffers ? buffers[i] : 0);
			}
		}
		inline void COpenGL_FunctionTableBase::extGlBindBuffersRange(const GLenum& target, const GLuint& first, const GLsizei& count, const GLuint* buffers, const GLintptr* offsets, const GLsizeiptr* sizes)
		{
			// TODO impl in GL
			for (GLsizei i = 0; i < count; i++)
			{
				if (buffers)
				{
					if (glBuffer.pglBindBufferRange)
						glBuffer.pglBindBufferRange(target, first + i, buffers[i], offsets[i], sizes[i]);
				}
				else
				{
					if (glBuffer.pglBindBufferBase)
						glBuffer.pglBindBufferBase(target, first + i, 0);
				}
			}
		}
		inline void COpenGL_FunctionTableBase::extGlNamedBufferSubData(GLuint buffer, GLintptr offset, GLsizeiptr size, const void* data)
		{
			// TODO impl in GL
			GLint bound;
			glGeneral.pglGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bound);
			glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, buffer);
			glBuffer.pglBufferSubData(GL_ARRAY_BUFFER, offset, size, data);
			glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, bound);
		}
		inline void* COpenGL_FunctionTableBase::extGlMapNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length, GLbitfield access)
		{
			// TODO impl in GL
			if (glBuffer.pglMapBufferRange && glBuffer.pglBindBuffer)
			{
				GLvoid* retval;
				GLint bound;
				glGeneral.pglGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bound);
				glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, buffer);
				retval = glBuffer.pglMapBufferRange(GL_ARRAY_BUFFER, offset, length, access);
				glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, bound);
				return retval;
			}
			return NULL;
		}
		inline void COpenGL_FunctionTableBase::extGlFlushMappedNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length)
		{
			// TODO impl in GL
			if (glBuffer.pglFlushMappedBufferRange && glBuffer.pglBindBuffer)
			{
				GLint bound;
				glGeneral.pglGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bound);
				glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, buffer);
				glBuffer.pglFlushMappedBufferRange(GL_ARRAY_BUFFER, offset, length);
				glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, bound);
			}
		}
		inline GLboolean COpenGL_FunctionTableBase::extGlUnmapNamedBuffer(GLuint buffer)
		{
			// TODO impl in GL
			if (glBuffer.pglUnmapBuffer && glBuffer.pglBindBuffer)
			{
				GLboolean retval;
				GLint bound;
				glGeneral.pglGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bound);
				glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, buffer);
				retval = glBuffer.pglUnmapBuffer(GL_ARRAY_BUFFER);
				glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, bound);
				return retval;
			}
			return false;
		}

		inline void COpenGL_FunctionTableBase::extGlCopyNamedBufferSubData(GLuint readBuffer, GLuint writeBuffer, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size)
		{
			// TODO impl in GL
			if (glBuffer.pglCopyBufferSubData && glBuffer.pglBindBuffer)
			{
				GLint boundRead, boundWrite;
				glGeneral.pglGetIntegerv(GL_COPY_READ_BUFFER_BINDING, &boundRead);
				glGeneral.pglGetIntegerv(GL_COPY_WRITE_BUFFER_BINDING, &boundWrite);
				glBuffer.pglBindBuffer(GL_COPY_READ_BUFFER, readBuffer);
				glBuffer.pglBindBuffer(GL_COPY_WRITE_BUFFER, writeBuffer);
				glBuffer.pglCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, readOffset, writeOffset, size);
				glBuffer.pglBindBuffer(GL_COPY_READ_BUFFER, boundRead);
				glBuffer.pglBindBuffer(GL_COPY_WRITE_BUFFER, boundWrite);
			}
		}
		inline GLboolean COpenGL_FunctionTableBase::extGlIsBuffer(GLuint buffer)
		{
			if (glBuffer.pglIsBuffer)
				return glBuffer.pglIsBuffer(buffer);
			return false;
		}
		inline void COpenGL_FunctionTableBase::extGlGetNamedBufferParameteriv(const GLuint& buffer, const GLenum& value, GLint* data)
		{
			// TODO impl in GL
			if (glBuffer.pglGetBufferParameteriv && glBuffer.pglBindBuffer)
			{
				GLint bound;
				glGeneral.pglGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bound);
				glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, buffer);
				glBuffer.pglGetBufferParameteriv(GL_ARRAY_BUFFER, value, data);
				glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, bound);
			}
		}
		inline void COpenGL_FunctionTableBase::extGlGetNamedBufferParameteri64v(const GLuint& buffer, const GLenum& value, GLint64* data)
		{
			// TODO impl in GL
			if (glBuffer.pglGetBufferParameteri64v && glBuffer.pglBindBuffer)
			{
				GLint bound;
				glGeneral.pglGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bound);
				glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, buffer);
				glBuffer.pglGetBufferParameteri64v(GL_ARRAY_BUFFER, value, data);
				glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, bound);
			}
		}
		inline void COpenGL_FunctionTableBase::extGlCreateVertexArrays(GLsizei n, GLuint* arrays)
		{
			// TODO impl in GL
			if (glVertex.pglGenVertexArrays)
				glVertex.pglGenVertexArrays(n, arrays);
			else
				memset(arrays, 0, sizeof(GLuint) * n);
		}
		inline void COpenGL_FunctionTableBase::extGlVertexArrayElementBuffer(GLuint vaobj, GLuint buffer)
		{
			// TODO impl in GL
			if (glBuffer.pglBindBuffer && glVertex.pglBindVertexArray)
			{
				// Save the previous bound vertex array
				GLint restoreVertexArray;
				glGeneral.pglGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
				glVertex.pglBindVertexArray(vaobj);
				glBuffer.pglBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
				glVertex.pglBindVertexArray(restoreVertexArray);
			}
		}
		inline void COpenGL_FunctionTableBase::extGlVertexArrayVertexBuffer(GLuint vaobj, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride)
		{
			// TODO impl in GL
			if (glVertex.pglBindVertexBuffer && glVertex.pglBindVertexArray)
			{
				// Save the previous bound vertex array
				GLint restoreVertexArray;
				glGeneral.pglGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
				glVertex.pglBindVertexArray(vaobj);
				glVertex.pglBindVertexBuffer(bindingindex, buffer, offset, stride);
				glVertex.pglBindVertexArray(restoreVertexArray);
			}
		}
		inline void COpenGL_FunctionTableBase::extGlVertexArrayAttribBinding(GLuint vaobj, GLuint attribindex, GLuint bindingindex)
		{
			// TODO impl in GL
			if (glVertex.pglVertexAttribBinding && glVertex.pglBindVertexArray)
			{
				// Save the previous bound vertex array
				GLint restoreVertexArray;
				glGeneral.pglGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
				glVertex.pglBindVertexArray(vaobj);
				glVertex.pglVertexAttribBinding(attribindex, bindingindex);
				glVertex.pglBindVertexArray(restoreVertexArray);
			}
		}
		inline void COpenGL_FunctionTableBase::extGlEnableVertexArrayAttrib(GLuint vaobj, GLuint index)
		{
			// TODO impl in GL
			if (glVertex.pglEnableVertexAttribArray && glVertex.pglBindVertexArray)
			{
				// Save the previous bound vertex array
				GLint restoreVertexArray;
				glGeneral.pglGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
				glVertex.pglBindVertexArray(vaobj);
				glVertex.pglEnableVertexAttribArray(index);
				glVertex.pglBindVertexArray(restoreVertexArray);
			}
		}
		inline void COpenGL_FunctionTableBase::extGlDisableVertexArrayAttrib(GLuint vaobj, GLuint index)
		{
			// TODO impl in GL
			if (glVertex.pglDisableVertexAttribArray && glVertex.pglBindVertexArray)
			{
				// Save the previous bound vertex array
				GLint restoreVertexArray;
				glGeneral.pglGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
				glVertex.pglBindVertexArray(vaobj);
				glVertex.pglDisableVertexAttribArray(index);
				glVertex.pglBindVertexArray(restoreVertexArray);
			}
		}
		inline void COpenGL_FunctionTableBase::extGlVertexArrayAttribFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset)
		{
			// TODO impl in GL
			if (glVertex.pglVertexAttribFormat && glVertex.pglBindVertexArray)
			{
				// Save the previous bound vertex array
				GLint restoreVertexArray;
				glGeneral.pglGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
				glVertex.pglBindVertexArray(vaobj);
				glVertex.pglVertexAttribFormat(attribindex, size, type, normalized, relativeoffset);
				glVertex.pglBindVertexArray(restoreVertexArray);
			}
		}
		inline void COpenGL_FunctionTableBase::extGlVertexArrayAttribIFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset)
		{
			// TODO impl in GL
			if (glVertex.pglVertexAttribIFormat && glVertex.pglBindVertexArray)
			{
				// Save the previous bound vertex array
				GLint restoreVertexArray;
				glGeneral.pglGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
				glVertex.pglBindVertexArray(vaobj);
				glVertex.pglVertexAttribIFormat(attribindex, size, type, relativeoffset);
				glVertex.pglBindVertexArray(restoreVertexArray);
			}
		}
		inline void COpenGL_FunctionTableBase::extGlVertexArrayBindingDivisor(GLuint vaobj, GLuint bindingindex, GLuint divisor)
		{
			// TODO impl in GL
			if (glVertex.pglVertexBindingDivisor && glVertex.pglBindVertexArray)
			{
				// Save the previous bound vertex array
				GLint restoreVertexArray;
				glGeneral.pglGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
				glVertex.pglBindVertexArray(vaobj);
				glVertex.pglVertexBindingDivisor(bindingindex, divisor);
				glVertex.pglBindVertexArray(restoreVertexArray);
			}
		}
		inline void COpenGL_FunctionTableBase::extGlCreateQueries(GLenum target, GLsizei n, GLuint* ids)
		{
			// TODO impl in GL
			if (glQuery.pglGenQueries)
				glQuery.pglGenQueries(n, ids);
		}
		inline void COpenGL_FunctionTableBase::extGlGetInternalformativ(GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint* params)
		{
			if (glGeneral.pglGetInternalformativ)
				glGeneral.pglGetInternalformativ(target, internalformat, pname, bufSize, params);
		}
		inline void COpenGL_FunctionTableBase::extGlSwapInterval(int interval)
		{
			m_egl->call.peglSwapInterval(interval);
		}


		}		//namespace video
	}		//namespace nbl

#endif