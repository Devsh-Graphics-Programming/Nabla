#ifndef __NBL_I_OPEN_GL_FUNCTION_TABLE_H_INCLUDED__
#define __NBL_I_OPEN_GL_FUNCTION_TABLE_H_INCLUDED__

#include <atomic>
#include "nbl/video/COpenGLFeatureMap.h"
#include "nbl/core/string/UniqueStringLiteralType.h"
#include "nbl/system/DynamicFunctionCaller.h"
#include "nbl/video/CEGL.h"
#define GL_GLEXT_LEGACY 1
#include "GL/gl.h"
#undef GL_GLEXT_LEGACY
#define GL_GLEXT_PROTOTYPES
#include "GL/glext.h"
#undef GL_GLEXT_PROTOTYPES

namespace nbl {
	namespace video {

		// This class contains pointers to functions common in GL 4.6 and GLES 3.2
		// And implements (at least a common part) extGl* methods which can be implemented with those pointers
		class IOpenGL_FunctionTable
		{
			static std::atomic_uint32_t s_guidGenerator;

		public:
			// tokens common for GL 4.6 and GLES 3.2 assuming presence of some extensions
			static inline constexpr GLenum MAP_PERSISTENT_BIT				= GL_MAP_PERSISTENT_BIT;
			static inline constexpr GLenum MAP_COHERENT_BIT					= GL_MAP_COHERENT_BIT;
			static inline constexpr GLenum DYNAMIC_STORAGE_BIT				= GL_DYNAMIC_STORAGE_BIT;
			static inline constexpr GLenum CLIENT_STORAGE_BIT				= GL_CLIENT_STORAGE_BIT;
			static inline constexpr GLenum BUFFER_IMMUTABLE_STORAGE			= GL_BUFFER_IMMUTABLE_STORAGE;
			static inline constexpr GLenum BUFFER_STORAGE_FLAGS				= GL_BUFFER_STORAGE_FLAGS;
			static inline constexpr GLbitfield CLIENT_MAPPED_BUFFER_BARRIER_BIT = GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT;
			static inline constexpr GLenum TEXTURE_MAX_ANISOTROPY			= GL_TEXTURE_MAX_ANISOTROPY;
			static inline constexpr GLenum TEXTURE_LOD_BIAS					= GL_TEXTURE_LOD_BIAS;
			static inline constexpr GLenum DEPTH_CLAMP						= GL_DEPTH_CLAMP;
			static inline constexpr GLenum FRAMEBUFFER_SRGB					= GL_FRAMEBUFFER_SRGB; // GL_EXT_sRGB_write_control GLES extension (i think we got to require this)
			// clip control
			static inline constexpr GLenum UPPER_LEFT						= GL_UPPER_LEFT;
			static inline constexpr GLenum ZERO_TO_ONE						= GL_ZERO_TO_ONE;

			//desktop GL only
			static inline constexpr GLenum TEXTURE_1D						= GL_TEXTURE_1D;
			static inline constexpr GLenum TEXTURE_1D_ARRAY					= GL_TEXTURE_1D_ARRAY;
			static inline constexpr GLenum MIRROR_CLAMP_TO_EDGE				= GL_MIRROR_CLAMP_TO_EDGE;
			static inline constexpr GLenum MIRROR_CLAMP_TO_BORDER			= GL_MIRROR_CLAMP_TO_BORDER_EXT;
			static inline constexpr GLenum DOUBLE							= GL_DOUBLE;
			static inline constexpr GLenum COLOR_LOGIC_OP					= GL_COLOR_LOGIC_OP;
			static inline constexpr GLenum PRIMITIVE_RESTART				= GL_PRIMITIVE_RESTART;
			static inline constexpr GLenum MULTISAMPLE						= GL_MULTISAMPLE;
			static inline constexpr GLenum POLYGON_OFFSET_POINT				= GL_POLYGON_OFFSET_POINT;
			static inline constexpr GLenum POLYGON_OFFSET_LINE				= GL_POLYGON_OFFSET_LINE;
			static inline constexpr GLenum TEXTURE_CUBE_MAP_SEAMLESS		= GL_TEXTURE_CUBE_MAP_SEAMLESS;

			class OpenGLFunctionLoader final : public system::FuncPtrLoader
			{
				const egl::CEGL* egl;

			public:
				OpenGLFunctionLoader() : egl(nullptr) {}
				explicit OpenGLFunctionLoader(const egl::CEGL* _egl) : egl(_egl) {}

				inline bool isLibraryLoaded() override final
				{
					return true;
				}

				inline void* loadFuncPtr(const char* funcname) override final
				{
					return reinterpret_cast<void*>(egl->call.peglGetProcAddress(funcname));
				}
			};

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
			NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GL_sync, OpenGLFunctionLoader
				, glFenceSync
				, glDeleteSync
				, glClientWaitSync
				, glWaitSync
				, glMemoryBarrier
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
				, glDeleteTextures
				, glGenTextures
				, glTexParameteriv
				, glCopyImageSubData
			);
			NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLshader, OpenGLFunctionLoader
				, glCreateShader
				, glCreateShaderProgramv
				//, glCreateProgramPipelines
				, glGenProgramPipelines
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
				, glFrontFace
				, glCullFace
				, glGetProgramPipelineiv
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
				, glDepthRangef
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
				, glViewport
			);
			NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLcompute, OpenGLFunctionLoader
				, glDispatchCompute
				, glDispatchComputeIndirect
			);

			GLframeBuffer glFramebuffer;
			GL_sync glSync;
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
			const COpenGLFeatureMap* features;

			const uint32_t m_guid;


			virtual bool isGLES() const = 0;


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
			virtual inline void extGlBlitNamedFramebuffer(GLuint readFramebuffer, GLuint drawFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
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
			virtual void extGlVertexArrayAttribLFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset) = 0;
			virtual inline void extGlVertexArrayAttribFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset);
			virtual inline void extGlVertexArrayAttribIFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset);
			virtual inline void extGlVertexArrayBindingDivisor(GLuint vaobj, GLuint bindingindex, GLuint divisor);
			virtual inline void extGlCreateQueries(GLenum target, GLsizei n, GLuint* ids);
			virtual inline void extGlGetInternalformativ(GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint* params);
			virtual void extGlViewportArrayv(GLuint first, GLsizei count, const GLfloat* v) = 0;
			virtual void extGlDepthRangeArrayv(GLuint first, GLsizei count, const double* v) = 0;
			virtual void extGlClipControl(GLenum origin, GLenum depth) = 0;
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
			virtual inline void extGlTextureParameteriv(GLuint texture, GLenum target, GLenum pname, const GLint* params)
			{
				GLint bound;
				switch (target)
				{
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
			virtual void extGlCopyImageSubData(
				GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ,
				GLuint dstName, GLenum dstTarget, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ,
				GLsizei srcWidth, GLsizei srcHeight, GLsizei srcDepth
			) = 0;

			virtual void extGlDrawArraysInstancedBaseInstance(GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance)
			{
				if (baseinstance == 0u)
					glDrawing.pglDrawArraysInstanced(mode, first, count, instancecount);
#ifdef _NBL_DEBUG
				else
				{
					m_logger.log("GlDrawArraysInstancedBaseInstance unsupported!", system::ILogger::ELL_ERROR);
				}
#endif
			}

			virtual void extGlDrawElementsInstancedBaseInstance(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount, GLint baseinstance)
			{
				if (baseinstance == 0u)
					glDrawing.pglDrawElementsInstanced(mode, count, type, indices, instancecount);
#ifdef _NBL_DEBUG
				else
				{
					m_logger.log("GlDrawElementsInstancedBaseInstance unsupported!", system::ILogger::ELL_ERROR);
				}
#endif
			}

			virtual void extGlDrawElementsInstancedBaseVertex(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount, GLint basevertex)
			{
				if (basevertex == 0u)
					glDrawing.pglDrawElementsInstanced(mode, count, type, indices, instancecount);
#ifdef _NBL_DEBUG
				else
				{
					m_logger.log("GlDrawElementsInstancedBaseVertex unsupported!", system::ILogger::ELL_ERROR);
				}
#endif
			}

			virtual void extGlDrawElementsInstancedBaseVertexBaseInstance(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance)
			{
				if (basevertex == 0u)
					extGlDrawElementsInstancedBaseInstance(mode, count, type, indices, instancecount, baseinstance);
				else if (baseinstance == 0u)
					extGlDrawElementsInstancedBaseVertex(mode, count, type, indices, instancecount, basevertex);
#ifdef _NBL_DEBUG
				else
				{
					m_logger.log("GlDrawElementsInstancedBaseVertexBaseInstance unsupported!", system::ILogger::ELL_ERROR);
				}
#endif
			}

			virtual void extGlMultiDrawArraysIndirect(GLenum mode, const void* indirect, GLsizei drawcount, GLsizei stride) = 0;

			virtual void extGlMultiDrawElementsIndirect(GLenum mode, GLenum type, const void* indirect, GLsizei drawcount, GLsizei stride) = 0;

			virtual void extGlLogicOp(GLenum opcode) = 0;

			virtual void extGlPolygonMode(GLenum face, GLenum mode) = 0;


			const COpenGLFeatureMap* getFeatures() const { return features; }

			uint32_t getGUID() const { return m_guid; }

			system::logger_opt_smart_ptr m_logger;

			// constructor
			IOpenGL_FunctionTable(const egl::CEGL* _egl, const COpenGLFeatureMap* _features, system::logger_opt_smart_ptr&& logger) :
				m_logger(std::move(logger)),
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
				m_egl(_egl),
				features(_features),
				m_guid(s_guidGenerator++)
			{

			}

			~IOpenGL_FunctionTable()
			{
				--s_guidGenerator;
			}
		};	// end of class IOpenGL_FunctionTable

		void IOpenGL_FunctionTable::extGlCreateTextures(GLenum target, GLsizei n, GLuint* textures)
		{
			glTexture.pglGenTextures(n, textures);
		}

		inline void IOpenGL_FunctionTable::extGlTextureStorage3D(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth)
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
			case GL_TEXTURE_CUBE_MAP_ARRAY:
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP_ARRAY, &bound);
				break;
			default:
				m_logger.log("DevSH would like to ask you what are you doing!!??\n", system::ILogger::ELL_ERROR);
				return;
			}
			glTexture.pglBindTexture(target, texture);
			glTexture.pglTexStorage3D(target, levels, internalformat, width, height, depth);
			glTexture.pglBindTexture(target, bound);
		}
		inline void IOpenGL_FunctionTable::extGlTextureStorage2DMultisample(GLuint texture, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations)
		{
			
			GLint bound;
			if (target != GL_TEXTURE_2D_MULTISAMPLE)
			{
				m_logger.log("DevSH would like to ask you what are you doing!!??\n", system::ILogger::ELL_ERROR);
				return;
			}
			else
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE, &bound);
			glTexture.pglBindTexture(GL_TEXTURE_2D_MULTISAMPLE, texture);
			glTexture.pglTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, internalformat, width, height, fixedsamplelocations);
			glTexture.pglBindTexture(GL_TEXTURE_2D_MULTISAMPLE, bound);
		}
		inline void IOpenGL_FunctionTable::extGlTextureStorage3DMultisample(GLuint texture, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations)
		{
			GLint bound;
			if (target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
			{
				m_logger.log("DevSH would like to ask you what are you doing!!??\n", system::ILogger::ELL_ERROR);
				return;
			}
			else
				glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY, &bound);
			glTexture.pglBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, texture);
			glTexture.pglTexStorage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, samples, internalformat, width, height, depth, fixedsamplelocations);
			glTexture.pglBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, bound);
		}
		inline void IOpenGL_FunctionTable::extGlTextureSubImage3D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void* pixels)
		{
			
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
				m_logger.log("DevSH would like to ask you what are you doing!!??\n", system::ILogger::ELL_ERROR);
				return;
			}
			glTexture.pglBindTexture(target, texture);
			glTexture.pglTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
			glTexture.pglBindTexture(target, bound);
		}

		inline void IOpenGL_FunctionTable::extGlCompressedTextureSubImage3D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void* data)
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
				m_logger.log("DevSH would like to ask you what are you doing!!??\n", system::ILogger::ELL_ERROR);
				return;
			}
			glTexture.pglBindTexture(target, texture);
			glTexture.pglCompressedTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
			glTexture.pglBindTexture(target, bound);
		}

		inline void IOpenGL_FunctionTable::extGlCreateSamplers(GLsizei n, GLuint* samplers)
		{
			
			if (glTexture.pglGenSamplers)
				glTexture.pglGenSamplers(n, samplers);
			else memset(samplers, 0, 4 * n);
		}
		inline void IOpenGL_FunctionTable::extGlBindSamplers(const GLuint& first, const GLsizei& count, const GLuint* samplers)
		{
			
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
		inline void IOpenGL_FunctionTable::extGlBindImageTextures(GLuint first, GLsizei count, const GLuint* textures, const GLenum* formats)
		{
			
			for (GLsizei i = 0; i < count; i++)
			{
				if (!textures || textures[i] == 0u)
					glTexture.pglBindImageTexture(first + i, 0u, 0u, GL_FALSE, 0, GL_READ_WRITE, GL_R8);
				else
					glTexture.pglBindImageTexture(first + i, textures[i], 0, GL_TRUE, 0, GL_READ_WRITE, formats[i]);
			}
		}
		inline void IOpenGL_FunctionTable::extGlCreateFramebuffers(GLsizei n, GLuint* framebuffers)
		{
			glFramebuffer.pglGenFramebuffers(n, framebuffers);
		}
		inline GLenum IOpenGL_FunctionTable::extGlCheckNamedFramebufferStatus(GLuint framebuffer, GLenum target)
		{
			
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
		inline void IOpenGL_FunctionTable::extGlNamedFramebufferTexture(GLuint framebuffer, GLenum attachment, GLuint texture, GLint level, GLenum textureType)
		{
			GLuint bound;
			glGeneral.pglGetIntegerv(GL_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(&bound));

			if (bound != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
			glFramebuffer.pglFramebufferTexture(GL_FRAMEBUFFER, attachment, texture, level);
			if (bound != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, bound);
		}
		inline void IOpenGL_FunctionTable::extGlBlitNamedFramebuffer(GLuint readFramebuffer, GLuint drawFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter)
		{
			

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
		inline void IOpenGL_FunctionTable::extGlNamedFramebufferReadBuffer(GLuint framebuffer, GLenum mode)
		{
			

			GLint boundFBO;
			glGeneral.pglGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &boundFBO);

			if (static_cast<GLuint>(boundFBO) != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
			glFramebuffer.pglReadBuffer(mode);
			if (static_cast<GLuint>(boundFBO) != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_READ_FRAMEBUFFER, boundFBO);
		}
		inline void IOpenGL_FunctionTable::extGlNamedFramebufferDrawBuffers(GLuint framebuffer, GLsizei n, const GLenum* bufs)
		{
			

			GLint boundFBO;
			glGeneral.pglGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &boundFBO);

			if (static_cast<GLuint>(boundFBO) != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
			glFramebuffer.pglDrawBuffers(n, bufs);
			if (static_cast<GLuint>(boundFBO) != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_DRAW_FRAMEBUFFER, boundFBO);
		}
		inline void IOpenGL_FunctionTable::extGlClearNamedFramebufferiv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLint* value)
		{
			

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
		inline void IOpenGL_FunctionTable::extGlClearNamedFramebufferuiv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLuint* value)
		{
			

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
		inline void IOpenGL_FunctionTable::extGlClearNamedFramebufferfv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLfloat* value)
		{
			

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
		inline void IOpenGL_FunctionTable::extGlClearNamedFramebufferfi(GLuint framebuffer, GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil)
		{
			

			GLint boundFBO = -1;
			glGeneral.pglGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &boundFBO);
			if (boundFBO < 0)
				return;
			glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
			glFramebuffer.pglClearBufferfi(buffer, drawbuffer, depth, stencil);
			glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, boundFBO);
		}
		inline void IOpenGL_FunctionTable::extGlCreateBuffers(GLsizei n, GLuint* buffers)
		{
			

			if (glBuffer.pglGenBuffers)
				glBuffer.pglGenBuffers(n, buffers);
			else if (buffers)
				memset(buffers, 0, n * sizeof(GLuint));
		}
		inline void IOpenGL_FunctionTable::extGlBindBuffersBase(const GLenum& target, const GLuint& first, const GLsizei& count, const GLuint* buffers)
		{
			
			for (GLsizei i = 0; i < count; i++)
			{
				if (glBuffer.pglBindBufferBase)
					glBuffer.pglBindBufferBase(target, first + i, buffers ? buffers[i] : 0);
			}
		}
		inline void IOpenGL_FunctionTable::extGlBindBuffersRange(const GLenum& target, const GLuint& first, const GLsizei& count, const GLuint* buffers, const GLintptr* offsets, const GLsizeiptr* sizes)
		{
			
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
		inline void IOpenGL_FunctionTable::extGlNamedBufferSubData(GLuint buffer, GLintptr offset, GLsizeiptr size, const void* data)
		{
			
			GLint bound;
			glGeneral.pglGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bound);
			glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, buffer);
			glBuffer.pglBufferSubData(GL_ARRAY_BUFFER, offset, size, data);
			glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, bound);
		}
		inline void* IOpenGL_FunctionTable::extGlMapNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length, GLbitfield access)
		{
			
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
		inline void IOpenGL_FunctionTable::extGlFlushMappedNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length)
		{
			
			if (glBuffer.pglFlushMappedBufferRange && glBuffer.pglBindBuffer)
			{
				GLint bound;
				glGeneral.pglGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bound);
				glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, buffer);
				glBuffer.pglFlushMappedBufferRange(GL_ARRAY_BUFFER, offset, length);
				glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, bound);
			}
		}
		inline GLboolean IOpenGL_FunctionTable::extGlUnmapNamedBuffer(GLuint buffer)
		{
			
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

		inline void IOpenGL_FunctionTable::extGlCopyNamedBufferSubData(GLuint readBuffer, GLuint writeBuffer, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size)
		{
			
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
		inline GLboolean IOpenGL_FunctionTable::extGlIsBuffer(GLuint buffer)
		{
			if (glBuffer.pglIsBuffer)
				return glBuffer.pglIsBuffer(buffer);
			return false;
		}
		inline void IOpenGL_FunctionTable::extGlGetNamedBufferParameteriv(const GLuint& buffer, const GLenum& value, GLint* data)
		{
			
			if (glBuffer.pglGetBufferParameteriv && glBuffer.pglBindBuffer)
			{
				GLint bound;
				glGeneral.pglGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bound);
				glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, buffer);
				glBuffer.pglGetBufferParameteriv(GL_ARRAY_BUFFER, value, data);
				glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, bound);
			}
		}
		inline void IOpenGL_FunctionTable::extGlGetNamedBufferParameteri64v(const GLuint& buffer, const GLenum& value, GLint64* data)
		{
			
			if (glBuffer.pglGetBufferParameteri64v && glBuffer.pglBindBuffer)
			{
				GLint bound;
				glGeneral.pglGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bound);
				glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, buffer);
				glBuffer.pglGetBufferParameteri64v(GL_ARRAY_BUFFER, value, data);
				glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, bound);
			}
		}
		inline void IOpenGL_FunctionTable::extGlCreateVertexArrays(GLsizei n, GLuint* arrays)
		{
			
			if (glVertex.pglGenVertexArrays)
				glVertex.pglGenVertexArrays(n, arrays);
			else
				memset(arrays, 0, sizeof(GLuint) * n);
		}
		inline void IOpenGL_FunctionTable::extGlVertexArrayElementBuffer(GLuint vaobj, GLuint buffer)
		{
			
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
		inline void IOpenGL_FunctionTable::extGlVertexArrayVertexBuffer(GLuint vaobj, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride)
		{
			
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
		inline void IOpenGL_FunctionTable::extGlVertexArrayAttribBinding(GLuint vaobj, GLuint attribindex, GLuint bindingindex)
		{
			
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
		inline void IOpenGL_FunctionTable::extGlEnableVertexArrayAttrib(GLuint vaobj, GLuint index)
		{
			
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
		inline void IOpenGL_FunctionTable::extGlDisableVertexArrayAttrib(GLuint vaobj, GLuint index)
		{
			
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
		inline void IOpenGL_FunctionTable::extGlVertexArrayAttribFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset)
		{
			
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
		inline void IOpenGL_FunctionTable::extGlVertexArrayAttribIFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset)
		{
			
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
		inline void IOpenGL_FunctionTable::extGlVertexArrayBindingDivisor(GLuint vaobj, GLuint bindingindex, GLuint divisor)
		{
			
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
		inline void IOpenGL_FunctionTable::extGlCreateQueries(GLenum target, GLsizei n, GLuint* ids)
		{
			
			if (glQuery.pglGenQueries)
				glQuery.pglGenQueries(n, ids);
		}
		inline void IOpenGL_FunctionTable::extGlGetInternalformativ(GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint* params)
		{
			if (glGeneral.pglGetInternalformativ)
				glGeneral.pglGetInternalformativ(target, internalformat, pname, bufSize, params);
		}
		inline void IOpenGL_FunctionTable::extGlSwapInterval(int interval)
		{
			m_egl->call.peglSwapInterval(m_egl->display, interval);
		}


		}		//namespace video
	}		//namespace nbl

#endif