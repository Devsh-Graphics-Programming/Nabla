#ifndef __NBL_C_OPENGLES_FUNCTION_TABLE_H_INCLUDED__
#define __NBL_C_OPENGLES_FUNCTION_TABLE_H_INCLUDED__

#include "nbl/video/IOpenGL_FunctionTable.h"
#include "nbl/video/COpenGLFeatureMap.h"

#define GL_GLEXT_PROTOTYPES
#define GL_APICALL extern
#define GL_APIENTRY // im not sure about calling convention...
#undef GL_KHR_debug
#include "GLES3/gl2ext.h"

namespace nbl::video
{

/**
Extensions being loaded:
* KHR_debug
* GL_EXT_base_instance
* GL_EXT_draw_elements_base_vertex
* GL_OES_draw_elements_base_vertex
* GL_EXT_multi_draw_indirect
* OES_copy_image
* OES_draw_buffers_indexed
* EXT_color_buffer_float
* OES_texture_buffer
* OES_texture_cube_map_array
* OES_sample_shading
* OES_shader_multisample_interpolation
* OES_sample_variables
* GL_OES_shader_image_atomic
* GL_OES_texture_view
* GL_EXT_texture_view
* GL_OES_viewport_array
* GL_EXT_multi_draw_indirect
* GL_EXT_clip_control
*/
class COpenGLESFunctionTable final : public IOpenGL_FunctionTable
{
public:
	using features_t = COpenGLFeatureMap;
	constexpr static inline auto EGL_API_TYPE = EGL_OPENGL_ES_API;

	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLESgeneral, OpenGLFunctionLoader
		, glEnableiOES
		, glDisableiOES
		, glBlendEquationiOES
		, glBlendEquationSeparateiOES
		, glBlendFunciOES
		, glBlendFuncSeparateiOES
		, glColorMaskiOES
		, glIsEnablediOES
		, glDepthRangeArrayfvOES
		, glDepthRangef
		, glViewportArrayvOES
		, glClipControlEXT
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLESbuffer, OpenGLFunctionLoader
		, glBufferStorageEXT
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLEStexture, OpenGLFunctionLoader
		, glTexBufferOES
		, glTexBufferRangeOES
		, glCopyImageSubDataOES
		, glTextureViewOES
		, glTextureViewEXT
		, glCopyImageSubDataEXT
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLESdebug, OpenGLFunctionLoader
		, glDebugMessageControlKHR
		, glDebugMessageControl
		, glDebugMessageCallbackKHR
		, glDebugMessageCallback
		, glLabelObjectEXT
		, glObjectLabel // since ES 3.2
		, glGetObjectLabelEXT
		, glGetObjectLabel // since ES 3.2
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLESdrawing, OpenGLFunctionLoader
		, glDrawElementsBaseVertexEXT
		, glDrawRangeElementsBaseVertexEXT
		, glDrawElementsInstancedBaseVertexEXT
		, glDrawElementsInstancedBaseInstanceEXT
		, glDrawArraysInstancedBaseInstanceEXT
		, glDrawElementsBaseVertexOES
		, glDrawElementsInstancedBaseVertexOES
		, glDrawElementsInstancedBaseVertexBaseInstanceEXT
		, glDrawRangeElementsBaseVertexOES
		, glMultiDrawArraysIndirectEXT
		, glMultiDrawElementsIndirectEXT
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLESfragment, OpenGLFunctionLoader
		, glMinSampleShading
		, glMinSampleShadingOES
	);

	GLESgeneral glesGeneral;
	GLESbuffer glesBuffer;
	GLEStexture glesTexture;
	GLESdebug glesDebug;
	GLESdrawing glesDrawing;
	GLESfragment glesFragment;

	COpenGLESFunctionTable(const egl::CEGL* _egl, const COpenGLFeatureMap* _features, system::logger_opt_smart_ptr&& logger) :
		IOpenGL_FunctionTable(_egl, _features, std::move(logger)),
		glesGeneral(_egl),
		glesBuffer(_egl),
		glesTexture(_egl),
		glesDebug(_egl),
		glesDrawing(_egl),
		glesFragment(_egl)
	{}

	bool isGLES() const override { return true; }

	void extGlDebugMessageControl(GLenum source, GLenum type, GLenum severity, GLsizei count, const GLuint* ids, GLboolean enabled) override
	{
		if (features->Version >= 320)
			glesDebug.pglDebugMessageControl(source, type, severity, count, ids, enabled);
		else if (glesDebug.pglDebugMessageControlKHR)
			glesDebug.pglDebugMessageControlKHR(source, type, severity, count, ids, enabled);
	}
	void extGlDebugMessageCallback(GLDebugCallbackType callback, const void* userParam) override
	{
		if (features->Version >= 320)
			glesDebug.pglDebugMessageCallback(callback, userParam);
		else if (glesDebug.pglDebugMessageCallbackKHR)
			glesDebug.pglDebugMessageCallbackKHR(callback, userParam);
	}

	void extGlBindTextures(const GLuint& first, const GLsizei& count, const GLuint* textures, const GLenum* targets) override
	{
		const GLenum supportedTargets[] = { GL_TEXTURE_2D,
											GL_TEXTURE_3D,GL_TEXTURE_CUBE_MAP,
											GL_TEXTURE_2D_ARRAY,GL_TEXTURE_BUFFER,
											GL_TEXTURE_CUBE_MAP_ARRAY,GL_TEXTURE_2D_MULTISAMPLE,GL_TEXTURE_2D_MULTISAMPLE_ARRAY };

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

	void extGlTextureView(GLuint texture, GLenum target, GLuint origtexture, GLenum internalformat, GLuint minlevel, GLuint numlevels, GLuint minlayer, GLuint numlayers) override
	{
		if (glesTexture.pglTextureViewOES)
			glesTexture.pglTextureViewOES(texture, target, origtexture, internalformat, minlevel, numlevels, minlayer, numlayers);
		else if (glesTexture.pglTextureViewEXT)
			glesTexture.pglTextureViewEXT(texture, target, origtexture, internalformat, minlevel, numlevels, minlayer, numlayers);
		else
		{
			m_logger.log("None of texture view extensions for GLES are supported, cannot create texture view!\n", system::ILogger::ELL_ERROR);
		}
	}

	void extGlTextureBuffer(GLuint texture, GLenum internalformat, GLuint buffer) override
	{
		if (features->Version >= 320)
			glTexture.pglTexBuffer(texture, internalformat, buffer);
		else if (glesTexture.pglTexBufferOES)
			glesTexture.pglTexBufferOES(texture, internalformat, buffer);
	}
	void extGlTextureBufferRange(GLuint texture, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizei length) override
	{
		GLint bound;
		glGeneral.pglGetIntegerv(GL_TEXTURE_BINDING_BUFFER, &bound);

		glTexture.pglBindTexture(GL_TEXTURE_BUFFER, texture);
		if (features->Version >= 320)
			glTexture.pglTexBufferRange(GL_TEXTURE_BUFFER, internalformat, buffer, offset, length);
		else if (glesTexture.pglTexBufferRangeOES)
			glesTexture.pglTexBufferRangeOES(GL_TEXTURE_BUFFER, internalformat, buffer, offset, length);

		glTexture.pglBindTexture(GL_TEXTURE_BUFFER, bound);
	}

	void extGlTextureStorage2D(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height) override
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
		default:
			m_logger.log("DevSH would like to ask you what are you doing!!??\n", system::ILogger::ELL_ERROR);
			return;
		}
		glTexture.pglBindTexture(target, texture);
		glTexture.pglTexStorage2D(target, levels, internalformat, width, height);
		glTexture.pglBindTexture(target, bound);
	}

	void extGlTextureStorage3DMultisample(GLuint texture, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations) override
	{
		if (features->Version >= 320)
			IOpenGL_FunctionTable::extGlTextureStorage3DMultisample(texture, target, samples, internalformat, width, height, depth, fixedsamplelocations);
	}

	void extGlTextureSubImage2D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void* pixels) override
	{
		GLint bound;
		switch (target)
		{
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
		default:
			m_logger.log("DevSH would like to ask you what are you doing!!??\n", system::ILogger::ELL_ERROR);
			return;
		}
		glTexture.pglBindTexture(target, texture);
		glTexture.pglTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixels);
		glTexture.pglBindTexture(target, bound);
	}

	void extGlCompressedTextureSubImage2D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void* data) override
	{
		GLint bound;
		switch (target)
		{
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

	void extGlGenerateTextureMipmap(GLuint texture, GLenum target) override
	{
		GLint bound;
		switch (target)
		{
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
		default:
			m_logger.log("DevSH would like to ask you what are you doing!!??\n", system::ILogger::ELL_ERROR);
			return;
		}
		glTexture.pglBindTexture(target, texture);
		glTexture.pglGenerateMipmap(target);
		glTexture.pglBindTexture(target, bound); 
	}

	void extGlNamedFramebufferDrawBuffer(GLuint framebuffer, GLenum buf) override
	{
		GLint boundFBO;
		glGeneral.pglGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &boundFBO);

		if (static_cast<GLuint>(boundFBO) != framebuffer)
			glFramebuffer.pglBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);

		GLint maxColorAttachments;
		glGeneral.pglGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &maxColorAttachments);
		const GLenum none[8]{ GL_NONE, GL_NONE, GL_NONE, GL_NONE, GL_NONE, GL_NONE, GL_NONE, GL_NONE };
		// glDrawBuffer will set the draw buffer for fragment colors other than zero to GL_NONE.
		glFramebuffer.pglDrawBuffers(maxColorAttachments, none);
		glFramebuffer.pglDrawBuffers(1, &buf);

		if (static_cast<GLuint>(boundFBO) != framebuffer)
			glFramebuffer.pglBindFramebuffer(GL_DRAW_FRAMEBUFFER, boundFBO);
	}

	void extGlNamedFramebufferTexture(GLuint framebuffer, GLenum attachment, GLuint texture, GLint level, GLenum textarget) override
	{
		if (features->Version >= 320)
			IOpenGL_FunctionTable::extGlNamedFramebufferTexture(framebuffer, attachment, texture, level, textarget);
		else
		{
			GLuint bound;
			glGeneral.pglGetIntegerv(GL_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(&bound));

			if (bound != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

			switch (textarget)
			{
			case GL_TEXTURE_2D:
			case GL_TEXTURE_2D_MULTISAMPLE:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
				glFramebuffer.pglFramebufferTexture2D(GL_FRAMEBUFFER, attachment, textarget, texture, level);
				break;
			default:
				m_logger.log("DevSH would like to ask you what are you doing!!??\n", system::ILogger::ELL_ERROR);
				break;
			}

			if (bound != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, bound);
		}
	}

	void extGlNamedFramebufferTextureLayer(GLuint framebuffer, GLenum attachment, GLuint texture, GLenum textureType, GLint level, GLint layer) override
	{
		if (textureType != GL_TEXTURE_CUBE_MAP)
		{
			GLuint bound;
			glGeneral.pglGetIntegerv(GL_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(&bound));

			if (bound != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
			glFramebuffer.pglFramebufferTextureLayer(GL_FRAMEBUFFER, attachment, texture, level, layer);
			if (bound != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, bound);
		}
		else
		{
			constexpr GLenum CubeMapFaceToCubeMapFaceGLenum[] = {
				GL_TEXTURE_CUBE_MAP_POSITIVE_X,GL_TEXTURE_CUBE_MAP_NEGATIVE_X,GL_TEXTURE_CUBE_MAP_POSITIVE_Y,GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,GL_TEXTURE_CUBE_MAP_POSITIVE_Z,GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
			};

			GLuint bound;
			glGeneral.pglGetIntegerv(GL_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(&bound));

			if (bound != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
			glFramebuffer.pglFramebufferTexture2D(GL_FRAMEBUFFER, attachment, CubeMapFaceToCubeMapFaceGLenum[layer], texture, level);
			if (bound != framebuffer)
				glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, bound);
		}
	}

	void extGlNamedBufferStorage(GLuint buffer, GLsizeiptr size, const void* data, GLbitfield flags) override
	{
		if (glesBuffer.pglBufferStorageEXT && glBuffer.pglBindBuffer)
		{
			GLint bound;
			glGeneral.pglGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bound);
			glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, buffer);
			glesBuffer.pglBufferStorageEXT(GL_ARRAY_BUFFER, size, data, flags);
			glBuffer.pglBindBuffer(GL_ARRAY_BUFFER, bound);
		}
	}

	void extGlClearNamedBufferData(GLuint buffer, GLenum internalformat, GLenum format, GLenum type, const void* data) override
	{
		// TODO need some workaround (theres no glClearbufferData in GLES)
	}
	void extGlClearNamedBufferSubData(GLuint buffer, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type, const void* data) override
	{
		// TODO need some workaround (theres no glClearbufferSubData in GLES)
	}

	void extGlEnablei(GLenum target, GLuint index) override
	{
		if (features->Version >= 320)
			IOpenGL_FunctionTable::extGlEnablei(target, index);
		else if (glesGeneral.pglEnableiOES)
			glesGeneral.pglEnableiOES(target, index);
	}
	void extGlDisablei(GLenum target, GLuint index) override
	{
		if (features->Version >= 320)
			IOpenGL_FunctionTable::extGlDisablei(target, index);
		else if (glesGeneral.pglDisableiOES)
			glesGeneral.pglDisableiOES(target, index);
	}
	void extGlBlendEquationi(GLuint buf, GLenum mode) override
	{
		if (features->Version >= 320)
			IOpenGL_FunctionTable::extGlBlendEquationi(buf, mode);
		else if (glesGeneral.pglBlendEquationiOES)
			glesGeneral.pglBlendEquationiOES(buf, mode);
	}
	void extGlBlendEquationSeparatei(GLuint buf, GLenum modeRGB, GLenum modeAlpha) override
	{
		if (features->Version >= 320)
			IOpenGL_FunctionTable::extGlBlendEquationSeparatei(buf, modeRGB, modeAlpha);
		else if (glesGeneral.pglBlendEquationSeparateiOES)
			glesGeneral.pglBlendEquationSeparateiOES(buf, modeRGB, modeAlpha);
	}
	void extGlBlendFunci(GLuint buf, GLenum src, GLenum dst) override
	{
		if (features->Version >= 320)
			IOpenGL_FunctionTable::extGlBlendFunci(buf, src, dst);
		else if (glesGeneral.pglBlendFunciOES)
			glesGeneral.pglBlendFunciOES(buf, src, dst);
	}
	void extGlBlendFuncSeparatei(GLuint buf, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha) override
	{
		if (features->Version >= 320)
			IOpenGL_FunctionTable::extGlBlendFuncSeparatei(buf, srcRGB, dstRGB, srcAlpha, dstAlpha);
		else if (glesGeneral.pglBlendFuncSeparateiOES)
			glesGeneral.pglBlendFuncSeparateiOES(buf, srcRGB, dstRGB, srcAlpha, dstAlpha);
	}
	void extGlColorMaski(GLuint buf, GLboolean r, GLboolean g, GLboolean b, GLboolean a) override
	{
		if (features->Version >= 320)
			IOpenGL_FunctionTable::extGlColorMaski(buf, r, g, b, a);
		else if (glesGeneral.pglColorMaskiOES)
			glesGeneral.pglColorMaskiOES(buf, r, g, b, a);
	}
	GLboolean extGlIsEnabledi(GLenum target, GLuint index) override
	{
		if (features->Version >= 320)
			return IOpenGL_FunctionTable::extGlIsEnabledi(target, index);
		else if (glesGeneral.pglIsEnablediOES)
			return glesGeneral.pglIsEnablediOES(target, index);
		return GL_FALSE;
	}
	void extGlMinSampleShading(GLfloat value) override
	{
		if (glesFragment.pglMinSampleShading)
			glesFragment.pglMinSampleShading(value);
		else if (glesFragment.pglMinSampleShadingOES)
			glesFragment.pglMinSampleShadingOES(value);
	}
	void extGlCopyImageSubData(
		GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ,
		GLuint dstName, GLenum dstTarget, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ,
		GLsizei srcWidth, GLsizei srcHeight, GLsizei srcDepth
	) override
	{
		if (features->Version >= 320)
			glTexture.pglCopyImageSubData(srcName, srcTarget, srcLevel, srcX, srcY, srcZ, dstName, dstTarget, dstLevel, dstX, dstY, dstZ, srcWidth, srcHeight, srcDepth);
		else if (features->isFeatureAvailable(COpenGLFeatureMap::NBL_OES_copy_image))
			glesTexture.pglCopyImageSubDataOES(srcName, srcTarget, srcLevel, srcX, srcY, srcZ, dstName, dstTarget, dstLevel, dstX, dstY, dstZ, srcWidth, srcHeight, srcDepth);
		else if (features->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_copy_image))
			glesTexture.pglCopyImageSubDataEXT(srcName, srcTarget, srcLevel, srcX, srcY, srcZ, dstName, dstTarget, dstLevel, dstX, dstY, dstZ, srcWidth, srcHeight, srcDepth);
	}

	void extGlDrawArraysInstancedBaseInstance(GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance) override
	{
		if (features->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_base_instance))
			glesDrawing.pglDrawArraysInstancedBaseInstanceEXT(mode, first, count, instancecount, baseinstance);
		else 
			IOpenGL_FunctionTable::extGlDrawArraysInstancedBaseInstance(mode, first, count, instancecount, baseinstance);
	}

	void extGlDrawElementsInstancedBaseInstance(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount, GLint baseinstance) override
	{
		if (features->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_base_instance))
			glesDrawing.pglDrawElementsInstancedBaseInstanceEXT(mode, count, type, indices, instancecount, baseinstance);
		else
			IOpenGL_FunctionTable::extGlDrawElementsInstancedBaseInstance(mode, count, type, indices, instancecount, baseinstance);
	}

	void extGlVertexArrayAttribLFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset) override
	{
		assert(false);
	}

	void extGlDrawElementsInstancedBaseVertex(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount, GLint basevertex) override
	{
		if (features->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_draw_elements_base_vertex))
			glesDrawing.pglDrawElementsInstancedBaseVertexEXT(mode, count, type, indices, instancecount, basevertex);
		else if (features->isFeatureAvailable(COpenGLFeatureMap::NBL_OES_draw_elements_base_vertex))
			glesDrawing.pglDrawElementsInstancedBaseVertexOES(mode, count, type, indices, instancecount, basevertex);
		else
			IOpenGL_FunctionTable::extGlDrawElementsInstancedBaseVertex(mode, count, type, indices, instancecount, basevertex);
	}

	void extGlDrawElementsInstancedBaseVertexBaseInstance(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance) override
	{
		if (features->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_base_instance))
			glesDrawing.pglDrawElementsInstancedBaseVertexBaseInstanceEXT(mode, count, type, indices, instancecount, basevertex, baseinstance);
		else
			IOpenGL_FunctionTable::extGlDrawElementsInstancedBaseVertexBaseInstance(mode, count, type, indices, instancecount, basevertex, baseinstance);
	}

	void extGlMultiDrawArraysIndirect(GLenum mode, const void* indirect, GLsizei drawcount, GLsizei stride) override
	{
		if (features->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_multi_draw_indirect))
			glesDrawing.pglMultiDrawArraysIndirectEXT(mode, indirect, drawcount, stride);
		else if (drawcount == 1)
			glDrawing.pglDrawArraysIndirect(mode, indirect);
#ifdef _NBL_DEBUG
		else
		{
			m_logger.log("MDI not supported!", system::ILogger::ELL_ERROR);
		}
#endif
	}
	void extGlMultiDrawElementsIndirect(GLenum mode, GLenum type, const void* indirect, GLsizei drawcount, GLsizei stride) override
	{
		if (features->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_multi_draw_indirect))
			glesDrawing.pglMultiDrawElementsIndirectEXT(mode, type, indirect, drawcount, stride);
		else if (drawcount == 1)
			glDrawing.pglDrawElementsIndirect(mode, type, indirect);
#ifdef _NBL_DEBUG
		else
		{
			m_logger.log("MDI not supported!", system::ILogger::ELL_ERROR);
		}
#endif
	}

	void extGlMultiDrawArraysIndirectCount(GLenum mode, const void* indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride) override
	{
		m_logger.log("MDI with DrawCount not supported!", system::ILogger::ELL_ERROR);
	}
	void extGlMultiDrawElementsIndirectCount(GLenum mode, GLenum type, const void* indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride) override
	{
		m_logger.log("MDI with DrawCount not supported!", system::ILogger::ELL_ERROR);
	}

	void extGlViewportArrayv(GLuint first, GLsizei count, const GLfloat* v) override
	{
		if (features->isFeatureAvailable(COpenGLFeatureMap::NBL_OES_viewport_array))
			glesGeneral.pglViewportArrayvOES(first, count, v);
		else if (first == 0u)
		{
#ifdef _NBL_DEBUG
			m_logger.log("Multiple viewports not supported, setting only first viewport!");
#endif
			glGeneral.pglViewport(v[0], v[1], v[2], v[3]);
		}
#ifdef _NBL_DEBUG
		else
		{
			m_logger.log("Multiple viewports not supported!");
		}
#endif
	}
	void extGlDepthRangeArrayv(GLuint first, GLsizei count, const GLdouble* v) override
	{
		GLfloat fv[2 * 16];
		for (GLsizei i = 0; i < 2*count; ++i)
		{
			fv[i] = v[i];
		}

		if (features->isFeatureAvailable(COpenGLFeatureMap::NBL_OES_viewport_array))
		{
			glesGeneral.pglDepthRangeArrayfvOES(first, count, fv);
		}
		else if (first == 0)
		{
#ifdef _NBL_DEBUG
			m_logger.log("Multiple viewports not supported, setting only first viewport!");
#endif
			glesGeneral.pglDepthRangef(fv[0], fv[1]);
		}
#ifdef _NBL_DEBUG
		else
		{
			m_logger.log("Multiple viewports not supported!");
		}
#endif
	}

	void extGlClipControl(GLenum origin, GLenum depth) override
	{
		if (features->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_clip_control))
			glesGeneral.pglClipControlEXT(origin, depth);
#ifdef _NBL_DEBUG
		else
			m_logger.log("GL_EXT_clip_control not supported on GLES backend!", system::ILogger::ELL_ERROR);
#endif
	}

	void extGlLogicOp(GLenum opcode) override
	{
		assert(false);
	}

	void extGlPolygonMode(GLenum face, GLenum mode) override
	{
		assert(false);
	}

	void extGlObjectLabel(GLenum id, GLuint name, GLsizei length, const char* label) override
	{
		if (features->Version >= 320u)
			glesDebug.pglObjectLabel(id, name, length, label);
		else if (features->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_debug_label))
		{
			switch (id)
			{
			case GL_BUFFER:				id = GL_BUFFER_OBJECT_EXT; break;
			case GL_SHADER:				id = GL_SHADER_OBJECT_EXT; break;
			case GL_PROGRAM:			id = GL_PROGRAM_OBJECT_EXT; break;
			case GL_VERTEX_ARRAY:		id = GL_VERTEX_ARRAY_OBJECT_EXT; break;
			case GL_QUERY:				id = GL_QUERY_OBJECT_EXT; break;
			case GL_PROGRAM_PIPELINE:	id = GL_PROGRAM_PIPELINE_OBJECT_EXT; break;
			// other types does not need to be translated into extension-specific enums
			}
			glesDebug.pglLabelObjectEXT(id, name, length, label);
		}
	}

	void extGlGetObjectLabel(GLenum identifier, GLuint name, GLsizei bufsize, GLsizei* length, GLchar* label) override
	{
		if (features->Version >= 320u)
			glesDebug.pglGetObjectLabel(identifier, name, bufsize, length, label);
		else if (features->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_debug_label))
		{
			switch (identifier)
			{
			case GL_BUFFER:				identifier = GL_BUFFER_OBJECT_EXT; break;
			case GL_SHADER:				identifier = GL_SHADER_OBJECT_EXT; break;
			case GL_PROGRAM:			identifier = GL_PROGRAM_OBJECT_EXT; break;
			case GL_VERTEX_ARRAY:		identifier = GL_VERTEX_ARRAY_OBJECT_EXT; break;
			case GL_QUERY:				identifier = GL_QUERY_OBJECT_EXT; break;
			case GL_PROGRAM_PIPELINE:	identifier = GL_PROGRAM_PIPELINE_OBJECT_EXT; break;
			// other types does not need to be translated into extension-specific enums
			}
			glesDebug.pglGetObjectLabelEXT(identifier, name, bufsize, length, label);
		}
	}
};

}

#undef GL_GLEXT_PROTOTYPES
#endif
