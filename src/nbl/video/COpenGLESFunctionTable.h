#ifndef __NBL_C_OPENGLES_FUNCTION_TABLE_H_INCLUDED__
#define __NBL_C_OPENGLES_FUNCTION_TABLE_H_INCLUDED__

#include "nbl/video/COpenGL_FunctionTableBase.h"
#include "nbl/video/COpenGLESFeatureMap.h"
#define GL_GLEXT_PROTOTYPES
#include "GLES3/gl2ext.h"

namespace nbl {
namespace video
{

/**
Extensions being loaded:
* KHR_debug
* OES_draw_elements_base_vertex
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
*/
class COpenGLESFunctionTable final : public COpenGL_FunctionTableBase
{
public:
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLESgeneral, OpenGLFunctionLoader
		, glEnableiOES
		, glDisableiOES
		, glBlendEquationiOES
		, glBlendEquationSeparateiOES
		, glBlendFunciOES
		, glBlendFuncSeparateiOES
		, glColorMaskiOES
		, glIsEnablediOES
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
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLESdebug, OpenGLFunctionLoader
		, glDebugMessageControlKHR
		, glDebugMessageControl
		, glDebugMessageCallbackKHR
		, glDebugMessageCallback
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLESdrawing, OpenGLFunctionLoader
		, glDrawElementsBaseVertexOES
		, glDrawRangeElementsBaseVertexOES
		, glDrawElementsInstancedBaseVertexOES
	);
	NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(GLESfragment, OpenGLFunctionLoader
		, glMinSampleShadingOES
	);

	GLESgeneral glesGeneral;
	GLESbuffer glesBuffer;
	GLEStexture glesTexture;
	GLESdebug glesDebug;
	GLESdrawing glesDrawing;
	GLESfragment glesFragment;

	COpenGLESFunctionTable(const egl::CEGL* _egl, const COpenGLESFeatureMap* _features) : 
		COpenGL_FunctionTableBase(_egl),
		features(_features),
		glesGeneral(_egl),
		glesBuffer(_egl),
		glesDebug(_egl),
		glesDrawing(_egl),
		glesFragment(_egl)
	{}

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
			os::Printer::log("None of texture view extensions for GLES are supported, cannot create texture view!\n", ELL_ERROR);
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
		if (features->Version >= 320)
			glTexture.pglTexBufferRange(texture, internalformat, buffer);
		else if (glesTexture.pglTexBufferRangeOES)
			glesTexture.pglTexBufferRangeOES(texture, internalformat, buffer);
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
			os::Printer::log("DevSH would like to ask you what are you doing!!??\n", ELL_ERROR);
			return;
		}
		glTexture.pglBindTexture(target, texture);
		glTexture.pglTexStorage2D(target, levels, internalformat, width, height);
		glTexture.pglBindTexture(target, bound);
	}

	void extGlTextureStorage3DMultisample(GLuint texture, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations) override
	{
		if (features->Version >= 320)
			COpenGL_FunctionTableBase::extGlTextureStorage3DMultisample(texture, target, samples, internalformat, width, height, depth, fixedsamplelocations);
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
			os::Printer::log("DevSH would like to ask you what are you doing!!??\n", ELL_ERROR);
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
			os::Printer::log("DevSH would like to ask you what are you doing!!??\n", ELL_ERROR);
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
			os::Printer::log("DevSH would like to ask you what are you doing!!??\n", ELL_ERROR);
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
			COpenGL_FunctionTableBase::extGlNamedFramebufferTexture(framebuffer, attachment, texture, level, textarget);
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
				os::Printer::log("DevSH would like to ask you what are you doing!!??\n", ELL_ERROR);
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
			COpenGL_FunctionTableBase::extGlEnablei(target, index);
		else if (glesGeneral.pglEnableiOES)
			glesGeneral.pglEnableiOES(target, index);
	}
	void extGlDisablei(GLenum target, GLuint index) override
	{
		if (features->Version >= 320)
			COpenGL_FunctionTableBase::extGlDisablei(target, index);
		else if (glesGeneral.pglDisableiOES)
			glesGeneral.pglDisableiOES(target, index);
	}
	void extGlBlendEquationi(GLuint buf, GLenum mode) override
	{
		if (features->Version >= 320)
			COpenGL_FunctionTableBase::extGlBlendEquationi(buf, mode);
		else if (glesGeneral.pglBlendEquationiOES)
			glesGeneral.pglBlendEquationiOES(buf, mode);
	}
	void extGlBlendEquationSeparatei(GLuint buf, GLenum modeRGB, GLenum modeAlpha) override
	{
		if (features->Version >= 320)
			COpenGL_FunctionTableBase::extGlBlendEquationSeparatei(buf, modeRGB, modeAlpha);
		else if (glesGeneral.pglBlendEquationSeparateiOES)
			glesGeneral.pglBlendEquationSeparateiOES(buf, modeRGB, modeAlpha);
	}
	void extGlBlendFunci(GLuint buf, GLenum src, GLenum dst) override
	{
		if (features->Version >= 320)
			COpenGL_FunctionTableBase::extGlBlendFunci(buf, src, dst);
		else if (glesGeneral.pglBlendFunciOES)
			glesGeneral.pglBlendFunciOES(buf, src, dst);
	}
	void extGlBlendFuncSeparatei(GLuint buf, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha) override
	{
		if (features->Version >= 320)
			COpenGL_FunctionTableBase::extGlBlendFuncSeparatei(buf, srcRGB, dstRGB, srcAlpha, dstAlpha);
		else if (glesGeneral.pglBlendFuncSeparateiOES)
			glesGeneral.pglBlendFuncSeparateiOES(buf, srcRGB, dstRGB, srcAlpha, dstAlpha);
	}
	void extGlColorMaski(GLuint buf, GLboolean r, GLboolean g, GLboolean b, GLboolean a) override
	{
		if (features->Version >= 320)
			COpenGL_FunctionTableBase::extGlColorMaski(buf, r, g, b, a);
		else if (glesGeneral.pglColorMaskiOES)
			glesGeneral.pglColorMaskiOES(buf, r, g, b, a);
	}
	GLboolean extGlIsEnabledi(GLenum target, GLuint index) override
	{
		if (features->Version >= 320)
			return COpenGL_FunctionTableBase::extGlIsEnabledi(target, index);
		else if (glesGeneral.pglIsEnablediOES)
			return glesGeneral.pglIsEnablediOES(target, index);
		return GL_FALSE;
	}
	void extGlMinSampleShading(GLfloat value) override
	{
		if (glesFragment.pglMinSampleShadingOES)
			glesFragment.pglMinSampleShadingOES(value);
	}

private:
	const COpenGLESFeatureMap* features;
};

}
}

#undef GL_GLEXT_PROTOTYPES
#endif
