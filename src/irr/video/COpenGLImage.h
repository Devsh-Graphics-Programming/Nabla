#ifndef __C_OPEN_GL_IMAGE_H_INCLUDED__
#define __C_OPEN_GL_IMAGE_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "irr/video/IGPUImage.h"

#include "irr/video/COpenGLCommon.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_


namespace irr
{
namespace video
{

class COpenGLImage final : public IGPUImage, public IDriverMemoryAllocation
{
	protected:
		virtual ~COpenGLImage()
		{
			if (name)
				glDeleteTextures(1,&name);
			#ifdef OPENGL_LEAK_DEBUG
				COpenGLExtensionHandler::textureLeaker.deregisterObj(this);
			#endif // OPENGL_LEAK_DEBUG
		}

		GLenum internalFormat;
		GLenum target;
		GLuint name;

	public:
		//! constructor
		COpenGLImage(IGPUImage::SCreationParams&& _params) : IGPUImage(std::move(_params)),
			internalFormat(GL_INVALID_ENUM), target(GL_INVALID_ENUM), name(0u)
		{
			#ifdef OPENGL_LEAK_DEBUG
				COpenGLExtensionHandler::textureLeaker.registerObj(this);
			#endif // OPENGL_LEAK_DEBUG
			internalFormat = getSizedOpenGLFormatFromOurFormat(params.format);
			switch (params.type)
			{
				case IGPUImage::ET_1D:
					target = GL_TEXTURE_1D_ARRAY;
					COpenGLExtensionHandler::extGlCreateTextures(target, 1, &name);
					COpenGLExtensionHandler::extGlTextureStorage1D(	name, target, params.mipLevels, internalFormat,
																	params.extent.width);
					break;
				case IGPUImage::ET_2D:
					target = GL_TEXTURE_2D_ARRAY;
					COpenGLExtensionHandler::extGlCreateTextures(target, 1, &name);
					COpenGLExtensionHandler::extGlTextureStorage2D(	name, target, params.mipLevels, internalFormat,
																	params.extent.width, params.extent.height);
					break;
				case IGPUImage::ET_3D:
					target = GL_TEXTURE_3D;
					COpenGLExtensionHandler::extGlCreateTextures(target, 1, &name);
					COpenGLExtensionHandler::extGlTextureStorage3D(	name, target, params.mipLevels, internalFormat,
																	params.extent.width, params.extent.height, params.extent.depth);
					break;
				default:
					assert(false);
					break;
			}
		}

		//! returns the opengl texture type
		inline GLuint getOpenGLName() const { return name; }
		//inline GLenum getOpenGLTextureType() const {return target;}

		virtual bool resize(const uint32_t* size, const uint32_t& mipLevels=0);


		inline size_t getAllocationSize() const override { return this->getSize(); }
		inline IDriverMemoryAllocation* getBoundMemory() override { return this; }
		inline const IDriverMemoryAllocation* getBoundMemory() const override { return this; }
		inline size_t getBoundMemoryOffset() const override { return 0ll; }

		inline E_SOURCE_MEMORY_TYPE getType() const override { return ESMT_DEVICE_LOCAL; }
		inline void unmapMemory() override {}
		inline bool isDedicated() const override { return true; }
};


} // end namespace video
} // end namespace irr

#endif // _IRR_COMPILE_WITH_OPENGL_

#endif

#if 0

			//!
			const uint64_t& hasOpenGLNameChanged() const { return TextureNameHasChanged; }

			//! returns the opengl texture type
			virtual GLenum getOpenGLTextureType() const = 0;



			//!
			static bool isInternalFormatCompressed(GLenum format);

			//! Get the OpenGL color format parameters based on the given Irrlicht color format
			static void getOpenGLFormatAndParametersFromColorFormat(const asset::E_FORMAT& format, GLenum& colorformat, GLenum& type); //kill this


		protected:
			
			//! for resizes
			void recreateName(const GLenum& textureType_Target);

			uint64_t TextureNameHasChanged;





		//! .
		class COpenGLFilterableTexture : public ITexture, public COpenGLTexture, public IDriverMemoryAllocation
		{
		public:
			virtual IRenderableVirtualTexture::E_VIRTUAL_TEXTURE_TYPE getVirtualTextureType() const { return IRenderableVirtualTexture::EVTT_OPAQUE_FILTERABLE; }

			//! Get size
			virtual core::dimension2du getRenderableSize() const { return *reinterpret_cast<const core::dimension2du*>(TextureSize); }

			//! returns driver type of texture (=the driver, that created it)
			virtual E_DRIVER_TYPE getDriverType() const { return EDT_OPENGL; }

			//! returns color format of texture
			virtual asset::E_FORMAT getColorFormat() const { return ColorFormat; }

			//! returns pitch of texture (in bytes)
			virtual core::rational<uint32_t> getPitch() const { return asset::getTexelOrBlockBytesize(ColorFormat) * TextureSize[0]; }

			//!
			GLint getOpenGLInternalFormat() const { return InternalFormat; }

			virtual uint32_t getMipMapLevelCount() const { return MipLevelsStored; }

			//! return whether this texture has mipmaps
			virtual bool hasMipMaps() const { return MipLevelsStored > 1; }



#endif
#endif // _IRR_COMPILE_WITH_OPENGL_


#endif

