// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

#include "irrTypes.h"
#include "COpenGLTexture.h"
#include "COpenGLDriver.h"
#include "os.h"
#include "CColorConverter.h"

#include "irrString.h"

namespace irr
{
namespace video
{

//! constructor for usual textures
COpenGLTexture::COpenGLTexture(IImage* origImage, const io::path& name, void* mipmapData, COpenGLDriver* driver, u32 mipmapLevels)
	: ITexture(name), ColorFormat(ECF_A8R8G8B8), Driver(driver),
	TextureName(0), InternalFormat(GL_RGBA), MipLevelsStored(0x1),
	TextureNameHasChanged(0)
{
	#ifdef _DEBUG
	setDebugName("COpenGLTexture");
	#endif
	core::dimension2du ImageSize2 = getImageValues(origImage);

    u32 defaultMipMapCount = 1u+u32(floorf(log2(float(core::min_(core::max_(TextureSize.Width,TextureSize.Height),Driver->MaxTextureSize)))));
    if (mipmapLevels==0)
    {
        HasMipMaps = Driver->getTextureCreationFlag(ETCF_CREATE_MIP_MAPS);
        MipLevelsStored = HasMipMaps ? defaultMipMapCount:1;
    }
    else
    {
        HasMipMaps = mipmapLevels>1;
        MipLevelsStored = core::min_(mipmapLevels,defaultMipMapCount);
    }


    Driver->extGlCreateTextures(GL_TEXTURE_2D,1,&TextureName);

    GLenum PixelFormat = GL_BGRA;
    GLenum PixelType = GL_UNSIGNED_BYTE;
	InternalFormat = getOpenGLFormatAndParametersFromColorFormat(ColorFormat, PixelFormat, PixelType);
    IImage* Image = 0;

	if (ImageSize2!=TextureSize||ColorFormat!=origImage->getColorFormat())
	{
		Image = Driver->createImage(ColorFormat, TextureSize);
		// scale texture
		origImage->copyToScaling(Image);
		os::Printer::log("DevSH is very disappointed with you for creating a weird size texture.", ELL_ERROR);
	}
	void* data = Image ? Image->lock():origImage->lock();
    //! we're going to have problems with uploading lower mip levels
    uint32_t bpp = IImage::getBitsPerPixelFromFormat(ColorFormat);

    Driver->extGlTextureStorage2D(TextureName,GL_TEXTURE_2D,MipLevelsStored,InternalFormat,TextureSize.Width, TextureSize.Height);


    uint8_t* tmpMipmapDataPTr = NULL;
    if (data)
    {
        size_t levelByteSize;
        if (ColorFormat>=ECF_RGB_BC1&&ColorFormat<=ECF_RG_BC5)
        {
            levelByteSize = (((TextureSize.Width+3)&0xfffffc)*((TextureSize.Height+3)&0xfffffc)*bpp)/8;
            Driver->extGlCompressedTextureSubImage2D(TextureName,GL_TEXTURE_2D,0,0,0,TextureSize.Width, TextureSize.Height, InternalFormat, levelByteSize,data);
        }
        else
        {
            levelByteSize = (TextureSize.Width*TextureSize.Height*bpp)/8;
            Driver->setPixelUnpackAlignment((TextureSize.Width*bpp)/8,data);
            Driver->extGlTextureSubImage2D(TextureName,GL_TEXTURE_2D,0,0,0,TextureSize.Width, TextureSize.Height, PixelFormat,PixelType,data);
        }

       tmpMipmapDataPTr = ((uint8_t*)data)+levelByteSize;
    }

    if (mipmapLevels>1&&tmpMipmapDataPTr)
    {
        for (u32 i=1; i<MipLevelsStored; i++)
        {
            core::dimension2d<u32> tmpSize = TextureSize;
            tmpSize.Width = core::max_(tmpSize.Width/(0x1u<<i),0x1u);
            tmpSize.Height = core::max_(tmpSize.Height/(0x1u<<i),0x1u);
            size_t levelByteSize;
            if (ColorFormat>=ECF_RGB_BC1&&ColorFormat<=ECF_RG_BC5)
            {
                levelByteSize = (((tmpSize.Width+3)&0xfffffc)*((tmpSize.Height+3)&0xfffffc)*bpp)/8;
                Driver->extGlCompressedTextureSubImage2D(TextureName,GL_TEXTURE_2D,i,0,0, tmpSize.Width,tmpSize.Height, InternalFormat,levelByteSize,tmpMipmapDataPTr);
            }
            else
            {
                levelByteSize = (tmpSize.Width*tmpSize.Height*bpp)/8;
                Driver->setPixelUnpackAlignment((tmpSize.Width*bpp)/8,tmpMipmapDataPTr);
                Driver->extGlTextureSubImage2D(TextureName,GL_TEXTURE_2D,i,0,0, tmpSize.Width,tmpSize.Height, PixelFormat, PixelType, (void*)tmpMipmapDataPTr);
            }
            tmpMipmapDataPTr += levelByteSize;
        }
    }
    else if (HasMipMaps)
    {
        Driver->extGlGenerateTextureMipmap(TextureName,GL_TEXTURE_2D);
    }

	if (Image)
	{
		Image->drop();
	}
}

COpenGLTexture::COpenGLTexture(GLenum internalFormat, core::dimension2du size, void* data, GLenum inDataFmt, GLenum inDataTpe, const io::path& name, void* mipmapData, COpenGLDriver* driver, u32 mipmapLevels)
	: ITexture(name), Driver(driver), TextureName(0), TextureSize(size),
	InternalFormat(internalFormat),
	TextureNameHasChanged(0)
{
	#ifdef _DEBUG
	setDebugName("COpenGLTexture");
	#endif

    u32 defaultMipMapCount = 1u+u32(floorf(log2(float(core::min_(core::max_(size.Width,size.Height),Driver->MaxTextureSize)))));
    if (mipmapLevels==0)
    {
        HasMipMaps = Driver->getTextureCreationFlag(ETCF_CREATE_MIP_MAPS);
        MipLevelsStored = HasMipMaps ? defaultMipMapCount:1;
    }
    else
    {
        HasMipMaps = mipmapLevels>1;
        MipLevelsStored = core::min_(mipmapLevels,defaultMipMapCount);
    }


    Driver->extGlCreateTextures(GL_TEXTURE_2D,1,&TextureName);

    ColorFormat = ECF_UNKNOWN;
    bool compressed = COpenGLTexture::isInternalFormatCompressed(InternalFormat);
    //! we're going to have problems with uploading lower mip levels
    uint32_t bpp = getOpenGLFormatBpp(InternalFormat);



    Driver->extGlTextureStorage2D(TextureName,GL_TEXTURE_2D,MipLevelsStored,InternalFormat,size.Width, size.Height);
    if (data)
    {
        if (compressed)
        {
            size_t levelByteSize = (((size.Width+3)&0xfffffc)*((size.Height+3)&0xfffffc)*bpp)/8;
            Driver->extGlCompressedTextureSubImage2D(TextureName,GL_TEXTURE_2D,0,0,0,size.Width, size.Height, InternalFormat,levelByteSize,data);
        }
        else
        {
            Driver->setPixelUnpackAlignment((TextureSize.Width*bpp)/8,data);
            Driver->extGlTextureSubImage2D(TextureName,GL_TEXTURE_2D,0,0,0,size.Width, size.Height, inDataFmt,inDataTpe,data);
        }
    }

    if (mipmapData)
    {
        uint8_t* tmpMipmapDataPTr = (uint8_t*)mipmapData;
        for (u32 i=1; i<MipLevelsStored; i++)
        {
            core::dimension2d<u32> tmpSize = size;
            tmpSize.Width = core::max_(tmpSize.Width/(0x1u<<i),0x1u);
            tmpSize.Height = core::max_(tmpSize.Height/(0x1u<<i),0x1u);
            size_t levelByteSize;
            if (compressed)
            {
                levelByteSize = (((tmpSize.Width+3)&0xfffffc)*((tmpSize.Height+3)&0xfffffc)*bpp)/8;
                Driver->extGlCompressedTextureSubImage2D(TextureName,GL_TEXTURE_2D,i,0,0, tmpSize.Width,tmpSize.Height, InternalFormat,levelByteSize,tmpMipmapDataPTr);
            }
            else
            {
                levelByteSize = (tmpSize.Width*tmpSize.Height*bpp)/8;
                Driver->setPixelUnpackAlignment((tmpSize.Width*bpp)/8,tmpMipmapDataPTr);
                Driver->extGlTextureSubImage2D(TextureName,GL_TEXTURE_2D,i,0,0, tmpSize.Width,tmpSize.Height, inDataFmt, inDataTpe, (void*)tmpMipmapDataPTr);
            }
            tmpMipmapDataPTr += levelByteSize;
        }
    }
    else if (HasMipMaps)
    {
        Driver->extGlGenerateTextureMipmap(TextureName,GL_TEXTURE_2D);
    }
}


//! constructor for basic setup (only for derived classes)
COpenGLTexture::COpenGLTexture(const io::path& name, COpenGLDriver* driver)
	: ITexture(name), ColorFormat(ECF_UNKNOWN), Driver(driver),
	TextureName(0), InternalFormat(GL_RGBA), MipLevelsStored(0), HasMipMaps(false),
	TextureNameHasChanged(0)
{
	#ifdef _DEBUG
	setDebugName("COpenGLTexture");
	#endif
}


//! destructor
COpenGLTexture::~COpenGLTexture()
{
	if (TextureName)
		glDeleteTextures(1, &TextureName);
}


//! Choose best matching color format, based on texture creation flags
ECOLOR_FORMAT COpenGLTexture::getBestColorFormat(ECOLOR_FORMAT format)
{
	ECOLOR_FORMAT destFormat = ECF_A8R8G8B8;
	switch (format)
	{
		case ECF_A1R5G5B5:
			if (!Driver->getTextureCreationFlag(ETCF_ALWAYS_32_BIT))
				destFormat = ECF_A1R5G5B5;
		break;
		case ECF_R5G6B5:
			if (!Driver->getTextureCreationFlag(ETCF_ALWAYS_32_BIT))
				destFormat = ECF_R5G6B5;
		break;
		//! NO IRRLICHT, JUST NO - you will strip the ALPHA CHANNEL!
		/*
		case ECF_A8R8G8B8:
			if (Driver->getTextureCreationFlag(ETCF_ALWAYS_16_BIT) ||
					Driver->getTextureCreationFlag(ETCF_OPTIMIZED_FOR_SPEED))
				destFormat = ECF_A1R5G5B5;
		break;*/
		case ECF_R8G8B8:
			if (Driver->getTextureCreationFlag(ETCF_ALWAYS_16_BIT) ||
					Driver->getTextureCreationFlag(ETCF_OPTIMIZED_FOR_SPEED))
				destFormat = ECF_R5G6B5;
            else
                destFormat = ECF_R8G8B8;
		default:
            destFormat = format;
		break;
	}
	if (Driver->getTextureCreationFlag(ETCF_NO_ALPHA_CHANNEL))
	{
		switch (destFormat)
		{
			case ECF_A1R5G5B5:
				destFormat = ECF_R5G6B5;
			break;
			case ECF_A8R8G8B8:
			case ECF_R8G8B8A8:
				destFormat = ECF_R8G8B8;
			break;
			default:
			break;
		}
	}
	return destFormat;
}

uint32_t COpenGLTexture::getOpenGLFormatBpp(const GLenum& colorformat) const
{
    switch(colorformat)
    {
        case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
            return 4;
            break;
        case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
            return 8;
            break;
        case GL_COMPRESSED_RED_RGTC1_EXT:
            return 4;
        case GL_COMPRESSED_RED_GREEN_RGTC2_EXT:
            return 8;
            break;
        case GL_STENCIL_INDEX8:
        case GL_RGBA2:
        case GL_R3_G3_B2:
        case GL_R8:
        case GL_R8I:
        case GL_R8UI:
        case GL_R8_SNORM:
            return 8;
            break;
        case GL_RGB4:
            return 12;
            break;
        case GL_RGB5:
            return 15;
            break;
        case GL_DEPTH_COMPONENT16:
        case GL_RGBA4:
        case GL_RGB5_A1:
        case GL_RG8:
        case GL_RG8I:
        case GL_RG8UI:
        case GL_RG8_SNORM:
        case GL_R16:
        case GL_R16I:
        case GL_R16UI:
        case GL_R16_SNORM:
        case GL_R16F:
            return 16;
            break;
        case GL_DEPTH_COMPONENT24:
        case GL_RGB8:
        case GL_RGB8I:
        case GL_RGB8UI:
        case GL_RGB8_SNORM:
        case GL_SRGB8:
            return 24;
            break;
        case GL_RGB10:
            return 30;
            break;
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
        case GL_RGBA8:
        case GL_RGBA8I:
        case GL_RGBA8UI:
        case GL_RGBA8_SNORM:
        case GL_SRGB8_ALPHA8:
        case GL_RGB10_A2:
        case GL_RGB10_A2UI:
        case GL_R11F_G11F_B10F:
        case GL_RGB9_E5:
        case GL_RG16:
        case GL_RG16I:
        case GL_RG16UI:
        case GL_RG16F:
        case GL_R32I:
        case GL_R32UI:
        case GL_R32F:
            return 32;
            break;
        case GL_RGB12:
            return 36;
            break;
        case GL_DEPTH32F_STENCIL8:
            return 40;
            break;
        case GL_RGBA12:
        case GL_RGB16:
        case GL_RGB16I:
        case GL_RGB16UI:
        case GL_RGB16_SNORM:
        case GL_RGB16F:
            return 48;
            break;
        case GL_RGBA16:
        case GL_RGBA16I:
        case GL_RGBA16UI:
        case GL_RGBA16F:
        case GL_RG32I:
        case GL_RG32UI:
        case GL_RG32F:
            return 64;
            break;
        case GL_RGB32I:
        case GL_RGB32UI:
        case GL_RGB32F:
            return 96;
            break;
        case GL_RGBA32I:
        case GL_RGBA32UI:
        case GL_RGBA32F:
            return 128;
            break;
        default:
            return 0xdeadu;
    }
    return 0xdeadu;
}


bool COpenGLTexture::isInternalFormatCompressed(GLenum format)
{
    switch (format)
    {
        case GL_COMPRESSED_ALPHA:
        case GL_COMPRESSED_INTENSITY:
        case GL_COMPRESSED_LUMINANCE:
        case GL_COMPRESSED_LUMINANCE_ALPHA:
        case GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT:
        case GL_COMPRESSED_LUMINANCE_LATC1_EXT:
        case GL_COMPRESSED_R11_EAC:
        case GL_COMPRESSED_RED:
        case GL_COMPRESSED_RED_RGTC1:
        case GL_COMPRESSED_RG:
        case GL_COMPRESSED_RG11_EAC:
        case GL_COMPRESSED_RGB:
        case GL_COMPRESSED_RGB8_ETC2:
        case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2:
        case GL_COMPRESSED_RGBA:
        case GL_COMPRESSED_RGBA8_ETC2_EAC:
        case GL_COMPRESSED_RGBA_ASTC_10x10_KHR:
        case GL_COMPRESSED_RGBA_ASTC_10x5_KHR:
        case GL_COMPRESSED_RGBA_ASTC_10x6_KHR:
        case GL_COMPRESSED_RGBA_ASTC_10x8_KHR:
        case GL_COMPRESSED_RGBA_ASTC_12x10_KHR:
        case GL_COMPRESSED_RGBA_ASTC_12x12_KHR:
        case GL_COMPRESSED_RGBA_ASTC_4x4_KHR:
        case GL_COMPRESSED_RGBA_ASTC_5x4_KHR:
        case GL_COMPRESSED_RGBA_ASTC_5x5_KHR:
        case GL_COMPRESSED_RGBA_ASTC_6x5_KHR:
        case GL_COMPRESSED_RGBA_ASTC_6x6_KHR:
        case GL_COMPRESSED_RGBA_ASTC_8x5_KHR:
        case GL_COMPRESSED_RGBA_ASTC_8x6_KHR:
        case GL_COMPRESSED_RGBA_ASTC_8x8_KHR:
        case GL_COMPRESSED_RGBA_BPTC_UNORM:
        case GL_COMPRESSED_RGBA_FXT1_3DFX:
        case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
        case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
        case GL_COMPRESSED_RGB_FXT1_3DFX:
        case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
        case GL_COMPRESSED_RG_RGTC2:
        case GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT:
        case GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT:
        case GL_COMPRESSED_SIGNED_R11_EAC:
        case GL_COMPRESSED_SIGNED_RED_RGTC1:
        case GL_COMPRESSED_SIGNED_RG11_EAC:
        case GL_COMPRESSED_SIGNED_RG_RGTC2:
        case GL_COMPRESSED_SLUMINANCE:
        case GL_COMPRESSED_SLUMINANCE_ALPHA:
        case GL_COMPRESSED_SRGB:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:
        case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
        case GL_COMPRESSED_SRGB8_ETC2:
        case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:
        case GL_COMPRESSED_SRGB_ALPHA:
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
        case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT:
        case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT:
        case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT:
        case GL_COMPRESSED_SRGB_S3TC_DXT1_EXT:
            return true;
            break;
    }
    return false;
}

//! Get opengl values for the GPU texture storage
GLint COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(const ECOLOR_FORMAT &format,
				GLenum& colorformat,
				GLenum& type)
{
	// default
	colorformat = GL_RGBA;
	type = GL_UNSIGNED_BYTE;
	GLenum internalformat = GL_RGBA;

	switch(format)
	{
		case ECF_A1R5G5B5:
			colorformat=GL_BGRA_EXT;
			type=GL_UNSIGNED_SHORT_1_5_5_5_REV;
			internalformat =  GL_RGB5_A1;
			break;
		case ECF_R5G6B5:
			colorformat=GL_RGB;
			type=GL_UNSIGNED_SHORT_5_6_5;
			internalformat =  GL_RGB565;
			break;
		case ECF_R8G8B8:
			colorformat=GL_RGB;
			type=GL_UNSIGNED_BYTE;
			internalformat =  GL_RGB8;
			break;
		case ECF_A8R8G8B8:
			colorformat=GL_BGRA_EXT;
            type=GL_UNSIGNED_INT_8_8_8_8_REV;
			internalformat =  GL_RGBA8;
			break;
		case ECF_R8G8B8A8:
			colorformat=GL_RGBA;
            type=GL_UNSIGNED_BYTE;
			internalformat =  GL_RGBA8;
			break;
		// Floating Point texture formats. Thanks to Patryk "Nadro" Nadrowski.
		case ECF_R16F:
		{
			colorformat = GL_RED;
			type = GL_FLOAT;

			internalformat =  GL_R16F;
		}
			break;
		case ECF_G16R16F:
		{
			colorformat = GL_RG;
			type = GL_FLOAT;

			internalformat =  GL_RG16F;
		}
			break;
		case ECF_A16B16G16R16F:
		{
			colorformat = GL_RGBA;
			type = GL_FLOAT;

			internalformat =  GL_RGBA16F_ARB;
		}
			break;
		case ECF_R32F:
		{
			colorformat = GL_RED;
			type = GL_FLOAT;

			internalformat =  GL_R32F;
		}
			break;
		case ECF_G32R32F:
		{
			colorformat = GL_RG;
			type = GL_FLOAT;

			internalformat =  GL_RG32F;
		}
			break;
		case ECF_A32B32G32R32F:
		{
			colorformat = GL_RGBA;
			type = GL_FLOAT;

			internalformat =  GL_RGBA32F_ARB;
		}
			break;
		case ECF_R8:
		{
			colorformat = GL_RED;
			type = GL_UNSIGNED_BYTE;

			internalformat =  GL_R8;
		}
			break;
		case ECF_R8G8:
		{
			colorformat = GL_RG;
			type = GL_UNSIGNED_BYTE;

			internalformat =  GL_RGB8;
		}
			break;
		case ECF_RGB_BC1:
		{
			colorformat = GL_RGB;
			type = GL_UNSIGNED_BYTE;

			internalformat =  GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
		}
			break;
		case ECF_RGBA_BC1:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;

			internalformat =  GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
		}
			break;
		case ECF_RGBA_BC2:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;

			internalformat =  GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
		}
			break;
		case ECF_RGBA_BC3:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;

			internalformat =  GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
		}
			break;
		case ECF_R_BC4:
		{
			colorformat = GL_RED;
			type = GL_UNSIGNED_BYTE;

			internalformat =  GL_COMPRESSED_RED_RGTC1_EXT;
		}
			break;
		case ECF_RG_BC5:
		{
			colorformat = GL_RG;
			type = GL_UNSIGNED_BYTE;

			internalformat =  GL_COMPRESSED_RED_GREEN_RGTC2_EXT;
		}
			break;
		case ECF_8BIT_PIX:
		{
			colorformat = GL_RED;
			type = GL_UNSIGNED_BYTE;

			internalformat =  GL_R8;
		}
			break;
		case ECF_16BIT_PIX:
		{
			colorformat = GL_RG;
			type = GL_UNSIGNED_BYTE;

			internalformat =  GL_RG8;
		}
			break;
		case ECF_24BIT_PIX:
		{
			colorformat = GL_RGB;
			type = GL_UNSIGNED_BYTE;

			internalformat =  GL_RGB8;
		}
			break;
		case ECF_32BIT_PIX:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_BYTE;

			internalformat =  GL_RGBA8;
		}
			break;
		case ECF_48BIT_PIX:
		{
			colorformat = GL_RGB;
			type = GL_UNSIGNED_SHORT;

			internalformat =  GL_RGB16;
		}
			break;
		case ECF_64BIT_PIX:
		{
			colorformat = GL_RGBA;
			type = GL_UNSIGNED_SHORT;

			internalformat =  GL_RGBA16;
		}
			break;
		case ECF_96BIT_PIX:
		{
			colorformat = GL_RGB;
			type = GL_FLOAT;

			internalformat =  GL_RGB32F;
		}
			break;
		case ECF_128BIT_PIX:
		{
			colorformat = GL_RGBA;
			type = GL_FLOAT;

			internalformat =  GL_RGBA32F;
		}
			break;
        /// this is totally wrong but safe - most probs have to reupload
		case ECF_DEPTH16:
		{
			colorformat = GL_DEPTH;
			type = GL_UNSIGNED_SHORT;

			internalformat =  GL_DEPTH_COMPONENT16;
		}
			break;
		case ECF_DEPTH24:
		{
			colorformat = GL_DEPTH;
			type = GL_UNSIGNED_SHORT;

			internalformat =  GL_DEPTH_COMPONENT24;
		}
			break;
		case ECF_DEPTH24_STENCIL8:
		{
			colorformat = GL_DEPTH_STENCIL;
			type = GL_UNSIGNED_INT_24_8_EXT;

			internalformat =  GL_DEPTH24_STENCIL8;
		}
			break;
		case ECF_DEPTH32F:
		{
			colorformat = GL_DEPTH;
			type = GL_FLOAT;

			internalformat =  GL_DEPTH_COMPONENT32F;
		}
			break;
		case ECF_DEPTH32F_STENCIL8:
		{
			colorformat = GL_DEPTH_STENCIL;
			type = GL_UNSIGNED_BYTE;

			internalformat =  GL_DEPTH32F_STENCIL8;
		}
			break;
		default:
		{
			os::Printer::log("Unsupported texture format", ELL_ERROR);
			internalformat =  GL_RGBA8;
		}
	}

#ifndef GL_ARB_texture_rg
    os::Printer::log("DevSH recommends you send this machine to a museum, GL_ARB_texture_rg unsupported.", ELL_ERROR);
#endif

	return internalformat;
}


// prepare values ImageSize, TextureSize, and ColorFormat based on image
core::dimension2du COpenGLTexture::getImageValues(IImage* image)
{
	if (!image)
	{
		os::Printer::log("No image for OpenGL texture.", ELL_ERROR);
		return core::dimension2du(0,0);
	}

	core::dimension2du ImageSize = image->getDimension();

	if ( !ImageSize.Width || !ImageSize.Height)
	{
		os::Printer::log("Invalid size of image for OpenGL Texture.", ELL_ERROR);
		return ImageSize;
	}

	const f32 ratio = (f32)ImageSize.Width/(f32)ImageSize.Height;
	if ((ImageSize.Width>Driver->MaxTextureSize) && (ratio >= 1.0f))
	{
		ImageSize.Width = Driver->MaxTextureSize;
		ImageSize.Height = (u32)(Driver->MaxTextureSize/ratio);
	}
	else if (ImageSize.Height>Driver->MaxTextureSize)
	{
		ImageSize.Height = Driver->MaxTextureSize;
		ImageSize.Width = (u32)(Driver->MaxTextureSize*ratio);
	}
	TextureSize=ImageSize;

	ColorFormat = getBestColorFormat(image->getColorFormat());

	return ImageSize;
}


//! Returns size of the texture.
const core::dimension2d<u32>& COpenGLTexture::getSize() const
{
	return TextureSize;
}


//! returns driver type of texture, i.e. the driver, which created the texture
E_DRIVER_TYPE COpenGLTexture::getDriverType() const
{
	return EDT_OPENGL;
}


//! returns color format of texture
ECOLOR_FORMAT COpenGLTexture::getColorFormat() const
{
	return ColorFormat;
}


//! returns pitch of texture (in bytes)
u32 COpenGLTexture::getPitch() const
{
	return IImage::getBitsPerPixelFromFormat(ColorFormat)*TextureSize.Width/8;
}


GLint COpenGLTexture::getOpenGLInternalFormat() const
{
    return InternalFormat;
}/**
GLenum COpenGLTexture::getOpenGLPixelFormat() const
{
    return PixelFormat;
}
GLenum COpenGLTexture::getOpenGLPixelType() const
{
    return PixelType;
}**/


//! Returns whether this texture has mipmaps
bool COpenGLTexture::hasMipMaps() const
{
	return HasMipMaps;
}


//! Regenerates the mip map levels of the texture. Useful after locking and
//! modifying the texture
void COpenGLTexture::regenerateMipMapLevels()
{
	if (!HasMipMaps)
		return;

    Driver->extGlGenerateTextureMipmap(TextureName,getOpenGLTextureType());
}



void COpenGLTexture::resize(const core::dimension2du &newSize, const u32 &mipmapLevels)
{
    TextureSize = newSize;

    if (TextureName)
        glDeleteTextures(1,&TextureName);
    Driver->extGlCreateTextures(getOpenGLTextureType(),1,&TextureName);
    u32 defaultMipMapCount = 1u+u32(floorf(log2(float(core::min_(core::max_(newSize.Width,newSize.Height),Driver->MaxTextureSize)))));
    if (HasMipMaps)
    {
        if (mipmapLevels==0)
            MipLevelsStored = defaultMipMapCount;
        else
            MipLevelsStored = core::min_(mipmapLevels,defaultMipMapCount);
    }
    else
        MipLevelsStored = 1;
	Driver->extGlTextureStorage2D(TextureName,getOpenGLTextureType(), MipLevelsStored, InternalFormat, TextureSize.Width, TextureSize.Height);

	TextureNameHasChanged = os::Timer::getRealTime();
}




} // end namespace video
} // end namespace irr

#endif // _IRR_COMPILE_WITH_OPENGL_

