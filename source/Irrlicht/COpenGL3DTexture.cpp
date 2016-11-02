#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

#include "irrTypes.h"
#include "COpenGL3DTexture.h"
#include "COpenGLDriver.h"


#include "irrString.h"

namespace irr
{
namespace video
{

COpenGL3DTexture::COpenGL3DTexture(core::vector3d<u32> size, GLenum format, GLenum inDataFmt, GLenum inDataTpe, const io::path& name, const void* data, void* mipmapData, COpenGLDriver* driver, u32 mipLevels) : COpenGLTexture(name, driver)
{
    TextureSize[0] = size.X;
    TextureSize[1] = size.Y;
    TextureSize[2] = size.Z;

    u32 defaultMipMapCount = 1u+u32(floorf(log2(float(core::min_(core::max_(core::max_(size.X,size.Y),size.Z),Driver->getMaxTextureSize(ETT_3D)[0])))));
    if (mipLevels==0)
    {
        HasMipMaps = Driver->getTextureCreationFlag(ETCF_CREATE_MIP_MAPS);
        MipLevelsStored = HasMipMaps ? defaultMipMapCount:1;
    }
    else
    {
        HasMipMaps = mipLevels>1;
        MipLevelsStored = core::min_(mipLevels,defaultMipMapCount);
    }


    COpenGLExtensionHandler::extGlCreateTextures(GL_TEXTURE_3D,1,&TextureName);

    InternalFormat = format;
    GLenum PixelFormat = inDataFmt;
    GLenum PixelType = inDataTpe;
    bool compressed = COpenGLTexture::isInternalFormatCompressed(InternalFormat);

    //! we're going to have problems with uploading lower mip levels
    uint32_t bpp = COpenGLTexture::getOpenGLFormatBpp(format);


    COpenGLExtensionHandler::extGlTextureStorage3D(TextureName,GL_TEXTURE_3D,MipLevelsStored,InternalFormat,size.X, size.Y, size.Z);
    size_t levelByteSize;
    if (data)
    {
        if (compressed)
        {
            levelByteSize = size.Z;
            levelByteSize *= (((size.X+3)&0xfffffc)*((size.Y+3)&0xfffffc)*bpp)/8;
            COpenGLExtensionHandler::extGlCompressedTextureSubImage3D(TextureName,GL_TEXTURE_3D,0,0,0,0,size.X, size.Y, size.Z,InternalFormat,levelByteSize,data);
        }
        else
        {
            COpenGLExtensionHandler::setPixelUnpackAlignment((size.X*bpp)/8,const_cast<void*>(data));
            COpenGLExtensionHandler::extGlTextureSubImage3D(TextureName,GL_TEXTURE_3D,0,0,0,0,size.X, size.Y, size.Z,PixelFormat,PixelType,data);
        }
    }

    if (mipmapData)
    {
        uint8_t* tmpMipmapDataPTr = (uint8_t*)mipmapData;
        for (u32 i=1; i<MipLevelsStored; i++)
        {
            core::vector3d<u32> tmpSize = size/(0x1u<<i);
            tmpSize.X = core::max_(tmpSize.X,0x1u);
            tmpSize.Y = core::max_(tmpSize.Y,0x1u);
            tmpSize.Z = core::max_(tmpSize.Z,0x1u);
            levelByteSize = tmpSize.Z;
            if (compressed)
            {
                levelByteSize *= (((tmpSize.X+3)&0xfffffc)*((tmpSize.Y+3)&0xfffffc)*bpp)/8;
                COpenGLExtensionHandler::extGlCompressedTextureSubImage3D(TextureName,GL_TEXTURE_3D,i,0,0,0, tmpSize.X,tmpSize.Y,tmpSize.Z, InternalFormat,levelByteSize,tmpMipmapDataPTr);
            }
            else
            {
                levelByteSize *= (tmpSize.X*tmpSize.Y*bpp)/8;
                COpenGLExtensionHandler::setPixelUnpackAlignment((tmpSize.X*bpp)/8,tmpMipmapDataPTr);
                COpenGLExtensionHandler::extGlTextureSubImage3D(TextureName,GL_TEXTURE_3D,i,0,0,0, tmpSize.X,tmpSize.Y,tmpSize.Z, PixelFormat, PixelType, (void*)tmpMipmapDataPTr);
            }
            tmpMipmapDataPTr += levelByteSize;
        }
    }
    else if (HasMipMaps)
    {
        COpenGLExtensionHandler::extGlGenerateTextureMipmap(TextureName,GL_TEXTURE_3D);
    }

}

/*
	ReadOnlyLock(false)
*/
bool COpenGL3DTexture::updateSubRegion(const ECOLOR_FORMAT &inDataColorFormat, const void* data, const uint32_t* minimum, const uint32_t* maximum, s32 mipmap)
{
    bool compressed = COpenGLTexture::isInternalFormatCompressed(InternalFormat);
    if (compressed&&(minimum[0]||minimum[1]||minimum[2]))
        return false;

    GLenum pixFmt,pixType;
	getOpenGLFormatAndParametersFromColorFormat(inDataColorFormat, pixFmt, pixType);

    if (compressed)
    {
        size_t levelByteSize = (maximum[2]-minimum[2])/(0x1u<<mipmap);
        levelByteSize *= ((((maximum[0]-minimum[0])/(0x1u<<mipmap)+3)&0xfffffc)*(((maximum[1]-minimum[1])/(0x1u<<mipmap)+3)&0xfffffc)*COpenGLTexture::getOpenGLFormatBpp(InternalFormat))/8;

        COpenGLExtensionHandler::extGlCompressedTextureSubImage3D(TextureName,GL_TEXTURE_3D, mipmap, minimum[0],minimum[1],minimum[2], maximum[0]-minimum[0],maximum[1]-minimum[1],maximum[2]-minimum[2], InternalFormat, levelByteSize, data);
    }
    else
    {
        //! we're going to have problems with uploading lower mip levels
        uint32_t bpp = IImage::getBitsPerPixelFromFormat(inDataColorFormat);
        uint32_t pitchInBits = ((maximum[0]-minimum[0])>>mipmap)*bpp/8;

        COpenGLExtensionHandler::setPixelUnpackAlignment(pitchInBits,const_cast<void*>(data));
        COpenGLExtensionHandler::extGlTextureSubImage3D(TextureName,GL_TEXTURE_3D, mipmap, minimum[0],minimum[1],minimum[2], maximum[0]-minimum[0],maximum[1]-minimum[1],maximum[2]-minimum[2], pixFmt, pixType, data);
    }
    return true;
}

bool COpenGL3DTexture::resize(const uint32_t* size, u32 mipLevels)
{
    if (TextureSize[0]==size[0]&&TextureSize[1]==size[1]&&TextureSize[2]==size[2])
        return true;

    bool compressed = COpenGLTexture::isInternalFormatCompressed(InternalFormat);

    if (TextureName)
        glDeleteTextures(1, &TextureName);
    COpenGLExtensionHandler::extGlCreateTextures(GL_TEXTURE_3D,1,&TextureName);
	TextureNameHasChanged = CNullDriver::incrementAndFetchReallocCounter();

    u32 defaultMipMapCount = 1u+u32(floorf(log2(float(core::min_(core::max_(core::max_(size[0],size[1]),size[2]),Driver->getMaxTextureSize(ETT_3D)[0])))));
    if (HasMipMaps)
    {
        if (mipLevels==0)
            MipLevelsStored = defaultMipMapCount;
        else
            MipLevelsStored = core::min_(mipLevels,defaultMipMapCount);
    }
    else
        MipLevelsStored = 1;

    Driver->extGlTextureStorage3D(TextureName,GL_TEXTURE_3D,MipLevelsStored,InternalFormat,size[0], size[1], size[2]);

    memcpy(TextureSize,size,12);
    return true;
}

u32 COpenGL3DTexture::getPitch() const
{
    return TextureSize[0]*COpenGLTexture::getOpenGLFormatBpp(InternalFormat)/8;
}


GLenum COpenGL3DTexture::getOpenGLTextureType() const
{
    return GL_TEXTURE_3D;
}


























COpenGL2DTextureArray::COpenGL2DTextureArray(core::vector3d<u32> size, ECOLOR_FORMAT format, const io::path& name, void* mipmapData, COpenGLDriver* driver, u32 mipmapLevels) : COpenGLTexture(name, driver)
{
    TextureSize[0] = size.X;
    TextureSize[1] = size.Y;
    TextureSize[2] = size.Z;

    u32 defaultMipMapCount = 1u+u32(floorf(log2(float(core::min_(core::max_(size.X,size.Y),Driver->getMaxTextureSize(ETT_2D_ARRAY)[0])))));
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


    COpenGLExtensionHandler::extGlCreateTextures(GL_TEXTURE_2D_ARRAY,1,&TextureName);

    GLenum PixelFormat = GL_BGRA;
    GLenum PixelType = GL_UNSIGNED_BYTE;
    ColorFormat = format;

	InternalFormat = getOpenGLFormatAndParametersFromColorFormat(ColorFormat, PixelFormat, PixelType);

    //! we're going to have problems with uploading lower mip levels
    uint32_t bpp = IImage::getBitsPerPixelFromFormat(ColorFormat);

    COpenGLExtensionHandler::extGlTextureStorage3D(TextureName,GL_TEXTURE_2D_ARRAY,MipLevelsStored,InternalFormat,size.X, size.Y, size.Z);
    size_t levelByteSize;
    if (mipmapData)
    {
        levelByteSize = size.Z;
        if (ColorFormat>=ECF_RGB_BC1&&ColorFormat<=ECF_RG_BC5)
        {
            levelByteSize *= (((size.X+3)&0xfffffc)*((size.Y+3)&0xfffffc)*bpp)/8;
            COpenGLExtensionHandler::extGlCompressedTextureSubImage3D(TextureName,GL_TEXTURE_2D_ARRAY,0,0,0,0,size.X, size.Y, size.Z,InternalFormat,levelByteSize,mipmapData);
        }
        else
        {
            COpenGLExtensionHandler::setPixelUnpackAlignment((size.X*bpp)/8,mipmapData);
            COpenGLExtensionHandler::extGlTextureSubImage3D(TextureName,GL_TEXTURE_2D_ARRAY,0,0,0,0,size.X, size.Y, size.Z,PixelFormat,PixelType,mipmapData);
            levelByteSize *= (size.X*size.Y*bpp)/8;
        }

        if (mipmapLevels==0)
        {
            uint8_t* tmpMipmapDataPTr = ((uint8_t*)mipmapData)+levelByteSize;
            for (u32 i=1; i<MipLevelsStored; i++)
            {
                core::vector3d<u32> tmpSize = size;
                tmpSize.X = core::max_(tmpSize.X/(0x1u<<i),0x1u);
                tmpSize.Y = core::max_(tmpSize.Y/(0x1u<<i),0x1u);
                levelByteSize = tmpSize.Z;
                if (ColorFormat>=ECF_RGB_BC1&&ColorFormat<=ECF_RG_BC5)
                {
                    levelByteSize *= (((tmpSize.X+3)&0xfffffc)*((tmpSize.Y+3)&0xfffffc)*bpp)/8;
                    COpenGLExtensionHandler::extGlCompressedTextureSubImage3D(TextureName,GL_TEXTURE_2D_ARRAY,i,0,0,0, tmpSize.X,tmpSize.Y,tmpSize.Z, InternalFormat,levelByteSize,tmpMipmapDataPTr);
                }
                else
                {
                    levelByteSize *= (tmpSize.X*tmpSize.Y*bpp)/8;
                    COpenGLExtensionHandler::setPixelUnpackAlignment((tmpSize.X*bpp)/8,tmpMipmapDataPTr);
                    COpenGLExtensionHandler::extGlTextureSubImage3D(TextureName,GL_TEXTURE_2D_ARRAY,i,0,0,0, tmpSize.X,tmpSize.Y,tmpSize.Z, PixelFormat, PixelType, (void*)tmpMipmapDataPTr);
                }
                tmpMipmapDataPTr += levelByteSize;
            }
        }
        else
        {
            COpenGLExtensionHandler::extGlGenerateTextureMipmap(TextureName,GL_TEXTURE_2D_ARRAY);
        }
    }
}

/*
	ReadOnlyLock(false)
*/

bool COpenGL2DTextureArray::updateSubRegion(const ECOLOR_FORMAT &inDataColorFormat, const void* data, const uint32_t* minimum, const uint32_t* maximum, s32 mipmap)
{
    bool compressed = COpenGLTexture::isInternalFormatCompressed(InternalFormat);
    if (compressed&&(minimum[0]||minimum[1]))
        return false;

    GLenum pixFmt,pixType;
	getOpenGLFormatAndParametersFromColorFormat(inDataColorFormat, pixFmt, pixType);

    if (compressed)
    {
        size_t levelByteSize = (maximum[2]-minimum[2]);
        levelByteSize *= ((((maximum[0]-minimum[0])/(0x1u<<mipmap)+3)&0xfffffc)*(((maximum[1]-minimum[1])/(0x1u<<mipmap)+3)&0xfffffc)*COpenGLTexture::getOpenGLFormatBpp(InternalFormat))/8;

        COpenGLExtensionHandler::extGlCompressedTextureSubImage3D(TextureName,GL_TEXTURE_2D_ARRAY, mipmap, minimum[0],minimum[1],minimum[2], maximum[0]-minimum[0],maximum[1]-minimum[1],maximum[2]-minimum[2], InternalFormat, levelByteSize, data);
    }
    else
    {
        //! we're going to have problems with uploading lower mip levels
        uint32_t bpp = IImage::getBitsPerPixelFromFormat(inDataColorFormat);
        uint32_t pitchInBits = ((maximum[0]-minimum[0])>>mipmap)*bpp/8;

        COpenGLExtensionHandler::setPixelUnpackAlignment(pitchInBits,const_cast<void*>(data));
        COpenGLExtensionHandler::extGlTextureSubImage3D(TextureName,GL_TEXTURE_2D_ARRAY, mipmap, minimum[0],minimum[1],minimum[2], maximum[0]-minimum[0],maximum[1]-minimum[1],maximum[2]-minimum[2], pixFmt, pixType, data);
    }
    return true;
}

bool COpenGL2DTextureArray::resize(const uint32_t* size, u32 mipLevels)
{
    if (TextureSize[0]==size[0]&&TextureSize[1]==size[1]&&TextureSize[2]==size[2])
        return true;

    if (TextureName)
        glDeleteTextures(1, &TextureName);
    COpenGLExtensionHandler::extGlCreateTextures(GL_TEXTURE_2D_ARRAY,1,&TextureName);
	TextureNameHasChanged = CNullDriver::incrementAndFetchReallocCounter();

    u32 defaultMipMapCount = 1u+u32(floorf(log2(float(core::min_(core::max_(size[0],size[1]),Driver->getMaxTextureSize(ETT_2D_ARRAY)[0])))));
    if (HasMipMaps)
    {
        if (mipLevels==0)
            MipLevelsStored = defaultMipMapCount;
        else
            MipLevelsStored = core::min_(mipLevels,defaultMipMapCount);
    }
    else
        MipLevelsStored = 1;


    COpenGLExtensionHandler::extGlTextureStorage3D(TextureName,GL_TEXTURE_2D_ARRAY,MipLevelsStored,InternalFormat,size[0], size[1], size[2]);

    memcpy(TextureSize,size,12);
    return true;
}

u32 COpenGL2DTextureArray::getPitch() const
{
    return TextureSize[0]*COpenGLTexture::getOpenGLFormatBpp(InternalFormat)/8;
}

GLenum COpenGL2DTextureArray::getOpenGLTextureType() const
{
    return GL_TEXTURE_2D_ARRAY;
}

}
}
#endif
