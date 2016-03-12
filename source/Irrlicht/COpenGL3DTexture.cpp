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

COpenGL3DTexture::COpenGL3DTexture(core::vector3d<u32> size, GLenum format, GLenum inDataFmt, GLenum inDataTpe, const io::path& name, void* data, void* mipmapData, COpenGLDriver* driver, u32 mipLevels) : COpenGLTexture(name, driver), OrigSize3D(size), TextureSize3D(size)
{
    u32 defaultMipMapCount = 1u+u32(floorf(log2(float(core::min_(core::max_(core::max_(size.X,size.Y),size.Z),Driver->MaxTextureSize)))));
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


    Driver->extGlCreateTextures(GL_TEXTURE_3D,1,&TextureName);

    InternalFormat = format;
    GLenum PixelFormat = inDataFmt;
    GLenum PixelType = inDataTpe;
    bool compressed = COpenGLTexture::isInternalFormatCompressed(InternalFormat);

    //! we're going to have problems with uploading lower mip levels
    uint32_t bpp = COpenGLTexture::getOpenGLFormatBpp(format);


    Driver->extGlTextureStorage3D(TextureName,GL_TEXTURE_3D,MipLevelsStored,InternalFormat,size.X, size.Y, size.Z);
    size_t levelByteSize;
    if (data)
    {
        if (compressed)
        {
            levelByteSize = size.Z;
            levelByteSize *= (((size.X+3)&0xfffffc)*((size.Y+3)&0xfffffc)*bpp)/8;
            Driver->extGlCompressedTextureSubImage3D(TextureName,GL_TEXTURE_3D,0,0,0,0,size.X, size.Y, size.Z,InternalFormat,levelByteSize,data);
        }
        else
        {
            Driver->setPixelUnpackAlignment((size.X*bpp)/8,data);
            Driver->extGlTextureSubImage3D(TextureName,GL_TEXTURE_3D,0,0,0,0,size.X, size.Y, size.Z,PixelFormat,PixelType,data);
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
                Driver->extGlCompressedTextureSubImage3D(TextureName,GL_TEXTURE_3D,i,0,0,0, tmpSize.X,tmpSize.Y,tmpSize.Z, InternalFormat,levelByteSize,tmpMipmapDataPTr);
            }
            else
            {
                levelByteSize *= (tmpSize.X*tmpSize.Y*bpp)/8;
                Driver->setPixelUnpackAlignment((tmpSize.X*bpp)/8,tmpMipmapDataPTr);
                Driver->extGlTextureSubImage3D(TextureName,GL_TEXTURE_3D,i,0,0,0, tmpSize.X,tmpSize.Y,tmpSize.Z, PixelFormat, PixelType, (void*)tmpMipmapDataPTr);
            }
            tmpMipmapDataPTr += levelByteSize;
        }
    }
    else if (HasMipMaps)
    {
        Driver->extGlGenerateTextureMipmap(TextureName,GL_TEXTURE_3D);
    }

}

/*
	ReadOnlyLock(false)
*/

COpenGL3DTexture::~COpenGL3DTexture()
{
	if (TextureName)
		glDeleteTextures(1, &TextureName);
}

void COpenGL3DTexture::updateSubRegion(const ECOLOR_FORMAT &inDataColorFormat, void* data, u32 minX, u32 minY, u32 minZ, u32 maxX, u32 maxY, u32 maxZ, s32 mipmap)
{
    bool compressed = COpenGLTexture::isInternalFormatCompressed(InternalFormat);
    if (compressed&&(minX||minY||minZ))
        return;

    GLenum pixFmt,pixType;
	getOpenGLFormatAndParametersFromColorFormat(inDataColorFormat, pixFmt, pixType);

    if (compressed)
    {
        size_t levelByteSize = (maxZ-minZ)/(0x1u<<mipmap);
        levelByteSize *= ((((maxX-minX)/(0x1u<<mipmap)+3)&0xfffffc)*(((maxY-minY)/(0x1u<<mipmap)+3)&0xfffffc)*COpenGLTexture::getOpenGLFormatBpp(InternalFormat))/8;

        Driver->extGlCompressedTextureSubImage3D(TextureName,GL_TEXTURE_3D, mipmap, minX,minY,minZ, maxX-minX,maxY-minY,maxZ-minZ, InternalFormat, levelByteSize, data);
    }
    else
    {
        //! we're going to have problems with uploading lower mip levels
        uint32_t bpp = IImage::getBitsPerPixelFromFormat(inDataColorFormat);
        uint32_t pitchInBits = ((maxX-minX)>>mipmap)*bpp/8;

        Driver->setPixelUnpackAlignment(pitchInBits,data);
        Driver->extGlTextureSubImage3D(TextureName,GL_TEXTURE_3D, mipmap, minX,minY,minZ, maxX-minX,maxY-minY,maxZ-minZ, pixFmt, pixType, data);
    }
}

void COpenGL3DTexture::resize(core::vector3d<u32> size, u32 mipLevels)
{
    if (TextureSize3D==size)
        return;

    bool compressed = COpenGLTexture::isInternalFormatCompressed(InternalFormat);

    if (TextureName)
        glDeleteTextures(1, &TextureName);
    Driver->extGlCreateTextures(GL_TEXTURE_3D,1,&TextureName);
	TextureNameHasChanged = os::Timer::getRealTime();

    u32 defaultMipMapCount = 1u+u32(floorf(log2(float(core::min_(core::max_(core::max_(size.X,size.Y),size.Z),Driver->MaxTextureSize)))));
    if (HasMipMaps)
    {
        if (mipLevels==0)
            MipLevelsStored = defaultMipMapCount;
        else
            MipLevelsStored = core::min_(mipLevels,defaultMipMapCount);
    }
    else
        MipLevelsStored = 1;

    Driver->extGlTextureStorage3D(TextureName,GL_TEXTURE_3D,MipLevelsStored,InternalFormat,size.X, size.Y, size.Z);

    OrigSize3D = TextureSize3D = size;
}

const core::vector3d<u32>& COpenGL3DTexture::getSize3D() const
{
    return TextureSize3D;
}

u32 COpenGL3DTexture::getPitch() const
{
    return TextureSize3D.X*COpenGLTexture::getOpenGLFormatBpp(InternalFormat)/8;
}


GLenum COpenGL3DTexture::getOpenGLTextureType() const
{
    return GL_TEXTURE_3D;
}


























COpenGL2DTextureArray::COpenGL2DTextureArray(core::vector3d<u32> size, ECOLOR_FORMAT format, const io::path& name, void* mipmapData, COpenGLDriver* driver, u32 mipmapLevels) : COpenGLTexture(name, driver), OrigSize3D(size), TextureSize3D(size)
{
    u32 defaultMipMapCount = 1u+u32(floorf(log2(float(core::min_(core::max_(size.X,size.Y),Driver->MaxTextureSize)))));
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


    Driver->extGlCreateTextures(GL_TEXTURE_2D_ARRAY,1,&TextureName);

    GLenum PixelFormat = GL_BGRA;
    GLenum PixelType = GL_UNSIGNED_BYTE;
    ColorFormat = format;

	InternalFormat = getOpenGLFormatAndParametersFromColorFormat(ColorFormat, PixelFormat, PixelType);

    //! we're going to have problems with uploading lower mip levels
    uint32_t bpp = IImage::getBitsPerPixelFromFormat(ColorFormat);

    Driver->extGlTextureStorage3D(TextureName,GL_TEXTURE_2D_ARRAY,MipLevelsStored,InternalFormat,size.X, size.Y, size.Z);
    size_t levelByteSize;
    if (mipmapData)
    {
        levelByteSize = size.Z;
        if (ColorFormat>=ECF_RGB_BC1&&ColorFormat<=ECF_RG_BC5)
        {
            levelByteSize *= (((size.X+3)&0xfffffc)*((size.Y+3)&0xfffffc)*bpp)/8;
            Driver->extGlCompressedTextureSubImage3D(TextureName,GL_TEXTURE_2D_ARRAY,0,0,0,0,size.X, size.Y, size.Z,InternalFormat,levelByteSize,mipmapData);
        }
        else
        {
            Driver->setPixelUnpackAlignment((size.X*bpp)/8,mipmapData);
            Driver->extGlTextureSubImage3D(TextureName,GL_TEXTURE_2D_ARRAY,0,0,0,0,size.X, size.Y, size.Z,PixelFormat,PixelType,mipmapData);
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
                    Driver->extGlCompressedTextureSubImage3D(TextureName,GL_TEXTURE_2D_ARRAY,i,0,0,0, tmpSize.X,tmpSize.Y,tmpSize.Z, InternalFormat,levelByteSize,tmpMipmapDataPTr);
                }
                else
                {
                    levelByteSize *= (tmpSize.X*tmpSize.Y*bpp)/8;
                    Driver->setPixelUnpackAlignment((tmpSize.X*bpp)/8,tmpMipmapDataPTr);
                    Driver->extGlTextureSubImage3D(TextureName,GL_TEXTURE_2D_ARRAY,i,0,0,0, tmpSize.X,tmpSize.Y,tmpSize.Z, PixelFormat, PixelType, (void*)tmpMipmapDataPTr);
                }
                tmpMipmapDataPTr += levelByteSize;
            }
        }
        else
        {
            Driver->extGlGenerateTextureMipmap(TextureName,GL_TEXTURE_2D_ARRAY);
        }
    }
}

/*
	ReadOnlyLock(false)
*/

COpenGL2DTextureArray::~COpenGL2DTextureArray()
{
	if (TextureName)
		glDeleteTextures(1, &TextureName);
}

void COpenGL2DTextureArray::updateSubRegion(const ECOLOR_FORMAT &inDataColorFormat, void* data, u32 minX, u32 minY, u32 minZ, u32 maxX, u32 maxY, u32 maxZ, s32 mipmap)
{
    bool compressed = COpenGLTexture::isInternalFormatCompressed(InternalFormat);
    if (compressed&&(minX||minY))
        return;

    GLenum pixFmt,pixType;
	getOpenGLFormatAndParametersFromColorFormat(inDataColorFormat, pixFmt, pixType);

    if (compressed)
    {
        size_t levelByteSize = (maxZ-minZ);
        levelByteSize *= ((((maxX-minX)/(0x1u<<mipmap)+3)&0xfffffc)*(((maxY-minY)/(0x1u<<mipmap)+3)&0xfffffc)*COpenGLTexture::getOpenGLFormatBpp(InternalFormat))/8;

        Driver->extGlCompressedTextureSubImage3D(TextureName,GL_TEXTURE_2D_ARRAY, mipmap, minX,minY,minZ, maxX-minX,maxY-minY,maxZ-minZ, InternalFormat, levelByteSize, data);
    }
    else
    {
        //! we're going to have problems with uploading lower mip levels
        uint32_t bpp = IImage::getBitsPerPixelFromFormat(inDataColorFormat);
        uint32_t pitchInBits = ((maxX-minX)>>mipmap)*bpp/8;

        Driver->setPixelUnpackAlignment(pitchInBits,data);
        Driver->extGlTextureSubImage3D(TextureName,GL_TEXTURE_2D_ARRAY, mipmap, minX,minY,minZ, maxX-minX,maxY-minY,maxZ-minZ, pixFmt, pixType, data);
    }
}

void COpenGL2DTextureArray::resize(core::vector3d<u32> size, u32 mipLevels)
{
    if (TextureSize3D==size)
        return;

    if (TextureName)
        glDeleteTextures(1, &TextureName);
    Driver->extGlCreateTextures(GL_TEXTURE_2D_ARRAY,1,&TextureName);
	TextureNameHasChanged = os::Timer::getRealTime();

    u32 defaultMipMapCount = 1u+u32(floorf(log2(float(core::min_(core::max_(size.X,size.Y),Driver->MaxTextureSize)))));
    if (HasMipMaps)
    {
        if (mipLevels==0)
            MipLevelsStored = defaultMipMapCount;
        else
            MipLevelsStored = core::min_(mipLevels,defaultMipMapCount);
    }
    else
        MipLevelsStored = 1;


    Driver->extGlTextureStorage3D(TextureName,GL_TEXTURE_2D_ARRAY,MipLevelsStored,InternalFormat,size.X, size.Y, size.Z);


    OrigSize3D = TextureSize3D = size;
}
const core::vector3d<u32>& COpenGL2DTextureArray::getSize3D() const
{
    return TextureSize3D;
}

u32 COpenGL2DTextureArray::getPitch() const
{
    return TextureSize3D.X*COpenGLTexture::getOpenGLFormatBpp(InternalFormat)/8;
}

GLenum COpenGL2DTextureArray::getOpenGLTextureType() const
{
    return GL_TEXTURE_2D_ARRAY;
}

}
}
#endif
