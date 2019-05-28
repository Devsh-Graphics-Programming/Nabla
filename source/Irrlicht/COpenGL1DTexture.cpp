#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

#include "irr/core/Types.h"
#include "COpenGL1DTexture.h"
#include "COpenGLDriver.h"




namespace irr
{
namespace video
{

COpenGL1DTexture::COpenGL1DTexture(GLenum internalFormat, const uint32_t* size, uint32_t mipmapLevels, const io::path& name) : COpenGLFilterableTexture(name,getOpenGLTextureType())
{
#ifdef _IRR_DEBUG
	setDebugName("COpenGL1DTexture");
#endif
    TextureSize[0] = size[0];
    TextureSize[1] = 1;
    TextureSize[2] = 1;
    MipLevelsStored = mipmapLevels;

    InternalFormat = internalFormat;
    COpenGLExtensionHandler::extGlTextureStorage1D(TextureName,GL_TEXTURE_1D,MipLevelsStored,InternalFormat,TextureSize[0]);

    ColorFormat = getColorFormatFromSizedOpenGLFormat(InternalFormat);
}

bool COpenGL1DTexture::updateSubRegion(const asset::E_FORMAT &inDataColorFormat, const void* data, const uint32_t* minimum, const uint32_t* maximum, int32_t mipmap, const uint32_t& unpackRowByteAlignment)
{
    bool sourceCompressed = isBlockCompressionFormat(inDataColorFormat);

    bool destinationCompressed = COpenGLTexture::isInternalFormatCompressed(InternalFormat);
    if ((!destinationCompressed)&&sourceCompressed)
        return false;

    if (destinationCompressed)
    {
        if (minimum[0])
            return false;

        uint32_t adjustedTexSize[1] = {TextureSize[0]};
        adjustedTexSize[0] /= 0x1u<<mipmap;
        /*
        adjustedTexSize[0] += 3u;
        adjustedTexSize[0] &= 0xfffffc;
        */
        if (maximum[0]!=adjustedTexSize[0])
            return false;
    }

    if (sourceCompressed)
    {
        size_t levelByteSize = (((maximum[0]-minimum[0]+3)&0xfffffc)*COpenGLTexture::getOpenGLFormatBpp(InternalFormat))/8;

        COpenGLExtensionHandler::extGlCompressedTextureSubImage1D(TextureName,GL_TEXTURE_1D, mipmap, minimum[0],maximum[0]-minimum[0], InternalFormat, levelByteSize, data);
    }
    else
    {
        GLenum pixFmt,pixType;
        getOpenGLFormatAndParametersFromColorFormat(inDataColorFormat, pixFmt, pixType);
        //! replace with
        ///COpenGLExtensionHandler::extGlGetInternalFormativ(GL_TEXTURE_1D,InternalFormat,GL_TEXTURE_IMAGE_FORMAT,1,&pixFmt);
        ///COpenGLExtensionHandler::extGlGetInternalFormativ(GL_TEXTURE_1D,InternalFormat,GL_TEXTURE_IMAGE_FORMAT,1,&pixType);

        //! we're going to have problems with uploading lower mip levels
        uint32_t bpp = video::getBitsPerPixelFromFormat(inDataColorFormat);
        uint32_t pitchInBits = ((maximum[0]-minimum[0])*bpp)/8;

        COpenGLExtensionHandler::setPixelUnpackAlignment(pitchInBits,const_cast<void*>(data),unpackRowByteAlignment);
        COpenGLExtensionHandler::extGlTextureSubImage1D(TextureName, GL_TEXTURE_1D, mipmap, minimum[0], maximum[0]-minimum[0], pixFmt, pixType, data);
    }
    return true;
}

bool COpenGL1DTexture::resize(const uint32_t* size, const uint32_t &mipLevels)
{
    if (TextureSize[0]==size[0])
        return true;

    recreateName(getOpenGLTextureType());

    uint32_t defaultMipMapCount = 1u+uint32_t(floorf(log2(float(size[0]))));
    if (MipLevelsStored>1)
    {
        if (mipLevels==0)
            MipLevelsStored = defaultMipMapCount;
        else
            MipLevelsStored = core::min_(mipLevels,defaultMipMapCount);
    }

    TextureSize[0] = size[0];
	COpenGLExtensionHandler::extGlTextureStorage1D(TextureName,getOpenGLTextureType(), MipLevelsStored, InternalFormat, TextureSize[0]);
	return true;
}

}
}
#endif


