#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

#include "irr/core/Types.h"
#include "COpenGL2DTexture.h"
#include "COpenGLDriver.h"




namespace irr
{
namespace video
{

/**
    if (mipmapData)
    {
        uint8_t* tmpMipmapDataPTr = (uint8_t*)mipmapData;
        for (uint32_t i=1; i<MipLevelsStored; i++)
        {
            core::dimension2d<uint32_t> tmpSize = size;
            tmpSize.Width = core::max_(tmpSize.Width/(0x1u<<i),0x1u);
            tmpSize.Height = core::max_(tmpSize.Height/(0x1u<<i),0x1u);
            size_t levelByteSize;
            if (compressed)
            {
                levelByteSize = (((tmpSize.Width+3)&0xfffffc)*((tmpSize.Height+3)&0xfffffc)*bpp)/8;
                COpenGLExtensionHandler::extGlCompressedTextureSubImage2D(TextureName,GL_TEXTURE_2D,i,0,0, tmpSize.Width,tmpSize.Height, InternalFormat,levelByteSize,tmpMipmapDataPTr);
            }
            else
            {
                levelByteSize = (tmpSize.Width*tmpSize.Height*bpp)/8;
                COpenGLExtensionHandler::setPixelUnpackAlignment((tmpSize.Width*bpp)/8,tmpMipmapDataPTr);
                COpenGLExtensionHandler::extGlTextureSubImage2D(TextureName,GL_TEXTURE_2D,i,0,0, tmpSize.Width,tmpSize.Height, inDataFmt, inDataTpe, (void*)tmpMipmapDataPTr);
            }
            tmpMipmapDataPTr += levelByteSize;
        }
    }
**/
COpenGL2DTexture::COpenGL2DTexture(GLenum internalFormat, const uint32_t* size, uint32_t mipmapLevels, const io::path& name) : COpenGLFilterableTexture(name,getOpenGLTextureType())
{
#ifdef _IRR_DEBUG
	setDebugName("COpenGL2DTexture");
#endif
    TextureSize[0] = size[0];
    TextureSize[1] = size[1];
    TextureSize[2] = 1;
    MipLevelsStored = mipmapLevels;

    InternalFormat = internalFormat;
    COpenGLExtensionHandler::extGlTextureStorage2D(TextureName,GL_TEXTURE_2D,MipLevelsStored,InternalFormat,TextureSize[0],TextureSize[1]);

    ColorFormat = getColorFormatFromSizedOpenGLFormat(InternalFormat);
}

bool COpenGL2DTexture::updateSubRegion(const asset::E_FORMAT &inDataColorFormat, const void* data, const uint32_t* minimum, const uint32_t* maximum, int32_t mipmap, const uint32_t& unpackRowByteAlignment)
{
    bool sourceCompressed = isBlockCompressionFormat(inDataColorFormat);

    bool destinationCompressed = COpenGLTexture::isInternalFormatCompressed(InternalFormat);
    if ((!destinationCompressed)&&sourceCompressed)
        return false;

    if (destinationCompressed)
    {
        if (minimum[0]||minimum[1])
            return false;

        uint32_t adjustedTexSize[2] = {TextureSize[0],TextureSize[1]};
        adjustedTexSize[0] /= 0x1u<<mipmap;
        adjustedTexSize[1] /= 0x1u<<mipmap;
        /*
        adjustedTexSize[0] += 3u;
        adjustedTexSize[1] += 3u;
        adjustedTexSize[0] &= 0xfffffc;
        adjustedTexSize[1] &= 0xfffffc;
        */
        if (maximum[0]!=adjustedTexSize[0]||maximum[1]!=adjustedTexSize[1])
            return false;
    }

    if (sourceCompressed)
    {
        size_t levelByteSize = (((maximum[0]-minimum[0]+3)&0xfffffc)*((maximum[1]-minimum[1]+3)&0xfffffc)*COpenGLTexture::getOpenGLFormatBpp(InternalFormat))/8;

        COpenGLExtensionHandler::extGlCompressedTextureSubImage2D(TextureName,GL_TEXTURE_2D, mipmap, minimum[0],minimum[1],maximum[0]-minimum[0],maximum[1]-minimum[1], InternalFormat, levelByteSize, data);
    }
    else
    {
        GLenum pixFmt,pixType;
        getOpenGLFormatAndParametersFromColorFormat(inDataColorFormat, pixFmt, pixType);
        //! replace with
        ///COpenGLExtensionHandler::extGlGetInternalFormativ(GL_TEXTURE_2D,InternalFormat,GL_TEXTURE_IMAGE_FORMAT,1,&pixFmt);
        ///COpenGLExtensionHandler::extGlGetInternalFormativ(GL_TEXTURE_2D,InternalFormat,GL_TEXTURE_IMAGE_FORMAT,1,&pixType);

        //! we're going to have problems with uploading lower mip levels
        uint32_t bpp = video::getBitsPerPixelFromFormat(inDataColorFormat);
        uint32_t pitchInBits = ((maximum[0]-minimum[0])*bpp)/8;

        COpenGLExtensionHandler::setPixelUnpackAlignment(pitchInBits,const_cast<void*>(data),unpackRowByteAlignment);
        COpenGLExtensionHandler::extGlTextureSubImage2D(TextureName, GL_TEXTURE_2D, mipmap, minimum[0],minimum[1], maximum[0]-minimum[0],maximum[1]-minimum[1], pixFmt, pixType, data);
    }
    return true;
}

bool COpenGL2DTexture::resize(const uint32_t* size, const uint32_t &mipLevels)
{
    if (TextureSize[0]==size[0]&&TextureSize[1]==size[1])
        return true;

    recreateName(getOpenGLTextureType());

    uint32_t defaultMipMapCount = 1u+uint32_t(floorf(log2(float(core::max_(size[0],size[1])))));
    if (MipLevelsStored>1)
    {
        if (mipLevels==0)
            MipLevelsStored = defaultMipMapCount;
        else
            MipLevelsStored = core::min_(mipLevels,defaultMipMapCount);
    }

    TextureSize[0] = size[0];
    TextureSize[1] = size[1];
	COpenGLExtensionHandler::extGlTextureStorage2D(TextureName,getOpenGLTextureType(), MipLevelsStored, InternalFormat, TextureSize[0], TextureSize[1]);
	return true;
}

}
}
#endif

