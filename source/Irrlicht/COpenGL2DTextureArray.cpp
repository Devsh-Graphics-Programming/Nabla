#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

#include "irrTypes.h"
#include "COpenGL2DTextureArray.h"
#include "COpenGLDriver.h"


#include "irrString.h"

namespace irr
{
namespace video
{


COpenGL2DTextureArray::COpenGL2DTextureArray(GLenum internalFormat, const uint32_t* size, uint32_t mipmapLevels, const io::path& name) : COpenGLFilterableTexture(name)
{
#ifdef _DEBUG
	setDebugName("COpenGL2DTextureArray");
#endif
    TextureSize[0] = size[0];
    TextureSize[1] = size[1];
    TextureSize[2] = size[2];
    MipLevelsStored = mipmapLevels;

    COpenGLExtensionHandler::extGlCreateTextures(GL_TEXTURE_2D_ARRAY,1,&TextureName);

    InternalFormat = internalFormat;
    COpenGLExtensionHandler::extGlTextureStorage3D(TextureName,GL_TEXTURE_2D_ARRAY,MipLevelsStored,InternalFormat,TextureSize[0],TextureSize[1],TextureSize[2]);

    ColorFormat = getColorFormatFromSizedOpenGLFormat(InternalFormat);
}

bool COpenGL2DTextureArray::updateSubRegion(const ECOLOR_FORMAT &inDataColorFormat, const void* data, const uint32_t* minimum, const uint32_t* maximum, int32_t mipmap, const uint32_t& unpackRowByteAlignment)
{
    GLenum pixFmt,pixType;
	GLenum pixFmtSized = getOpenGLFormatAndParametersFromColorFormat(inDataColorFormat, pixFmt, pixType);
    bool sourceCompressed = COpenGLTexture::isInternalFormatCompressed(pixFmtSized);

    bool destinationCompressed = COpenGLTexture::isInternalFormatCompressed(InternalFormat);
    if ((!destinationCompressed)&&sourceCompressed)
        return false;

    if (destinationCompressed&&(minimum[0]||minimum[1]))
        return false;

    if (destinationCompressed)
    {
        size_t levelByteSize = (maximum[2]-minimum[2]);
        levelByteSize *= ((((maximum[0]-minimum[0])/(0x1u<<mipmap)+3)&0xfffffc)*(((maximum[1]-minimum[1])/(0x1u<<mipmap)+3)&0xfffffc)*COpenGLTexture::getOpenGLFormatBpp(InternalFormat))/8;

        COpenGLExtensionHandler::extGlCompressedTextureSubImage3D(TextureName,GL_TEXTURE_2D_ARRAY, mipmap, minimum[0],minimum[1],minimum[2], maximum[0]-minimum[0],maximum[1]-minimum[1],maximum[2]-minimum[2], InternalFormat, levelByteSize, data);
    }
    else
    {
        //! we're going to have problems with uploading lower mip levels
        uint32_t bpp = video::getBitsPerPixelFromFormat(inDataColorFormat);
        uint32_t pitchInBits = ((maximum[0]-minimum[0])>>mipmap)*bpp/8;

        COpenGLExtensionHandler::setPixelUnpackAlignment(pitchInBits,const_cast<void*>(data),unpackRowByteAlignment);
        COpenGLExtensionHandler::extGlTextureSubImage3D(TextureName,GL_TEXTURE_2D_ARRAY, mipmap, minimum[0],minimum[1],minimum[2], maximum[0]-minimum[0],maximum[1]-minimum[1],maximum[2]-minimum[2], pixFmt, pixType, data);
    }
    return true;
}

bool COpenGL2DTextureArray::resize(const uint32_t* size, const uint32_t& mipLevels)
{
    if (TextureSize[0]==size[0]&&TextureSize[1]==size[1]&&TextureSize[2]==size[2])
        return true;

    if (TextureName)
        glDeleteTextures(1, &TextureName);
    COpenGLExtensionHandler::extGlCreateTextures(GL_TEXTURE_2D_ARRAY,1,&TextureName);
	TextureNameHasChanged = CNullDriver::incrementAndFetchReallocCounter();

    uint32_t defaultMipMapCount = 1u+uint32_t(floorf(log2(float(core::max_(size[0],size[1])))));
    if (MipLevelsStored>1)
    {
        if (mipLevels==0)
            MipLevelsStored = defaultMipMapCount;
        else
            MipLevelsStored = core::min_(mipLevels,defaultMipMapCount);
    }


    COpenGLExtensionHandler::extGlTextureStorage3D(TextureName,GL_TEXTURE_2D_ARRAY,MipLevelsStored,InternalFormat,size[0], size[1], size[2]);

    memcpy(TextureSize,size,12);
    return true;
}

}
}
#endif

