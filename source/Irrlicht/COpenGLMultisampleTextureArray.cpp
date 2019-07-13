#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

#include "COpenGLDriver.h"

#include "COpenGLMultisampleTextureArray.h"


namespace irr
{
namespace video
{

COpenGLMultisampleTextureArray::COpenGLMultisampleTextureArray(GLenum internalFormat, const uint32_t& samples, const uint32_t* size, const bool& fixedSampleLocations)
                                                        :   COpenGLTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY), IMultisampleTexture(IDriverMemoryBacked::SDriverMemoryRequirements{{0,0,0},0,0,1,1}),
                                                            SampleCount(samples), FixedSampleLocations(fixedSampleLocations), InternalFormat(internalFormat)
{
#ifdef _IRR_DEBUG
	setDebugName("COpenGLMultisampleTextureArray");
	assert(core::isPoT(samples));
#endif
    TextureSize[0] = size[0];
    TextureSize[1] = size[1];
    TextureSize[2] = size[2];

    COpenGLExtensionHandler::extGlTextureStorage3DMultisample(TextureName,GL_TEXTURE_2D_MULTISAMPLE_ARRAY,SampleCount,InternalFormat,TextureSize[0],TextureSize[1],TextureSize[2],FixedSampleLocations);

    ColorFormat = getColorFormatFromSizedOpenGLFormat(InternalFormat);
}

bool COpenGLMultisampleTextureArray::resize(const uint32_t* size, const uint32_t& sampleCount)
{
    uint32_t newSampleCount = sampleCount ? sampleCount:SampleCount;
    return resize(size,newSampleCount,FixedSampleLocations);
}

bool COpenGLMultisampleTextureArray::resize(const uint32_t* size, const uint32_t& sampleCount, const bool& fixedSampleLocations)
{
    if (TextureSize[0]==size[0]&&TextureSize[1]==size[1]&&TextureSize[2]==size[2]&&SampleCount==sampleCount&&FixedSampleLocations==fixedSampleLocations)
        return true;

    if (core::isNPoT(sampleCount))
        return false;

    recreateName(getOpenGLTextureType());

    memcpy(TextureSize,size,12);
    SampleCount = sampleCount;
    FixedSampleLocations = fixedSampleLocations;
    COpenGLExtensionHandler::extGlTextureStorage3DMultisample(TextureName,GL_TEXTURE_2D_MULTISAMPLE_ARRAY,SampleCount,InternalFormat,TextureSize[0],TextureSize[1],TextureSize[2],FixedSampleLocations);
    return true;
}

}
}
#endif

