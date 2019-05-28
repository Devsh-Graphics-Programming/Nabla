#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

#include "COpenGLDriver.h"

#include "COpenGLMultisampleTexture.h"


namespace irr
{
namespace video
{

COpenGLMultisampleTexture::COpenGLMultisampleTexture(GLenum internalFormat, const uint32_t& samples, const uint32_t* size, const bool& fixedSampleLocations)
                                                        :   COpenGLTexture(GL_TEXTURE_2D_MULTISAMPLE), IMultisampleTexture(IDriverMemoryBacked::SDriverMemoryRequirements{{0,0,0},0,0,0,0}),
                                                            SampleCount(samples), FixedSampleLocations(fixedSampleLocations), InternalFormat(internalFormat)
{
#ifdef _IRR_DEBUG
	setDebugName("COpenGLMultisampleTexture");
	assert(core::isPoT(samples));
#endif
    TextureSize[0] = size[0];
    TextureSize[1] = size[1];

    COpenGLExtensionHandler::extGlTextureStorage2DMultisample(TextureName,GL_TEXTURE_2D_MULTISAMPLE,SampleCount,InternalFormat,TextureSize[0],TextureSize[1],FixedSampleLocations);

    ColorFormat = getColorFormatFromSizedOpenGLFormat(InternalFormat);
}

bool COpenGLMultisampleTexture::resize(const uint32_t* size, const uint32_t& sampleCount)
{
    uint32_t newSampleCount = sampleCount ? sampleCount:SampleCount;
    return resize(size,newSampleCount,FixedSampleLocations);
}

bool COpenGLMultisampleTexture::resize(const uint32_t* size, const uint32_t& sampleCount, const bool& fixedSampleLocations)
{
    if (TextureSize[0]==size[0]&&TextureSize[1]==size[1]&&SampleCount==sampleCount&&FixedSampleLocations==fixedSampleLocations)
        return true;

    if (core::isNPoT(sampleCount))
        return false;

    recreateName(getOpenGLTextureType());

    memcpy(TextureSize,size,8);
    SampleCount = sampleCount;
    FixedSampleLocations = fixedSampleLocations;
    COpenGLExtensionHandler::extGlTextureStorage2DMultisample(TextureName,GL_TEXTURE_2D_MULTISAMPLE,SampleCount,InternalFormat,TextureSize[0],TextureSize[1],FixedSampleLocations);
    return true;
}

}
}
#endif
