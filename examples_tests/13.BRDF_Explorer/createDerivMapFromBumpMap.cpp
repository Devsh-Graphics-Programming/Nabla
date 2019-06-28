#include <irrlicht.h>
#include "createComputeShader.h"
#include "../source/Irrlicht/COpenGL2DTexture.h"
#include "../source/Irrlicht/COpenGLDriver.h"
#include "CComputeShaderManager.h"

using namespace irr;

video::IVirtualTexture* createDerivMapFromBumpMap(video::IVirtualTexture* _bumpMap, IrrlichtDevice* _device, float _heightFactor)
{
    asset::IAssetManager& assetMgr = _device->getAssetManager();
    asset::IAssetLoader::SAssetLoadParams lparams;

    video::IVideoDriver* driver = _device->getVideoDriver();

    const uint32_t* derivMap_sz = _bumpMap->getSize();
    video::ITexture* derivMap = driver->createGPUTexture(video::ITexture::ETT_2D, derivMap_sz, std::log2(std::max(derivMap_sz[0], derivMap_sz[1])), asset::EF_R8G8_SNORM);

    {
        video::STextureSamplingParams params;
        params.UseMipmaps = 0;
        params.MaxFilter = params.MinFilter = video::ETFT_LINEAR_NO_MIP;
        params.TextureWrapU = params.TextureWrapV = video::ETC_CLAMP_TO_EDGE;
        const_cast<video::COpenGLDriver::SAuxContext*>(reinterpret_cast<video::COpenGLDriver*>(driver)->getThreadContext())->setActiveTexture(7, _bumpMap, params);
    }

    GLuint deriv_map_gen_cs = CComputeShaderManager::getShader("../deriv_map_gen.comp");

    GLint previousProgram;
    glGetIntegerv(GL_CURRENT_PROGRAM, &previousProgram);

    video::COpenGLExtensionHandler::extGlBindImageTexture(0, static_cast<const video::COpenGL2DTexture*>(derivMap)->getOpenGLName(),
        0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG8_SNORM);

    video::COpenGLExtensionHandler::extGlUseProgram(deriv_map_gen_cs);
    video::COpenGLExtensionHandler::extGlProgramUniform1fv(deriv_map_gen_cs, 0, 1u, &_heightFactor);
    video::COpenGLExtensionHandler::extGlDispatchCompute((derivMap_sz[0]+15u) / 16u, (derivMap_sz[1]+15u) / 16u, 1u);
    video::COpenGLExtensionHandler::extGlMemoryBarrier(
        GL_TEXTURE_FETCH_BARRIER_BIT |
        GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
        GL_PIXEL_BUFFER_BARRIER_BIT |
        GL_TEXTURE_UPDATE_BARRIER_BIT |
        GL_FRAMEBUFFER_BARRIER_BIT
    );

    video::COpenGLExtensionHandler::extGlBindImageTexture(0u, 0u, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R8); //unbind image
    { //unbind texture
        video::STextureSamplingParams params;
        const_cast<video::COpenGLDriver::SAuxContext*>(reinterpret_cast<video::COpenGLDriver*>(driver)->getThreadContext())->setActiveTexture(7, nullptr, params);
    }
    video::COpenGLExtensionHandler::extGlUseProgram(previousProgram); //rebind previously bound program

    derivMap->regenerateMipMapLevels();

    return derivMap;
}