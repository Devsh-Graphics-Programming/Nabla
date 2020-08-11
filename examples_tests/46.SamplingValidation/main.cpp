#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>
#include "../../ext/ScreenShot/ScreenShot.h"
#include "../../ext/FullScreenTriangle/FullScreenTriangle.h"
#include "../common/QToQuitEventReceiver.h"

using namespace irr;
using namespace core;

int main()
{
    // create device with full flexibility over creation parameters
    // you can add more parameters if desired, check irr::SIrrlichtCreationParameters
    irr::SIrrlichtCreationParameters params;
    params.Bits = 24; //may have to set to 32bit for some platforms
    params.ZBufferBits = 24; //we'd like 32bit here
    params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
    params.WindowSize = dimension2d<uint32_t>(1280, 720);
    params.Fullscreen = false;
    params.Vsync = true; //! If supported by target platform
    params.Doublebuffer = true;
    params.Stencilbuffer = false; //! This will not even be a choice soon
    auto device = createDeviceEx(params);

    if (!device)
        return 1; // could not create selected driver.

    //! disable mouse cursor, since camera will force it to the middle
    //! and we don't want a jittery cursor in the middle distracting us
    device->getCursorControl()->setVisible(false);

    //! Since our cursor will be enslaved, there will be no way to close the window
    //! So we listen for the "Q" key being pressed and exit the application
    QToQuitEventReceiver receiver;
    device->setEventReceiver(&receiver);

    auto* driver = device->getVideoDriver();
    auto* smgr = device->getSceneManager();
    auto* am = device->getAssetManager();
    auto* filesystem = am->getFileSystem();
    auto* glslc = am->getGLSLCompiler();
    auto* gc = am->getGeometryCreator();

	video::IGPUImage::SCreationParams imgInfo;
	imgInfo.format = asset::EF_R16G16B16A16_SFLOAT;
	imgInfo.type = asset::ICPUImage::ET_2D;
	imgInfo.extent.width = driver->getScreenSize().Width;
	imgInfo.extent.height = driver->getScreenSize().Height;
	imgInfo.extent.depth = 1u;
	imgInfo.mipLevels = 1u;
	imgInfo.arrayLayers = 1u;
	imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
	imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);

	auto image = driver->createGPUImageOnDedMem(std::move(imgInfo), driver->getDeviceLocalGPUMemoryReqs());
	const auto texelFormatBytesize = getTexelOrBlockBytesize(image->getCreationParameters().format);

	video::IGPUImageView::SCreationParams imgViewInfo;
	imgViewInfo.format = image->getCreationParameters().format;
	imgViewInfo.image = std::move(image);
	imgViewInfo.viewType = asset::IImageView<video::IGPUImage>::ET_2D;
	imgViewInfo.flags = static_cast<video::IGPUImageView::E_CREATE_FLAGS>(0u);
	imgViewInfo.subresourceRange.baseArrayLayer = 0u;
	imgViewInfo.subresourceRange.baseMipLevel = 0u;
	imgViewInfo.subresourceRange.layerCount = imgInfo.arrayLayers;
	imgViewInfo.subresourceRange.levelCount = imgInfo.mipLevels;

    auto imageView = driver->createGPUImageView(std::move(imgViewInfo));

    auto* fbo = driver->addFrameBuffer();
    fbo->attach(video::EFAP_COLOR_ATTACHMENT0, core::smart_refctd_ptr(imageView));

    core::smart_refctd_ptr<video::IGPUSpecializedShader> fs;
    {
        auto* file = filesystem->createAndOpenFile("../fullscreen.frag");

        auto cpufs_unspec = glslc->resolveIncludeDirectives(file, asset::ISpecializedShader::ESS_FRAGMENT, "../fullscreen.frag");
        auto fs_unspec = driver->createGPUShader(std::move(cpufs_unspec));

        asset::ISpecializedShader::SInfo info(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_FRAGMENT);
        fs = driver->createGPUSpecializedShader(fs_unspec.get(), info);
    }

    auto fsTri = ext::FullScreenTriangle::createFullScreenTriangle(std::move(fs), driver->createGPUPipelineLayout(), am, driver);

    while (device->run() && receiver.keepOpen())
    {
        driver->beginScene(true, false, video::SColor(255, 255, 255, 255));

        driver->bindGraphicsPipeline(fsTri->getPipeline());
        driver->drawMeshBuffer(fsTri.get());

        driver->endScene();
    }

    ext::ScreenShot::createScreenShot(device, fbo->getAttachment(video::EFAP_COLOR_ATTACHMENT0), "ss.exr");

    return 0;
}