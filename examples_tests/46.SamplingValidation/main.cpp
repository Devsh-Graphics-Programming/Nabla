// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

using namespace nbl;
using namespace core;

enum E_TEST_CASE : uint32_t
{
    ETC_LAMBERT,
    ETC_GGX,
    ETC_BECKMANN,
    ETC_LAMBERT_TRANSMIT,
    ETC_GGX_TRANSMIT,
    ETC_BECKMANN_TRANSMIT
};
class EventReceiver : public nbl::IEventReceiver
{
public:
	bool OnEvent(const nbl::SEvent& event)
	{
        constexpr float da = 0.1f;
		if (event.EventType == nbl::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
		{
			switch (event.KeyInput.Key)
			{
				case nbl::KEY_KEY_Q:
					running = false;
					return true;
                case nbl::KEY_KEY_1:
                    test = ETC_LAMBERT;
                    return true;
                case nbl::KEY_KEY_2:
                    test = ETC_GGX;
                    return true;
                case nbl::KEY_KEY_3:
                    test = ETC_BECKMANN;
                    return true;
                case nbl::KEY_KEY_4:
                    test = ETC_LAMBERT_TRANSMIT;
                    return true;
                case nbl::KEY_KEY_5:
                    test = ETC_GGX_TRANSMIT;
                    return true;
                case nbl::KEY_KEY_6:
                    test = ETC_BECKMANN_TRANSMIT;
                    return true;
                case nbl::KEY_KEY_S:
                    ss = true;
                    return true;

                case nbl::KEY_KEY_Z:
                    m_ax = std::max(0.f, m_ax-da);
                    return true;
                case nbl::KEY_KEY_X:
                    m_ax = std::min(1.f, m_ax+da);
                    return true;

                case nbl::KEY_KEY_C:
                    m_ay = std::max(0.f, m_ay-da);
                    return true;
                case nbl::KEY_KEY_V:
                    m_ay = std::min(1.f, m_ay+da);
                    return true;
				default:
					break;
			}
		}

		return false;
	}

	inline bool keepOpen() const { return running; }
    inline E_TEST_CASE getTestCase() const { return test; }
    inline bool screenshot()
    {
        bool v = ss;
        ss = false;
        return v;
    }
    inline float ax() const { return m_ax; }
    inline float ay() const { return m_ay; }

private:
    E_TEST_CASE test = ETC_GGX;
	bool running = true;
    bool ss = false;
    float m_ax = 0.5f, m_ay = 0.5f;
};

int main()
{
    // create device with full flexibility over creation parameters
    // you can add more parameters if desired, check nbl::SIrrlichtCreationParameters
    nbl::SIrrlichtCreationParameters params;
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
    EventReceiver receiver;
    device->setEventReceiver(&receiver);

    auto* driver = device->getVideoDriver();
    auto* smgr = device->getSceneManager();
    auto* am = device->getAssetManager();
    auto* filesystem = am->getFileSystem();
    auto* glslc = am->getGLSLCompiler();
    auto* gc = am->getGeometryCreator();

    core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> pipeline_ggx;
    core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> pipeline_beckmann;
    core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> pipeline_cosw;

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

    video::IGPUImageView::SCreationParams imgViewInfo;
    imgViewInfo.format = image->getCreationParameters().format;
    imgViewInfo.image = std::move(image);
    imgViewInfo.viewType = asset::IImageView<video::IGPUImage>::ET_2D;
    imgViewInfo.flags = static_cast<video::IGPUImageView::E_CREATE_FLAGS>(0u);
    imgViewInfo.subresourceRange.baseArrayLayer = 0u;
    imgViewInfo.subresourceRange.baseMipLevel = 0u;
    imgViewInfo.subresourceRange.layerCount = imgInfo.arrayLayers;
    imgViewInfo.subresourceRange.levelCount = imgInfo.mipLevels;

    auto imageView = driver->createImageView(std::move(imgViewInfo));

    auto* fbo = driver->addFrameBuffer();
    fbo->attach(video::EFAP_COLOR_ATTACHMENT0, core::smart_refctd_ptr(imageView));

    core::smart_refctd_ptr<video::IGPUSpecializedShader> fs;
    {
        auto* file = filesystem->createAndOpenFile("../fullscreen.frag");

        auto cpufs_unspec = glslc->resolveIncludeDirectives(file, asset::ISpecializedShader::ESS_FRAGMENT, "../fullscreen.frag");
        auto fs_unspec = driver->createShader(std::move(cpufs_unspec));

        asset::ISpecializedShader::SInfo info(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_FRAGMENT);
        fs = driver->createSpecializedShader(fs_unspec.get(), info);
    }

    struct SPushConsts
    {
        float ax;
        float ay;
        E_TEST_CASE test;
    };
    asset::SPushConstantRange rng;
    rng.offset = 0u;
    rng.size = sizeof(SPushConsts);
    rng.stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
    auto fsTri = ext::FullScreenTriangle::createFullScreenTriangle(std::move(fs), driver->createPipelineLayout(&rng,&rng+1), am, driver);

    uint32_t ssNum = 0u;
    while (device->run() && receiver.keepOpen())
    {
        driver->setRenderTarget(fbo);
        driver->beginScene(true, false, video::SColor(255, 255, 255, 255));

        SPushConsts pc;
        pc.ax = receiver.ax();
        pc.ay = receiver.ay();
        pc.test = receiver.getTestCase();
        driver->pushConstants(fsTri->getPipeline()->getLayout(), asset::ISpecializedShader::ESS_FRAGMENT, 0u, sizeof(pc), &pc);
        driver->bindGraphicsPipeline(fsTri->getPipeline());
        driver->drawMeshBuffer(fsTri.get());

        driver->blitRenderTargets(fbo, nullptr, false, false);

        driver->endScene();

        if (receiver.screenshot())
            ext::ScreenShot::createScreenShot(device, fbo->getAttachment(video::EFAP_COLOR_ATTACHMENT0), "screenshot" + std::to_string(ssNum++) + ".exr");
    }

    return 0;
}