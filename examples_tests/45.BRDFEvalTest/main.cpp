// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>
#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace core;

enum E_TEST_CASE
{
    ETC_GGX,
    ETC_BECKMANN,
    ETC_PHONG,
    ETC_AS,
    ETC_OREN_NAYAR,
    ETC_LAMBERT
};
class EventReceiver : public nbl::IEventReceiver
{
public:
	bool OnEvent(const nbl::SEvent& event)
	{
		if (event.EventType == nbl::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
		{
			switch (event.KeyInput.Key)
			{
				case nbl::KEY_KEY_Q:
					running = false;
					return true;
                case nbl::KEY_KEY_1:
                    test = ETC_GGX;
                    return true;
                case nbl::KEY_KEY_2:
                    test = ETC_BECKMANN;
                    return true;
                case nbl::KEY_KEY_3:
                    test = ETC_PHONG;
                    return true;
                case nbl::KEY_KEY_4:
                    test = ETC_AS;
                    return true;
                case nbl::KEY_KEY_5:
                    test = ETC_OREN_NAYAR;
                    return true;
                case nbl::KEY_KEY_6:
                    test = ETC_LAMBERT;
                    return true;

                case nbl::KEY_KEY_S:
                    ss = true;
                    break;
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

private:
    E_TEST_CASE test = ETC_GGX;
	bool running = true;
    bool ss = false;
};

core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> createGraphicsPipeline(video::IVideoDriver* _driver, core::smart_refctd_ptr<video::IGPUPipelineLayout>&& _layout, video::IGPUSpecializedShader* _vs, const std::string& _fsBaseSource, const char* _testName, const asset::IGeometryCreator::return_type& _dat)
{
    using namespace std::string_literals;

    std::string fsSrc = _fsBaseSource;
    const size_t _2ndLine = fsSrc.find('\n');
    fsSrc.insert(_2ndLine+1u, "#define "s + _testName + "\n");

    auto cpufs = core::make_smart_refctd_ptr<asset::ICPUShader>(fsSrc.c_str());
    auto fs = _driver->createShader(std::move(cpufs));
    asset::ISpecializedShader::SInfo fsinfo(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_FRAGMENT, "../shader.frag");
    auto fs_spec = _driver->createSpecializedShader(fs.get(), fsinfo);

    video::IGPUSpecializedShader* shaders[2] {_vs,fs_spec.get()};
    asset::SRasterizationParams raster;
    return _driver->createRenderpassIndependentPipeline(nullptr, std::move(_layout), shaders, shaders+2, _dat.inputParams, asset::SBlendParams{}, _dat.assemblyParams, raster);
}

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
    auto* fs = am->getFileSystem();
    auto* glslc = am->getGLSLCompiler();
    auto* gc = am->getGeometryCreator();

    struct SPushConsts
    {
        core::matrix4SIMD VP;
        core::vectorSIMDf campos;
    };

    core::smart_refctd_ptr<video::IGPUPipelineLayout> layout;
    {
        asset::SPushConstantRange rng[2];
        rng[0].offset = 0u;
        rng[0].size = sizeof(SPushConsts::VP);
        rng[0].stageFlags = asset::ISpecializedShader::ESS_VERTEX;
        rng[1].offset = offsetof(SPushConsts,campos);
        rng[1].size = sizeof(SPushConsts::campos);
        rng[1].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;

        layout = driver->createPipelineLayout(rng, rng+2);
    }

    constexpr uint32_t INSTANCE_COUNT = 10u;

    core::smart_refctd_ptr<video::IGPUMeshBuffer> sphere;
    core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> pipeline_ggx;
    core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> pipeline_beckmann;
    core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> pipeline_phong;
    core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> pipeline_as;
    core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> pipeline_orennayar;
    core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> pipeline_lambert;
    {
        auto dat = gc->createSphereMesh(0.5f, 50u, 50u);
        
        auto cpusphere = core::make_smart_refctd_ptr<asset::ICPUMeshBuffer>(nullptr, nullptr, dat.bindings, std::move(dat.indexBuffer));
        cpusphere->setBoundingBox(dat.bbox);
        cpusphere->setIndexType(dat.indexType);
        cpusphere->setIndexCount(dat.indexCount);
        cpusphere->setInstanceCount(INSTANCE_COUNT);

        io::IReadFile* file = fs->createAndOpenFile("../shader.vert");
        auto cpuvs = glslc->resolveIncludeDirectives(file, asset::ISpecializedShader::ESS_VERTEX, "../shader.vert");
        auto vs = driver->createShader(std::move(cpuvs));
        file->drop();
        asset::ISpecializedShader::SInfo vsinfo(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_VERTEX, "../shader.vert");
        auto vs_spec = driver->createSpecializedShader(vs.get(), vsinfo);

        file = fs->createAndOpenFile("../shader.frag");
        std::string fsSrc;
        fsSrc.resize(file->getSize());
        file->read(fsSrc.data(), fsSrc.size());
        file->drop();

        pipeline_ggx = createGraphicsPipeline(driver, core::smart_refctd_ptr(layout), vs_spec.get(), fsSrc, "TEST_GGX", dat);
        pipeline_beckmann = createGraphicsPipeline(driver, core::smart_refctd_ptr(layout), vs_spec.get(), fsSrc, "TEST_BECKMANN", dat);
        pipeline_phong = createGraphicsPipeline(driver, core::smart_refctd_ptr(layout), vs_spec.get(), fsSrc, "TEST_PHONG", dat);
        pipeline_as = createGraphicsPipeline(driver, core::smart_refctd_ptr(layout), vs_spec.get(), fsSrc, "TEST_AS", dat);
        pipeline_orennayar = createGraphicsPipeline(driver, core::smart_refctd_ptr(layout), vs_spec.get(), fsSrc, "TEST_OREN_NAYAR", dat);
        pipeline_lambert = createGraphicsPipeline(driver, core::smart_refctd_ptr(layout), vs_spec.get(), fsSrc, "TEST_LAMBERT", dat);

        sphere = driver->getGPUObjectsFromAssets(&cpusphere.get(), &cpusphere.get()+1)->front();
    }

    //! we want to move around the scene and view it from different angles
    scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(nullptr , 100.0f, 0.005f);

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

    auto imageView = driver->createImageView(std::move(imgViewInfo));

    auto* fbo = driver->addFrameBuffer();
    fbo->attach(video::EFAP_COLOR_ATTACHMENT0, core::smart_refctd_ptr(imageView));

    camera->setLeftHanded(false);
    camera->setPosition(core::vector3df(6.75f, 2.f, 6.f));
    camera->setTarget(core::vector3df(6.75f, 0.f, -1.f));
    camera->setFOV(core::radians(60.f));
    camera->setNearValue(0.01f);
    camera->setFarValue(5000.0f);

    smgr->setActiveCamera(camera);

    SPushConsts pc;
    uint32_t ssNum = 0u;
    while (device->run() && receiver.keepOpen())
    {
        driver->setRenderTarget(fbo);
        const float clear[4] {0.f,0.f,0.f,1.f};
        driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0, clear);
        driver->beginScene(true, false, video::SColor(255, 0, 0, 0));

        camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
        camera->render();

        pc.VP = camera->getConcatenatedMatrix();
        pc.campos = core::vectorSIMDf(&camera->getPosition().X);
        driver->pushConstants(layout.get(), asset::ISpecializedShader::ESS_VERTEX|asset::ISpecializedShader::ESS_FRAGMENT, 0u, sizeof(pc), &pc);
        switch (receiver.getTestCase())
        {
        case ETC_GGX:
            driver->bindGraphicsPipeline(pipeline_ggx.get()); break;
        case ETC_BECKMANN:
            driver->bindGraphicsPipeline(pipeline_beckmann.get()); break;
        case ETC_PHONG:
            driver->bindGraphicsPipeline(pipeline_phong.get()); break;
        case ETC_AS:
            driver->bindGraphicsPipeline(pipeline_as.get()); break;
        case ETC_OREN_NAYAR:
            driver->bindGraphicsPipeline(pipeline_orennayar.get()); break;
        case ETC_LAMBERT:
            driver->bindGraphicsPipeline(pipeline_lambert.get()); break;
        }
        driver->drawMeshBuffer(sphere.get());

        driver->blitRenderTargets(fbo, nullptr, false, false);

        driver->endScene();

        if (receiver.screenshot())
            ext::ScreenShot::createScreenShot(device, fbo->getAttachment(video::EFAP_COLOR_ATTACHMENT0), "screenshot" + std::to_string(ssNum++) + ".exr");
    }


    return 0;
}