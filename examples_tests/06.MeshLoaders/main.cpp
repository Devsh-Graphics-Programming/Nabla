#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

//! I advise to check out this file, its a basic input handler
#include "../common/QToQuitEventReceiver.h"

//#include "../../ext/ScreenShot/ScreenShot.h"


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

    asset::IAssetLoader::SAssetLoadParams lp;
    auto meshes_bundle = am->getAsset("../../media/sponza/sponza.obj", lp);
    assert(!meshes_bundle.isEmpty());
    auto mesh = meshes_bundle.getContents().first[0];
    auto mesh_raw = static_cast<asset::ICPUMesh*>(mesh.get());

    asset::ICPUDescriptorSetLayout* ds1layout = mesh_raw->getMeshBuffer(0u)->getPipeline()->getLayout()->getDescriptorSetLayout(1u);
    //pipelines attached to meshbuffers from OBJ loader has DS1 layout being "irr/builtin/ds_layout/default_ds1_layout"
    //that is designed for use with "irr/builtin/ds/default_ds1", so let's use it!
    auto ds1_bundle = am->getAsset("irr/builtin/descriptor_set/basic_view_parameters", lp);
    //you might want to IAsset::clone() this DS in order to have exact copy but not the same object. However in this case won't be doing it.
    //(same CPU object -> same GPU object returned from driver->getGPUObjectsFromAssets())
    //use IAsset::clone()'s optional parameter to clone the buffer (being the only descriptor in default DS1) as well
    auto ds1 = ds1_bundle.getContents().first[0];
    auto ds1_raw = static_cast<asset::ICPUDescriptorSet*>(ds1.get());
    asset::ICPUBuffer* ubo = static_cast<asset::ICPUBuffer*>(ds1_raw->getDescriptors(0u).begin()->desc.get());

    auto gpuds1 = driver->getGPUObjectsFromAssets(&ds1_raw,&ds1_raw+1)->front();
    //video::IGPUBuffer* gpuubo = gpuds1->getDescriptors()...//TODO GPU DS needs some (constant?) getter to get its descriptors
    auto gpuubo = driver->getGPUObjectsFromAssets(&ubo,&ubo+1)->front();

    auto gpumesh = driver->getGPUObjectsFromAssets(&mesh_raw, &mesh_raw + 1)->front();

	//! we want to move around the scene and view it from different angles
	scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0,100.0f,0.5f);

	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(10000.0f);

    smgr->setActiveCamera(camera);

	uint64_t lastFPSTime = 0;
	while(device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255,255,255,255) );

        //! This animates (moves) the camera and sets the transforms
		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();

        asset::SBasicViewParameters uboData;
		core::matrix4SIMD mvp = camera->getConcatenatedMatrix();
        memcpy(uboData.MVP, mvp.pointer(), sizeof(uboData.MVP));
        core::matrix3x4SIMD MV3x4;
        MV3x4.set(camera->getViewMatrix());
        core::matrix4SIMD MV(MV3x4);
        memcpy(uboData.MV, MV.pointer(), sizeof(uboData.MV));
        memcpy(uboData.NormalMat, MV.pointer(), sizeof(uboData.NormalMat));
        driver->updateBufferRangeViaStagingBuffer(gpuubo->getBuffer(), gpuubo->getOffset(), sizeof(uboData), &uboData);

        for (uint32_t i = 0u; i < gpumesh->getMeshBufferCount(); ++i)
        {
            video::IGPUMeshBuffer* gpumb = gpumesh->getMeshBuffer(i);
            const video::IGPURenderpassIndependentPipeline* pipeline = gpumb->getPipeline();
            const video::IGPUDescriptorSet* ds3 = gpumb->getAttachedDescriptorSet();

            driver->bindGraphicsPipeline(pipeline);
            const video::IGPUDescriptorSet* gpuds1_ptr = gpuds1.get();
            driver->bindDescriptorSets(video::EPBP_GRAPHICS, pipeline->getLayout(), 1u, 1u, &gpuds1_ptr, nullptr);
            const video::IGPUDescriptorSet* gpuds3_ptr = gpumb->getAttachedDescriptorSet();
            if (gpuds3_ptr)
                driver->bindDescriptorSets(video::EPBP_GRAPHICS, pipeline->getLayout(), 3u, 1u, &gpuds3_ptr, nullptr);
            driver->pushConstants(pipeline->getLayout(), video::IGPUSpecializedShader::ESS_FRAGMENT, 0u, gpumb->MAX_PUSH_CONSTANT_BYTESIZE, gpumb->getPushConstantsDataPtr());

            driver->drawMeshBuffer(gpumb);
        }

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"GPU Mesh Demo - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str().c_str());
			lastFPSTime = time;
		}
	}

	//create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		//ext::ScreenShot::dirtyCPUStallingScreenshot(device, "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}

	return 0;
}