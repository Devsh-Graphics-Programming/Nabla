// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>
#include "../common/QToQuitEventReceiver.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

#include "nbl/ext/Blur/CBlurPerformer.h"

using namespace nbl;
using namespace core;


//!Same As Last Example
class MyEventReceiver : public QToQuitEventReceiver
{
		using Base = QToQuitEventReceiver;
		const float BLUR_RADIUS_MIN = 0.f, BLUR_RADIUS_MAX = 0.5f;

	public:
		ext::Blur::CBlurPerformer* blurPerf = nullptr;

		bool OnEvent(const SEvent& event)
		{
			if (!Base::OnEvent(event))
			if (event.EventType == nbl::EET_MOUSE_INPUT_EVENT && blurPerf)
			{
				float r = blurPerf->getRadius() + event.MouseInput.Wheel/500.f;
				blurPerf->setRadius(std::max(BLUR_RADIUS_MIN, std::min(r, BLUR_RADIUS_MAX)));
				//return true;
			}

			return false;
		}
};

class SimpleCallBack : public video::IShaderConstantSetCallBack
{
    int32_t mvpUniformLocation;
    int32_t cameraDirUniformLocation;
    video::E_SHADER_CONSTANT_TYPE mvpUniformType;
    video::E_SHADER_CONSTANT_TYPE cameraDirUniformType;

public:
    SimpleCallBack() : cameraDirUniformLocation(-1), cameraDirUniformType(video::ESCT_FLOAT_VEC3) {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
    {
        for (size_t i = 0; i<constants.size(); i++)
        {
            if (constants[i].name == "MVP")
            {
                mvpUniformLocation = constants[i].location;
                mvpUniformType = constants[i].type;
            }
            else if (constants[i].name == "cameraPos")
            {
                cameraDirUniformLocation = constants[i].location;
                cameraDirUniformType = constants[i].type;
            }
        }
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        core::vectorSIMDf modelSpaceCamPos;
        modelSpaceCamPos.set(services->getVideoDriver()->getTransform(video::E4X3TS_WORLD_VIEW_INVERSE).getTranslation());
        services->setShaderConstant(&modelSpaceCamPos, cameraDirUniformLocation, cameraDirUniformType, 1);
        services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(), mvpUniformLocation, mvpUniformType, 1);
    }

    virtual void OnUnsetMaterial() {}
};

//! Use scroll to adjust blur radius
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
    IrrlichtDevice* device = createDeviceEx(params);

    if (device == 0)
        return 1; // could not create selected driver.

    video::IVideoDriver* driver = device->getVideoDriver();

    SimpleCallBack* cb = new SimpleCallBack();
    video::E_MATERIAL_TYPE newMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../cube.vert",
        "", "", "", //! No Geometry or Tessellation Shaders
        "../cube.frag",
        3, video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
        cb, //! Our Shader Callback
        0); //! No custom user data
    cb->drop();


    scene::ISceneManager* smgr = device->getSceneManager();
    driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
    scene::ICameraSceneNode* camera =
        smgr->addCameraSceneNodeFPS(0, 100.0f, 0.01f);
    camera->setPosition(core::vector3df(-4, 0, 0));
    camera->setTarget(core::vector3df(0, 0, 0));
    camera->setNearValue(0.01f);
    camera->setFarValue(100.0f);
    smgr->setActiveCamera(camera);
    device->getCursorControl()->setVisible(false);
    MyEventReceiver receiver;
    device->setEventReceiver(&receiver);


    auto cpumesh = device->getAssetManager()->getGeometryCreator()->createCubeMesh();
    asset::IAssetLoader::SAssetLoadParams lparams;
    auto cputexture = core::smart_refctd_ptr_static_cast<asset::ICPUTexture>(*device->getAssetManager()->getAsset("../tex.jpg", lparams).getContents().first);
    auto texture = driver->getGPUObjectsFromAssets(&cputexture.get(), (&cputexture.get())+1)->front();

    const core::vector2d<uint32_t> dsFactor{ 1u, 1u };
    ext::Blur::CBlurPerformer* blur = ext::Blur::CBlurPerformer::instantiate(driver, device->getIncludeHandler(), 0.01f, dsFactor, 2u);
    receiver.blurPerf = blur;

    auto outputTex = blur->createOutputTexture(texture.get());

    cpumesh->getMeshBuffer(0)->getMaterial().TextureLayer[0].SamplingParams.TextureWrapU = video::ETC_CLAMP_TO_EDGE;
    cpumesh->getMeshBuffer(0)->getMaterial().TextureLayer[0].SamplingParams.TextureWrapV = video::ETC_CLAMP_TO_EDGE;

    auto gpumesh = driver->getGPUObjectsFromAssets(&cpumesh.get(), (&cpumesh.get())+1)->front();
    video::SGPUMaterial& mutableMaterial = smgr->addMeshSceneNode(core::smart_refctd_ptr(gpumesh))->getMesh()->getMeshBuffer(0)->getMaterial();
    mutableMaterial.MaterialType = static_cast<video::E_MATERIAL_TYPE>(newMaterialType);
    mutableMaterial.TextureLayer[0].Texture = outputTex;

    cpumesh = device->getAssetManager()->getGeometryCreator()->createCubeMesh();
    cpumesh->getMeshBuffer(0)->getMaterial().TextureLayer[0].SamplingParams.TextureWrapU = video::ETC_CLAMP_TO_EDGE;
    cpumesh->getMeshBuffer(0)->getMaterial().TextureLayer[0].SamplingParams.TextureWrapV = video::ETC_CLAMP_TO_EDGE;

    gpumesh = driver->getGPUObjectsFromAssets(&cpumesh.get(), (&cpumesh.get())+1)->front();
    video::SGPUMaterial& mat2 = smgr->addMeshSceneNode(core::smart_refctd_ptr(gpumesh), nullptr, -1, vector3df(10.f, 0.f, 0.f))->getMesh()->getMeshBuffer(0)->getMaterial();
    mat2.MaterialType = static_cast<video::E_MATERIAL_TYPE>(newMaterialType);
    mat2.TextureLayer[0].Texture = texture;

    uint64_t lastFPSTime = 0;

    while (device->run() && receiver.keepOpen() )
    {
        driver->beginScene(true, true, video::SColor(255, 0, 0, 255));

        blur->blurTexture(texture.get(), outputTex.get());

        smgr->drawAll();

        driver->endScene();

        // display frames per second in window title
        uint64_t time = device->getTimer()->getRealTime();
        if (time - lastFPSTime > 1000)
        {
            std::wostringstream sstr;
            sstr << L"Builtin Nodes Demo - Irrlicht Engine FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

            device->setWindowCaption(sstr.str().c_str());
            lastFPSTime = time;
        }
    }

    blur->drop();

	//create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		ext::ScreenShot::dirtyCPUStallingScreenshot(device, "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}

    device->drop();

    return 0;
}
