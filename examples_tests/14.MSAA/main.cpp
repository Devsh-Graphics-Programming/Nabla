// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>
#include <iostream>
#include <cstdio>

#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

#include "../common/QToQuitEventReceiver.h"


#include "COpenGLStateManager.h"

using namespace nbl;
using namespace core;

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
        for (size_t i=0; i<constants.size(); i++)
        {
            if (constants[i].name=="MVP")
            {
                mvpUniformLocation = constants[i].location;
                mvpUniformType = constants[i].type;
            }
            else if (constants[i].name=="cameraPos")
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
        if (cameraDirUniformLocation!=-1)
            services->setShaderConstant(&modelSpaceCamPos,cameraDirUniformLocation,cameraDirUniformType,1);
        if (mvpUniformLocation!=-1)
            services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(),mvpUniformLocation,mvpUniformType,1);

    }

    virtual void OnUnsetMaterial() {}
};

class PostProcCallBack : public video::IShaderConstantSetCallBack
{
    int32_t sampleCountUniformLocation;
    video::E_SHADER_CONSTANT_TYPE sampleCountUniformType;
public:
    PostProcCallBack() {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
    {
        /**
        Shader Unigorms get saved as Program (Shader state)
        So we can perma-assign texture slots to sampler uniforms
        **/
        for (size_t i=0; i<constants.size(); i++)
        {
            if (constants[i].name=="sampleCount")
            {
                sampleCountUniformLocation = constants[i].location;
                sampleCountUniformType = constants[i].type;
            }
        }
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        if (sampleCountUniformLocation!=-1)
            services->setShaderConstant(&userData,sampleCountUniformLocation,sampleCountUniformType,1);
    }

    virtual void OnUnsetMaterial() {}
};


int main()
{
	printf("Enter the number of samples to use for MSAA: ");

	uint32_t numberOfSamples = 8;
	std::cin >> numberOfSamples;
	// yeah yeah I'll add a query mechanism for the max in IVideoDriver
	if (numberOfSamples>64)
        numberOfSamples = 64;
    if (numberOfSamples<=1)
        numberOfSamples = 2;

    if (core::isNPoT(numberOfSamples))
    {
        printf("\nError, number of samples must be a power-of-two!\n");
        return 2;
    }

    //You may find while experimenting that you can only create a texture with 8 samples but renderbuffer with 32 !
    printf("\nUsing %d samples.\n",numberOfSamples);

	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check nbl::SIrrlichtCreationParameters
	nbl::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = false;
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.


	video::IVideoDriver* driver = device->getVideoDriver();

    SimpleCallBack* cb = new SimpleCallBack();
    video::E_MATERIAL_TYPE newMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
                                                        "","","", //! No Geometry or Tessellation Shaders
                                                        "../mesh.frag",
                                                        3,video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
                                                        cb); //! Our Shader Callback
    cb->drop();



	scene::ISceneManager* smgr = device->getSceneManager();
	scene::ICameraSceneNode* camera =
		smgr->addCameraSceneNodeFPS(0,100.0f,0.01f);
	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(250.0f);
    smgr->setActiveCamera(camera);


	device->getCursorControl()->setVisible(false);
	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);


#define kInstanceSquareSize 10
	scene::ISceneNode* instancesToRemove[kInstanceSquareSize*kInstanceSquareSize] = { 0 };

    asset::IAssetLoader::SAssetLoadParams lparams;
	auto cpumesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(*device->getAssetManager()->getAsset("../../media/dwarf.baw", lparams).getContents().first);
	if (cpumesh&&cpumesh->getMeshType() == asset::EMT_ANIMATED_SKINNED)
	{
		scene::ISkinnedMeshSceneNode* anode = 0;
		auto gpumesh = driver->getGPUObjectsFromAssets(&cpumesh.get(), (&cpumesh.get())+1)->front();

		auto setMaterialTypeAndBufferViewOnAllMaterials = [](auto* mesh, auto newMaterialType, auto* tbo)
		{
			for (auto i = 0u; i < mesh->getMeshBufferCount(); i++)
			{
				auto& material = mesh->getMeshBuffer(i)->getMaterial();
				material.MaterialType = newMaterialType;
				material.setTexture(3u, core::smart_refctd_ptr<video::ITextureBufferObject>(tbo));
			}
		};

		for (size_t x = 0; x<kInstanceSquareSize; x++)
			for (size_t z = 0; z<kInstanceSquareSize; z++)
			{
				instancesToRemove[x + kInstanceSquareSize*z] = anode = smgr->addSkinnedMeshSceneNode(core::smart_refctd_ptr_static_cast<video::IGPUSkinnedMesh>(gpumesh));
				anode->setScale(core::vector3df(0.05f));
				anode->setPosition(core::vector3df(x, 0.f, z)*4.f);
				anode->setAnimationSpeed(18.f*float(x + 1 + (z + 1)*kInstanceSquareSize) / float(kInstanceSquareSize*kInstanceSquareSize));
				setMaterialTypeAndBufferViewOnAllMaterials(anode->getMesh(), newMaterialType, anode->getBonePoseTBO());
			}

		gpumesh->drop();
	}

    auto fsTriMeshBuffer = ext::FullScreenTriangle::createFullScreenTriangle(driver);
    //! We use a renderbuffer because we don't intend on reading from it
    video::IMultisampleTexture* colorMT=NULL,* depthMT=NULL;
    video::SGPUMaterial postProcMaterial;
    video::IFrameBuffer* framebuffer = driver->addFrameBuffer();
    {
        colorMT = driver->addMultisampleTexture(video::IMultisampleTexture::EMTT_2D,numberOfSamples,&params.WindowSize.Width,asset::EF_B8G8R8A8_UNORM);
        depthMT = driver->addMultisampleTexture(video::IMultisampleTexture::EMTT_2D,numberOfSamples,&params.WindowSize.Width,asset::EF_D32_SFLOAT);
        framebuffer->attach(video::EFAP_COLOR_ATTACHMENT0,colorMT);
        framebuffer->attach(video::EFAP_DEPTH_ATTACHMENT,depthMT);

        PostProcCallBack* callBack = new PostProcCallBack();
        //! First need to make a material other than default to be able to draw with custom shader
        postProcMaterial.BackfaceCulling = false; //! Triangles will be visible from both sides
        postProcMaterial.ZBuffer = video::ECFN_ALWAYS; //! Ignore Depth Test
        postProcMaterial.ZWriteEnable = false; //! Why even write depth?
        postProcMaterial.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../fullscreentri.vert",
                                                                            "","","", //! No Geometry or Tessellation Shaders
                                                                            "../postproc.frag",
                                                                            3,video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only)
                                                                            callBack,
                                                                            NULL,0, //! Xform feedback stuff, irrelevant here
                                                                            numberOfSamples); //! custom user data
        //! Need to bind our Multisample Textures to the correct texture units upon draw
        postProcMaterial.setTexture(0,core::smart_refctd_ptr<video::IMultisampleTexture>(colorMT));
        postProcMaterial.setTexture(1,core::smart_refctd_ptr<video::IMultisampleTexture>(depthMT));
        callBack->drop();
    }


	uint64_t lastFPSTime = 0;

	while(device->run() && receiver.keepOpen())
	{
		driver->beginScene(false, false);

		driver->setRenderTarget(framebuffer);
		vectorSIMDf clearColor(1.f,1.f,1.f,1.f);
        driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0,clearColor.pointer);
		driver->clearZBuffer();
        //! This animates (moves) the camera and sets the transforms
        //! Also draws the meshbuffer

        //yeah we dont have the state tracker yet
        glEnable(GL_MULTISAMPLE);
        smgr->drawAll();
        glDisable(GL_MULTISAMPLE);

        /**
        We could use blit, since works on FBOs it would work here to as a resolve.

        But instead we show off programmable resolve with a shader.
        **/
        if (false)
        {
            //notice how I dont even have to set the current FBO (render target) to 0 (the screen) for results to display
            const bool needToCopyDepth = false;
            driver->blitRenderTargets(framebuffer,0,needToCopyDepth);
        }
        else
        {
            driver->setRenderTarget(0);
            driver->setMaterial(postProcMaterial);
            driver->drawMeshBuffer(fsTriMeshBuffer.get());
        }

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Builtin Nodes Demo - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str());
			lastFPSTime = time;
		}
	}

	driver->removeFrameBuffer(framebuffer);
    driver->removeMultisampleTexture(colorMT);
    driver->removeMultisampleTexture(depthMT);

    for (size_t x=0; x<kInstanceSquareSize; x++)
    for (size_t z=0; z<kInstanceSquareSize; z++)
        instancesToRemove[x+kInstanceSquareSize*z]->remove();

	//create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		ext::ScreenShot::dirtyCPUStallingScreenshot(device, "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}

	device->drop();

	return 0;
}
