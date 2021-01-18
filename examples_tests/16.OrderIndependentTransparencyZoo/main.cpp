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
    int32_t selfPosLocation;
    int32_t texUniformLocation[4];
    video::E_SHADER_CONSTANT_TYPE mvpUniformType;
    video::E_SHADER_CONSTANT_TYPE selfPosType;
    video::E_SHADER_CONSTANT_TYPE texUniformType[4];
public:
    SimpleCallBack() : selfPosLocation(-1), selfPosType(video::ESCT_FLOAT_VEC3) {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
    {
        for (size_t i=0; i<constants.size(); i++)
        {
            if (constants[i].name=="MVP")
            {
                mvpUniformLocation = constants[i].location;
                mvpUniformType = constants[i].type;
            }
            else if (constants[i].name=="selfPos")
            {
                selfPosLocation = constants[i].location;
                selfPosType = constants[i].type;
            }
        }
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        core::vectorSIMDf selfPos = services->getVideoDriver()->getTransform(video::E4X3TS_WORLD).getTranslation3D();
        if (selfPosLocation!=-1)
            services->setShaderConstant(selfPos.pointer,selfPosLocation,selfPosType,1);
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
    PostProcCallBack() : sampleCountUniformLocation(-1) {}

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
	printf("\nChoose the Transparency Algorithm:\n");
	printf(" (0 : default) None\n");
	printf(" (1) Z-Sorted\n"); //would benefit from transmittance thresholding
	printf(" (2) Stencil Routed Original\n"); //records k-first fragments (needs sorting for depth complexity>k)
	printf(" (3) Stencil Routed A-la DevSH\n"); //records k nearest fragments from k disjoint sets
	printf(" (4) Stencil Routed Min-Transmission\n"); //records k most opaque fragments from k disjoint sets (by putting the alpha value into the Z-buffer)
	printf(" (5) A-Buffer\n");
	printf(" (6) Stochastic\n");
	printf(" (7) Adaptive OIT\n");
	printf(" (8) Moment Transparency\n");
	/** TODO
	+ A-Buffer http://www.icare3d.org/codes-and-projects/codes/opengl-4-0-abuffer-v2-0-linked-lists-of-fragment-pages.html
	+ Linked List
	+ Offset List
	+ Atomic Loop 64
	+ Intel Method
	+ AMD DX11 Method
	+ k+ Buffer
	**/

	uint32_t method=3;
	std::cin >> method;
    printf("\nUsing method %d.\n",method);

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

	scene::ISceneManager* smgr = device->getSceneManager();
	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
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
	core::matrix3x4SIMD instancePositions[kInstanceSquareSize*kInstanceSquareSize];

    asset::IAssetLoader::SAssetLoadParams lparams;
	auto cpumesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(*device->getAssetManager()->getAsset("../../media/dwarf.baw", lparams).getContents().first);
    core::smart_refctd_ptr<video::IGPUMesh> gpumesh;
    if (cpumesh)
    {
        gpumesh = driver->getGPUObjectsFromAssets(&cpumesh.get(), (&cpumesh.get()) + 1)->front();

        for (size_t z=0; z<kInstanceSquareSize; z++)
        for (size_t x=0; x<kInstanceSquareSize; x++)
        {
            auto& matrix = instancePositions[x+kInstanceSquareSize*z];
            matrix.setScale(core::vectorSIMDf(0.05f,0.05f,0.05f));
            matrix.setTranslation(core::vectorSIMDf(x,0.f,z)*4.f);
        }
    }

    //! Set up screen triangle for post-processing
    auto screenTriangleMeshBuffer = ext::FullScreenTriangle::createFullScreenTriangle(driver);


    //! Must be Power Of Two!
    const uint32_t transparencyLayers = 0x1u<<3;


    video::IFrameBuffer* framebuffer = driver->addFrameBuffer();
    video::IMultisampleTexture* colorMT=NULL,* depthMT=NULL;
    video::SGPUMaterial initMaterial,resolveMaterial;
    switch (method)
    {
        case 0:
        case 1:
            {
                SimpleCallBack* cb = new SimpleCallBack();
                video::E_MATERIAL_TYPE newMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
                                                                    "","","", //! No Geometry or Tessellation Shaders
                                                                    "../mesh.frag",
                                                                    3,nbl::video::EMT_TRANSPARENT_ALPHA_CHANNEL,
                                                                    cb, //! Our Shader Callback
                                                                    0); //! No custom user data
                cb->drop();
                for (size_t i=0; i<gpumesh->getMeshBufferCount(); i++)
                {
                    video::SGPUMaterial& mat = gpumesh->getMeshBuffer(i)->getMaterial();
                    mat.BlendOperation = video::EBO_ADD;
                    mat.ZWriteEnable = false;
                    mat.BackfaceCulling = false;
                    mat.MaterialType = (video::E_MATERIAL_TYPE)newMaterialType;
                }
            }
            break;
        case 2:
        case 3:
            {
                SimpleCallBack* cb = new SimpleCallBack();
                video::E_MATERIAL_TYPE newMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
                                                                    "","","", //! No Geometry or Tessellation Shaders
                                                                    "../mesh.frag",
                                                                    3,nbl::video::EMT_SOLID,
                                                                    cb, //! Our Shader Callback
                                                                    0); //! No custom user data
                cb->drop();
                for (size_t i=0; i<gpumesh->getMeshBufferCount(); i++)
                {
                    video::SGPUMaterial& mat = gpumesh->getMeshBuffer(i)->getMaterial();
                    if (method==2)
                    {
                        mat.ZBuffer = video::ECFN_ALWAYS;
                        mat.ZWriteEnable = true; //original bavoil has this off
                    }
                    mat.BackfaceCulling = false;
                    mat.MaterialType = (video::E_MATERIAL_TYPE)newMaterialType;
                }

                {
                    const uint32_t numberOfSamples = transparencyLayers;
                    colorMT = driver->addMultisampleTexture(video::IMultisampleTexture::EMTT_2D,numberOfSamples,&params.WindowSize.Width,asset::EF_B8G8R8A8_UNORM);
                    depthMT = driver->addMultisampleTexture(video::IMultisampleTexture::EMTT_2D,numberOfSamples,&params.WindowSize.Width,asset::EF_D32_SFLOAT_S8_UINT);
                    framebuffer->attach(video::EFAP_COLOR_ATTACHMENT0,colorMT);
                    framebuffer->attach(video::EFAP_DEPTH_STENCIL_ATTACHMENT,depthMT);


                    PostProcCallBack* callBack = new PostProcCallBack();


                    initMaterial.BackfaceCulling = false; //! Triangles will be visible from both sides
                    initMaterial.ZBuffer = video::ECFN_ALWAYS; //! Ignore Depth Test
                    initMaterial.ZWriteEnable = false; //! Why even write depth?
                    initMaterial.ColorMask = video::ECP_NONE; //! Why even write depth?
                    initMaterial.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../fullscreentri.vert",
                                                                                        "","","", //! No Geometry or Tessellation Shaders
                                                                                        "../stencilKClear.frag",
                                                                                        3,video::EMT_SOLID);

                    resolveMaterial.BackfaceCulling = false; //! Triangles will be visible from both sides
                    resolveMaterial.ZBuffer = video::ECFN_ALWAYS; //! Ignore Depth Test
                    resolveMaterial.ZWriteEnable = false; //! Why even write depth?
                    resolveMaterial.BlendOperation = video::EBO_ADD;
                    resolveMaterial.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../fullscreentri.vert",
                                                                                        "","","", //! No Geometry or Tessellation Shaders
                                                                                        "../stencilKResolve.frag",
                                                                                        3,video::EMT_TRANSPARENT_ALPHA_CHANNEL, //! 3 vertices per primitive (this is tessellation shader relevant only)
                                                                                        callBack,
                                                                                        NULL,0, //! Xform feedback stuff, irrelevant here
                                                                                        numberOfSamples); //! custom user data
                    //! Need to bind our Multisample Textures to the correct texture units upon draw
                    resolveMaterial.setTexture(0,core::smart_refctd_ptr<video::IMultisampleTexture>(colorMT,dont_grab));
                    resolveMaterial.setTexture(1,core::smart_refctd_ptr<video::IMultisampleTexture>(depthMT,dont_grab));


                    callBack->drop();
                }
            }
            break;
        case 4:
            {
                SimpleCallBack* cb = new SimpleCallBack();
                video::E_MATERIAL_TYPE newMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
                                                                    "","","", //! No Geometry or Tessellation Shaders
                                                                    "../mesh_minTrans.frag",
                                                                    3,nbl::video::EMT_SOLID,
                                                                    cb, //! Our Shader Callback
                                                                    0); //! No custom user data
                cb->drop();
                for (size_t i=0; i<gpumesh->getMeshBufferCount(); i++)
                {
                    video::SGPUMaterial& mat = gpumesh->getMeshBuffer(i)->getMaterial();
                    mat.BackfaceCulling = false;
                    mat.ZBuffer = video::ECFN_GREATER;
                    mat.MaterialType = (video::E_MATERIAL_TYPE)newMaterialType;
                }

                {
                    const uint32_t numberOfSamples = transparencyLayers;
                    colorMT = driver->addMultisampleTexture(video::IMultisampleTexture::EMTT_2D,numberOfSamples,&params.WindowSize.Width,asset::EF_B8G8R8A8_UNORM);
                    depthMT = driver->addMultisampleTexture(video::IMultisampleTexture::EMTT_2D,numberOfSamples,&params.WindowSize.Width,asset::EF_D32_SFLOAT_S8_UINT);
                    framebuffer->attach(video::EFAP_COLOR_ATTACHMENT0,colorMT);
                    framebuffer->attach(video::EFAP_DEPTH_STENCIL_ATTACHMENT,depthMT);


                    PostProcCallBack* callBack = new PostProcCallBack();


                    initMaterial.BackfaceCulling = false; //! Triangles will be visible from both sides
                    initMaterial.ZBuffer = video::ECFN_ALWAYS; //! Ignore Depth Test
                    initMaterial.ZWriteEnable = false; //! Why even write depth?
                    initMaterial.ColorMask = video::ECP_NONE; //! Why even write depth?
                    initMaterial.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../fullscreentri.vert",
                                                                                        "","","", //! No Geometry or Tessellation Shaders
                                                                                        "../stencilKClear.frag",
                                                                                        3,video::EMT_SOLID);

                    resolveMaterial.BackfaceCulling = false; //! Triangles will be visible from both sides
                    resolveMaterial.ZBuffer = video::ECFN_ALWAYS; //! Ignore Depth Test
                    resolveMaterial.ZWriteEnable = false; //! Why even write depth?
                    resolveMaterial.BlendOperation = video::EBO_ADD;
                    resolveMaterial.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../fullscreentri.vert",
                                                                                        "","","", //! No Geometry or Tessellation Shaders
                                                                                        "../minTransResolve.frag",
                                                                                        3,video::EMT_TRANSPARENT_ALPHA_CHANNEL, //! 3 vertices per primitive (this is tessellation shader relevant only)
                                                                                        callBack,
                                                                                        NULL,0, //! Xform feedback stuff, irrelevant here
                                                                                        numberOfSamples); //! custom user data
                    //! Need to bind our Multisample Textures to the correct texture units upon draw
					resolveMaterial.setTexture(0, core::smart_refctd_ptr<video::IMultisampleTexture>(colorMT, dont_grab));
					resolveMaterial.setTexture(1, core::smart_refctd_ptr<video::IMultisampleTexture>(depthMT, dont_grab));


                    callBack->drop();
                }
            }
            break;
        default:
            break;
    }


    uint64_t lastFPSTime = 0;

    while(device->run() && receiver.keepOpen())
    {
        driver->beginScene( false,false );

        //! This animates (moves) the camera and sets the transforms
        smgr->drawAll();

        switch (method)
        {
            case 0:
                {
                    vectorSIMDf clearColor(1.f,1.f,1.f,1.f);
                    driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0,clearColor.pointer);
                    driver->clearZBuffer();

                    for (size_t z=0; z<kInstanceSquareSize; z++)
                    for (size_t x=0; x<kInstanceSquareSize; x++)
                    {
                        driver->setTransform(video::E4X3TS_WORLD,instancePositions[x+kInstanceSquareSize*z]);
                        for (size_t i=0; i<gpumesh->getMeshBufferCount(); i++)
                        {
                            driver->setMaterial(gpumesh->getMeshBuffer(i)->getMaterial());
                            driver->drawMeshBuffer(gpumesh->getMeshBuffer(i));
                        }
                    }
                }
                break;
            case 1:
                {
                    vectorSIMDf clearColor(1.f,1.f,1.f,1.f);
                    driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0,clearColor.pointer);
                    driver->clearZBuffer();


                    std::pair<float,const core::matrix3x4SIMD*> distanceSortedInstances[kInstanceSquareSize*kInstanceSquareSize];
                    for (size_t z=0; z<kInstanceSquareSize; z++)
                    for (size_t x=0; x<kInstanceSquareSize; x++)
                    {
                        size_t offset = x+kInstanceSquareSize*z;
                        const auto* matrix = instancePositions+offset;
                        float dist = core::length(core::vectorSIMDf().set(camera->getAbsolutePosition())-matrix->getTranslation3D()).x;
                        distanceSortedInstances[offset] = std::pair<float,const core::matrix3x4SIMD*>(dist,matrix);
                    }
                    struct
                    {
                        bool operator()(const std::pair<float,const core::matrix3x4SIMD*>& a, const std::pair<float,const core::matrix3x4SIMD*>& b) const
                        {
                            return a.first > b.first;
                        }
                    } customLess;
                    std::stable_sort(distanceSortedInstances,distanceSortedInstances+kInstanceSquareSize*kInstanceSquareSize,customLess);


                    for (size_t ix=0; ix<kInstanceSquareSize*kInstanceSquareSize; ix++)
                    {
                        driver->setTransform(video::E4X3TS_WORLD,*distanceSortedInstances[ix].second);
                        for (size_t i=0; i<gpumesh->getMeshBufferCount(); i++)
                        {
                            driver->setMaterial(gpumesh->getMeshBuffer(i)->getMaterial());
                            driver->drawMeshBuffer(gpumesh->getMeshBuffer(i));
                        }
                    }
                }
                break;
            case 2:
                {
                    ///fix the shit between here
                    driver->setRenderTarget(framebuffer);
                    vectorSIMDf clearColor(0.f);
                    driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0,clearColor.pointer);
                    //to reset the depth mask, otherwise won't clear
                    glDepthMask(GL_TRUE);
                    driver->clearZStencilBuffers(0,2);

                    glEnable(GL_STENCIL_TEST);
                    glStencilOp(GL_KEEP,GL_KEEP,GL_REPLACE);

                    //init the stencil buffer
                    glEnable(GL_MULTISAMPLE);
                    glEnable(GL_SAMPLE_MASK);
                    for (uint32_t i=1; i<transparencyLayers; i++)
                    {
                        glStencilFunc(GL_ALWAYS,i+2,transparencyLayers-1);
                        video::COpenGLExtensionHandler::extGlSampleMaski(0,0x1u<<i);
                        driver->setMaterial(initMaterial);
                        driver->drawMeshBuffer(screenTriangleMeshBuffer.get());
                    }
                    video::COpenGLExtensionHandler::extGlSampleMaski(0,~0x0u);
                    glDisable(GL_SAMPLE_MASK);
                    glDisable(GL_MULTISAMPLE);

                    //draw our stuff
                    glStencilOp(GL_DECR,GL_DECR,GL_DECR);
                    glStencilFunc(GL_EQUAL,2,transparencyLayers-1);
                    for (size_t z=0; z<kInstanceSquareSize; z++)
                    for (size_t x=0; x<kInstanceSquareSize; x++)
                    {
                        driver->setTransform(video::E4X3TS_WORLD,instancePositions[x+kInstanceSquareSize*z]);
                        for (size_t i=0; i<gpumesh->getMeshBufferCount(); i++)
                        {
                            driver->setMaterial(gpumesh->getMeshBuffer(i)->getMaterial());
                            driver->drawMeshBuffer(gpumesh->getMeshBuffer(i));
                        }
                    }
                    glDisable(GL_STENCIL_TEST);

                    //! Resolve
                    {
                        driver->setRenderTarget(0);
                        clearColor = vectorSIMDf(1.f);
                        driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0,clearColor.pointer);

                        driver->setMaterial(resolveMaterial);
                        driver->drawMeshBuffer(screenTriangleMeshBuffer.get());
                    }
                    //to reset the depth mask
                    ///driver->setMaterial(video::SMaterial());
                }
                break;
            case 3:
            case 4: //only resolve differs
                {
                    ///fix the shit between here
                    driver->setRenderTarget(framebuffer);
                    vectorSIMDf clearColor(0.f);
                    driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0,clearColor.pointer);
                    //to reset the depth mask, otherwise won't clear
                    glDepthMask(GL_TRUE);
                    driver->clearZStencilBuffers(0,0);

                    glEnable(GL_STENCIL_TEST);
                    glStencilOp(GL_KEEP,GL_KEEP,GL_REPLACE);

                    //init the stencil buffer
                    glEnable(GL_MULTISAMPLE);
                    glEnable(GL_SAMPLE_MASK);
                    for (uint32_t i=1; i<transparencyLayers; i++)
                    {
                        glStencilFunc(GL_ALWAYS,i,transparencyLayers-1);
                        video::COpenGLExtensionHandler::extGlSampleMaski(0,0x1u<<i);
                        driver->setMaterial(initMaterial);
                        driver->drawMeshBuffer(screenTriangleMeshBuffer.get());
                    }
                    video::COpenGLExtensionHandler::extGlSampleMaski(0,~0x0u);
                    glDisable(GL_SAMPLE_MASK);
                    glDisable(GL_MULTISAMPLE);

                    //draw our stuff
                    glStencilOp(GL_DECR_WRAP,GL_DECR_WRAP,GL_DECR_WRAP);
                    glStencilFunc(GL_EQUAL,0,transparencyLayers-1);
                    for (size_t z=0; z<kInstanceSquareSize; z++)
                    for (size_t x=0; x<kInstanceSquareSize; x++)
                    {
                        driver->setTransform(video::E4X3TS_WORLD,instancePositions[x+kInstanceSquareSize*z]);
                        for (size_t i=0; i<gpumesh->getMeshBufferCount(); i++)
                        {
                            driver->setMaterial(gpumesh->getMeshBuffer(i)->getMaterial());
                            driver->drawMeshBuffer(gpumesh->getMeshBuffer(i));
                        }
                    }
                    glDisable(GL_STENCIL_TEST);

                    //! Resolve
                    {
                        driver->setRenderTarget(0);
                        clearColor = vectorSIMDf(1.f);
                        driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0,clearColor.pointer);

                        driver->setMaterial(resolveMaterial);
                        driver->drawMeshBuffer(screenTriangleMeshBuffer.get());
                    }
                    //to reset the depth mask
                    ///driver->setMaterial(video::SMaterial());
                }
                break;
            default:
                break;
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


    //! Cleanup
    driver->removeAllFrameBuffers();
    driver->removeAllMultisampleTextures();

	//create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		ext::ScreenShot::dirtyCPUStallingScreenshot(driver, device->getAssetManager(), "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}

	device->drop();

	return 0;
}
