// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "nbl/ext/ScreenShot/ScreenShot.h"

#include "../common/QToQuitEventReceiver.h"


#include "createComputeShader.h"
#include "../source/Irrlicht/COpenGL2DTexture.h"
#include "../source/Irrlicht/COpenGLDriver.h"

using namespace nbl;
using namespace core;

core::vector3df absoluteLightPos;
core::matrix4SIMD ViewProjCubeMatrices[6];

class SimpleCallBack : public video::IShaderConstantSetCallBack
{
    int32_t worldspaceLightPosUniformLocation;
    int32_t worldMatUniformLocation;
    int32_t normalMatUniformLocation;
    int32_t mvpUniformLocation;
    int32_t vpcmUniformLocation;
    video::E_SHADER_CONSTANT_TYPE worldspaceLightPosUniformType;
    video::E_SHADER_CONSTANT_TYPE worldMatUniformType;
    video::E_SHADER_CONSTANT_TYPE normalMatUniformType;
    video::E_SHADER_CONSTANT_TYPE mvpUniformType;
    video::E_SHADER_CONSTANT_TYPE vpcmUniformType;
public:
    SimpleCallBack() : worldspaceLightPosUniformLocation(-1), worldMatUniformLocation(-1), normalMatUniformLocation(-1), mvpUniformLocation(-1), vpcmUniformLocation(-1) {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
    {
        int32_t id[] = {0,1,2,3};
        for (size_t i=0; i<constants.size(); i++)
        {
            if (constants[i].name=="worldSpaceLightPos")
            {
                worldspaceLightPosUniformLocation = constants[i].location;
                worldspaceLightPosUniformType = constants[i].type;
            }
            else if (constants[i].name=="worldMat")
            {
                worldMatUniformLocation = constants[i].location;
                worldMatUniformType = constants[i].type;
            }
            else if (constants[i].name=="normalMat")
            {
                normalMatUniformLocation = constants[i].location;
                normalMatUniformType = constants[i].type;
            }
            else if (constants[i].name=="MVP")
            {
                mvpUniformLocation = constants[i].location;
                mvpUniformType = constants[i].type;
            }
            else if (constants[i].name=="ViewProjCubeMatrices"||constants[i].name=="ViewProjCubeMatrices[0]") //nvidia intel and amd report names differently
            {
                vpcmUniformLocation = constants[i].location;
                vpcmUniformType = constants[i].type;
            }
        }
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        if (worldspaceLightPosUniformLocation!=-1)
        {
            services->setShaderConstant(&absoluteLightPos.X,worldspaceLightPosUniformLocation,worldspaceLightPosUniformType,1);
        }
        if (worldMatUniformLocation!=-1)
            services->setShaderConstant(services->getVideoDriver()->getTransform(video::E4X3TS_WORLD).pointer(),worldMatUniformLocation,worldMatUniformType,1);
        if (normalMatUniformLocation!=-1)
        {
            float worldSpaceNormalMatrix[9]; //no view space like gl_NormalMatrix or E4X3TS_NORMAL_MATRIX
            services->getVideoDriver()->getTransform(video::E4X3TS_WORLD).getSub3x3InverseTranspose(worldSpaceNormalMatrix);
            services->setShaderConstant(worldSpaceNormalMatrix,normalMatUniformLocation,normalMatUniformType,1);
        }
        if (mvpUniformLocation!=-1)
            services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(),mvpUniformLocation,mvpUniformType,1);
        if (vpcmUniformLocation!=-1)
        {
            core::matrix4SIMD ModelViewProjCubeMatrices[6];
            for (size_t i=0; i<6; i++)
                ModelViewProjCubeMatrices[i] = core::concatenateBFollowedByA(ViewProjCubeMatrices[i],services->getVideoDriver()->getTransform(video::E4X3TS_WORLD));
            services->setShaderConstant(ModelViewProjCubeMatrices,vpcmUniformLocation,vpcmUniformType,6);
        }
    }

    virtual void OnUnsetMaterial() {}
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
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.


	video::IVideoDriver* driver = device->getVideoDriver();

    SimpleCallBack* cb = new SimpleCallBack();
    video::E_MATERIAL_TYPE skinnedMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../skinnedMesh.vert",
                                                        "","","", //! No Geometry or Tessellation Shaders
                                                        "../mesh.frag",
                                                        3,video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
                                                        cb); //! Our Shader Callback
    //! we could use the same shader callback for many materials, but then have to keep track of uniforms separately!
    cb->drop();
    cb = new SimpleCallBack();
    //! Oh the stuff we could do with this shader, output transform feedback for main-view drawing and saving the GPU skinning results.
    video::E_MATERIAL_TYPE skinnedShadowMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../skinnedMeshShadow.vert",
                                                        "","", //! Tessellation Shaders
                                                        "../cubeMapLayerDispatch.geom", //! Geometry Shader to amplify geometry and set gl_Layer
                                                        "../shadow.frag",
                                                        3,video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
                                                        cb); //! Our Shader Callback
    //! we could use the same shader callback for many materials, but then have to keep track of uniforms separately!
    cb->drop();
    cb = new SimpleCallBack();
    video::E_MATERIAL_TYPE litSolidMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
                                                        "","","", //! No Geometry or Tessellation Shaders
                                                        "../mesh.frag",
                                                        3,video::EMT_SOLID,
                                                        cb);
    //! we could use the same shader callback for many materials, but then have to keep track of uniforms separately!
    cb->drop();
    cb = new SimpleCallBack();
    video::E_MATERIAL_TYPE shadowMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../meshShadow.vert",
                                                        "","", //! Tessellation Shaders
                                                        "../cubeMapLayerDispatch.geom", //! Geometry Shader to amplify geometry and set gl_Layer
                                                        "../shadow.frag",
                                                        3,video::EMT_SOLID,
                                                        cb);
    cb->drop();

    #define kInstanceSquareSize 10
	scene::ISceneManager* smgr = device->getSceneManager();


    //! Create our dummy scene-node signfying the light and lets get the view and projection matrices!
    scene::IDummyTransformationSceneNode* dummyLightNode = smgr->addDummyTransformationSceneNode();
    dummyLightNode->setPosition(core::vector3df(2.f,0.5f,2.f)*kInstanceSquareSize);
    //scene::ISceneNodeAnimator* anim = smgr->createFlyCircleAnimator(dummyLightNode->getPosition(),10.f);
    //dummyLightNode->addAnimator(anim);
    //anim->drop();

    // could fish this proj matrix from the envMapCam, but I know better and that all of them would be equal
    // set near value to be as far as possible to increase our precision in Z-Buffer (definitely want it to be same size as the light-bulb)
    // set far value to be the range of the light (or farthest shadow caster away from the light)
    // aspect ratio and FOV must be 1 and 90 degrees to render a cube face
    core::matrix4SIMD ProjMatrix = ProjMatrix.buildProjectionMatrixPerspectiveFovRH(core::PI*0.5f,1.f,0.1f,250.f);
    ProjMatrix(0,0) = 1.f;
    ProjMatrix(1,1) = 1.f;
    core::matrix4x3 ViewMatricesWithoutTranslation[6];
    for (size_t i=0; i<6; i++)
    {
        //preconfig stuff for camera orientations
        core::vector3df lookat[6] = {core::vector3df( 1, 0, 0),core::vector3df(-1, 0, 0),core::vector3df( 0, 1, 0),core::vector3df( 0,-1, 0),core::vector3df( 0, 0, 1),core::vector3df( 0, 0,-1)};
        core::vector3df up[6] = {core::vector3df( 0, 1, 0),core::vector3df( 0, 1, 0),core::vector3df( 0, 0, -1),core::vector3df( 0, 0, 1),core::vector3df( 0, 1, 0),core::vector3df( 0, 1, 0)};

        ViewMatricesWithoutTranslation[i].buildCameraLookAtMatrixLH(core::vector3df(),lookat[i],up[i]);
    }


    #define kCubeMapSize 2048
    uint32_t size[3] = {kCubeMapSize,kCubeMapSize,6};
    auto cubeMap = driver->createGPUTexture(video::ITexture::ETT_CUBE_MAP,size,1u,asset::EF_D32_SFLOAT); //dat ZBuffer Precision, may be excessive
    //notice this FBO only has a depth attachment, no colour!
    video::IFrameBuffer* fbo = driver->addFrameBuffer();
    fbo->attach(video::EFAP_DEPTH_ATTACHMENT,cubeMap.get(),0); //attach all 6 faces at once
    //! REMEMBER THIS IS NOT THE END OF THE OPTIMIZATIONS, WE COULD ALWAYS USE TRANSFORM FEEDBACK TO SAVE GPU SKINNING AND NOT HAVE TO DO IT AGAIN (100% FASTER RENDER on second pass)

    auto* assetMgr = device->getAssetManager();
    asset::IAssetLoader::SAssetLoadParams lparams;

    uint32_t derivMap_sz[3]{ 512u, 512u, 1u };
    auto derivMap = driver->createGPUTexture(video::ITexture::ETT_2D, derivMap_sz, 5u, asset::EF_R8G8_SNORM);

	auto bumpMap_asset = core::smart_refctd_ptr_static_cast<asset::ICPUTexture>(*assetMgr->getAsset("../../media/bumpmap.jpg", lparams).getContents().first);
    auto bumpMap = driver->getGPUObjectsFromAssets(&bumpMap_asset.get(), (&bumpMap_asset.get())+1)->front();

    {
        video::STextureSamplingParams params;
        params.UseMipmaps = 0;
        params.MaxFilter = params.MinFilter = video::ETFT_LINEAR_NO_MIP;
        params.TextureWrapU = params.TextureWrapV = video::ETC_CLAMP_TO_EDGE;
        const_cast<video::COpenGLDriver::SAuxContext*>(reinterpret_cast<video::COpenGLDriver*>(driver)->getThreadContext())->setActiveTexture(7, bumpMap, params);
    }

    GLuint deriv_map_gen_cs = createComputeShaderFromFile("../deriv_map_gen.comp");

    GLint previousProgram;
    glGetIntegerv(GL_CURRENT_PROGRAM, &previousProgram);

    for (GLuint i = 0u; i < 5u; ++i)
        video::COpenGLExtensionHandler::extGlBindImageTexture(i, static_cast<const video::COpenGL2DTexture*>(derivMap.get())->getOpenGLName(),
            i, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG8_SNORM);

    video::COpenGLExtensionHandler::extGlUseProgram(deriv_map_gen_cs);
    video::COpenGLExtensionHandler::extGlDispatchCompute(derivMap_sz[0]/16u, derivMap_sz[1]/16u, 1u);
    video::COpenGLExtensionHandler::extGlMemoryBarrier(
        GL_TEXTURE_FETCH_BARRIER_BIT |
        GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
        GL_PIXEL_BUFFER_BARRIER_BIT |
        GL_TEXTURE_UPDATE_BARRIER_BIT |
        GL_FRAMEBUFFER_BARRIER_BIT
    );
    video::COpenGLExtensionHandler::extGlDeleteProgram(deriv_map_gen_cs);
    for (GLuint i = 0u; i < 5u; ++i)
        video::COpenGLExtensionHandler::extGlBindImageTexture(i, 0u, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R8); //unbind image
    { //unbind texture
        video::STextureSamplingParams params;
        const_cast<video::COpenGLDriver::SAuxContext*>(reinterpret_cast<video::COpenGLDriver*>(driver)->getThreadContext())->setActiveTexture(7, nullptr, params);
    }
    video::COpenGLExtensionHandler::extGlUseProgram(previousProgram); //rebind previously bound program

    //derivMap->regenerateMipMapLevels();

    auto wallTexture = core::smart_refctd_ptr_static_cast<asset::ICPUTexture>(*assetMgr->getAsset("../../media/wall.jpg", lparams).getContents().first);

	scene::ICameraSceneNode* camera =
		smgr->addCameraSceneNodeFPS(0,180.0f,0.01f);
	camera->setPosition(core::vector3df(-4,10,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(250.0f);
    smgr->setActiveCamera(camera);


	device->getCursorControl()->setVisible(false);
	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);


	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
	//add a floor
	auto* floor = smgr->addCubeSceneNode(kInstanceSquareSize*20.f,0,-1,core::vector3df(0,-0.75f,0),core::vector3df(0,0,0),core::vector3df(1.f,1.f/(kInstanceSquareSize*20.f),1.f));
	video::SGPUMaterial& floorMaterial = floor->getMesh()->getMeshBuffer(0)->getMaterial();
	floorMaterial.setTexture(0,core::smart_refctd_ptr(driver->getGPUObjectsFromAssets(&wallTexture.get(), (&wallTexture.get())+1)->front()));
	floorMaterial.setTexture(1,core::smart_refctd_ptr(cubeMap));
    floorMaterial.setTexture(4,core::smart_refctd_ptr(derivMap));
	floorMaterial.MaterialType = litSolidMaterialType;

	scene::ISkinnedMeshSceneNode* anodes[kInstanceSquareSize*kInstanceSquareSize] = {0};

	//! For Shadow Optimization
	scene::ISkinnedMeshSceneNode* fastestNode = NULL;
	//
	auto cpumesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(*assetMgr->getAsset("../../media/dwarf.baw", lparams).getContents().first);

	if (cpumesh&&cpumesh->getMeshType() == asset::EMT_ANIMATED_SKINNED)
	{
		scene::ISkinnedMeshSceneNode* anode = 0;
		auto gpumesh = driver->getGPUObjectsFromAssets(&cpumesh.get(), (&cpumesh.get())+1)->front();

		for (size_t x = 0; x<kInstanceSquareSize; x++)
			for (size_t z = 0; z<kInstanceSquareSize; z++)
			{
				anodes[x + kInstanceSquareSize*z] = anode = smgr->addSkinnedMeshSceneNode(core::smart_refctd_ptr_static_cast<video::IGPUSkinnedMesh>(gpumesh));
				anode->setScale(core::vector3df(0.05f));
				anode->setPosition(core::vector3df(x, 0.f, z)*4.f);
				anode->setAnimationSpeed(18.f*float(x + 1 + (z + 1)*kInstanceSquareSize) / float(kInstanceSquareSize*kInstanceSquareSize));

				auto setMaterialTypeAndDescriptorsOnAllMaterials = [=](auto* mesh)
				{
					for (auto i = 0u; i < mesh->getMeshBufferCount(); i++)
					{
						auto& material = mesh->getMeshBuffer(i)->getMaterial();
						material.MaterialType = skinnedMaterialType;
						material.setTexture(1, core::smart_refctd_ptr(cubeMap));
						material.setTexture(3, core::smart_refctd_ptr<video::ITextureBufferObject>(anode->getBonePoseTBO()));
						material.setTexture(4, core::smart_refctd_ptr(derivMap));
					}
				};
				setMaterialTypeAndDescriptorsOnAllMaterials(anode->getMesh());
			}
        fastestNode = anode;
		gpumesh->drop();
	}

	//! Material setting lambda
	auto setMaterialTypeOnAllMeshBuffers = [](auto* node, auto type)
	{
		auto* mesh = node->getMesh();
		for (auto i=0u; i<mesh->getMeshBufferCount(); i++)
			mesh->getMeshBuffer(i)->getMaterial().MaterialType = type;
	};

	uint64_t lastFPSTime = 0;
	float lastFastestMeshFrameNr = -1.f;

	while(device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255,255,255,255) );

		//! Animate first
		smgr->getRootSceneNode()->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());

		// without this optimization FPS is 400 instead of 1000 FPS
		if (fastestNode->getFrameNr()!=lastFastestMeshFrameNr)
        {
            lastFastestMeshFrameNr = fastestNode->getFrameNr();
            //! its a bit stupid that I update light position only when animations update
            //! but in the internals of the engine animations only update every 120Hz (we can set this individually per mesh)
            //! so I'm just syncing everything up to the fastest mesh
            absoluteLightPos = dummyLightNode->getAbsolutePosition();

            //! draw shadows
            smgr->setActiveCamera(NULL);

			setMaterialTypeOnAllMeshBuffers(floor,shadowMaterialType);
            for (size_t x=0; x<kInstanceSquareSize; x++)
            for (size_t z=0; z<kInstanceSquareSize; z++)
				setMaterialTypeOnAllMeshBuffers(anodes[x+kInstanceSquareSize*z],skinnedShadowMaterialType);

            driver->setRenderTarget(fbo,true);
            driver->clearZBuffer(0.f);
            for (size_t i=0; i<6; i++)
            {
                matrix4x3 viewMatModified(ViewMatricesWithoutTranslation[i]);

                //put the 'camera' position in
                ViewMatricesWithoutTranslation[i].mulSub3x3With3x1(&viewMatModified.getColumn(3).X,&absoluteLightPos.X);
                viewMatModified.getColumn(3) *= -1.f;

                //
                ViewProjCubeMatrices[i] = core::concatenateBFollowedByA(ProjMatrix,viewMatModified);
            }
            smgr->drawAll();

			setMaterialTypeOnAllMeshBuffers(floor,litSolidMaterialType);
            for (size_t x=0; x<kInstanceSquareSize; x++)
            for (size_t z=0; z<kInstanceSquareSize; z++)
				setMaterialTypeOnAllMeshBuffers(anodes[x+kInstanceSquareSize*z],skinnedMaterialType);

            driver->setRenderTarget(0,true);

            smgr->setActiveCamera(camera);
        }

        //! Draw the view
        smgr->drawAll();

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

	//create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		ext::ScreenShot::dirtyCPUStallingScreenshot(device, "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}


    for (size_t x=0; x<kInstanceSquareSize; x++)
    for (size_t z=0; z<kInstanceSquareSize; z++)
        anodes[x+kInstanceSquareSize*z]->remove();

	device->drop();

	return 0;
}
