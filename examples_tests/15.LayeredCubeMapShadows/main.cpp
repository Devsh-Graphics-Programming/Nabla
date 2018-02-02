#define _IRR_STATIC_LIB_
#include <irrlicht.h>


using namespace irr;
using namespace core;

bool quit = false;

//!Same As Last Example
class MyEventReceiver : public IEventReceiver
{
public:

	MyEventReceiver()
	{
	}

	bool OnEvent(const SEvent& event)
	{
        if (event.EventType == irr::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
        {
            switch (event.KeyInput.Key)
            {
            case irr::KEY_KEY_Q: // so we can quit
                quit = true;
                return true;
            default:
                break;
            }
        }

		return false;
	}

private:
};

scene::IDummyTransformationSceneNode* dummyLightNode = NULL;

class SimpleCallBack : public video::IShaderConstantSetCallBack
{
    int32_t worldspaceLightPosUniformLocation;
    int32_t worldMatUniformLocation;
    int32_t normalMatUniformLocation;
    int32_t mvpUniformLocation;
    video::E_SHADER_CONSTANT_TYPE worldspaceLightPosUniformType;
    video::E_SHADER_CONSTANT_TYPE worldMatUniformType;
    video::E_SHADER_CONSTANT_TYPE normalMatUniformType;
    video::E_SHADER_CONSTANT_TYPE mvpUniformType;
public:
    SimpleCallBack() : worldspaceLightPosUniformLocation(-1), worldMatUniformLocation(-1), normalMatUniformLocation(-1), mvpUniformLocation(-1) {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::array<video::SConstantLocationNamePair>& constants)
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
            } //! permabind texture slots
            else if (constants[i].name=="tex0")
                services->setShaderTextures(id+0,constants[i].location,constants[i].type,1);
            else if (constants[i].name=="tex1")
                services->setShaderTextures(id+1,constants[i].location,constants[i].type,1);
            else if (constants[i].name=="tex3")
                services->setShaderTextures(id+3,constants[i].location,constants[i].type,1);
        }
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        if (worldspaceLightPosUniformLocation!=-1 && dummyLightNode)
        {
            core::vector3df worldSpaceLightPos = dummyLightNode->getAbsolutePosition();
            services->setShaderConstant(&worldSpaceLightPos.X,worldspaceLightPosUniformLocation,worldspaceLightPosUniformType,1);
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
    }

    virtual void OnUnsetMaterial() {}
};


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
    video::E_MATERIAL_TYPE skinnedShadowMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../skinnedMesh.vert",
                                                        "","","", //! No Geometry or Tessellation Shaders
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
    video::E_MATERIAL_TYPE shadowMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
                                                        "","","", //! No Geometry or Tessellation Shaders
                                                        "../shadow.frag",
                                                        3,video::EMT_SOLID,
                                                        cb);
    cb->drop();


    #define kInstanceSquareSize 10
	scene::ISceneManager* smgr = device->getSceneManager();


    //! Create our dummy scene-node signfying the light and lets get the view and projection matrices!
    dummyLightNode = smgr->addDummyTransformationSceneNode();
    dummyLightNode->setPosition(core::vector3df(kInstanceSquareSize,5,kInstanceSquareSize)*2.f);

    //preconfig stuff for camera orientations
    core::vector3df lookat[6] = {core::vector3df( 1, 0, 0),core::vector3df(-1, 0, 0),core::vector3df( 0, 1, 0),core::vector3df( 0,-1, 0),core::vector3df( 0, 0, 1),core::vector3df( 0, 0,-1)};
    core::vector3df up[6] = {core::vector3df( 0, 1, 0),core::vector3df( 0, 1, 0),core::vector3df( 1, 0, 0),core::vector3df( 0, 0, 1),core::vector3df( 0, 1, 0),core::vector3df( 0, 1, 0)};
    // could fish this proj matrix from the envMapCam, but I know better and that all of them would be equal
    // set near value to be as far as possible to increase our precision in Z-Buffer (definitely want it to be same size as the light-bulb)
    // set far value to be the range of the light (or farthest shadow caster away from the light)
    // aspect ratio and FOV must be 1 and 90 degrees to render a cube face
    core::matrix4 ProjMatrix;
    ProjMatrix = ProjMatrix.buildProjectionMatrixPerspectiveFovLH(core::PI*0.5f,1.f,0.1f,250.f);
    ProjMatrix[0] = 1.f;
    ProjMatrix[5] = -1.f;
    core::matrix4x3 ViewMatricesWithoutTranslation[6];
    core::matrix4x3 ViewMatricesWithoutTranslationInv[6];
    scene::ICameraSceneNode* envMapCams[6];
    for (size_t i=0; i<6; i++)
    {
        scene::ICameraSceneNode* envMapCam = envMapCams[i] = smgr->addCameraSceneNode();
        envMapCam->setFOV(0.5f*core::PI);
        envMapCam->setAspectRatio(1.f);
        envMapCam->setNearValue(0.1f);
        envMapCam->setFarValue(250.f);
        envMapCam->setProjectionMatrix(ProjMatrix);
        envMapCam->setTarget(lookat[i]);
        envMapCam->setUpVector(up[i]);
        envMapCam->OnAnimate(0);
        envMapCam->render();
        ViewMatricesWithoutTranslation[i] = envMapCam->getViewMatrix();
        ViewMatricesWithoutTranslation[i].getInverse(ViewMatricesWithoutTranslationInv[i]); //need this to bring back translation
        //! COMING SOON with gl_Layer
        //envMapCam->remove();
        envMapCam->setParent(dummyLightNode);
    }
    //! COMING SOON with gl_Layer
    //scene::ICameraSceneNode* dummyCubeMapCam = NULL; //some sort of ortho camera to only have camera BBOX culling (without frustum)
    //dummyCubeMapCam->setParent(dummyLightNode);


    uint32_t size[3] = {2048,2048,6};
    video::ITexture* cubeMap = driver->addTexture(video::ITexture::ETT_CUBE_MAP,size,1,"shadowmap",video::ECF_DEPTH32F); //dat ZBuffer Precision, may be excessive
    //notice this FBO only has a depth attachment, no colour!
    //! COMING SOON with gl_Layer
    //video::IFrameBuffer* fbo = driver->addFrameBuffer();
    //fbo->attach(video::EFAP_DEPTH_ATTACHMENT,cubeMap,0); //attach all 6 faces at once
    video::IFrameBuffer* fbo[6];
    for (size_t i=0; i<6; i++)
    {
        fbo[i] = driver->addFrameBuffer();
        fbo[i]->attach(video::EFAP_DEPTH_ATTACHMENT,cubeMap,0,i);
    }


	scene::ICameraSceneNode* camera =
		smgr->addCameraSceneNodeFPS(0,100.0f,0.01f);
	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(250.0f);
    smgr->setActiveCamera(camera);
	device->getCursorControl()->setVisible(false);
	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);

	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
	//add a floor
	scene::ISceneNode* floor = smgr->addCubeSceneNode(200.f,0,-1,core::vector3df(0,-0.75f,0),core::vector3df(0,0,0),core::vector3df(1.f,1.f/200.f,1.f));
	video::SMaterial& floorMaterial = floor->getMaterial(0);
	floorMaterial.setTexture(0,driver->getTexture("../../media/wall.jpg"));
	floorMaterial.setTexture(1,cubeMap);
	floorMaterial.MaterialType = litSolidMaterialType;

	scene::ISceneNode* anodes[kInstanceSquareSize*kInstanceSquareSize] = {0};

	//! Test Loading of Obj
    scene::ICPUMesh* cpumesh = smgr->getMesh("../../media/dwarf.x");
    if (cpumesh&&cpumesh->getMeshType()==scene::EMT_ANIMATED_SKINNED)
    {
        scene::ISkinnedMeshSceneNode* anode = 0;
        scene::ICPUSkinnedMesh* animMesh = dynamic_cast<scene::ICPUSkinnedMesh*>(cpumesh);
        scene::IGPUMesh* gpumesh = driver->createGPUMeshFromCPU(cpumesh);
        smgr->getMeshCache()->removeMesh(cpumesh); //drops hierarchy

        for (size_t x=0; x<kInstanceSquareSize; x++)
        for (size_t z=0; z<kInstanceSquareSize; z++)
        {
            anodes[x+kInstanceSquareSize*z] = anode = smgr->addSkinnedMeshSceneNode(static_cast<scene::IGPUSkinnedMesh*>(gpumesh));
            anode->setScale(core::vector3df(0.05f));
            anode->setPosition((core::vector3df(x,0.f,z)+core::vector3df(0.5f,0.f,0.5f))*4.f);
            anode->setAnimationSpeed(18.f*float(x+1+(z+1)*kInstanceSquareSize)/float(kInstanceSquareSize*kInstanceSquareSize));
            anode->setMaterialType(skinnedMaterialType);
            anode->setMaterialFlag(video::EMF_BACK_FACE_CULLING,false);
            anode->setMaterialTexture(1,cubeMap);
            anode->setMaterialTexture(3,anode->getBonePoseTBO());
        }

        gpumesh->drop();
    }


	uint64_t lastFPSTime = 0;

	while(device->run()&&(!quit))
	//if (device->isWindowActive())
	{
		driver->beginScene(true, true, video::SColor(255,0,0,255) );

		//! draw shadows
		floor->setMaterialType(shadowMaterialType);
        for (size_t x=0; x<kInstanceSquareSize; x++)
        for (size_t z=0; z<kInstanceSquareSize; z++)
            anodes[x+kInstanceSquareSize*z]->setMaterialType(skinnedShadowMaterialType);

        for (size_t i=0; i<6; i++)
        {
            driver->setRenderTarget(fbo[i],i==0);
            driver->clearZBuffer();
            float vals[] = {1.f,0.f,0.f,1.f};
            driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0,vals);
            envMapCams[i]->setTarget(dummyLightNode->getAbsolutePosition()+lookat[i]);
            smgr->setActiveCamera(envMapCams[i]);
            smgr->drawAll();
        }
		//! COMING SOON with gl_Layer
		//driver->setRenderTarget(fbo,true);
		//driver->clearZBuffer();
		//dummyCubeMapCam->setTarget(dummyLightNode->getAbsolutePosition()+vector3df(0,-1,0));
        //smgr->setActiveCamera(dummyCubeMapCam);
		//smgr->drawAll();

		floor->setMaterialType(litSolidMaterialType);
        for (size_t x=0; x<kInstanceSquareSize; x++)
        for (size_t z=0; z<kInstanceSquareSize; z++)
            anodes[x+kInstanceSquareSize*z]->setMaterialType(skinnedMaterialType);

		driver->setRenderTarget(0,true);

        smgr->setActiveCamera(camera);
        //! This animates (moves) the camera and sets the transforms
        //! Also draws the meshbuffer
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

    for (size_t x=0; x<kInstanceSquareSize; x++)
    for (size_t z=0; z<kInstanceSquareSize; z++)
        anodes[x+kInstanceSquareSize*z]->remove();

	device->drop();

	return 0;
}
