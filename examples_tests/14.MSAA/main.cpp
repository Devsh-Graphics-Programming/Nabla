#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "COpenGLStateManager.h"

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
            case irr::KEY_KEY_Q: // switch wire frame mode
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

class SimpleCallBack : public video::IShaderConstantSetCallBack
{
    int32_t mvpUniformLocation;
    int32_t cameraDirUniformLocation;
    int32_t texUniformLocation[4];
    video::E_SHADER_CONSTANT_TYPE mvpUniformType;
    video::E_SHADER_CONSTANT_TYPE cameraDirUniformType;
    video::E_SHADER_CONSTANT_TYPE texUniformType[4];
public:
    SimpleCallBack() : cameraDirUniformLocation(-1), cameraDirUniformType(video::ESCT_FLOAT_VEC3) {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::array<video::SConstantLocationNamePair>& constants)
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
            else if (constants[i].name=="tex0")
            {
                texUniformLocation[0] = constants[i].location;
                texUniformType[0] = constants[i].type;
            }
            else if (constants[i].name=="tex3")
            {
                texUniformLocation[3] = constants[i].location;
                texUniformType[3] = constants[i].type;
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

        int32_t id[] = {0,1,2,3};
        if (texUniformLocation[0]!=-1)
            services->setShaderTextures(id+0,texUniformLocation[0],texUniformType[0],1);
        if (texUniformLocation[3]!=-1)
            services->setShaderTextures(id+3,texUniformLocation[3],texUniformType[3],1);
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
    video::E_MATERIAL_TYPE newMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
                                                        "","","", //! No Geometry or Tessellation Shaders
                                                        "../mesh.frag",
                                                        3,video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
                                                        cb, //! Our Shader Callback
                                                        0); //! No custom user data
    cb->drop();



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
	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);

        #define kInstanceSquareSize 10
	scene::ISceneNode* instancesToRemove[kInstanceSquareSize*kInstanceSquareSize] = {0};

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
            instancesToRemove[x+kInstanceSquareSize*z] = anode = smgr->addSkinnedMeshSceneNode(static_cast<scene::IGPUSkinnedMesh*>(gpumesh));
            anode->setScale(core::vector3df(0.05f));
            anode->setPosition(core::vector3df(x,0.f,z)*4.f);
            anode->setAnimationSpeed(18.f*float(x+1+(z+1)*kInstanceSquareSize)/float(kInstanceSquareSize*kInstanceSquareSize));
            anode->setMaterialType(newMaterialType);
            anode->setMaterialTexture(3,anode->getBonePoseTBO());
        }

        gpumesh->drop();
    }

    //! We use a renderbuffer because we don't intend on reading from it
    const uint32_t numberOfSamples = 8;
    video::IFrameBuffer* framebuffer = driver->addFrameBuffer();
    if (true)
    {
        video::IRenderBuffer* color = driver->addMultisampleRenderBuffer(numberOfSamples,params.WindowSize,video::ECF_A8R8G8B8);
        video::IRenderBuffer* depth = driver->addMultisampleRenderBuffer(numberOfSamples,params.WindowSize,video::ECF_DEPTH32F);
        framebuffer->attach(video::EFAP_COLOR_ATTACHMENT0,color);
        framebuffer->attach(video::EFAP_DEPTH_ATTACHMENT,depth);
    }/*
    else
    {
        numberOfSamples,&params.WindowSize.Width
        video::ITexture* color = driver->addTexture(,video::ECF_A8R8G8B8);
        video::ITexture* depth = driver->addMultisampleRenderBuffer(numberOfSamples,params.WindowSize,video::ECF_DEPTH32F);
        framebuffer->attach(video::EFAP_COLOR_ATTACHMENT0,color);
        framebuffer->attach(video::EFAP_DEPTH_ATTACHMENT,depth);
    }*/


	uint64_t lastFPSTime = 0;

	while(device->run()&&(!quit))
	//if (device->isWindowActive())
	{
		driver->beginScene( false,false );

		driver->setRenderTarget(framebuffer);
		vectorSIMDf clearColor(1.f,1.f,1.f,1.f);
        driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0,clearColor.pointer);
		driver->clearZBuffer();
        //! This animates (moves) the camera and sets the transforms
        //! Also draws the meshbuffer

        glEnable(GL_MULTISAMPLE);
        smgr->drawAll();
        glDisable(GL_MULTISAMPLE);

        const bool needToCopyDepth = false;
        driver->blitRenderTargets(framebuffer,0,needToCopyDepth);

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
	driver->removeRenderBuffer(color);
	driver->removeRenderBuffer(depth);

    for (size_t x=0; x<kInstanceSquareSize; x++)
    for (size_t z=0; z<kInstanceSquareSize; z++)
        instancesToRemove[x+kInstanceSquareSize*z]->remove();

	device->drop();

	return 0;
}
