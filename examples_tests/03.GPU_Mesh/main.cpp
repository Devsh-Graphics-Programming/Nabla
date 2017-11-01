#include <irrlicht.h>
#include "driverChoice.h"


using namespace irr;
using namespace core;


/**
We do cool stuff here, like make an event receiver to process input
**/
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
                exit(0);
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
    video::E_SHADER_CONSTANT_TYPE mvpUniformType;
public:
    SimpleCallBack() : mvpUniformLocation(-1), mvpUniformType(video::ESCT_FLOAT_VEC3) {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::array<video::SConstantLocationNamePair>& constants)
    {
        //! Normally we'd iterate through the array and check our actual constant names before mapping them to locations but oh well
        mvpUniformLocation = constants[0].location;
        mvpUniformType = constants[0].type;
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(),mvpUniformLocation,mvpUniformType,1);
    }

    virtual void OnUnsetMaterial() {}
};

#include "irrpack.h"
struct VertexStruct
{
    float Pos[3];
    uint8_t Col[2];
} PACK_STRUCT;
#include "irrunpack.h"

int main()
{
	printf("Please select the background:\n");
	printf(" (0 : default) Use SkyDome\n");
	printf(" (1) Use SkyBox\n");

	char c;
	std::cin >> c;

	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
    params.AntiAlias = 0; //No AA, yet
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
    SimpleCallBack* callBack = new SimpleCallBack();

    //! First need to make a material other than default to be able to draw with custom shader
    video::SMaterial material;
    material.BackfaceCulling = false; //! Triangles will be visible from both sides
    material.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
                                                        "","","", //! No Geometry or Tessellation Shaders
                                                        "../mesh.frag",
                                                        3,video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
                                                        callBack,
                                                        0); //! No custom user data
    callBack->drop();


	scene::ISceneManager* smgr = device->getSceneManager();




	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
	driver->setTextureCreationFlag(video::ETCF_CREATE_MIP_MAPS, false);

	// create skybox and skydome
	switch (c)
	{
        case '1':
            smgr->addSkyBoxSceneNode(
                driver->getTexture("../../media/irrlicht2_up.jpg"),
                driver->getTexture("../../media/irrlicht2_dn.jpg"),
                driver->getTexture("../../media/irrlicht2_lf.jpg"),
                driver->getTexture("../../media/irrlicht2_rt.jpg"),
                driver->getTexture("../../media/irrlicht2_ft.jpg"),
                driver->getTexture("../../media/irrlicht2_bk.jpg"));
            break;
        default:
            smgr->addSkyDomeSceneNode(driver->getTexture("../../media/skydome.jpg"),16,8,0.95f,2.0f,10.f);
            break;
	}

	driver->setTextureCreationFlag(video::ETCF_CREATE_MIP_MAPS, true);


	//! we want to move around the scene and view it from different angles
	scene::ICameraSceneNode* camera =
		smgr->addCameraSceneNodeFPS(0,100.0f,0.001f);

	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(10.0f);

    smgr->setActiveCamera(camera);


	//! disable mouse cursor, since camera will force it to the middle
	//! and we don't want a jittery cursor in the middle distracting us
	device->getCursorControl()->setVisible(false);

	//! Since our cursor will be enslaved, there will be no way to close the window
	//! So we listen for the "Q" key being pressed and exit the application
	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);

	uint64_t lastFPSTime = 0;
	printf("%d\n",sizeof(VertexStruct));

	while(device->run())
	//if (device->isWindowActive())
	{
		driver->beginScene(true, true, video::SColor(255,255,255,255) );

        //! This animates (moves) the camera and sets the transforms
        //! Also draws the meshbuffer
        smgr->drawAll();

        //! Stress test for memleaks aside from demo how to create meshes that live on the GPU RAM
        {
            scene::IGPUMeshBuffer* mb = new scene::IGPUMeshBuffer();
            scene::IGPUMeshDataFormatDesc* desc = driver->createGPUMeshDataFormatDesc();
            mb->setMeshDataAndFormat(desc);
            desc->drop();

            VertexStruct vertices[8];
            vertices[0].Pos[0] = -1.f;
            vertices[0].Pos[1] = -1.f;
            vertices[0].Pos[2] = -1.f;
            vertices[0].Col[0] = 0;
            vertices[0].Col[1] = 0;
            vertices[1].Pos[0] = 1.f;
            vertices[1].Pos[1] = -1.f;
            vertices[1].Pos[2] = -1.f;
            vertices[1].Col[0] = 127;
            vertices[1].Col[1] = 0;
            vertices[2].Pos[0] = -1.f;
            vertices[2].Pos[1] = 1.f;
            vertices[2].Pos[2] = -1.f;
            vertices[2].Col[0] = 255;
            vertices[2].Col[1] = 0;
            vertices[3].Pos[0] = 1.f;
            vertices[3].Pos[1] = 1.f;
            vertices[3].Pos[2] = -1.f;
            vertices[3].Col[0] = 0;
            vertices[3].Col[1] = 127;
            vertices[4].Pos[0] = -1.f;
            vertices[4].Pos[1] = -1.f;
            vertices[4].Pos[2] = 1.f;
            vertices[4].Col[0] = 127;
            vertices[4].Col[1] = 127;
            vertices[5].Pos[0] = 1.f;
            vertices[5].Pos[1] = -1.f;
            vertices[5].Pos[2] = 1.f;
            vertices[5].Col[0] = 255;
            vertices[5].Col[1] = 127;
            vertices[6].Pos[0] = -1.f;
            vertices[6].Pos[1] = 1.f;
            vertices[6].Pos[2] = 1.f;
            vertices[6].Col[0] = 0;
            vertices[6].Col[1] = 255;
            vertices[7].Pos[0] = 1.f;
            vertices[7].Pos[1] = 1.f;
            vertices[7].Pos[2] = 1.f;
            vertices[7].Col[0] = 127;
            vertices[7].Col[1] = 255;

            uint16_t indices_indexed16[] =
            {
                0,1,2,1,2,3,
                4,5,6,5,6,7,
                0,1,4,1,4,5,
                2,3,6,3,6,7,
                0,2,4,2,4,6,
                1,3,5,3,5,7
            };

            void* tmpMem = malloc(sizeof(vertices)+2+sizeof(indices_indexed16));
            memcpy(tmpMem,vertices,sizeof(vertices));
            memcpy(tmpMem+sizeof(vertices)+2,indices_indexed16,sizeof(indices_indexed16));
            video::IGPUBuffer* buff = driver->createGPUBuffer(sizeof(vertices)+sizeof(indices_indexed16)+2,tmpMem);
            free(tmpMem);

            desc->mapVertexAttrBuffer(buff,scene::EVAI_ATTR0,scene::ECPA_THREE,scene::ECT_FLOAT,sizeof(VertexStruct),0);
            desc->mapVertexAttrBuffer(buff,scene::EVAI_ATTR1,scene::ECPA_TWO,scene::ECT_NORMALIZED_UNSIGNED_BYTE,sizeof(VertexStruct),12);
            desc->mapIndexBuffer(buff);
            mb->setIndexBufferOffset(sizeof(vertices)+2);
            mb->setIndexType(video::EIT_16BIT);
            mb->setIndexCount(2*3*6);
            buff->drop();

            driver->setTransform(video::E4X3TS_WORLD,core::matrix4x3());
            driver->setMaterial(material);
            driver->drawMeshBuffer(mb);
            mb->drop();
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

	device->drop();

	return 0;
}
