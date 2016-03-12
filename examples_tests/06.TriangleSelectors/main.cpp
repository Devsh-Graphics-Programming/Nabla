#include <irrlicht.h>
#include "driverChoice.h"

#include "../source/Irrlicht/CGeometryCreator.h"

using namespace irr;
using namespace core;

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



int main()
{
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
    params.AntiAlias = 0; //No AA, yet
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<u32>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.


	video::IVideoDriver* driver = device->getVideoDriver();

    //! First need to make a material other than default to be able to draw with custom shader
    video::SMaterial material;
    material.BackfaceCulling = false; //! Triangles will be visible from both sides
    material.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
                                                        "","","", //! No Geometry or Tessellation Shaders
                                                        "../mesh.frag",
                                                        3,video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
                                                        NULL, //! No Shader Callback (we dont have any constants/uniforms to pass to the shader)
                                                        0); //! No custom user data



	scene::ISceneManager* smgr = device->getSceneManager();
	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
	scene::ICameraSceneNode* camera =
		smgr->addCameraSceneNodeFPS(0,100.0f,0.01f);
	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(100.0f);
    smgr->setActiveCamera(camera);
	device->getCursorControl()->setVisible(false);
	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);


	//! Test Creation Of Builtin
	scene::IMeshSceneNode* cube = dynamic_cast<scene::IMeshSceneNode*>(smgr->addCubeSceneNode(1.f,0,-1));
    cube->setRotation(core::vector3df(45,20,15));
    cube->getMaterial(0).setTexture(0,driver->getTexture("../../media/irrlicht2_dn.jpg"));

	scene::IMeshSceneNode* sphere = dynamic_cast<scene::IMeshSceneNode*>(smgr->addSphereSceneNode(2,512));
    sphere->getMaterial(0).setTexture(0,driver->getTexture("../../media/skydome.jpg"));
    sphere->setPosition(core::vector3df(4,0,0));

	scene::ISceneNode* billboard = smgr->addBillboardSceneNode(0,core::dimension2df(1.f,1.f),core::vector3df(-4,0,0));
    billboard->getMaterial(0).setTexture(0,driver->getTexture("../../media/wall.jpg"));

    //scene::CGeometryCreator* geom = new scene::CGeometryCreator();


	uint64_t lastFPSTime = 0;

	while(device->run())
	//if (device->isWindowActive())
	{
		driver->beginScene(true, true, video::SColor(255,0,0,255) );

        //! This animates (moves) the camera and sets the transforms
        //! Also draws the meshbuffer
        smgr->drawAll();

        //! Stress test for memleaks aside from demo how to create meshes that live on the GPU RAM
        {/*
            scene::IGPUMeshBuffer* mb = new scene::IGPUMeshBuffer();
            scene::IGPUMeshDataFormatDesc* desc = driver->createGPUMeshDataFormatDesc();
            mb->setMeshDataAndFormat(desc);
            desc->drop();

            uint16_t indices_indexed16[] = {
                0,1,2,1,2,3,
                4,5,6,5,6,7,
                0,1,4,1,4,5,
                2,3,6,3,6,7,
                0,2,4,2,4,6,
                1,3,5,3,5,7
            };
            video::IGPUBuffer* index = driver->createGPUBuffer(sizeof(indices_indexed16),indices_indexed16);
            desc->mapIndexBuffer(index);
            mb->setIndexType(video::EIT_16BIT);
            mb->setIndexCount(2*3*6);
            mb->setIndexRange(0,7);
            index->drop();

            float attrArr[] = {
                -1.f,-1.f,-1.f,0.f,0.f,
                 1.f,-1.f,-1.f,0.5f,0.f,
                -1.f, 1.f,-1.f,1.f,0.f,
                 1.f, 1.f,-1.f,0.f,0.5f,
                -1.f,-1.f, 1.f,0.5f,0.5f,
                 1.f,-1.f, 1.f,1.f,0.5f,
                -1.f, 1.f, 1.f,0.f,1.f,
                 1.f, 1.f, 1.f,0.5f,1.f
            };
            video::IGPUBuffer* attr0 = driver->createGPUBuffer(sizeof(attrArr),attrArr);
            desc->mapVertexAttrBuffer(attr0,scene::EVAI_ATTR0,scene::ECPA_THREE,scene::ECT_FLOAT,20,0);
            desc->mapVertexAttrBuffer(attr0,scene::EVAI_ATTR1,scene::ECPA_TWO,scene::ECT_FLOAT,20,3*4);
            attr0->drop();

            driver->setTransform(video::ETS_WORLD,core::matrix4());
            driver->setMaterial(material);
            driver->drawMeshBuffer(mb);
            mb->drop();*/
        }

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			stringw str = L"Builtin Nodes Demo - Irrlicht Engine [";
			str += driver->getName();
			str += "] FPS:";
			str += driver->getFPS();
			str += " PrimitvesDrawn:";
			str += driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.c_str());
			lastFPSTime = time;
		}
	}

	device->drop();

	return 0;
}
