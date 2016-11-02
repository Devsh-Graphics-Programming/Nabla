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



	scene::ISceneManager* smgr = device->getSceneManager();
	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
	scene::ICameraSceneNode* camera =
		smgr->addCameraSceneNodeFPS(0,100.0f,0.01f);
	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(100.0f);
    smgr->setActiveCamera(camera);
	//device->getCursorControl()->setVisible(false);
	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);


    core::SCollisionEngine* gCollEng = new core::SCollisionEngine();

	//! Test Creation Of Builtin
	scene::IMeshSceneNode* cube = dynamic_cast<scene::IMeshSceneNode*>(smgr->addCubeSceneNode(1.f,0,-1));
    cube->setRotation(core::vector3df(45,20,15));
    cube->setMaterialFlag(video::EMF_BACK_FACE_CULLING,false);
    cube->getMaterial(0).setTexture(0,driver->getTexture("../../media/irrlicht2_dn.jpg"));
	core::SCompoundCollider* compound = new core::SCompoundCollider();
	compound->AddBox(core::SAABoxCollider(cube->getBoundingBox()));
	core::SColliderData collData;
	collData.attachedNode = cube;
	compound->setColliderData(collData);
    gCollEng->addCompoundCollider(compound);
    compound->drop();

	scene::IMeshSceneNode* sphere = dynamic_cast<scene::IMeshSceneNode*>(smgr->addSphereSceneNode(2,32));
    sphere->setMaterialFlag(video::EMF_BACK_FACE_CULLING,false);
    sphere->getMaterial(0).setTexture(0,driver->getTexture("../../media/skydome.jpg"));
    sphere->getMaterial(0).MaterialType = material.MaterialType;
    sphere->setPosition(core::vector3df(4,0,0));
	compound = new core::SCompoundCollider();
	compound->AddEllipsoid(core::vectorSIMDf(),core::vectorSIMDf(2.f));
	collData.attachedNode = sphere;
	compound->setColliderData(collData);
    gCollEng->addCompoundCollider(compound);
    compound->drop();

	uint64_t lastFPSTime = 0;

    core::SColliderData hitPointData;
	while(device->run())
	//if (device->isWindowActive())
	{
		driver->beginScene(true, true, video::SColor(255,0,0,255) );

        //! This animates (moves) the camera and sets the transforms
        //! Also draws the meshbuffer
        smgr->drawAll();

		driver->endScene();

        cube->setMaterialFlag(video::EMF_WIREFRAME,false);
        sphere->setMaterialFlag(video::EMF_WIREFRAME,false);
        core::vectorSIMDf origin,dir;
        origin.set(camera->getAbsolutePosition());
        dir.set(camera->getTarget());
        dir -= origin;
        dir = core::normalize(dir);
        float outLen;
        if (gCollEng->FastCollide(hitPointData,outLen,origin,dir,10.f))
        {
            if (hitPointData.attachedNode)
                hitPointData.attachedNode->setMaterialFlag(video::EMF_WIREFRAME,true);
        }

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
