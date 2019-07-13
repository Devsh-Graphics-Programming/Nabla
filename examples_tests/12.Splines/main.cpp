#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include "../source/Irrlicht/COpenGLExtensionHandler.h"

using namespace irr;
using namespace core;

vector3df camPos;
vector<vectorSIMDf> controlPts;
ISpline* spline = NULL;

//!Same As Last Example
class MyEventReceiver : public IEventReceiver
{
public:

	MyEventReceiver() : wasLeftPressedBefore(false)
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
                    break;
                case KEY_KEY_T:
                    {
                        if (spline)
                            delete spline;
                        spline = NULL;
                        if (controlPts.size())
                            spline = new CLinearSpline(controlPts.data(),controlPts.size());

                        return true;
                    }
                    break;
                case KEY_KEY_Y:
                    {
                        if (spline)
                            delete spline;
                        spline = NULL;
                        if (controlPts.size())
                            spline = new CLinearSpline(controlPts.data(),controlPts.size(),true); //make it loop

                        return true;
                    }
                    break;
                case KEY_KEY_U:
                    {
                        if (spline)
                            delete spline;
                        spline = NULL;
                        if (controlPts.size())
                        {
                            spline = new irr::core::CQuadraticBSpline(controlPts.data(),controlPts.size(),false);
                            printf("Total Len %f\n",spline->getSplineLength());
                            for (size_t i=0; i<spline->getSegmentCount(); i++)
                                printf("Seg: %d \t\t %f\n",i,spline->getSegmentLength(i));
                        }

                        return true;
                    }
                    break;
                case KEY_KEY_I:
                    {
                        if (spline)
                            delete spline;
                        spline = NULL;
                        if (controlPts.size())
                        {
                            spline = new CQuadraticBSpline(controlPts.data(),controlPts.size(),true); //make it a loop
                            printf("Total Len %f\n",spline->getSplineLength());
                            for (size_t i=0; i<spline->getSegmentCount(); i++)
                                printf("Seg: %d \t\t %f\n",i,spline->getSegmentLength(i));
                        }

                        return true;
                    }
                case KEY_KEY_O:
                    {
                        return true;
                    }
                    break;
                case KEY_KEY_C:
                    {
                        controlPts.clear();
                        return true;
                    }
                    break;
                default:
                    break;
            }
        }
        else if (event.EventType == EET_MOUSE_INPUT_EVENT)
        {
            bool pressed = event.MouseInput.isLeftPressed();
            if (pressed && !wasLeftPressedBefore)
            {
                controlPts.push_back(core::vectorSIMDf(camPos.X,camPos.Y,camPos.Z));
            }
            wasLeftPressedBefore = pressed;
        }

		return false;
	}

private:
    bool wasLeftPressedBefore;
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


	scene::ISceneManager* smgr = device->getSceneManager();
	scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0,100.0f,0.01f);
	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(100.0f);
    smgr->setActiveCamera(camera);

	device->getCursorControl()->setVisible(false);
	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);

    asset::IAssetLoader::SAssetLoadParams lparams;
    asset::ICPUTexture* cputextures[]{
        static_cast<asset::ICPUTexture*>(device->getAssetManager().getAsset("../../media/irrlicht2_dn.jpg", lparams)),
        static_cast<asset::ICPUTexture*>(device->getAssetManager().getAsset("../../media/wall.jpg", lparams))
    };
    core::vector<video::ITexture*> gputextures = driver->getGPUObjectsFromAssets(cputextures, cputextures+2);

	//! Test Creation Of Builtin
	scene::IMeshSceneNode* cube = dynamic_cast<scene::IMeshSceneNode*>(smgr->addCubeSceneNode(1.f,0,-1));
    cube->setRotation(core::vector3df(45,20,15));
    cube->getMaterial(0).setTexture(0,gputextures[0]);

	scene::ISceneNode* billboard = smgr->addCubeSceneNode(2.f,0,-1,core::vector3df(0,0,0));
    billboard->getMaterial(0).setTexture(0,gputextures[1]);

    float cubeDistance = 0.f;
    float cubeParameterHint = 0.f;
    uint32_t cubeSegment = 0;

    #define kCircleControlPts 3
    for (size_t i=0; i<kCircleControlPts; i++)
    {
        float x = float(i)*core::PI*2.f/float(kCircleControlPts);
        controlPts.push_back(vectorSIMDf(sin(x),0,-cos(x))*4.f);
    }


	uint64_t lastFPSTime = 0;

	uint64_t lastTime = device->getTimer()->getRealTime();
    uint64_t timeDelta = 0;

	while(device->run())
	{
		driver->beginScene(true, true, video::SColor(255,0,0,255) );

        uint64_t nowTime = device->getTimer()->getRealTime();
        timeDelta = nowTime-lastTime;
        lastTime = nowTime;

		if (spline)
        {
            vectorSIMDf newPos;
            cubeDistance += float(timeDelta)*0.001f; //1 unit per second
            cubeSegment = spline->getPos(newPos,cubeDistance,cubeSegment,&cubeParameterHint);
            if (cubeSegment>=0xdeadbeefu) //reached end of non-loop, or spline changed
            {
                cubeDistance = 0;
                cubeParameterHint = 0;
                cubeSegment = 0;
                cubeSegment = spline->getPos(newPos,cubeDistance,cubeSegment,&cubeParameterHint);
            }

            vectorSIMDf forwardDir; 
			bool success = spline->getUnnormDirection_fromParameter(forwardDir,cubeSegment,cubeParameterHint);
            assert(success); //must be TRUE
            forwardDir = normalize(forwardDir); //must normalize after
            vectorSIMDf sideDir = normalize(cross(forwardDir,vectorSIMDf(0,1,0))); // predefined up vector
            vectorSIMDf pseudoUp = cross(sideDir,forwardDir);



            matrix4x3 mat;
            mat.getColumn(0) = reinterpret_cast<vector3df&>(forwardDir);
            mat.getColumn(1) = reinterpret_cast<vector3df&>(pseudoUp);
            mat.getColumn(2) = reinterpret_cast<vector3df&>(sideDir);
            mat.setTranslation(reinterpret_cast<const vector3df&>(newPos));
            cube->setRelativeTransformationMatrix(mat);
        }

        //! This animates (moves) the camera and sets the transforms
        //! Also draws the meshbuffer
        smgr->drawAll();
        camPos = camera->getAbsolutePosition();

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Builtin Nodes Demo - Irrlicht Engine FPS:" << driver->getFPS() << " PrimitvesDrawn:";
			str << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str());
			lastFPSTime = time;
		}
	}

    if (spline)
        delete spline;

	device->drop();

	return 0;
}
