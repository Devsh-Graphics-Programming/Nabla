#include <irrlicht.h>
#include "driverChoice.h"

#include "../source/Irrlicht/CGeometryCreator.h"

using namespace irr;
using namespace core;

vector3df camPos;
array<vectorSIMDf> controlPts;
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
                            spline = new CLinearSpline(controlPts.pointer(),controlPts.size());

                        return true;
                    }
                    break;
                case KEY_KEY_Y:
                    {
                        if (spline)
                            delete spline;
                        spline = NULL;
                        if (controlPts.size())
                            spline = new CLinearSpline(controlPts.pointer(),controlPts.size(),true); //make it loop

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
                            spline = new CQuadraticSpline(controlPts.pointer(),controlPts.size(),false);
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
                            spline = new CQuadraticSpline(controlPts.pointer(),controlPts.size(),false,true); //make it be ready for first turn
                            printf("Total Len %f\n",spline->getSplineLength());
                            for (size_t i=0; i<spline->getSegmentCount(); i++)
                                printf("Seg: %d \t\t %f\n",i,spline->getSegmentLength(i));
                        }

                        return true;
                    }
                case KEY_KEY_O:
                    {
                        if (spline)
                            delete spline;
                        spline = NULL;
                        if (controlPts.size())
                        {
                            spline = new CQuadraticSpline(controlPts.pointer(),controlPts.size(),true); //make it loop
                            printf("Total Len %f\n",spline->getSplineLength());
                            for (size_t i=0; i<spline->getSegmentCount(); i++)
                                printf("Seg: %d \t\t %f\n",i,spline->getSegmentLength(i));
                        }

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



int main()
{
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
                                                        callBack, //! No Shader Callback (we dont have any constants/uniforms to pass to the shader)
                                                        0); //! No custom user data
    callBack->drop();


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

	scene::ISceneNode* billboard = smgr->addCubeSceneNode(2.f,0,-1,core::vector3df(0,0,0));
    billboard->getMaterial(0).setTexture(0,driver->getTexture("../../media/wall.jpg"));

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
	//if (device->isWindowActive())
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
            assert(spline->getUnnormDirection_fromParameter(forwardDir,cubeSegment,cubeParameterHint)); //must be TRUE
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
