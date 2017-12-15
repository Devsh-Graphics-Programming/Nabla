#include <irrlicht.h>
#include "driverChoice.h"

#include "../source/Irrlicht/CGeometryCreator.h"
#include "../source/Irrlicht/COpenGLExtensionHandler.h"
#include "../source/Irrlicht/COpenGLBuffer.h"

using namespace irr;
using namespace core;
using namespace video;


#define kNumHardwareInstancesX 30
#define kNumHardwareInstancesY 40
#define kNumHardwareInstancesZ 50

#define kHardwareInstancesTOTAL (kNumHardwareInstancesX*kNumHardwareInstancesY*kNumHardwareInstancesZ)




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

class SimpleCallBack : public video::IShaderConstantSetCallBack
{
    int32_t cameraDirUniformLocation;
    video::E_SHADER_CONSTANT_TYPE cameraDirUniformType;
public:
    SimpleCallBack() : cameraDirUniformLocation(-1), cameraDirUniformType(video::ESCT_FLOAT_VEC3) {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::array<video::SConstantLocationNamePair>& constants)
    {
        //! Normally we'd iterate through the array and check our actual constant names before mapping them to locations but oh well
        cameraDirUniformLocation = constants[0].location;
        cameraDirUniformType = constants[0].type;
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(),cameraDirUniformLocation,cameraDirUniformType,1);
    }

    virtual void OnUnsetMaterial() {}
};


void APIENTRY openGLCBFunc(GLenum source, GLenum type, GLuint id, GLenum severity,
                           GLsizei length, const GLchar* message, const void* userParam)
{
    core::stringc outStr;
    switch (severity)
    {
        //case GL_DEBUG_SEVERITY_HIGH:
        case GL_DEBUG_SEVERITY_HIGH_ARB:
            outStr = "[H.I.G.H]";
            break;
        //case GL_DEBUG_SEVERITY_MEDIUM:
        case GL_DEBUG_SEVERITY_MEDIUM_ARB:
            outStr = "[MEDIUM]";
            break;
        //case GL_DEBUG_SEVERITY_LOW:
        case GL_DEBUG_SEVERITY_LOW_ARB:
            outStr = "[  LOW  ]";
            break;
        case GL_DEBUG_SEVERITY_NOTIFICATION:
            outStr = "[  LOW  ]";
            break;
        default:
            outStr = "[UNKNOWN]";
            break;
    }
    switch (source)
    {
        //case GL_DEBUG_SOURCE_API:
        case GL_DEBUG_SOURCE_API_ARB:
            switch (type)
            {
                //case GL_DEBUG_TYPE_ERROR:
                case GL_DEBUG_TYPE_ERROR_ARB:
                    outStr += "[OPENGL  API ERROR]\t\t";
                    break;
                //case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
                case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB:
                    outStr += "[OPENGL  DEPRECATED]\t\t";
                    break;
                //case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
                case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB:
                    outStr += "[OPENGL   UNDEFINED]\t\t";
                    break;
                //case GL_DEBUG_TYPE_PORTABILITY:
                case GL_DEBUG_TYPE_PORTABILITY_ARB:
                    outStr += "[OPENGL PORTABILITY]\t\t";
                    break;
                //case GL_DEBUG_TYPE_PERFORMANCE:
                case GL_DEBUG_TYPE_PERFORMANCE_ARB:
                    outStr += "[OPENGL PERFORMANCE]\t\t";
                    break;
                default:
                    outStr += "[OPENGL       OTHER]\t\t";
                    ///return;
                    break;
            }
            outStr += message;
            break;
        //case GL_DEBUG_SOURCE_SHADER_COMPILER:
        case GL_DEBUG_SOURCE_SHADER_COMPILER_ARB:
            outStr += "[SHADER]\t\t";
            outStr += message;
            break;
        //case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM_ARB:
            outStr += "[WINDOW SYS]\t\t";
            outStr += message;
            break;
        //case GL_DEBUG_SOURCE_THIRD_PARTY:
        case GL_DEBUG_SOURCE_THIRD_PARTY_ARB:
            outStr += "[3RDPARTY]\t\t";
            outStr += message;
            break;
        //case GL_DEBUG_SOURCE_APPLICATION:
        case GL_DEBUG_SOURCE_APPLICATION_ARB:
            outStr += "[APP]\t\t";
            outStr += message;
            break;
        //case GL_DEBUG_SOURCE_OTHER:
        case GL_DEBUG_SOURCE_OTHER_ARB:
            outStr += "[OTHER]\t\t";
            outStr += message;
            break;
        default:
            break;
    }
    outStr += "\n";
    printf("%s",outStr.c_str());
}

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
    if (COpenGLExtensionHandler::FeatureAvailable[COpenGLExtensionHandler::IRR_KHR_debug])
    {
        glEnable(GL_DEBUG_OUTPUT);
        //glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        COpenGLExtensionHandler::pGlDebugMessageControl(GL_DONT_CARE,GL_DONT_CARE,GL_DONT_CARE,0,NULL,true);

        COpenGLExtensionHandler::pGlDebugMessageCallback(openGLCBFunc,NULL);
    }
    else
    {
        //glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
        COpenGLExtensionHandler::pGlDebugMessageControlARB(GL_DONT_CARE,GL_DONT_CARE,GL_DONT_CARE,0,NULL,true);

        COpenGLExtensionHandler::pGlDebugMessageCallbackARB(openGLCBFunc,NULL);
    }

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
	camera->setFarValue(100.0f);
    smgr->setActiveCamera(camera);
	device->getCursorControl()->setVisible(false);
	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);


    core::vectorSIMDf instanceAngularSpeed[kNumHardwareInstancesZ][kNumHardwareInstancesY][kNumHardwareInstancesX];
    core::vector3df instancePositions[kNumHardwareInstancesZ][kNumHardwareInstancesY][kNumHardwareInstancesX];
    core::vector3df instanceNewPositions[kNumHardwareInstancesZ][kNumHardwareInstancesY][kNumHardwareInstancesX];

    video::IGPUBuffer* instancePosBuf;
    scene::IGPUMesh* gpumesh = smgr->getGeometryCreator()->createCubeMeshGPU(driver,core::vector3df(1.f));

        smgr->addMeshSceneNode(gpumesh,0,-1)->setMaterialType(newMaterialType);
        gpumesh->drop();

        //! Special Juice for INSTANCING
        for (size_t z=0; z<kNumHardwareInstancesZ; z++)
        for (size_t y=0; y<kNumHardwareInstancesY; y++)
        {
            srand(device->getTimer()->getRealTime64());
            for (size_t x=0; x<kNumHardwareInstancesX; x++)
            {
                instancePositions[z][y][x].set(x,y,z);
                instancePositions[z][y][x] *= 2.f;
                instanceAngularSpeed[z][y][x].set(rand()%16741,rand()%16741,rand()%16741);
                instanceAngularSpeed[z][y][x] /= 16741.f;
            }
        }

        //instancePosBuf = driver->createGPUBuffer(sizeof(core::vector3df)*kHardwareInstancesTOTAL,instancePositions,true);
        instancePosBuf = driver->createPersistentlyMappedBuffer(sizeof(core::vector3df)*kHardwareInstancesTOTAL,instancePositions,EGBA_WRITE,false,false);
        void* ptr;
        GLuint handle;
        {
            COpenGLBuffer* bufGL = dynamic_cast<COpenGLBuffer*>(instancePosBuf);
            handle = bufGL->getOpenGLName();
            COpenGLExtensionHandler::extGlUnmapNamedBuffer(handle);

            ptr = COpenGLExtensionHandler::extGlMapNamedBufferRange(handle,0,instancePosBuf->getSize(),GL_MAP_WRITE_BIT|GL_MAP_PERSISTENT_BIT|GL_MAP_FLUSH_EXPLICIT_BIT);
        }
        gpumesh->getMeshBuffer(0)->getMeshDataAndFormat()->mapVertexAttrBuffer(instancePosBuf,scene::EVAI_ATTR2,scene::ECPA_THREE,scene::ECT_FLOAT,12,0,1);
        instancePosBuf->drop();

        //set instance count on mesh
        gpumesh->getMeshBuffer(0)->setInstanceCount(kHardwareInstancesTOTAL);

        //new bbox is necessary
        core::aabbox3df newBBox = gpumesh->getMeshBuffer(0)->getBoundingBox();
        newBBox.MaxEdge += vector3df(kNumHardwareInstancesX,kNumHardwareInstancesY,kNumHardwareInstancesZ)*2.f;
        newBBox.MaxEdge -= 2.f;
        gpumesh->getMeshBuffer(0)->setBoundingBox(newBBox);
        gpumesh->setBoundingBox(newBBox);



	uint64_t lastFPSTime = 0;

	while(device->run())
	//if (device->isWindowActive())
	{
		driver->beginScene(true, true, video::SColor(255,0,0,255) );

		float timeLocal = device->getTimer()->getTime();
        for (size_t z=0; z<kNumHardwareInstancesZ; z++)
        for (size_t y=0; y<kNumHardwareInstancesY; y++)
        for (size_t x=0; x<kNumHardwareInstancesX; x++)
		{
		    core::vectorSIMDf rot = instanceAngularSpeed[z][y][x]*timeLocal;
		    rot /= 1000.f;
		    instanceNewPositions[z][y][x] = instancePositions[z][y][x]+core::vector3df(cosf(rot.X),cosf(rot.Y),cosf(rot.Z));
		}
		///case 0:
		//instancePosBuf->updateSubRange(0,instancePosBuf->getSize(),instanceNewPositions);
		///case 1:
        memcpy(ptr,instanceNewPositions,instancePosBuf->getSize());
        COpenGLExtensionHandler::extGlFlushMappedNamedBufferRange(handle,0,instancePosBuf->getSize());

        //! This animates (moves) the camera and sets the transforms
        //! Also draws the meshbuffer
        smgr->drawAll();

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			stringw str;/**
			str = L"Builtin Nodes Demo - Irrlicht Engine [";
			str += driver->getName();
			str += "] FPS:";
			str += driver->getFPS();
			str += " PrimitvesDrawn:";
			str += driver->getPrimitiveCountDrawn();*/

			device->setWindowCaption(str.c_str());
			lastFPSTime = time;
		}
	}
    COpenGLExtensionHandler::extGlUnmapNamedBuffer(handle);

	device->drop();

	return 0;
}
