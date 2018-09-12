#include <irrlicht.h>
#include "driverChoice.h"
#include "SilverLining.h"
#include "ResourceLoader.h"

#include "../source/Irrlicht/COpenGLDriver.h"

using namespace irr;
using namespace core;
using namespace video;

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
            return;
            outStr = "[  LOW  ]";
            break;
        case GL_DEBUG_SEVERITY_NOTIFICATION:
            return;
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


IrrlichtDevice* device = NULL;


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
                    device->closeDevice();
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

class SimpleCallBack : public video::IShaderConstantSetCallBack
{
    int32_t mvpUniformLocation;
    video::E_SHADER_CONSTANT_TYPE mvpUniformType;
public:
    SimpleCallBack() : mvpUniformLocation(-1), mvpUniformType(video::ESCT_FLOAT_VEC3) {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
    {
        //! Normally we'd iterate through the vector and check our actual constant names before mapping them to locations but oh well
        mvpUniformLocation = constants[0].location;
        mvpUniformType = constants[0].type;
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(),mvpUniformLocation,mvpUniformType,1);
    }

    virtual void OnUnsetMaterial() {}
};


// All SilverLining objects are in the SilverLining namespace.
using namespace SilverLining;

class SilverLoader : public ResourceLoader
{
    public:
        SilverLoader(io::IFileSystem* fileSystem, const std::string& archivePath) : fSys(fileSystem), resourceArch(NULL)
        {
            fSys->addFileArchive(archivePath.c_str(),false,false,io::EFAT_ZIP,"",&resourceArch);
        }
        virtual                        ~SilverLoader()
        {
            fSys->removeFileArchive(resourceArch);
        }
    /// Sets the path to the SilverLining resources folder, which will be pre-pended to all resource filenames
    /// passed into LoadResource(). This path is also used to locate the renderer DLL's inside the SilverLining
    /// resources folder. It should be called after constructing the ResourceLoader and before calling LoadResource().
        void SILVERLINING_API           SetResourceDirPath(const char *path)
        {
            resDir = path;
        }

    /// Retrieves the path set by SetResourceDirPath().
        const char* SILVERLINING_API    GetResourceDirPath() const
        {
            return resDir.c_str();
        }

    /** Load a resource from mass storage; the default implementation uses the POSIX functions fopen(), fread(), and fclose()
       to do this, but you may override this method to load resources however you wish. The caller is responsible for calling
       FreeResource() when it's done consuming the resource data in order to free its memory.

       \param pathName The path to the desired resource, relative to the location of the resources folder previously specified
       in SetResourceDirPath().
       \param data A reference to a char * that will return the resource's data upon a successful load.
       \param dataLen A reference to an unsigned int that will return the number of bytes loaded upon a successful load.
       \param text True if the resource is a text file, such as a shader. If true, a terminating null character will be appended
       to the resulting data and the file will be opened in text mode.
       \return True if the resource was located and loaded successfully, false otherwise.

       \sa SetResourceDirPath
     */
        virtual bool SILVERLINING_API   LoadResource(const char *pathName, char*& data, unsigned int& dataLen, bool text)
        {
            if (!resourceArch)
                return false;

            io::IReadFile* file = resourceArch->createAndOpenFile(pathName);
            if (!file)
                return false;

            dataLen = file->getSize();
            if (text)
            {
                data = (char*)malloc(dataLen+1);
                file->read(data,dataLen);
                data[dataLen] = 0;
            }
            else
            {
                data = (char*)malloc(dataLen);
                file->read(data,dataLen);
            }

            file->drop();
            return true;
        }

    /** Frees the resource data memory that was returned from LoadResource(). The data pointer will be invalid following
       this call. */
        virtual void SILVERLINING_API   FreeResource(char *data)
        {
            free(data);
        }

    /** Retrieves a list of file names within the directory path specified (relative to the resource path specified
       with SetResourceDirPath().
       \param pathName The path to the directory underneath the resources directory. The path to the resources directory will
       be pre-pended to this path.
       \param dirContents A reference that will receive a vector of strings of the file names found inside this path, if any.
       \return True if the path was found and scanned successfully, false otherwise.
     */
        virtual bool SILVERLINING_API   GetFilesInDirectory(const char *pathName, SL_VECTOR(SL_STRING)& dirContents)
        {
            return false;
        }

    private:
        io::IFileSystem* fSys;
        io::IFileArchive* resourceArch;
        std::string resDir;
};

// Statics and defines for a simple, self-contained demo application
static Atmosphere *atm = 0; // The Atmosphere object is the main interface to SilverLining.
static SilverLoader* silverLoader = 0;
static float aspectRatio, yaw = 0;

// Simulated visibility in meters, for fog effects.
#define kVisibility  20000.0f
// Configure high cirrus clouds.
static void SetupCirrusClouds()
{
    CloudLayer *cirrusCloudLayer;

    cirrusCloudLayer = CloudLayerFactory::Create(CIRRUS_FIBRATUS);
    cirrusCloudLayer->SetBaseAltitude(6000);
    cirrusCloudLayer->SetThickness(0);
    cirrusCloudLayer->SetBaseLength(100000);
    cirrusCloudLayer->SetBaseWidth(100000);
    cirrusCloudLayer->SetLayerPosition(0, 0);
    cirrusCloudLayer->SeedClouds(*atm);

    atm->GetConditions()->AddCloudLayer(cirrusCloudLayer);
}

// Add a cumulus congestus deck with 80% sky coverage.
static void SetupCumulusCongestusClouds()
{
    CloudLayer *cumulusCongestusLayer;

    cumulusCongestusLayer = CloudLayerFactory::Create(CUMULUS_CONGESTUS_HI_RES);
    cumulusCongestusLayer->SetIsInfinite(true);
    cumulusCongestusLayer->SetBaseAltitude(1500);
    cumulusCongestusLayer->SetThickness(100);
    cumulusCongestusLayer->SetBaseLength(30000);
    cumulusCongestusLayer->SetBaseWidth(30000);
    cumulusCongestusLayer->SetDensity(0.8);
    cumulusCongestusLayer->SetLayerPosition(0, 0);
    cumulusCongestusLayer->SetCloudAnimationEffects(0.1, false);
    cumulusCongestusLayer->SeedClouds(*atm);
    cumulusCongestusLayer->SetAlpha(0.5);
    cumulusCongestusLayer->SetFadeTowardEdges(true);

    atm->GetConditions()->AddCloudLayer(cumulusCongestusLayer);
}

// Sets up a solid stratus deck.
static void SetupStratusClouds()
{
    CloudLayer *stratusLayer;

    stratusLayer = CloudLayerFactory::Create(STRATUS);
    stratusLayer->SetIsInfinite(true);
    stratusLayer->SetBaseAltitude(1000);
    stratusLayer->SetThickness(600);
    stratusLayer->SetDensity(0.5);
    stratusLayer->SetLayerPosition(0, 0);
    stratusLayer->SeedClouds(*atm);

    atm->GetConditions()->AddCloudLayer(stratusLayer);
}

// A thunderhead; note a Cumulonimbus cloud layer contains a single cloud.
static void SetupCumulonimbusClouds()
{
    CloudLayer *cumulonimbusLayer;

    cumulonimbusLayer = CloudLayerFactory::Create(CUMULONIMBUS_CAPPILATUS);
    cumulonimbusLayer->SetBaseAltitude(1000);
    cumulonimbusLayer->SetThickness(3000);
    cumulonimbusLayer->SetBaseLength(3000);
    cumulonimbusLayer->SetBaseWidth(5000);
    cumulonimbusLayer->SetLayerPosition(0, -5000);
    cumulonimbusLayer->SeedClouds(*atm);

    atm->GetConditions()->AddCloudLayer(cumulonimbusLayer);
}

// Cumulus mediocris are little, puffy clouds. Keep the density low for realism, otherwise
// you'll have a LOT of clouds because they are small.
static void SetupCumulusMediocrisClouds()
{
    CloudLayer *cumulusMediocrisLayer;

    cumulusMediocrisLayer = CloudLayerFactory::Create(CUMULUS_MEDIOCRIS);
    cumulusMediocrisLayer->SetIsInfinite(true);
    cumulusMediocrisLayer->SetBaseAltitude(1000);
    cumulusMediocrisLayer->SetThickness(200);
    cumulusMediocrisLayer->SetBaseLength(20000);
    cumulusMediocrisLayer->SetBaseWidth(20000);
    cumulusMediocrisLayer->SetDensity(0.9);
    cumulusMediocrisLayer->SetLayerPosition(0, 0);
    cumulusMediocrisLayer->SeedClouds(*atm);

    atm->GetConditions()->AddCloudLayer(cumulusMediocrisLayer);
}

// Stratocumulus clouds are rendered with GPU ray-casting. On systems that can support it
// (Shader model 3.0+) this enables very dense cloud layers with per-fragment lighting.
static void SetupStratocumulusClouds()
{
    CloudLayer *stratocumulusLayer;

    stratocumulusLayer = CloudLayerFactory::Create(STRATOCUMULUS);
    stratocumulusLayer->SetBaseAltitude(1000);
    stratocumulusLayer->SetThickness(3000);
    stratocumulusLayer->SetBaseLength(kVisibility);
    stratocumulusLayer->SetBaseWidth(kVisibility);
    stratocumulusLayer->SetDensity(0.5);
    stratocumulusLayer->SetIsInfinite(true);
    stratocumulusLayer->SetAlpha(1.0);
    stratocumulusLayer->SetFadeTowardEdges(true);
    stratocumulusLayer->SetLayerPosition(0, 0);
    stratocumulusLayer->SeedClouds(*atm);

    atm->GetConditions()->AddCloudLayer(stratocumulusLayer);
}

// Sandstorms should be positioned at ground level. There is no need to set their
// density or thickness.
static void SetupSandstorm()
{
    CloudLayer *sandstormLayer;

    sandstormLayer = CloudLayerFactory::Create(SANDSTORM);
    sandstormLayer->SetIsInfinite(false);
    sandstormLayer->SetLayerPosition(0, -24000);
    sandstormLayer->SetBaseAltitude(0);
    sandstormLayer->SetBaseLength(50000);
    sandstormLayer->SetBaseWidth(50000);
    sandstormLayer->SeedClouds(*atm);

    atm->GetConditions()->AddCloudLayer(sandstormLayer);
}

// Configure SilverLining for the desired wind, clouds, and visibility.
static void SetupAtmosphericConditions()
{
    assert(atm);

    // Set up the desired cloud types.
    SetupCirrusClouds();
    //SetupCumulusCongestusClouds();
    //SetupStratusClouds();
    //SetupCumulonimbusClouds();
    //SetupCumulusMediocrisClouds();
    SetupStratocumulusClouds();
    //SetupSandstorm();

    // Set up wind blowing northeast at 50 meters/sec
    WindVolume wv;
    wv.SetDirection(225);
    wv.SetMinAltitude(0);
    wv.SetMaxAltitude(10000);
    wv.SetWindSpeed(50);
    atm->GetConditions()->SetWind(wv);

    // Set visibility
    atm->GetConditions()->SetVisibility(kVisibility);
}

// Sets the simulated location and local time.
// Note, it's important that your longitude in the Location agrees with
// the time zone in the LocalTime.
void SetTimeAndLocation()
{
    Location loc;
    loc.SetLatitude(45);
    loc.SetLongitude(-122);

    LocalTime tm;
    tm.SetYear(1971);
    tm.SetMonth(8);
    tm.SetDay(7);
    tm.SetHour(12);
    tm.SetMinutes(30);
    tm.SetSeconds(0);
    tm.SetObservingDaylightSavingsTime(true);
    tm.SetTimeZone(PST);

    atm->GetConditions()->SetTime(tm);
    atm->GetConditions()->SetLocation(loc);
}



int main()
{
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 32; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	///params.HandleSRGB = true;
	device = createDeviceEx(params);


	if (device == 0)
		return 1; // could not create selected driver.


	video::COpenGLDriver* driver = dynamic_cast<COpenGLDriver*>(device->getVideoDriver());

#define OPENGL_SUPERLOG
    if (COpenGLExtensionHandler::FeatureAvailable[COpenGLExtensionHandler::IRR_KHR_debug])
    {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        COpenGLExtensionHandler::pGlDebugMessageControl(GL_DONT_CARE,GL_DONT_CARE,GL_DONT_CARE,0,NULL,true);
#ifndef OPENGL_SUPERLOG
        COpenGLExtensionHandler::pGlDebugMessageControl(GL_DONT_CARE,GL_DONT_CARE,GL_DEBUG_SEVERITY_LOW,0,NULL,false);
        COpenGLExtensionHandler::pGlDebugMessageControl(GL_DONT_CARE,GL_DEBUG_TYPE_OTHER,GL_DONT_CARE,0,NULL,false);
        COpenGLExtensionHandler::pGlDebugMessageControl(GL_DONT_CARE,GL_DONT_CARE,GL_DEBUG_SEVERITY_NOTIFICATION,0,NULL,false);
#endif // OPENGL_SUPERLOG
        COpenGLExtensionHandler::pGlDebugMessageCallback(openGLCBFunc,NULL);
    }
    else
    {
        //glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
        COpenGLExtensionHandler::pGlDebugMessageControlARB(GL_DONT_CARE,GL_DONT_CARE,GL_DONT_CARE,0,NULL,true);
#ifndef OPENGL_SUPERLOG
        COpenGLExtensionHandler::pGlDebugMessageControlARB(GL_DONT_CARE,GL_DONT_CARE,GL_DEBUG_SEVERITY_LOW_ARB,0,NULL,false);
        COpenGLExtensionHandler::pGlDebugMessageControlARB(GL_DONT_CARE,GL_DEBUG_TYPE_OTHER_ARB,GL_DONT_CARE,0,NULL,false);
#endif // OPENGL_SUPERLOG
        COpenGLExtensionHandler::pGlDebugMessageCallbackARB(openGLCBFunc,NULL);
    }

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
		smgr->addCameraSceneNodeFPS(0,100.0f,0.55f);
	camera->setPosition(core::vector3df(-4,50.f,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.05f);
	camera->setFarValue(20000.f);
	camera->setFOV(core::PI/3.f);
    smgr->setActiveCamera(camera);
	device->getCursorControl()->setVisible(false);
	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);


	//! Test Creation Of Builtin
	scene::IMeshSceneNode* cube = dynamic_cast<scene::IMeshSceneNode*>(smgr->addCubeSceneNode(1.f,0,-1));
    cube->setRotation(core::vector3df(45,20,15));
    cube->getMaterial(0).setTexture(0,driver->getTexture("../../media/irrlicht2_dn.jpg"));

	scene::ISceneNode* billboard = smgr->addCubeSceneNode(kVisibility,0,-1,core::vector3df(0,0,0),core::vector3df(0,0,0),core::vector3df(1.f,1.f/kVisibility,1.f));
    billboard->getMaterial(0).setTexture(0,driver->getTexture("../../media/wall.jpg"));

    float cubeDistance = 0.f;
    float cubeParameterHint = 0.f;
    uint32_t cubeSegment = 0;

    #define kCircleControlPts 3
    for (size_t i=0; i<kCircleControlPts; i++)
    {
        float x = float(i)*core::PI*2.f/float(kCircleControlPts);
        controlPts.push_back(vectorSIMDf(sin(x),12.5f,-cos(x))*4.f);
    }
    cube->setPosition(controlPts[0].getAsVector3df());

    {
            COpenGLState preState = COpenGLState::collectGLState();
        // Instantiate an Atmosphere object. Substitute your own purchased license name and code here.
        atm = new Atmosphere("YOURNAME", "YOURKEY");

        silverLoader = new SilverLoader(device->getFileSystem(),"../../../../../client/silverlining.zip");
        ///atm->SetResourceLoader(silverLoader);

        int err;
        std::string relativeResourceDir = "../../../../../client/silverlining/";
    #ifdef WIN32
        std::replace(relativeResourceDir.begin(),relativeResourceDir.end(),'/','\\');
        err = atm->Initialize(Atmosphere::OPENGL32CORE, relativeResourceDir.c_str(), false, 0);
    #else
        err = atm->Initialize(Atmosphere::OPENGL32CORE, relativeResourceDir.c_str(), false, 0);
    #endif

        if (err == Atmosphere::E_NOERROR) {

            // If you want different clouds to be generated every time, remember to seed the
            // random number generator.
            atm->GetRandomNumberGenerator()->Seed(time(NULL));

            // Set your frame of reference (call this before setting up clouds!)
            atm->SetUpVector(0, 1, 0);
            atm->SetRightVector(1, 0, 0);

            //! can I call this shit only once?
            atm->SetWorldUnits(0.5);
            atm->SetDepthRange(1.f,0.f);
            atm->SetViewport(0,0,params.WindowSize.Width,params.WindowSize.Height);

            ///atm->SetHaze(0.784f/0.04f,0.707f/0.04f,0.543f/0.04f,600.f,0.0005f);

            if (true)
            {
                atm->SetOutputScale(0.04f);
                atm->EnableHDR(true);
            }
            else
                atm->EnableHDR(false);

            atm->DisableFarCulling(true);

            std::ifstream ifs("savedsky7.atm", std::ifstream::in);
            if (ifs.is_open())
                atm->Unserialize(ifs);
            else
            {
                // Set up all the clouds
                SetupAtmosphericConditions();

                // Configure where and when we want to be
                SetTimeAndLocation();
            }
        } else {
            printf("Error was %d\n", err);
        }

            executeGLDiff(preState^COpenGLState::collectGLState());
    }

    //! Little tutorial on Render Target Rendering
    uint32_t texSize[] = {params.WindowSize.Width/2,params.WindowSize.Height/2};
    video::IFrameBuffer* fbo = driver->addFrameBuffer();
    video::ITexture* tex = device->getVideoDriver()->addTexture(ITexture::ETT_2D,texSize,1,"depth_attach",ECF_DEPTH32F);
    fbo->attach(EFAP_DEPTH_ATTACHMENT,tex);
    tex = device->getVideoDriver()->addTexture(ITexture::ETT_2D,texSize,1,"color_attach",ECF_A16B16G16R16F);
    fbo->attach(EFAP_COLOR_ATTACHMENT0,tex);

    atm->SetViewport(0,0,tex->getSize()[0],tex->getSize()[1]);


	uint64_t lastFPSTime = 0;

	uint64_t lastTime = device->getTimer()->getRealTime();
    uint64_t timeDelta = 0;


	while (device->run())
	{
	    atm->UpdateEphemeris(); // to get new sun position

		driver->beginScene(false, false, video::SColor(0,0,0,0) );

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

        driver->setRenderTarget(fbo,true);
        driver->clearZBuffer();
        float color[4] = {0.f,0.f,0.f,0.f};
        driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT0,color);

        atm->UpdateSkyAndClouds(); // do this inbetween queries!!

        //! This animates (moves) the camera and sets the transforms
        //! Also draws the meshbuffer
        smgr->drawAll();
        camPos = camera->getAbsolutePosition();
        {
            double mv[16];
            for (size_t i=0; i<4; i++)
            {
                for (size_t j=0; j<3; j++)
                    ((double*)mv)[i*4+j] = reinterpret_cast<const float*>(&camera->getViewMatrix().getColumn(i))[j];
                ((double*)mv)[i*4+3] = i!=3 ? 0.f:1.f;
            }
            double proj[16];
            for (size_t i=0; i<4; i++)
            for (size_t j=0; j<4; j++)
                proj[i*4+j] = camera->getProjectionMatrix().pointer()[i*4+j];

            //atm stuff after this get done twice for reflections
            atm->SetCameraMatrix((double*)mv);
            atm->SetProjectionMatrix((double*)proj);

            core::matrix4 mpv_irr(driver->getTransform(video::EPTS_PROJ_VIEW));
            double mvp[16];
            for (size_t i=0; i<4; i++)
            for (size_t j=0; j<4; j++)
                mvp[i+j*4] = mpv_irr.pointer()[i*4+j];

            SilverLining::Frustum f;
            SilverLining::Matrix4(mvp).GetFrustum(f);
            atm->CullObjects(f,false);


            COpenGLState preState = COpenGLState::collectGLState();
            COpenGLState silverState;
            {
                // set pixel store unpack align 4 (implicit)
                silverState.glBindProgramPipeline_val = 0;
                silverState.setGlEnableBit(EGEB_MULTISAMPLE,false);
                silverState.setGlEnableBit(EGEB_DEPTH_CLAMP,true);
                silverState.setGlEnableBit(EGEB_DEPTH_TEST,true);
                silverState.setGlEnableBit(EGEB_CULL_FACE,true);
                for (uint32_t i=0; i<driver->MaxMultipleRenderTargets; i++)
                    silverState.setGlEnableiBit(EGEIB_BLEND,i,false);

                silverState.glFrontFace_val = GL_CW;
                silverState.glDepthFunc_val = GL_GEQUAL;
                silverState.glDepthMask_val = false;
                silverState.boundVAO = 0;
            }
            executeGLDiff(silverState.getStateDiff(preState,
                                                    false,  //careAboutHints
                                                    false,  //careAboutFBOs
                                                    false,  //careAboutPolygonOffset
                                                    true,   //careAboutPixelXferOps
                                                    false,  //careAboutSSBOAndAtomicCounters
                                                    true,   //careAboutXFormFeedback
                                                    true,   //careAboutProgram
                                                    false,  //careAboutPipeline
                                                    false,  //careAboutTesellationParams
                                                    false,  //careAboutViewports
                                                    true,   //careAboutDrawIndirectBuffers
                                                    false,  //careAboutPointSize
                                                    false,  //careAboutLineWidth
                                                    false,  //careAboutLogicOp
                                                    false,  //careAboutMultisampling -- CAN CHANGE IN THE FUTURE
                                                    true,   //careAboutBlending
                                                    true,   //careAboutColorWriteMasks
                                                    false,  //careAboutStencilFunc
                                                    false,  //careAboutStencilOp
                                                    true,   //careAboutStencilMask
                                                    true,   //careAboutDepthFunc
                                                    true,   //careAboutDepthMask
                                                    false,  //careAboutImages
                                                    true,   //careAboutTextures
                                                    true,   //careAboutFaceOrientOrCull
                                                    true)); //careAboutVAO


            atm->DrawSky(true,false,0,true,false,true,0,-2.f,0);
            // When you're done, call Atmosphere::DrawObjects() to draw all the clouds from back to front.
            atm->DrawObjects(true,true,true,0.f,false);
            atm->DrawLensFlare();

            executeGLDiff(preState.getStateDiff(COpenGLState::collectGLState(),
                                                    false,  //careAboutHints
                                                    false,  //careAboutFBOs
                                                    false,  //careAboutPolygonOffset
                                                    true,   //careAboutPixelXferOps
                                                    false,  //careAboutSSBOAndAtomicCounters
                                                    false,   //careAboutXFormFeedback
                                                    true,   //careAboutProgram
                                                    false,  //careAboutPipeline
                                                    false,  //careAboutTesellationParams
                                                    false,  //careAboutViewports
                                                    false,   //careAboutDrawIndirectBuffers
                                                    false,  //careAboutPointSize
                                                    false,  //careAboutLineWidth
                                                    false,  //careAboutLogicOp
                                                    false,  //careAboutMultisampling -- CAN CHANGE IN THE FUTURE
                                                    true,   //careAboutBlending
                                                    true,   //careAboutColorWriteMasks
                                                    false,  //careAboutStencilFunc
                                                    false,  //careAboutStencilOp
                                                    true,   //careAboutStencilMask
                                                    true,   //careAboutDepthFunc
                                                    true,   //careAboutDepthMask
                                                    false,  //careAboutImages
                                                    true,   //careAboutTextures
                                                    true,   //careAboutFaceOrientOrCull
                                                    true)); //careAboutVAO
        }

        driver->setRenderTarget(0,true);

	///glEnable(GL_FRAMEBUFFER_SRGB);
        driver->blitRenderTargets(fbo,0,false,core::recti(0,0,0,0),core::recti(0,0,0,0),true);
	///glDisable(GL_FRAMEBUFFER_SRGB);

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Silverlining Integration Demo - Irrlicht Engine FPS:" << driver->getFPS() << " PrimitvesDrawn:";
			str << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str());
			lastFPSTime = time;
		}
	}

	driver->removeFrameBuffer(fbo);

    if (spline)
        delete spline;

    std::ofstream ofs("savedsky.atm", std::ofstream::out|std::ofstream::trunc);
    if (ofs.is_open())
    {
        atm->Serialize(ofs);
        ofs.close();
    }

    delete atm;
    delete silverLoader;

    //create a screenshot
	video::IImage* screenshot = driver->createImage(video::ECF_A8R8G8B8,params.WindowSize);
    glReadPixels(0,0, params.WindowSize.Width,params.WindowSize.Height, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, screenshot->getData());
    {
        // images are horizontally flipped, so we have to fix that here.
        uint8_t* pixels = (uint8_t*)screenshot->getData();

        const int32_t pitch=screenshot->getPitch();
        uint8_t* p2 = pixels + (params.WindowSize.Height - 1) * pitch;
        uint8_t* tmpBuffer = new uint8_t[pitch];
        for (uint32_t i=0; i < params.WindowSize.Height; i += 2)
        {
            memcpy(tmpBuffer, pixels, pitch);
            memcpy(pixels, p2, pitch);
            memcpy(p2, tmpBuffer, pitch);
            pixels += pitch;
            p2 -= pitch;
        }
        delete [] tmpBuffer;
    }
	driver->writeImageToFile(screenshot,"./screenshot.png");
	screenshot->drop();

	device->drop();

	return 0;
}
