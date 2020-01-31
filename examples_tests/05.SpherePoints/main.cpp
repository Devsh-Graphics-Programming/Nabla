#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include "../../ext/ScreenShot/ScreenShot.h"
#include "../common/QToQuitEventReceiver.h"

using namespace irr;
using namespace core;



class SimpleCallBack : public video::IShaderConstantSetCallBack
{
    int32_t mvpUniformLocation;
    video::E_SHADER_CONSTANT_TYPE mvpUniformType;
public:
    SimpleCallBack() : mvpUniformLocation(-1), mvpUniformType(video::ESCT_FLOAT_VEC3) {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
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


/*
The start of the main function starts like in most other example. We ask the
user for the desired renderer and start it up.
*/
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
	params.Vsync = false;
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	video::IVideoDriver* driver = device->getVideoDriver();
	SimpleCallBack* callBack = new SimpleCallBack();

    //! First need to make a material other than default to be able to draw with custom shader
    video::SGPUMaterial material;
    //material.BackfaceCulling = false; //! Triangles will be visible from both sides
    material.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../points.vert",
                                                        "","","", //! No Geometry or Tessellation Shaders
                                                        "../points.frag",
                                                        3,video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
                                                        callBack, //! No Shader Callback (we dont have any constants/uniforms to pass to the shader)
                                                        0); //! No custom user data
    callBack->drop();



	scene::ISceneManager* smgr = device->getSceneManager();


    video::IGPUMeshBuffer* mb = new video::IGPUMeshBuffer();
    auto desc = driver->createGPUMeshDataFormatDesc();

    size_t xComps = 0x1u<<9;
    size_t yComps = 0x1u<<9;
    size_t zComps = 0x1u<<9;
    size_t verts = xComps*yComps*zComps;
    uint32_t bufSize = verts*sizeof(uint32_t);
    uint32_t* mem = (uint32_t*)malloc(bufSize);
    for (size_t i=0; i<xComps; i++)
    for (size_t j=0; j<yComps; j++)
    for (size_t k=0; k<zComps; k++)
    {
        mem[i+xComps*(j+yComps*k)] = (i<<20)|(j<<10)|(k);
    }



    //! By mapping we increase/grab() ref counter of positionBuf, any previously mapped buffer will have it's reference dropped
    desc->setVertexAttrBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(driver->createFilledDeviceLocalGPUBufferOnDedMem(bufSize,mem),core::dont_grab), // TODO: fix with std::move soon
        asset::EVAI_ATTR0, //! we use first attribute slot (out of a minimum of 16)
        asset::EF_A2B10G10R10_SSCALED_PACK32); //! there are 3 components per vertex and they are floats
	free(mem);
	mb->setMeshDataAndFormat(std::move(desc));


    mb->setIndexCount(verts);
    mb->setPrimitiveType(asset::EPT_POINTS);

    scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0,80.f,0.001f);
    smgr->setActiveCamera(camera);
    camera->setNearValue(0.001f);
    camera->setFarValue(10.f);
    device->getCursorControl()->setVisible(false);

	uint64_t lastFPSTime = 0;

	while (device->run() && receiver.keepOpen())
	if (device->isWindowActive())
	{
		driver->beginScene(true, true, video::SColor(255,0,0,255) );

        smgr->drawAll();

        driver->setTransform(video::E4X3TS_WORLD,core::matrix3x4SIMD());
        driver->setMaterial(material);
        //! draw back to front
        driver->drawMeshBuffer(mb);

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Sphere Points - Irrlicht Engine  FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str().c_str());
			lastFPSTime = time;
		}
	}
	mb->drop();

	//create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		ext::ScreenShot::dirtyCPUStallingScreenshot(device, "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}

	device->drop();

	return 0;
}
