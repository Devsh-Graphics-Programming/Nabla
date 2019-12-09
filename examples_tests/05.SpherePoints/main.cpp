#define _IRR_STATIC_LIB_
#include <irrlicht.h>
//#include "../../ext/ScreenShot/ScreenShot.h"
#include "../common/QToQuitEventReceiver.h"

using namespace irr;
using namespace core;
using namespace asset;
using namespace video;


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
	auto device = createDeviceEx(params);

	if (!device)
		return 1; // could not create selected driver.

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	auto assetMgr = device->getAssetManager();
	auto driver = device->getVideoDriver();

	auto loadShader = [&](const char* path) -> core::smart_refctd_ptr<IGPUSpecializedShader>
	{
		auto shaderSource = device->getAssetManager()->getAsset(path, {});
		auto contents = shaderSource.getContents();
		if (contents.first==contents.second)
			return nullptr;
		if (contents.first[0]->getAssetType()!=IAsset::ET_SPECIALIZED_SHADER)
			return nullptr;

		auto cpuSS = static_cast<ICPUSpecializedShader*>(contents.first->get());
		return driver->getGPUObjectsFromAssets(&cpuSS,&cpuSS+1u)->operator[](0);
	};

	SPushConstantRange range[1] = {asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD)};
	auto pLayout = driver->createGPUPipelineLayout(range,range+1u,nullptr,nullptr,nullptr,nullptr);
		
	core::smart_refctd_ptr<IGPUSpecializedShader> shaders[2] = {loadShader("../points.vert"),loadShader("../points.frag")};
	auto shadersPtr = reinterpret_cast<IGPUSpecializedShader**>(shaders);

	SVertexInputParams inputParams;
	inputParams.enabledAttribFlags = 0x1u;
	inputParams.enabledBindingFlags = 0x1u;
	inputParams.attributes[0].binding = 0u;
	inputParams.attributes[0].format = EF_A2B10G10R10_SSCALED_PACK32;
	inputParams.attributes[0].relativeOffset = 0u;
	inputParams.bindings[0].stride = sizeof(uint32_t);
	inputParams.bindings[0].inputRate = EVIR_PER_VERTEX;

	SBlendParams blendParams; // default

	SPrimitiveAssemblyParams assemblyParams = {EPT_POINT_LIST,false,1u};

	SRasterizationParams rasterParams;
	rasterParams.faceCullingMode = EFCM_NONE;
	rasterParams.depthCompareOp = ECO_ALWAYS;
	rasterParams.depthTestEnable = false;
	rasterParams.depthWriteEnable = false;

	auto pipeline = driver->createGPURenderpassIndependentPipeline(	nullptr,std::move(pLayout),shadersPtr,shadersPtr+sizeof(shaders)/sizeof(core::smart_refctd_ptr<IGPUSpecializedShader>),
																			inputParams,blendParams,assemblyParams,rasterParams);

    size_t xComps = 0x1u<<9;
    size_t yComps = 0x1u<<9;
    size_t zComps = 0x1u<<9;
    size_t verts = xComps*yComps*zComps;
    uint32_t bufSize = verts*static_cast<uint32_t>(sizeof(uint32_t));
    uint32_t* mem = (uint32_t*)malloc(bufSize);
    for (size_t i=0; i<xComps; i++)
    for (size_t j=0; j<yComps; j++)
    for (size_t k=0; k<zComps; k++)
    {
        mem[i+xComps*(j+yComps*k)] = (i<<20)|(j<<10)|(k);
    }
	
	SBufferBinding<IGPUBuffer> bindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
	bindings[0u] = {0u,driver->createFilledDeviceLocalGPUBufferOnDedMem(bufSize,mem)};
	auto mb = core::make_smart_refctd_ptr<IGPUMeshBuffer>(core::smart_refctd_ptr(pipeline),nullptr,bindings,SBufferBinding<IGPUBuffer>{0u,nullptr});
	free(mem);

    mb->setIndexCount(verts);


	auto smgr = device->getSceneManager();

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

		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();
		core::matrix4SIMD mvp = camera->getConcatenatedMatrix();

		driver->bindGraphicsPipeline(pipeline.get());
		driver->pushConstants(pipeline->getLayout(),asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD),mvp.pointer());
        driver->drawMeshBuffer(mb.get());

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

	//create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		//ext::ScreenShot::dirtyCPUStallingScreenshot(device, "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}

	return 0;
}
