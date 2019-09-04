#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include <iostream>
#include <cstdio>
#include <fstream>

#include "../../ext/FullScreenTriangle/FullScreenTriangle.h"
#include "../../ext/ScreenShot/ScreenShot.h"


using namespace irr;
using namespace core;

video::SGPUMaterial presentMaterial;

void presentImageOnScreen(IrrlichtDevice* device, video::IGPUMeshBuffer* fullScreenTriangle, core::smart_refctd_ptr<video::ITexture>&& inTex, core::smart_refctd_ptr<video::ITexture>&& outTex=nullptr)
{
	video::IVideoDriver* driver = device->getVideoDriver();

	auto framebuffer = driver->addFrameBuffer();
	if (outTex)
		framebuffer->attach(video::EFAP_COLOR_ATTACHMENT0,outTex.get(),0u);

	presentMaterial.setTexture(0u,std::move(inTex));

	device->run();
	driver->beginScene(true,true);
	// so we have something to download
	if (outTex)
	{
		driver->setRenderTarget(framebuffer);
		driver->setMaterial(presentMaterial);
		driver->drawMeshBuffer(fullScreenTriangle);
		driver->setRenderTarget(nullptr,false);
	}
	//
	driver->setMaterial(presentMaterial);
	driver->drawMeshBuffer(fullScreenTriangle);
	//
	driver->endScene();

	driver->removeFrameBuffer(framebuffer);
}

void dumpTextureToFile(IrrlichtDevice* device, video::ITexture* tex, const std::string& outname)
{
	video::IVideoDriver* driver = device->getVideoDriver();

	video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
	reqs.vulkanReqs.size = (tex->getSize()[1]*tex->getPitch()).getIntegerApprox();
	reqs.vulkanReqs.alignment = 64u;
	reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
	reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL;
	reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ|video::IDriverMemoryAllocation::EMCF_COHERENT|video::IDriverMemoryAllocation::EMCF_CACHED;
	auto buffer = driver->createGPUBufferOnDedMem(reqs);

	auto fence = ext::ScreenShot::createScreenShot(driver,tex,buffer);
	while (fence->waitCPU(1000ull,fence->canDeferredFlush())==video::EDFR_TIMEOUT_EXPIRED) {}

	auto alloc = buffer->getBoundMemory();
	alloc->mapMemoryRange(video::IDriverMemoryAllocation::EMCAF_READ,{0u,reqs.vulkanReqs.size});
	uint32_t minCoord[3] = {0u,0u,0u};
	uint32_t maxCoord[3] = {tex->getSize()[0],tex->getSize()[1],1u};
	auto img = core::make_smart_refctd_ptr<asset::CImageData>(alloc->getMappedPointer(),minCoord,maxCoord,0u,tex->getColorFormat(),1u);
	buffer->drop();

	asset::IAssetWriter::SAssetWriteParams wparams(img.get());
	device->getAssetManager()->writeAsset(outname, wparams);
}

void testImage(const std::string& path, IrrlichtDevice* device, video::IGPUMeshBuffer* fullScreenTriangle)
{
	os::Printer::log("Reading", path);
	
	auto* assetMgr = device->getAssetManager();
	auto* driver = device->getVideoDriver();

	auto cputex = assetMgr->getAsset(path, {});
	
	if (cputex.getContents().first!=cputex.getContents().second)
	{
		io::path filename, extension;
		core::splitFilename(path.c_str(), nullptr, &filename, &extension);
		filename += "."; filename += extension;
		
		bool writeable = (extension != "dds");
		
		auto actualcputex = core::smart_refctd_ptr_static_cast<asset::ICPUTexture>(*cputex.getContents().first);
		auto tex = driver->getGPUObjectsFromAssets(&actualcputex.get(),&actualcputex.get()+1u)->front();
		if (tex)
		{
			auto tmpTex = driver->createGPUTexture(video::ITexture::ETT_2D,tex->getSize(),1u,tex->getColorFormat());
			presentImageOnScreen(device, fullScreenTriangle, core::smart_refctd_ptr(tex), std::move(tmpTex));
			
			if (writeable)
				dumpTextureToFile(device, tex.get(), (io::path("screen_")+filename).c_str());
		}
		
		if (writeable)
		{
			asset::CImageData* img = actualcputex->getMipMap(0u).first[0];
			asset::IAssetWriter::SAssetWriteParams wparams(img);

			assetMgr->writeAsset((io::path("write_")+filename).c_str(), wparams);
		}
		assetMgr->removeAssetFromCache(cputex);
		assetMgr->removeCachedGPUObject(actualcputex.get(),tex);
	}
	else
		std::cout << "ERROR: CANNOT LOAD FILE!" << std::endl;
}

int main()
{
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<uint32_t>(1600, 900);
	params.Fullscreen = false;

	IrrlichtDevice* device = createDeviceEx(params);
	if (device == 0)
		return 1; // could not create selected driver.

	video::IVideoDriver* driver = device->getVideoDriver();

	auto fullScreenTriangle = ext::FullScreenTriangle::createFullScreenTriangle(driver);
	
	//! First need to make a material other than default to be able to draw with custom shader
	presentMaterial.BackfaceCulling = false; //! Triangles will be visible from both sides
	presentMaterial.ZBuffer = video::ECFN_ALWAYS; //! Ignore Depth Test
	presentMaterial.ZWriteEnable = false; //! Why even write depth?
	presentMaterial.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles(
																"../fullscreentri.vert","","","","../present.frag",3,video::EMT_SOLID);
	
	std::ifstream list("../testlist.txt");
	if (list.is_open())
	{
                std::string line;
                for (; std::getline(list, line); )
                {
                        if(line != "" && line[0] != ';')
                                testImage(line, device, fullScreenTriangle.get());
                }
	}
	device->drop();

	return 0;
}

