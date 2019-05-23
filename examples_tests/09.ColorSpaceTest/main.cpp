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

void presentImageOnScreen(IrrlichtDevice* device, video::IGPUMeshBuffer* fullScreenTriangle, video::ITexture* inTex, video::ITexture* outTex=nullptr)
{
	video::IVideoDriver* driver = device->getVideoDriver();

	auto framebuffer = driver->addFrameBuffer();
	if (outTex)
		framebuffer->attach(video::EFAP_COLOR_ATTACHMENT0,outTex,0u);

	presentMaterial.setTexture(0u,inTex);

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
	reqs.vulkanReqs.size = tex->getSize()[1]*tex->getPitch();
	reqs.vulkanReqs.alignment = 64u;
	reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
	reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL;
	reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ|video::IDriverMemoryAllocation::EMCF_COHERENT|video::IDriverMemoryAllocation::EMCF_CACHED;
	auto buffer = driver->createGPUBufferOnDedMem(reqs);

	auto fence = ext::ScreenShot::createScreenShot(driver,tex,buffer);
	while (fence->waitCPU(1000ull,fence->canDeferredFlush())==video::EDFR_TIMEOUT_EXPIRED) {}
	fence->drop();

	auto alloc = buffer->getBoundMemory();
	alloc->mapMemoryRange(video::IDriverMemoryAllocation::EMCAF_READ,{0u,reqs.vulkanReqs.size});
	uint32_t minCoord[3] = {0u,0u,0u};
	uint32_t maxCoord[3] = {tex->getSize()[0],tex->getSize()[1],1u};
	asset::CImageData* img = new asset::CImageData(alloc->getMappedPointer(),minCoord,maxCoord,0u,tex->getColorFormat(),1u);
	buffer->drop();

	asset::IAssetWriter::SAssetWriteParams wparams(img);
	device->getAssetManager().writeAsset(outname, wparams);
	img->drop();
}

void testImage(const std::string& path, IrrlichtDevice* device, video::IGPUMeshBuffer* fullScreenTriangle)
{
	os::Printer::log("Reading", path);
	
	auto& assetMgr = device->getAssetManager();
	auto* driver = device->getVideoDriver();

	asset::ICPUTexture* cputex[1] = { static_cast<asset::ICPUTexture*>(assetMgr.getAsset(path, {})) };
	
	if (cputex[0])
	{
		io::path filename, extension;
		core::splitFilename(path.c_str(), nullptr, &filename, &extension);
		filename += "."; filename += extension;
		
		bool writeable = (extension != "dds") && (extension != "bmp");
		
		auto tex = driver->getGPUObjectsFromAssets(cputex,cputex+1u).front();
		{
			auto tmpTex = driver->createGPUTexture(video::ITexture::ETT_2D,tex->getSize(),1u,tex->getColorFormat());
			presentImageOnScreen(device, fullScreenTriangle, tex, tmpTex);
			
			if (writeable)
				dumpTextureToFile(device, tex, (io::path("screen_")+filename).c_str());
			
			tmpTex->drop();
		}
		
		if (writeable)
		{
			asset::CImageData* img = *cputex[0]->getMipMap(0u).first;
			asset::IAssetWriter::SAssetWriteParams wparams(img);

			device->getAssetManager().writeAsset((io::path("write_")+filename).c_str(), wparams);
		}
		assetMgr.removeAssetFromCache(cputex[0]);
		assetMgr.removeCachedGPUObject(cputex[0],tex);
	}
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

	video::IGPUMeshBuffer* fullScreenTriangle = ext::FullScreenTriangle::createFullScreenTriangle(driver);
	
	//! First need to make a material other than default to be able to draw with custom shader
	presentMaterial.BackfaceCulling = false; //! Triangles will be visible from both sides
	presentMaterial.ZBuffer = video::ECFN_ALWAYS; //! Ignore Depth Test
	presentMaterial.ZWriteEnable = false; //! Why even write depth?
	presentMaterial.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles(
																"../fullscreentri.vert","","","","../present.frag",3,video::EMT_SOLID);
	
	std::ifstream list("./testlist.txt");
	if (list.is_open())
	{
                std::string line;
                for (; std::getline(list, line); )
                {
                        if(line != "" && line[0] != ';')
                                testImage(line, device, fullScreenTriangle);
                }
	}
	
	fullScreenTriangle->drop();
	device->drop();

	return 0;
}

