#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include <iostream>
#include <cstdio>


#include "../ext/LumaMeter/CLumaMeter.h"
#include "../source/Irrlicht/COpenGLDriver.h"

#include "../common/QToQuitEventReceiver.h"

using namespace irr;
using namespace irr::core;
using namespace irr::asset;
using namespace irr::video;


int main()
{
	irr::SIrrlichtCreationParameters deviceParams;
	deviceParams.Bits = 24; //may have to set to 32bit for some platforms
	deviceParams.ZBufferBits = 24; //we'd like 32bit here
	deviceParams.DriverType = EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	deviceParams.WindowSize = dimension2d<uint32_t>(1280, 720);
	deviceParams.Fullscreen = false;
	deviceParams.Vsync = true; //! If supported by target platform
	deviceParams.Doublebuffer = true;
	deviceParams.Stencilbuffer = false; //! This will not even be a choice soon

	auto device = createDeviceEx(deviceParams);
	if (!device)
		return 1; // could not create selected driver.

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	IVideoDriver* driver = device->getVideoDriver();
	
	irr::io::IFileSystem* filesystem = device->getFileSystem();
	IAssetManager* am = device->getAssetManager();

	IAssetLoader::SAssetLoadParams lp;
	auto imageBundle = am->getAsset("../../media/OpenEXR/56_render_0_2_256.exr", lp);

	smart_refctd_ptr<IGPUImageView> imgToTonemap,outImg,outImgStorage;
	{
		auto cpuImg = IAsset::castDown<ICPUImage>(imageBundle.getContents().first[0]);
		IGPUImage::SCreationParams imgInfo = cpuImg->getCreationParameters();

		auto gpuImages = driver->getGPUObjectsFromAssets(&cpuImg.get(),&cpuImg.get()+1);
		auto gpuImage = gpuImages->operator[](0u);

		IGPUImageView::SCreationParams imgViewInfo;
		imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		imgViewInfo.image = std::move(gpuImage);
		imgViewInfo.viewType = IGPUImageView::ET_2D_ARRAY;
		imgViewInfo.format = imgInfo.format;
		imgViewInfo.subresourceRange.baseMipLevel = 0;
		imgViewInfo.subresourceRange.levelCount = 1;
		imgViewInfo.subresourceRange.baseArrayLayer = 0;
		imgViewInfo.subresourceRange.layerCount = 1;
		imgToTonemap = driver->createGPUImageView(IGPUImageView::SCreationParams(imgViewInfo));

		imgInfo.format = EF_R8G8B8A8_SRGB;
		imgViewInfo.image = driver->createDeviceLocalGPUImageOnDedMem(std::move(imgInfo));
		imgViewInfo.format = EF_R8G8B8A8_SRGB;
		outImg = driver->createGPUImageView(IGPUImageView::SCreationParams(imgViewInfo));

		imgViewInfo.format = EF_R32_UINT;
		outImgStorage = driver->createGPUImageView(std::move(imgViewInfo));
	}

// TODO: redo
/**
REQUIREMENTS:
- Input from storage image or buffer
- Output to storage image or buffer
- Flexible input and output formats 
- Flexible color semantics
- ACES and Reinhard tonemapping modes
- Automatic exposure metering or given
- Flexible SSBO/UBO data sourcing
- Mergable tonemapping shader (rippable from dedicated CS shader)
- Exposure measuring modes MEAN vs. MODE
- Overridable Tonemapping Parameter preparation (for OptiX and stuff)


CToneMapper:
- take input parameters (different headers for each struct, comes in via override)
- load input (override) + transform to XYZ
- do tonemap (different headers for that)
- transform from XYZ and write output (override)
- a facotry method that makes default compute pipelines
- a factory method for a default combined autoexposure + tonemapping with/without temporal
- add autoexposure and temporal adjustment somehow
**/
#if 0
	auto tonemapper = ext::AutoExposure::CToneMapper::create(driver,imgToTonemap->getCreationParameters().format,am->getGLSLCompiler());

	// TODO: employ luma histograming
	auto params = ext::AutoExposure::ReinhardParams::fromKeyAndBurn(0.18, 0.95, 8.0, 16.0);
	auto parameterBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(ext::AutoExposure::ReinhardParams),&params);

	auto descriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>(tonemapper->getDescriptorSetLayout()));
	{
		video::IGPUDescriptorSet::SDescriptorInfo pInfos[3];
		pInfos[0].desc = parameterBuffer;
		pInfos[0].buffer.offset = 0u;
		pInfos[0].buffer.size = video::IGPUBufferView::whole_buffer;
		pInfos[1].desc = imgToTonemap;
		pInfos[1].image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);
		using S = asset::ISampler;
		pInfos[1].image.sampler = driver->createGPUSampler({   {S::ETC_CLAMP_TO_EDGE,S::ETC_CLAMP_TO_EDGE,S::ETC_CLAMP_TO_EDGE,
																S::ETBC_FLOAT_OPAQUE_BLACK,
																S::ETF_NEAREST,S::ETF_NEAREST,S::ESMM_NEAREST,0u,
																false,asset::ECO_ALWAYS},
															0.f,-FLT_MAX,FLT_MAX});
		pInfos[2].desc = outImgStorage;
		pInfos[2].image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);
		pInfos[2].image.sampler = nullptr;

		video::IGPUDescriptorSet::SWriteDescriptorSet pWrites[3];
		pWrites[0].dstSet = descriptorSet.get();
		pWrites[0].binding = 0u;
		pWrites[0].arrayElement = 0u;
		pWrites[0].count = 1u;
		pWrites[0].descriptorType = asset::EDT_UNIFORM_BUFFER_DYNAMIC;
		pWrites[0].info = pInfos+0u;
		pWrites[1].dstSet = descriptorSet.get();
		pWrites[1].binding = 1u;
		pWrites[1].arrayElement = 0u;
		pWrites[1].count = 1u;
		pWrites[1].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
		pWrites[1].info = pInfos+1u;
		pWrites[2].dstSet = descriptorSet.get();
		pWrites[2].binding = 2u;
		pWrites[2].arrayElement = 0u;
		pWrites[2].count = 1u;
		pWrites[2].descriptorType = asset::EDT_STORAGE_IMAGE;
		pWrites[2].info = pInfos+2u;
		driver->updateDescriptorSets(3u,pWrites,0u,nullptr);
	}

	auto blitFBO = driver->addFrameBuffer();
	blitFBO->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(outImg));

	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(false, false);

		tonemapper->tonemap(imgToTonemap.get(), descriptorSet.get(), 0u);

		COpenGLExtensionHandler::extGlMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
		driver->blitRenderTargets(blitFBO, nullptr, false, false);

		driver->endScene();
	}
	// TODO: Histogramming
	//toneMapper->CalculateFrameExposureFactors(frameUniformBuffer,frameUniformBuffer,core::smart_refctd_ptr(hdrTex));
#endif

	return 0;
}