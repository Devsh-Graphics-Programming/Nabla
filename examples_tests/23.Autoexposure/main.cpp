#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include <iostream>
#include <cstdio>


#include "../ext/ToneMapper/CToneMapper.h"
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

	E_FORMAT inFormat;
	constexpr auto outFormat = EF_R8G8B8A8_SRGB;
	smart_refctd_ptr<IGPUImageView> imgToTonemap,outImg;
	{
		auto cpuImg = IAsset::castDown<ICPUImage>(imageBundle.getContents().first[0]);
		IGPUImage::SCreationParams imgInfo = cpuImg->getCreationParameters();
		inFormat = imgInfo.format;

		auto gpuImages = driver->getGPUObjectsFromAssets(&cpuImg.get(),&cpuImg.get()+1);
		auto gpuImage = gpuImages->operator[](0u);

		IGPUImageView::SCreationParams imgViewInfo;
		imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		imgViewInfo.image = std::move(gpuImage);
		imgViewInfo.viewType = IGPUImageView::ET_2D_ARRAY;
		imgViewInfo.format = inFormat;
		imgViewInfo.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u);
		imgViewInfo.subresourceRange.baseMipLevel = 0;
		imgViewInfo.subresourceRange.levelCount = 1;
		imgViewInfo.subresourceRange.baseArrayLayer = 0;
		imgViewInfo.subresourceRange.layerCount = 1;
		imgToTonemap = driver->createGPUImageView(IGPUImageView::SCreationParams(imgViewInfo));

		imgInfo.format = outFormat;
		imgViewInfo.image = driver->createDeviceLocalGPUImageOnDedMem(std::move(imgInfo));
		imgViewInfo.format = outFormat;
		outImg = driver->createGPUImageView(IGPUImageView::SCreationParams(imgViewInfo));
	}


	constexpr bool usingLumaMeter = false;
	constexpr auto meterMode = ext::LumaMeter::CLumaMeter::EMM_COUNT;

	constexpr bool usingTemporalAdapatation = false;

	auto tonemappingShader = ext::ToneMapper::CToneMapper::createShader(am->getGLSLCompiler(),
		std::make_tuple(inFormat,ECP_SRGB,EOTF_IDENTITY),
		std::make_tuple(outFormat,ECP_SRGB,OETF_sRGB),
		ext::ToneMapper::CToneMapper::EO_REINHARD,
		usingLumaMeter,meterMode,usingTemporalAdapatation
	);

	auto outImgStorage = ext::ToneMapper::CToneMapper::createViewForImage(driver,false,core::smart_refctd_ptr(outImg),{static_cast<IImage::E_ASPECT_FLAGS>(0u),0,1,0,1});

	constexpr float Key = 0.18;
	auto params = ext::ToneMapper::CToneMapper::ReinhardParams_t::fromKeyAndBurn(Key, 0.95, 0.1, 16.0);
/*
TODO:
- tone mapper double buffered parameters
- tone mapper luma adaptation
- descriptor sets final
- ACES trials
- adaptation speeds
*/
#if 0
	auto parameterBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(ext::AutoExposure::ReinhardParams),&params);
#endif

	auto commonPipelineLayout = ext::ToneMapper::CToneMapper::getDefaultPipelineLayout(driver,usingLumaMeter);

	auto tonemappingPipeline = driver->createGPUComputePipeline(nullptr,std::move(commonPipelineLayout),std::move(tonemappingShader));

	auto dynamicOffsetArray = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint32_t> >();
	auto commonDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(commonPipelineLayout->getDescriptorSetLayout(0u)));
	if constexpr (false)
	{
		video::IGPUDescriptorSet::SDescriptorInfo pInfos[3];
		pInfos[0].desc = parameterBuffer;
		pInfos[0].buffer.offset = 0u;
		pInfos[0].buffer.size = video::IGPUBufferView::whole_buffer;
		pInfos[1].desc = imgToTonemap;
		pInfos[1].image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);
		pInfos[1].image.sampler = nullptr;
		pInfos[2].desc = outImgStorage;
		pInfos[2].image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);
		pInfos[2].image.sampler = nullptr;

		video::IGPUDescriptorSet::SWriteDescriptorSet pWrites[3];
		pWrites[0].dstSet = commonDescriptorSet.get();
		pWrites[0].binding = 0u;
		pWrites[0].arrayElement = 0u;
		pWrites[0].count = 1u;
		pWrites[0].descriptorType = asset::EDT_UNIFORM_BUFFER_DYNAMIC;
		pWrites[0].info = pInfos + 0u;
		pWrites[1].dstSet = commonDescriptorSet.get();
		pWrites[1].binding = 1u;
		pWrites[1].arrayElement = 0u;
		pWrites[1].count = 1u;
		pWrites[1].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
		pWrites[1].info = pInfos + 1u;
		pWrites[2].dstSet = commonDescriptorSet.get();
		pWrites[2].binding = 2u;
		pWrites[2].arrayElement = 0u;
		pWrites[2].count = 1u;
		pWrites[2].descriptorType = asset::EDT_STORAGE_IMAGE;
		pWrites[2].info = pInfos + 2u;
		driver->updateDescriptorSets(3u, pWrites, 0u, nullptr);
	}


	auto blitFBO = driver->addFrameBuffer();
	blitFBO->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(outImg));


	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(false, false);

		driver->bindComputePipeline(tonemappingPipeline.get());
		driver->bindDescriptorSets(EPBP_COMPUTE,tonemappingPipeline->getLayout(),0u,1u,&commonDescriptorSet.get(),&dynamicOffsetArray.get());
		ext::ToneMapper::CToneMapper::dispatchHelper(driver,outImgStorage.get(),true);

		driver->blitRenderTargets(blitFBO, nullptr, false, false);

		driver->endScene();
	}

	return 0;
}