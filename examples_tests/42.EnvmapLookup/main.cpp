#include <irrlicht.h>

#include "../../ext/FullScreenTriangle/FullScreenTriangle.h"
#include "../common/QToQuitEventReceiver.h"
#include "../../ext/ScreenShot/ScreenShot.h"

using namespace irr;
using namespace core;
using namespace asset;
using namespace video;

// Modification of
//
// No Geometry 360 Video
// By KylBlz
// https://www.shadertoy.com/view/Ml33z2


constexpr std::string_view FRAGMENT_SHADER_GLSL = // TODO shader sampling envmap properly
R"(
#version 430 core

layout(set = 3, binding = 0) uniform sampler2D envMap; 
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 pixelColor;

void main()
{
    vec3 hdrColor = texture(envMap, uv).rgb;
  
    // reinhard tone mapping
    vec3 mapped = hdrColor / (hdrColor + vec3(1.0));
  
    pixelColor = vec4(mapped, 1.0);
}	
 )";

irr::video::IFrameBuffer* createHDRFramebuffer(core::smart_refctd_ptr<IrrlichtDevice> device, asset::E_FORMAT colorFormat)
{
	auto driver = device->getVideoDriver();

	auto createAttachement = [&](bool colorBuffer)
	{
		asset::ICPUImage::SCreationParams imgInfo;
		imgInfo.format = colorBuffer ? colorFormat : asset::EF_D24_UNORM_S8_UINT;
		imgInfo.type = asset::ICPUImage::ET_2D;
		imgInfo.extent.width = driver->getScreenSize().Width;
		imgInfo.extent.height = driver->getScreenSize().Height;
		imgInfo.extent.depth = 1u;
		imgInfo.mipLevels = 1u;
		imgInfo.arrayLayers = 1u;
		imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
		imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);

		auto image = asset::ICPUImage::create(std::move(imgInfo));
		const auto texelFormatBytesize = getTexelOrBlockBytesize(image->getCreationParameters().format);

		auto texelBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(image->getImageDataSizeInBytes());
		auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(1u);
		asset::ICPUImage::SBufferCopy& region = regions->front();

		region.imageSubresource.mipLevel = 0u;
		region.imageSubresource.baseArrayLayer = 0u;
		region.imageSubresource.layerCount = 1u;
		region.bufferOffset = 0u;
		region.bufferRowLength = image->getCreationParameters().extent.width;
		region.bufferImageHeight = 0u;
		region.imageOffset = { 0u, 0u, 0u };
		region.imageExtent = image->getCreationParameters().extent;

		image->setBufferAndRegions(std::move(texelBuffer), regions);

		asset::ICPUImageView::SCreationParams imgViewInfo;
		imgViewInfo.image = std::move(image);
		imgViewInfo.format = colorBuffer ? colorFormat : asset::EF_D24_UNORM_S8_UINT;
		imgViewInfo.viewType = asset::IImageView<asset::ICPUImage>::ET_2D;
		imgViewInfo.flags = static_cast<asset::ICPUImageView::E_CREATE_FLAGS>(0u);
		imgViewInfo.subresourceRange.baseArrayLayer = 0u;
		imgViewInfo.subresourceRange.baseMipLevel = 0u;
		imgViewInfo.subresourceRange.layerCount = imgInfo.arrayLayers;
		imgViewInfo.subresourceRange.levelCount = imgInfo.mipLevels;

		auto imageView = asset::ICPUImageView::create(std::move(imgViewInfo));
		auto gpuImageView = driver->getGPUObjectsFromAssets(&imageView.get(), &imageView.get() + 1)->front();

		return std::move(gpuImageView);
	};

	auto gpuImageViewDepthBuffer = createAttachement(false);
	auto gpuImageViewColorBuffer = createAttachement(true);

	auto frameBuffer = driver->addFrameBuffer();
	frameBuffer->attach(video::EFAP_DEPTH_ATTACHMENT, std::move(gpuImageViewDepthBuffer));
	frameBuffer->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(gpuImageViewColorBuffer));

	return frameBuffer;
}

int main()
{
	irr::SIrrlichtCreationParameters params;
	params.Bits = 32;
	params.ZBufferBits = 24;
	params.DriverType = video::EDT_OPENGL;
	params.WindowSize = dimension2d<uint32_t>(1600, 900);
	params.Fullscreen = false;
	params.Doublebuffer = true;
	params.Vsync = true;
	params.Stencilbuffer = false;

	auto device = createDeviceEx(params);
	if (!device)
		return false;

	device->getCursorControl()->setVisible(false);
	auto driver = device->getVideoDriver();
	auto assetManager = device->getAssetManager();
	auto sceneManager = device->getSceneManager();

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);
	auto fullScreenTriangle = ext::FullScreenTriangle::createFullScreenTriangle(device->getAssetManager(), device->getVideoDriver());

	scene::ICameraSceneNode* camera = sceneManager->addCameraSceneNodeFPS(0, 100.0f, 0.001f);

	camera->setPosition(core::vector3df(0, 0, 0));
	camera->setTarget(core::vector3df(0, 0, 0));
	camera->setNearValue(0.01f);
	camera->setFarValue(10.0f);

	// camera->getRotation()

	sceneManager->setActiveCamera(camera); // TODO

	IGPUDescriptorSetLayout::SBinding samplerBinding { 0u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IGPUSpecializedShader::ESS_FRAGMENT, nullptr };
	IGPUDescriptorSetLayout::SBinding uboBinding {0, asset::EDT_UNIFORM_BUFFER, 1u, static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_VERTEX | IGPUSpecializedShader::ESS_FRAGMENT), nullptr};

	auto gpuDescriptorSetLayout1 = driver->createGPUDescriptorSetLayout(&uboBinding, &uboBinding + 1u);
	auto gpuDescriptorSetLayout3 = driver->createGPUDescriptorSetLayout(&samplerBinding, &samplerBinding + 1u);

	auto createGPUPipeline = [&]() -> core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>
	{
		auto gpuFragmentShader = driver->createGPUShader(core::make_smart_refctd_ptr<asset::ICPUShader>(FRAGMENT_SHADER_GLSL.data()));
		const asset::ISpecializedShader::SInfo specInfo({}, nullptr, "main", asset::ISpecializedShader::ESS_FRAGMENT);

		auto gpuFragmentSpecialedShader = driver->createGPUSpecializedShader(gpuFragmentShader.get(), std::move(specInfo));
		IGPUSpecializedShader* shaders[2] = { std::get<0>(fullScreenTriangle).get(), gpuFragmentSpecialedShader.get() };

		SBlendParams blendParams;
		blendParams.logicOpEnable = false;
		blendParams.logicOp = ELO_NO_OP;
		for (size_t i = 0ull; i < SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
			blendParams.blendParams[i].attachmentEnabled = (i == 0ull);
		SRasterizationParams rasterParams;
		rasterParams.faceCullingMode = EFCM_NONE;
		rasterParams.depthCompareOp = ECO_ALWAYS;
		rasterParams.minSampleShading = 1.f;
		rasterParams.depthWriteEnable = false;
		rasterParams.depthTestEnable = false;

		auto gpuPipelineLayout = driver->createGPUPipelineLayout(nullptr, nullptr, nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout1), nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout3));

		return driver->createGPURenderpassIndependentPipeline(nullptr, std::move(gpuPipelineLayout), shaders, shaders + sizeof(shaders) / sizeof(IGPUSpecializedShader*),
			std::get<SVertexInputParams>(fullScreenTriangle), blendParams,
			std::get<SPrimitiveAssemblyParams>(fullScreenTriangle), rasterParams);
	};

	auto gpuPipeline = createGPUPipeline();

	SBufferBinding<IGPUBuffer> idxBinding{ 0ull, nullptr };
	auto gpuMeshBuffer = core::make_smart_refctd_ptr<IGPUMeshBuffer>(nullptr, nullptr, nullptr, std::move(idxBinding));
	gpuMeshBuffer->setIndexCount(3u);
	gpuMeshBuffer->setInstanceCount(1u);

	auto pathToTexture = "../../media/envmap/wooden_motel_2k_EXR.exr";
	IAssetLoader::SAssetLoadParams lp(0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES);
	auto cpuTexture = assetManager->getAsset(pathToTexture, lp);
	auto cpuTextureContents = cpuTexture.getContents();

	io::path filename, extension, finalFileNameWithExtension;
	core::splitFilename(pathToTexture, nullptr, &filename, &extension);
	finalFileNameWithExtension = filename + ".";
	finalFileNameWithExtension += extension;

	auto asset = *cpuTextureContents.first;

	ICPUImageView::SCreationParams viewParams;
	viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
	viewParams.image = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset);
	viewParams.format = viewParams.image->getCreationParameters().format;
	viewParams.viewType = IImageView<ICPUImage>::ET_2D;
	viewParams.subresourceRange.baseArrayLayer = 0u;
	viewParams.subresourceRange.layerCount = 1u;
	viewParams.subresourceRange.baseMipLevel = 0u;
	viewParams.subresourceRange.levelCount = 1u;

	auto cpuImageView = ICPUImageView::create(std::move(viewParams));
	auto gpuImageView = driver->getGPUObjectsFromAssets(&cpuImageView.get(), &cpuImageView.get() + 1u)->front();

	auto gpuubo = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(SBasicViewParameters));
	auto uboDescriptorSet1 = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout1));
	{
		video::IGPUDescriptorSet::SWriteDescriptorSet uboWriteDescriptorSet;
		uboWriteDescriptorSet.dstSet = uboDescriptorSet1.get();
		uboWriteDescriptorSet.binding = 0;
		uboWriteDescriptorSet.count = 1u;
		uboWriteDescriptorSet.arrayElement = 0u;
		uboWriteDescriptorSet.descriptorType = asset::EDT_UNIFORM_BUFFER;
		video::IGPUDescriptorSet::SDescriptorInfo info;
		{
			info.desc = gpuubo;
			info.buffer.offset = 0ull;
			info.buffer.size = sizeof(SBasicViewParameters);
		}
		uboWriteDescriptorSet.info = &info;
		driver->updateDescriptorSets(1u, &uboWriteDescriptorSet, 0u, nullptr);
	}

	auto samplerDescriptorSet3 = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout3));
	{
		IGPUDescriptorSet::SWriteDescriptorSet samplerWriteDescriptorSet;
		samplerWriteDescriptorSet.dstSet = samplerDescriptorSet3.get();
		samplerWriteDescriptorSet.binding = 0u;
		samplerWriteDescriptorSet.arrayElement = 0u;
		samplerWriteDescriptorSet.count = 1u;
		samplerWriteDescriptorSet.descriptorType = EDT_COMBINED_IMAGE_SAMPLER;

		IGPUDescriptorSet::SDescriptorInfo samplerDescriptorInfo;
		{
			samplerDescriptorInfo.desc = gpuImageView;
			ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
			samplerDescriptorInfo.image.sampler = driver->createGPUSampler(samplerParams);
			samplerDescriptorInfo.image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
		}
		samplerWriteDescriptorSet.info = &samplerDescriptorInfo;
		driver->updateDescriptorSets(1u, &samplerWriteDescriptorSet, 0u, nullptr);
	}

	auto HDRFramebuffer = createHDRFramebuffer(device, gpuImageView->getCreationParameters().format);
	float colorClearValues[] = { 1.f, 1.f, 1.f, 1.f };

	while (device->run())
	{
		driver->setRenderTarget(HDRFramebuffer, false);
		driver->clearZBuffer();
		driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0, colorClearValues);

		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();

		const auto viewProjection = camera->getConcatenatedMatrix();
		core::matrix3x4SIMD modelMatrix; // indentity, but we can use it however

		auto mv = core::concatenateBFollowedByA(camera->getViewMatrix(), modelMatrix);
		auto mvp = core::concatenateBFollowedByA(viewProjection, modelMatrix);
		core::matrix3x4SIMD normalMat;
		mv.getSub3x3InverseTranspose(normalMat);

		SBasicViewParameters uboData;
		memcpy(uboData.MV, mv.pointer(), sizeof(mv));
		memcpy(uboData.MVP, mvp.pointer(), sizeof(mvp));
		memcpy(uboData.NormalMat, normalMat.pointer(), sizeof(normalMat));
		driver->updateBufferRangeViaStagingBuffer(gpuubo.get(), 0ull, sizeof(uboData), &uboData);

		driver->bindGraphicsPipeline(gpuPipeline.get());
		driver->bindDescriptorSets(EPBP_GRAPHICS, gpuPipeline->getLayout(), 1u, 1u, &uboDescriptorSet1.get(), nullptr);
		driver->bindDescriptorSets(EPBP_GRAPHICS, gpuPipeline->getLayout(), 3u, 1u, &samplerDescriptorSet3.get(), nullptr);
		driver->drawMeshBuffer(gpuMeshBuffer.get());

		driver->setRenderTarget(nullptr, false);
		driver->blitRenderTargets(HDRFramebuffer, nullptr, false, false);

		driver->endScene();
	}

	assetManager->removeCachedGPUObject(asset.get(), gpuImageView);
	assetManager->removeAssetFromCache(cpuTexture);
}
