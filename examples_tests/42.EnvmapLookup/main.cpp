#include <irrlicht.h>

#include "../common/QToQuitEventReceiver.h"
#include "../../ext/ScreenShot/ScreenShot.h"

using namespace irr;
using namespace core;
using namespace asset;
using namespace video;

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
	auto geometryCreator = device->getAssetManager()->getGeometryCreator();
	auto cubeGeometry = geometryCreator->createCubeMesh(vector3df(1, 1, 1));
	auto icosphereGeometry = geometryCreator->createIcoSphere(0.1f, 4);

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	scene::ICameraSceneNode* camera = sceneManager->addCameraSceneNodeFPS(0, 100.0f, 0.001f);

	camera->setPosition(core::vector3df(0, 0, 0));
	camera->setTarget(core::vector3df(0, 0, 0));
	camera->setNearValue(0.01f);
	camera->setFarValue(100.0f);

	sceneManager->setActiveCamera(camera);

	IGPUDescriptorSetLayout::SBinding samplerBinding { 0u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IGPUSpecializedShader::ESS_FRAGMENT, nullptr };
	IGPUDescriptorSetLayout::SBinding uboBinding {0, asset::EDT_UNIFORM_BUFFER, 1u, static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_VERTEX | IGPUSpecializedShader::ESS_FRAGMENT), nullptr};

	auto gpuDescriptorSetLayout1 = driver->createGPUDescriptorSetLayout(&uboBinding, &uboBinding + 1u);
	auto gpuDescriptorSetLayout3 = driver->createGPUDescriptorSetLayout(&samplerBinding, &samplerBinding + 1u);

	auto createGpuResources = [&](asset::IGeometryCreator::return_type& geometry, std::string pathToShadersWithoutExtension, bool isSphere) -> std::pair<core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>, core::smart_refctd_ptr<video::IGPUMeshBuffer>>
	{
		auto cpuVertexSpecializedShader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(assetManager->getAsset(pathToShadersWithoutExtension + ".vert", { 0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES }).getContents().first[0]);
		auto cpuFragmentSpecializedShader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(assetManager->getAsset(pathToShadersWithoutExtension + ".frag", { 0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES }).getContents().first[0]);

		auto gpuVertexSpecialedShader = driver->getGPUObjectsFromAssets(&cpuVertexSpecializedShader.get(), &cpuVertexSpecializedShader.get() + 1)->front();
		auto gpuFragmentSpecialedShader = driver->getGPUObjectsFromAssets(&cpuFragmentSpecializedShader.get(), &cpuFragmentSpecializedShader.get() + 1)->front();
		IGPUSpecializedShader* shaders[2] = { gpuVertexSpecialedShader.get(), gpuFragmentSpecialedShader.get() };

		auto gpuPipelineLayout = driver->createGPUPipelineLayout(nullptr, nullptr, nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout1), nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout3));

		asset::SBlendParams blendParams;
		asset::SRasterizationParams rasterParams;
		rasterParams.faceCullingMode = asset::EFCM_NONE;

		/*
		TODO: actually I should render it on far plane somehow, it doesn't work anyways

		if (isSphere)
			rasterParams.depthCompareOp = asset::ECO_LESS_OR_EQUAL;
		else 
			rasterParams.depthCompareOp = asset::ECO_LESS;
		*/

		auto gpuPipeline = driver->createGPURenderpassIndependentPipeline(nullptr, std::move(gpuPipelineLayout), shaders, shaders + sizeof(shaders) / sizeof(IGPUSpecializedShader*), geometry.inputParams, blendParams, geometry.assemblyParams, rasterParams);

		constexpr auto MAX_ATTR_BUF_BINDING_COUNT = video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT;
		constexpr auto MAX_DATA_BUFFERS = MAX_ATTR_BUF_BINDING_COUNT + 1;

		core::vector<asset::ICPUBuffer*> cpubuffers;
		cpubuffers.reserve(MAX_DATA_BUFFERS);
		for (auto i = 0; i < MAX_ATTR_BUF_BINDING_COUNT; i++)
		{
			auto buf = geometry.bindings[i].buffer.get();
			if (buf)
				cpubuffers.push_back(buf);
		}
		auto cpuindexbuffer = geometry.indexBuffer.buffer.get();
		if (cpuindexbuffer)
			cpubuffers.push_back(cpuindexbuffer);

		auto gpubuffers = driver->getGPUObjectsFromAssets(cpubuffers.data(), cpubuffers.data() + cpubuffers.size());

		asset::SBufferBinding<video::IGPUBuffer> bindings[MAX_DATA_BUFFERS];
		for (auto i = 0, j = 0; i < MAX_ATTR_BUF_BINDING_COUNT; i++)
		{
			if (!geometry.bindings[i].buffer)
				continue;
			auto buffPair = gpubuffers->operator[](j++);
			bindings[i].offset = buffPair->getOffset();
			bindings[i].buffer = core::smart_refctd_ptr<video::IGPUBuffer>(buffPair->getBuffer());
		}
		if (cpuindexbuffer)
		{
			auto buffPair = gpubuffers->back();
			bindings[MAX_ATTR_BUF_BINDING_COUNT].offset = buffPair->getOffset();
			bindings[MAX_ATTR_BUF_BINDING_COUNT].buffer = core::smart_refctd_ptr<video::IGPUBuffer>(buffPair->getBuffer());
		}

		core::smart_refctd_ptr<video::IGPUMeshBuffer> gpuMeshBuffer = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(core::smart_refctd_ptr(gpuPipeline), nullptr, bindings, std::move(bindings[MAX_ATTR_BUF_BINDING_COUNT]));
		{
			gpuMeshBuffer->setIndexType(geometry.indexType);
			gpuMeshBuffer->setIndexCount(geometry.indexCount);
			gpuMeshBuffer->setBoundingBox(geometry.bbox);
		}

		return { gpuPipeline, gpuMeshBuffer };
	};

	auto gpuEnvmapResources = createGpuResources(cubeGeometry, "../envCubeMapShaders/envmap", false);
	auto gpuEnvmapPipeline = gpuEnvmapResources.first;
	auto gpuEnvmapMeshBuffer = gpuEnvmapResources.second;

	auto gpuSphereResources = createGpuResources(icosphereGeometry, "../sphereShaders/icosphere", true);
	auto gpuSpherePipeline = gpuSphereResources.first;
	auto gpuSphereMeshBuffer = gpuSphereResources.second;

	auto createGPUImageView = [&](std::string pathToOpenEXRHDRIImage)
	{
		auto pathToTexture = pathToOpenEXRHDRIImage;
		IAssetLoader::SAssetLoadParams lp(0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES);
		auto cpuTexture = assetManager->getAsset(pathToTexture, lp);
		auto cpuTextureContents = cpuTexture.getContents();

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

		return gpuImageView;
	};

	auto gpuEnvmapImageView = createGPUImageView("../../media/envmap/envmap_0.exr");
	auto gpuSphereImageView = createGPUImageView("../../media/envmap/envmap_1.exr");

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

	auto envmapSamplerDescriptorSet3 = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout3));
	auto sphereSamplerDescriptorSet3 = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout3));
	{
		auto updateDescriptorSets = [&](core::smart_refctd_ptr<video::IGPUDescriptorSet> samplerGPUDescriptorSet3 , core::smart_refctd_ptr<video::IGPUImageView> gpuImageView)
		{
			IGPUDescriptorSet::SWriteDescriptorSet samplerWriteDescriptorSet;
			samplerWriteDescriptorSet.dstSet = samplerGPUDescriptorSet3.get();
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
		};

		updateDescriptorSets(envmapSamplerDescriptorSet3, gpuEnvmapImageView);
		updateDescriptorSets(sphereSamplerDescriptorSet3, gpuSphereImageView);
	}

	auto HDRFramebuffer = createHDRFramebuffer(device, gpuEnvmapImageView->getCreationParameters().format);
	float colorClearValues[] = { 1.f, 1.f, 1.f, 1.f };

	uint64_t lastFPSTime = 0;
	while (device->run() && receiver.keepOpen())
	{
		driver->setRenderTarget(HDRFramebuffer, false);
		driver->clearZBuffer();
		driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0, colorClearValues);

		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();

		const auto viewMatrix = camera->getViewMatrix();
		const auto projectionMatrix = camera->getProjectionMatrix();
		const auto viewProjectionMatrix = core::concatenateBFollowedByA(projectionMatrix, viewMatrix);

		// sphere handle
		{
			auto mv = viewMatrix;
			auto mvp = viewProjectionMatrix;
			core::matrix3x4SIMD normalMat;
			mv.getSub3x3InverseTranspose(normalMat);

			SBasicViewParameters uboData;
			memcpy(uboData.MV, mv.pointer(), sizeof(mv));
			memcpy(uboData.MVP, mvp.pointer(), sizeof(mvp));
			memcpy(uboData.NormalMat, normalMat.pointer(), sizeof(normalMat));
			driver->updateBufferRangeViaStagingBuffer(gpuubo.get(), 0ull, sizeof(uboData), &uboData);

			driver->bindGraphicsPipeline(gpuSpherePipeline.get());
			driver->bindDescriptorSets(EPBP_GRAPHICS, gpuSpherePipeline->getLayout(), 1u, 1u, &uboDescriptorSet1.get(), nullptr); // we use the same descriptor, but it is done for formality
			driver->bindDescriptorSets(EPBP_GRAPHICS, gpuSpherePipeline->getLayout(), 3u, 1u, &sphereSamplerDescriptorSet3.get(), nullptr);
			driver->drawMeshBuffer(gpuSphereMeshBuffer.get());
		}

		// cube envmap handle
		{
			const auto newViewMatrix = viewMatrix.getSub3x3TransposeCofactors();
			const auto newProjectionMatrix = projectionMatrix;
			const auto newViewProjectionMatrix = core::concatenateBFollowedByA(projectionMatrix, newViewMatrix);

			auto mv = viewMatrix;
			auto mvp = newViewProjectionMatrix;
			core::matrix3x4SIMD normalMat;
			mv.getSub3x3InverseTranspose(normalMat);

			SBasicViewParameters uboData;
			memcpy(uboData.MV, mv.pointer(), sizeof(mv));
			memcpy(uboData.MVP, mvp.pointer(), sizeof(mvp));
			memcpy(uboData.NormalMat, normalMat.pointer(), sizeof(normalMat));
			driver->updateBufferRangeViaStagingBuffer(gpuubo.get(), 0ull, sizeof(uboData), &uboData);

			driver->bindGraphicsPipeline(gpuEnvmapPipeline.get());
			driver->bindDescriptorSets(EPBP_GRAPHICS, gpuEnvmapPipeline->getLayout(), 1u, 1u, &uboDescriptorSet1.get(), nullptr);
			driver->bindDescriptorSets(EPBP_GRAPHICS, gpuEnvmapPipeline->getLayout(), 3u, 1u, &envmapSamplerDescriptorSet3.get(), nullptr);
			driver->drawMeshBuffer(gpuEnvmapMeshBuffer.get());
		}

		driver->setRenderTarget(nullptr, false);
		driver->blitRenderTargets(HDRFramebuffer, nullptr, false, false);

		driver->endScene();

		uint64_t time = device->getTimer()->getRealTime();
		if (time - lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Envmap Example - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str().c_str());
			lastFPSTime = time;
		}
	}
}
