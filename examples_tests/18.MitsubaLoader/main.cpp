// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "nbl/ext/ScreenShot/ScreenShot.h"

#include "../common/Camera.hpp"
#include "../common/CommonAPI.h"

#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"
#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"

#define USE_ENVMAP

using namespace nbl;
using namespace core;
using namespace asset;
using namespace video;

#include "nbl/nblpack.h"
//std430-compatible
struct SLight
{
	core::vectorSIMDf position;
	core::vectorSIMDf intensity;
} PACK_STRUCT;
#include "nbl/nblunpack.h"

int main(int argc, char** argv)
{
	system::path CWD = system::path(argv[0]).parent_path().generic_string() + "/";
	constexpr uint32_t WIN_W = 1024;
	constexpr uint32_t WIN_H = 720;
	constexpr uint32_t SC_IMG_COUNT = 3u;
	constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	auto initOutput = CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(video::EAT_OPENGL, "MitsubaLoader", nbl::asset::EF_D32_SFLOAT);
	auto window = std::move(initOutput.window);
	auto gl = std::move(initOutput.apiConnection);
	auto surface = std::move(initOutput.surface);
	auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
	auto logicalDevice = std::move(initOutput.logicalDevice);
	auto queues = std::move(initOutput.queues);
	auto swapchain = std::move(initOutput.swapchain);
	auto renderpass = std::move(initOutput.renderpass);
	auto fbos = std::move(initOutput.fbo);
	auto commandPool = std::move(initOutput.commandPool);
	auto assetManager = std::move(initOutput.assetManager);
	auto logger = std::move(initOutput.logger);
	auto inputSystem = std::move(initOutput.inputSystem);
	auto system = std::move(initOutput.system);
	auto windowCallback = std::move(initOutput.windowCb);
	auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
	auto utilities = std::move(initOutput.utilities);

	core::smart_refctd_ptr<video::IGPUFence> gpuTransferFence;
	core::smart_refctd_ptr<video::IGPUFence> gpuComputeFence;
	nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
	{
		cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &gpuTransferFence;
		cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &gpuComputeFence;
	}

	// Select mitsuba file
	asset::SAssetBundle meshes;
	core::smart_refctd_ptr<const ext::MitsubaLoader::CMitsubaMetadata> globalMeta;
	{
		asset::CQuantNormalCache* qnc = assetManager->getMeshManipulator()->getQuantNormalCache();

		auto serializedLoader = core::make_smart_refctd_ptr<nbl::ext::MitsubaLoader::CSerializedLoader>(assetManager.get());
		auto mitsubaLoader = core::make_smart_refctd_ptr<nbl::ext::MitsubaLoader::CMitsubaLoader>(assetManager.get(), system.get());
		serializedLoader->initialize();
		mitsubaLoader->initialize();
		assetManager->addAssetLoader(std::move(serializedLoader));
		assetManager->addAssetLoader(std::move(mitsubaLoader));

		std::string filePath = "../../media/mitsuba/bathroom.zip";
		system::path parentPath;
		#define MITSUBA_LOADER_TESTS
#ifndef MITSUBA_LOADER_TESTS
		pfd::message("Choose file to load", "Choose mitsuba XML file to load or ZIP containing an XML. \nIf you cancel or choosen file fails to load staircase will be loaded.", pfd::choice::ok);
		pfd::open_file file("Choose XML or ZIP file", (CWD/"../../media/mitsuba").string(), { "ZIP files (.zip)", "*.zip", "XML files (.xml)", "*.xml" });
		if (!file.result().empty())
			filePath = file.result()[0];
#endif
		if (core::hasFileExtension(filePath, "zip", "ZIP"))
		{
			const system::path archPath = CWD/filePath;
			core::smart_refctd_ptr<system::IFileArchive> arch = nullptr;
			arch = system->openFileArchive(archPath);

			if (!arch)
				arch = system->openFileArchive(CWD/ "../../media/mitsuba/bathroom.zip");
			if (!arch)
				return 2;

			system->mount(std::move(arch), "resources");

			auto flist = arch->getArchivedFiles();
			if (flist.empty())
				return 3;

			for (auto it = flist.begin(); it != flist.end(); )
			{
				if (core::hasFileExtension(it->fullName, "xml", "XML"))
					it++;
				else
					it = flist.erase(it);
			}
			if (flist.size() == 0u)
				return 4;

			std::cout << "Choose File (0-" << flist.size() - 1ull << "):" << std::endl;
			for (auto i = 0u; i < flist.size(); i++)
				std::cout << i << ": " << flist[i].fullName << std::endl;
			uint32_t chosen = 0;
#ifndef MITSUBA_LOADER_TESTS
			std::cin >> chosen;
#endif
			if (chosen >= flist.size())
				chosen = 0u;

			filePath = flist[chosen].name.string();
			parentPath = flist[chosen].fullName.parent_path();
		}

		//! read cache results -- speeds up mesh generation
		qnc->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), "../../tmp/normalCache101010.sse");
		//! load the mitsuba scene
		asset::IAssetLoader::SAssetLoadParams loadParams;
		loadParams.workingDirectory = "resources"/parentPath;
		loadParams.logger = logger.get();
		meshes = assetManager->getAsset(filePath, loadParams);
		assert(!meshes.getContents().empty());
		//! cache results -- speeds up mesh generation on second run
		qnc->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), "../../tmp/normalCache101010.sse");

		auto contents = meshes.getContents();
		if (contents.begin() >= contents.end())
			return 2;

		auto firstmesh = *contents.begin();
		if (!firstmesh)
			return 3;

		globalMeta = core::smart_refctd_ptr<const ext::MitsubaLoader::CMitsubaMetadata>(meshes.getMetadata()->selfCast<const ext::MitsubaLoader::CMitsubaMetadata>());
		if (!globalMeta)
			return 4;
	}

	//TODO:
	//// recreate wth resolution
	//params.WindowSize = dimension2d<uint32_t>(1280, 720);
	//// set resolution
	//if (globalMeta->m_global.m_sensors.size())
	//{
	//	const auto& film = globalMeta->m_global.m_sensors.front().film;
	//	params.WindowSize.Width = film.width;
	//	params.WindowSize.Height = film.height;
	//}
	//else return 1; // no cameras

	// process metadata

	const auto& sensor = globalMeta->m_global.m_sensors.front(); //always choose frist one
	auto isOkSensorType = [](const ext::MitsubaLoader::CElementSensor& sensor) -> bool {
		return sensor.type == ext::MitsubaLoader::CElementSensor::Type::PERSPECTIVE || sensor.type == ext::MitsubaLoader::CElementSensor::Type::THINLENS;
	};

	if (!isOkSensorType(sensor))
		return 1;

	bool leftHandedCamera = false;
	auto cameraTransform = sensor.transform.matrix.extractSub3x4();
	{
		if (cameraTransform.getPseudoDeterminant().x < 0.f)
			leftHandedCamera = true;
	}

	// gather all meshes into core::vector and modify their pipelines

	core::vector<core::smart_refctd_ptr<asset::ICPUMesh>> cpuMeshes;
	{
		auto contents = meshes.getContents();

		cpuMeshes.reserve(contents.size());
		for (auto it = contents.begin(); it != contents.end(); ++it)
			cpuMeshes.push_back(core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(std::move(*it)));

		auto cpuds0 = globalMeta->m_global.m_ds0;

		asset::ICPUDescriptorSetLayout* ds1layout = cpuMeshes.front()->getMeshBuffers().begin()[0u]->getPipeline()->getLayout()->getDescriptorSetLayout(1u);
		uint32_t ds1UboBinding = 0u;
		for (const auto& bnd : ds1layout->getBindings())
		if (bnd.type == asset::EDT_UNIFORM_BUFFER)
		{
			ds1UboBinding = bnd.binding;
			break;
		}
	}

	// convert cpu meshes to gpu meshes

	// TODO: shit doesn't work, probably need to set pipeline for every mesh
	auto orgPplnVtxParams = cpuMeshes[0]->getMeshBuffers().begin()[0]->getPipeline()->getVertexInputParams();
	core::vector<core::smart_refctd_ptr<video::IGPUMesh>> gpuMeshes;
	{
		gpuMeshes.reserve(gpuMeshes.size());
		for (auto cpuMesh : cpuMeshes)
		{
			auto meshRaw = dynamic_cast<asset::ICPUMesh*>(cpuMesh.get());
			assert(meshRaw);
			auto meshBuffers = meshRaw->getMeshBufferVector();
			for (auto meshBuffer : meshBuffers)
				meshBuffer->setPipeline(nullptr);

			auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&meshRaw, &meshRaw + 1, cpu2gpuParams);
			assert(!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0]);
			gpuMeshes.push_back((*gpu_array)[0]);
			cpu2gpuParams.waitForCreationToComplete();
		}
	}

	// TMP
	// set shared pipeline
	IDriverMemoryBacked::SDriverMemoryRequirements memReq;
	memReq.vulkanReqs.size = sizeof(SBasicViewParameters);
	core::smart_refctd_ptr<IGPUBuffer> cameraUBO = logicalDevice->createGPUBufferOnDedMem(memReq, true);
	core::smart_refctd_ptr<IGPUDescriptorSetLayout> ds0Layout;
	core::smart_refctd_ptr<IGPUDescriptorSet> ds0;
	core::smart_refctd_ptr<IDescriptorPool> descriptorPool;
	{
		{
			IGPUDescriptorSetLayout::SBinding bindings[1];
			bindings[0].binding = 0u;
			bindings[0].count = 1u;
			bindings[0].type = E_DESCRIPTOR_TYPE::EDT_UNIFORM_BUFFER;
			bindings[0].stageFlags = ISpecializedShader::E_SHADER_STAGE::ESS_FRAGMENT;
			bindings[0].samplers = nullptr;

			ds0Layout = logicalDevice->createGPUDescriptorSetLayout(bindings, bindings + sizeof(bindings) / sizeof(IGPUDescriptorSetLayout::SBinding));
			descriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &ds0Layout.get(), &ds0Layout.get() + 1);

			ds0 = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), ds0Layout);
			IGPUDescriptorSet::SDescriptorInfo infos[1];
			infos[0].desc = cameraUBO;
			infos[0].buffer.offset = 0u;
			infos[0].buffer.size = cameraUBO->getSize();

			IGPUDescriptorSet::SWriteDescriptorSet writes[1];
			writes[0].dstSet = ds0.get();
			writes[0].binding = 0u;
			writes[0].arrayElement = 0u;
			writes[0].count = 1u;
			writes[0].descriptorType = EDT_UNIFORM_BUFFER;
			writes[0].info = infos;

			logicalDevice->updateDescriptorSets(sizeof(writes) / sizeof(IGPUDescriptorSet::SWriteDescriptorSet), writes, 0u, nullptr);
		}
	}

	// pipeline shared among all mesh buffers
	core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> renderpassIndependentPpln;
	core::smart_refctd_ptr<IGPUGraphicsPipeline> graphicsPpln;
	{
		asset::IAssetLoader::SAssetLoadParams lp;
		lp.logger = logger.get();
		lp.workingDirectory = CWD;
		auto cpuVtxShader = IAsset::castDown<ICPUSpecializedShader>(*assetManager->getAsset("../test.vert", lp).getContents().begin());
		auto cpuFragShader = IAsset::castDown<ICPUSpecializedShader>(*assetManager->getAsset("../test.frag", lp).getContents().begin());
		assert(cpuVtxShader);
		assert(cpuFragShader);
		assert(cpuVtxShader->getUnspecialized()->containsGLSL());
		assert(cpuFragShader->getUnspecialized()->containsGLSL());

		core::smart_refctd_ptr<ICPUSpecializedShader> cpuSpecializedShaders[2] = {
			cpuVtxShader, cpuFragShader
		};

		auto gpuSpecializedShaders = cpu2gpu.getGPUObjectsFromAssets(cpuSpecializedShaders, cpuSpecializedShaders + 2u, cpu2gpuParams);
		cpu2gpuParams.waitForCreationToComplete();

		IGPUSpecializedShader* shaders[2] = {
			gpuSpecializedShaders->begin()[0].get(),
			gpuSpecializedShaders->begin()[1].get(),
		};

		assert(shaders[0]);
		_NBL_DEBUG_BREAK_IF(shaders[0] == nullptr);

		renderpassIndependentPpln = logicalDevice->createGPURenderpassIndependentPipeline(nullptr, logicalDevice->createGPUPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr(ds0Layout)), shaders, shaders + 2u, orgPplnVtxParams, SBlendParams(), SPrimitiveAssemblyParams(), SRasterizationParams());

		nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
		graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr(renderpassIndependentPpln);
		graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);

		graphicsPpln = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
	}

	// TODO: do i even need that?
	/*for (auto gpuMesh : gpuMeshes)
	{
		auto meshBuffersRange = gpuMesh->getMeshBuffers();
		for (auto it = meshBuffersRange.begin(); it != meshBuffersRange.end(); it++)
			(*it)->setPipeline(core::smart_refctd_ptr(renderpassIndependentPpln));
	}*/

	// mesh convertion

	/*const asset::COBJMetadata::CRenderpassIndependentPipeline* pipelineMetadata = nullptr;
	uint32_t cameraUBOBinding = 0u;
	core::smart_refctd_ptr<video::IGPUBuffer> cameraUBO;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> perCameraDescSet;*/
	{
		/*auto contents = meshes.getContents();
		auto mesh = dynamic_cast<asset::ICPUMesh*>(contents.begin()->get());

		_NBL_DEBUG_BREAK_IF(mesh == nullptr);*/

		//const auto meshbuffers = meshRaw->getMeshBuffers();

		//core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuds1layout;
		//size_t neededDS1UBOsz = 0ull;
		//{
		//	// we can safely assume that all meshbuffers within mesh loaded from OBJ has same DS1 layout (used for camera-specific data)
		//	const asset::ICPUMeshBuffer* const firstMeshBuffer = *meshbuffers.begin();
		//	pipelineMetadata = metaOBJ->getAssetSpecificMetadata(firstMeshBuffer->getPipeline());

		//	// so we can create just one DS
		//	const asset::ICPUDescriptorSetLayout* ds1layout = firstMeshBuffer->getPipeline()->getLayout()->getDescriptorSetLayout(1u);
		//	for (const auto& bnd : ds1layout->getBindings())
		//		if (bnd.type == asset::EDT_UNIFORM_BUFFER)
		//		{
		//			cameraUBOBinding = bnd.binding;
		//			break;
		//		}

		//	for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
		//		if (shdrIn.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set == 1u && shdrIn.descriptorSection.uniformBufferObject.binding == cameraUBOBinding)
		//			neededDS1UBOsz = std::max<size_t>(neededDS1UBOsz, shdrIn.descriptorSection.uniformBufferObject.relByteoffset + shdrIn.descriptorSection.uniformBufferObject.bytesize);

		//	auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&ds1layout, &ds1layout + 1, cpu2gpuParams);
		//	assert(!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0]);
		//	gpuds1layout = (*gpu_array)[0];
		//	cpu2gpuParams.waitForCreationToComplete();
		//}

		//auto descriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &gpuds1layout.get(), &gpuds1layout.get() + 1);

		//auto ubomemreq = logicalDevice->getDeviceLocalGPUMemoryReqs();
		//ubomemreq.vulkanReqs.size = neededDS1UBOsz;
		//cameraUBO = logicalDevice->createGPUBufferOnDedMem(ubomemreq, true);
		//perCameraDescSet = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), std::move(gpuds1layout));
		//{
		//	video::IGPUDescriptorSet::SWriteDescriptorSet write;
		//	write.dstSet = perCameraDescSet.get();
		//	write.binding = cameraUBOBinding;
		//	write.count = 1u;
		//	write.arrayElement = 0u;
		//	write.descriptorType = asset::EDT_UNIFORM_BUFFER;
		//	video::IGPUDescriptorSet::SDescriptorInfo info;
		//	{
		//		info.desc = cameraUBO;
		//		info.buffer.offset = 0ull;
		//		info.buffer.size = neededDS1UBOsz;
		//	}
		//	write.info = &info;
		//	logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
		//}
	}

	// setup

	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

	core::vectorSIMDf cameraPosition(0, 5, -10);
	matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 0.1, 1000);
	Camera camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 10.f, 1.f);
	auto lastTime = std::chrono::system_clock::now();

	constexpr size_t NBL_FRAMES_TO_AVERAGE = 100ull;
	bool frameDataFilled = false;
	size_t frame_count = 0ull;
	double time_sum = 0;
	double dtList[NBL_FRAMES_TO_AVERAGE] = {};
	for (size_t i = 0ull; i < NBL_FRAMES_TO_AVERAGE; ++i)
		dtList[i] = 0.0;

	core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];
	logicalDevice->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

	core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };

	for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
	{
		imageAcquire[i] = logicalDevice->createSemaphore();
		renderFinished[i] = logicalDevice->createSemaphore();
	}

	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	uint32_t acquiredNextFBO = {};
	auto resourceIx = -1;
	
	float lastFastestMeshFrameNr = -1.f;

	// MAIN LOOP
	while (windowCallback->isWindowOpen())
	{
		++resourceIx;
		if (resourceIx >= FRAMES_IN_FLIGHT)
			resourceIx = 0;

		auto& commandBuffer = commandBuffers[resourceIx];
		auto& fence = frameComplete[resourceIx];

		if (fence)
			while (logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT) {}
		else
			fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		auto renderStart = std::chrono::system_clock::now();
		const auto renderDt = std::chrono::duration_cast<std::chrono::milliseconds>(renderStart - lastTime).count();
		lastTime = renderStart;
		{ // Calculate Simple Moving Average for FrameTime
			time_sum -= dtList[frame_count];
			time_sum += renderDt;
			dtList[frame_count] = renderDt;
			frame_count++;
			if (frame_count >= NBL_FRAMES_TO_AVERAGE)
			{
				frameDataFilled = true;
				frame_count = 0;
			}

		}
		const double averageFrameTime = frameDataFilled ? (time_sum / (double)NBL_FRAMES_TO_AVERAGE) : (time_sum / frame_count);

#ifdef NBL_MORE_LOGS
		logger->log("renderDt = %f ------ averageFrameTime = %f", system::ILogger::ELL_INFO, renderDt, averageFrameTime);
#endif // NBL_MORE_LOGS

		auto averageFrameTimeDuration = std::chrono::duration<double, std::milli>(averageFrameTime);
		auto nextPresentationTime = renderStart + averageFrameTimeDuration;
		auto nextPresentationTimeStamp = std::chrono::duration_cast<std::chrono::microseconds>(nextPresentationTime.time_since_epoch());

		inputSystem->getDefaultMouse(&mouse);
		inputSystem->getDefaultKeyboard(&keyboard);

		camera.beginInputProcessing(nextPresentationTimeStamp);
		mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
		keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, logger.get());
		camera.endInputProcessing(nextPresentationTimeStamp);

		const auto& viewMatrix = camera.getViewMatrix();
		const auto& viewProjectionMatrix = camera.getConcatenatedMatrix();

		commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
		commandBuffer->begin(0);

		asset::SViewport viewport;
		viewport.minDepth = 1.f;
		viewport.maxDepth = 0.f;
		viewport.x = 0u;
		viewport.y = 0u;
		viewport.width = WIN_W;
		viewport.height = WIN_H;
		commandBuffer->setViewport(0u, 1u, &viewport);

		swapchain->acquireNextImage(MAX_TIMEOUT, imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);

		nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		{
			VkRect2D area;
			area.offset = { 0,0 };
			area.extent = { WIN_W, WIN_H };
			asset::SClearValue clear[2] = {};
			clear[0].color.float32[0] = 1.f;
			clear[0].color.float32[1] = 1.f;
			clear[0].color.float32[2] = 1.f;
			clear[0].color.float32[3] = 1.f;
			clear[1].depthStencil.depth = 0.f;

			beginInfo.clearValueCount = 2u;
			beginInfo.framebuffer = fbos[acquiredNextFBO];
			beginInfo.renderpass = renderpass;
			beginInfo.renderArea = area;
			beginInfo.clearValues = clear;
		}

		commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);
		
		// draw shit
		
		SBasicViewParameters viewParams;
		std::memcpy(viewParams.MVP, viewProjectionMatrix.pointer(), sizeof(core::matrix4SIMD));
		std::memcpy(viewParams.MV, viewMatrix.pointer(), sizeof(core::matrix3x4SIMD));
		commandBuffer->updateBuffer(cameraUBO.get(), 0ull, cameraUBO->getSize(), &viewParams) == false;
		for (auto gpuMesh : gpuMeshes)
		{
			for (size_t i = 0; i < gpuMesh->getMeshBuffers().size(); ++i)
			{
				auto gpuMeshBuffer = gpuMesh->getMeshBuffers().begin()[i];

				//const video::IGPURenderpassIndependentPipeline* gpuRenderpassIndependentPipeline = gpuMeshBuffer->getPipeline();
				const video::IGPURenderpassIndependentPipeline* gpuRenderpassIndependentPipeline = renderpassIndependentPpln.get();
				commandBuffer->bindGraphicsPipeline(graphicsPpln.get());

				const video::IGPUDescriptorSet* ds0Ptr = ds0.get();
				commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 0u, 1u, &ds0Ptr, nullptr);

				commandBuffer->drawMeshBuffer(gpuMeshBuffer);
			}
		}

		commandBuffer->endRenderPass();
		commandBuffer->end();

		CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queues[decltype(initOutput)::EQT_GRAPHICS], imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
		CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[decltype(initOutput)::EQT_GRAPHICS], renderFinished[resourceIx].get(), acquiredNextFBO);
	}
}