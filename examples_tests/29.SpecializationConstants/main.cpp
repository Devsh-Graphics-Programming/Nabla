// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"
// #include "CFileSystem.h"
using namespace nbl;
using namespace core;

struct UBOCompute
{
	//xyz - gravity point, w - dt
	core::vectorSIMDf gravPointAndDt;
};
int main()
{
	constexpr uint32_t WIN_W = 1280;
	constexpr uint32_t WIN_H = 720;
	constexpr uint32_t SC_IMG_COUNT = 3u;
	constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT>SC_IMG_COUNT);

	CommonAPI::SFeatureRequest<video::IAPIConnection::E_FEATURE> requiredInstanceFeatures = {};
	requiredInstanceFeatures.count = 1u;
	video::IAPIConnection::E_FEATURE requiredFeatures_Instance[] = { video::IAPIConnection::EF_SURFACE };
	requiredInstanceFeatures.features = requiredFeatures_Instance;

	CommonAPI::SFeatureRequest<video::IAPIConnection::E_FEATURE> optionalInstanceFeatures = {};

	CommonAPI::SFeatureRequest<video::ILogicalDevice::E_FEATURE> requiredDeviceFeatures = {};
	requiredDeviceFeatures.count = 1u;
	video::ILogicalDevice::E_FEATURE requiredFeatures_Device[] = { video::ILogicalDevice::EF_SWAPCHAIN };
	requiredDeviceFeatures.features = requiredFeatures_Device;

	CommonAPI::SFeatureRequest< video::ILogicalDevice::E_FEATURE> optionalDeviceFeatures = {};

	const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_STORAGE_BIT);
	const video::ISurface::SFormat surfaceFormat(asset::EF_B8G8R8A8_UNORM, asset::ECP_COUNT, asset::EOTF_UNKNOWN);
	const asset::E_FORMAT depthFormat = asset::EF_UNKNOWN;

	auto initOutp = CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(
		video::EAT_VULKAN,
		"29.SpecializationConstants",
		requiredInstanceFeatures,
		optionalInstanceFeatures,
		requiredDeviceFeatures,
		optionalDeviceFeatures,
		swapchainImageUsage,
		surfaceFormat,
		depthFormat,
		true);

	auto win = std::move(initOutp.window);
	auto windowCb = std::move(initOutp.windowCb);
	auto api = std::move(initOutp.apiConnection);
	auto surface = std::move(initOutp.surface);
	auto device = std::move(initOutp.logicalDevice);
	auto gpu = std::move(initOutp.physicalDevice);
	auto queue = std::move(initOutp.queues[decltype(initOutp)::EQT_GRAPHICS]);
	auto sc = std::move(initOutp.swapchain);
	auto renderpass = std::move(initOutp.renderpass);
	auto fbo = std::move(initOutp.fbo);
	auto cmdpool = std::move(initOutp.commandPool);
	auto assetManager = std::move(initOutp.assetManager);
	auto filesystem = std::move(initOutp.system);
	auto cpu2gpuParams = std::move(initOutp.cpu2gpuParams);
	auto utils = std::move(initOutp.utilities);

	video::IDescriptorPool::SDescriptorPoolSize poolSize[2];
	poolSize[0].count = 1;
	poolSize[0].type = asset::EDT_STORAGE_BUFFER;
	poolSize[1].count = 1;
	poolSize[1].type = asset::EDT_UNIFORM_BUFFER;

	auto dscPool = device->createDescriptorPool(video::IDescriptorPool::ECF_FREE_DESCRIPTOR_SET_BIT, 2, 2, poolSize);

	video::IGPUObjectFromAssetConverter CPU2GPU;
	core::vectorSIMDf cameraPosition(0, 0, -10);
	matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(90), float(WIN_W) / WIN_H, 0.01, 100);
	matrix3x4SIMD view = matrix3x4SIMD::buildCameraLookAtMatrixRH(cameraPosition, core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 1, 0));
	auto viewProj = matrix4SIMD::concatenateBFollowedByA(proj, matrix4SIMD(view));
	auto camFront = view[2];

	auto glslExts = device->getSupportedGLSLExtensions();
	asset::CShaderIntrospector introspector(assetManager->getGLSLCompiler());

	const char* pathToCompShader = "../particles.comp";
	core::smart_refctd_ptr<asset::ICPUShader> computeUnspec = nullptr;
	{
		auto csBundle = assetManager->getAsset(pathToCompShader, {});
		auto csContents = csBundle.getContents();
		if (csContents.empty())
			assert(false);

		asset::ICPUSpecializedShader* csSpec = static_cast<nbl::asset::ICPUSpecializedShader*>(csContents.begin()->get());
		computeUnspec = core::smart_refctd_ptr<asset::ICPUShader>(csSpec->getUnspecialized());
		computeUnspec = assetManager->getGLSLCompiler()->resolveIncludeDirectives(
			(const char*)computeUnspec->getSPVorGLSL()->getPointer(), asset::IShader::ESS_COMPUTE, pathToCompShader);
	}

#if 0
	{

		auto file = filesystem->createAndOpenFile("../particles.comp");
		computeUnspec = assetManager->getGLSLCompiler()->resolveIncludeDirectives(file, asset::IShader::ESS_COMPUTE, file->getFileName().c_str());
		file->drop();
	}
#endif

	const asset::CIntrospectionData* introspection = nullptr;
	{
		asset::CShaderIntrospector::SIntrospectionParams params;
		params.entryPoint = "main";
		params.filePathHint = "../particles.comp";
		params.GLSLextensions = glslExts;
		params.stage = asset::IShader::ESS_COMPUTE;
		introspection = introspector.introspect(computeUnspec.get(), params);
	}

	constexpr uint32_t COMPUTE_SET = 0u;
	constexpr uint32_t PARTICLE_BUF_BINDING = 0u;
	constexpr uint32_t COMPUTE_DATA_UBO_BINDING = 1u;

	constexpr uint32_t WORKGROUP_SIZE = 256u;
	constexpr uint32_t PARTICLE_COUNT = 1u << 21;
	constexpr uint32_t PARTICLE_COUNT_PER_AXIS = 1u << 7;
	constexpr uint32_t POS_BUF_IX = 0u;
	constexpr uint32_t VEL_BUF_IX = 1u;
	constexpr uint32_t BUF_COUNT = 2u;
	asset::ISpecializedShader::SInfo specInfo;
	{
		struct SpecConstants
		{
			int32_t wg_size;
			int32_t particle_count;
			int32_t pos_buf_ix;
			int32_t vel_buf_ix;
			int32_t buf_count;
		};
		SpecConstants sc{ WORKGROUP_SIZE, PARTICLE_COUNT, POS_BUF_IX, VEL_BUF_IX, BUF_COUNT };

		auto it_particleBufDescIntro = std::find_if(introspection->descriptorSetBindings[COMPUTE_SET].begin(), introspection->descriptorSetBindings[COMPUTE_SET].end(),
			[=](auto b) { return b.binding == PARTICLE_BUF_BINDING; }
		);
		assert(it_particleBufDescIntro->descCountIsSpecConstant);
		const uint32_t buf_count_specID = it_particleBufDescIntro->count_specID;
		auto& particleDataArrayIntro = it_particleBufDescIntro->get<asset::ESRT_STORAGE_BUFFER>().members.array[0];
		assert(particleDataArrayIntro.countIsSpecConstant);
		const uint32_t particle_count_specID = particleDataArrayIntro.count_specID;

		auto backbuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(sc));
		memcpy(backbuf->getPointer(), &sc, sizeof(sc));
		auto entries = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ISpecializedShader::SInfo::SMapEntry>>(5u);
		(*entries)[0] = { 0u,offsetof(SpecConstants,wg_size),sizeof(int32_t) };//currently local_size_{x|y|z}_id is not queryable via introspection API
		(*entries)[1] = { particle_count_specID,offsetof(SpecConstants,particle_count),sizeof(int32_t) };
		(*entries)[2] = { 2u,offsetof(SpecConstants,pos_buf_ix),sizeof(int32_t) };
		(*entries)[3] = { 3u,offsetof(SpecConstants,vel_buf_ix),sizeof(int32_t) };
		(*entries)[4] = { buf_count_specID,offsetof(SpecConstants,buf_count),sizeof(int32_t) };

		specInfo = asset::ISpecializedShader::SInfo(std::move(entries), std::move(backbuf), "main");
	}

	auto compute = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(computeUnspec), std::move(specInfo));

	// auto computePipeline = introspector.createApproximateComputePipelineFromIntrospection(compute.get(), glslExts->begin(), glslExts->end());
	auto computePipeline = introspector.createApproximateComputePipelineFromIntrospection(compute.get(), nullptr, nullptr);
	auto computeLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(nullptr,nullptr,core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(computePipeline->getLayout()->getDescriptorSetLayout(0)));
	computePipeline->setLayout(core::smart_refctd_ptr(computeLayout));

	// cpu2gpuParams.sharingMode = asset::ESM_CONCURRENT; // Why concurrent?

	// These conversions don't require command buffers
	core::smart_refctd_ptr<video::IGPUComputePipeline> gpuComputePipeline = CPU2GPU.getGPUObjectsFromAssets(&computePipeline.get(), &computePipeline.get() + 1, cpu2gpuParams)->front();
	auto* ds0layoutCompute = computeLayout->getDescriptorSetLayout(0);
	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuDs0layoutCompute = CPU2GPU.getGPUObjectsFromAssets(&ds0layoutCompute,&ds0layoutCompute+1,cpu2gpuParams)->front();

	UBOCompute uboComputeData;

	core::vector<core::vector3df_SIMD> particlePos;
	particlePos.reserve(PARTICLE_COUNT);
	for (int32_t i = 0; i < PARTICLE_COUNT_PER_AXIS; ++i)
		for (int32_t j = 0; j < PARTICLE_COUNT_PER_AXIS; ++j)
			for (int32_t k = 0; k < PARTICLE_COUNT_PER_AXIS; ++k)
				particlePos.push_back(core::vector3df_SIMD(i, j, k) * 0.5f);

	constexpr size_t BUF_SZ = 4ull * sizeof(float) * PARTICLE_COUNT;
	video::IGPUBuffer::SCreationParams bufferCreationParams = {};
	bufferCreationParams.usage = static_cast<asset::IBuffer::E_USAGE_FLAGS>(asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_STORAGE_BUFFER_BIT | asset::IBuffer::EUF_VERTEX_BUFFER_BIT);
	auto gpuParticleBuf = device->createDeviceLocalGPUBufferOnDedMem(bufferCreationParams, 2ull * BUF_SZ);
	asset::SBufferRange<video::IGPUBuffer> range;
	range.buffer = gpuParticleBuf;
	range.offset = POS_BUF_IX * BUF_SZ;
	range.size = BUF_SZ;
	utils->updateBufferRangeViaStagingBuffer(queue, range, particlePos.data());
	particlePos.clear();

	auto devLocalReqs = device->getDeviceLocalGPUMemoryReqs();

	devLocalReqs.vulkanReqs.size = core::roundUp(sizeof(UBOCompute), 64ull);

	video::IGPUBuffer::SCreationParams uboComputeCreationParams = {};
	uboComputeCreationParams.usage = static_cast<asset::IBuffer::E_USAGE_FLAGS>(asset::IBuffer::EUF_UNIFORM_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT);
	auto gpuUboCompute = device->createGPUBufferOnDedMem(uboComputeCreationParams, devLocalReqs, true);
	auto gpuds0Compute = device->createGPUDescriptorSet(dscPool.get(), std::move(gpuDs0layoutCompute));
	{
		video::IGPUDescriptorSet::SDescriptorInfo i[3];
		video::IGPUDescriptorSet::SWriteDescriptorSet w[2];
		w[0].arrayElement = 0u;
		w[0].binding = PARTICLE_BUF_BINDING;
		w[0].count = BUF_COUNT;
		w[0].descriptorType = asset::EDT_STORAGE_BUFFER;
		w[0].dstSet = gpuds0Compute.get();
		w[0].info = i;
		w[1].arrayElement = 0u;
		w[1].binding = COMPUTE_DATA_UBO_BINDING;
		w[1].count = 1u;
		w[1].descriptorType = asset::EDT_UNIFORM_BUFFER;
		w[1].dstSet = gpuds0Compute.get();
		w[1].info = i+2u;
		i[0].desc = gpuParticleBuf;
		i[0].buffer.offset = 0ull;
		i[0].buffer.size = BUF_SZ;
		i[1].desc = gpuParticleBuf;
		i[1].buffer.offset = BUF_SZ;
		i[1].buffer.size = BUF_SZ;
		i[2].desc = gpuUboCompute;
		i[2].buffer.offset = 0ull;
		i[2].buffer.size = gpuUboCompute->getBufferSize();

		device->updateDescriptorSets(2u, w, 0u, nullptr);
	}

	asset::SBufferBinding<video::IGPUBuffer> vtxBindings[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
	vtxBindings[0].buffer = gpuParticleBuf;
	vtxBindings[0].offset = 0u;
	//auto meshbuffer = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(nullptr, nullptr, vtxBindings, asset::SBufferBinding<video::IGPUBuffer>{});
	//meshbuffer->setIndexCount(PARTICLE_COUNT);
	//meshbuffer->setIndexType(asset::EIT_UNKNOWN);

	auto createSpecShader = [&](const char* filepath, asset::IShader::E_SHADER_STAGE stage)
	{
		auto shaderBundle = assetManager->getAsset(filepath, {});
		auto shaderContents = shaderBundle.getContents();
		if (shaderContents.empty())
			assert(false);

		return core::smart_refctd_ptr<asset::ICPUSpecializedShader>(static_cast<nbl::asset::ICPUSpecializedShader*>(shaderContents.begin()->get()));
#if 0
		
		auto file = am->getFileSystem()->createAndOpenFile(filepath);
		auto unspec = am->getGLSLCompiler()->resolveIncludeDirectives(file, stage, file->getFileName().c_str());

		asset::ISpecializedShader::SInfo info(nullptr, nullptr, "main", stage, file->getFileName().c_str());
		file->drop();
		return core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspec), std::move(info));
#endif
	};
	auto vs = createSpecShader("../particles.vert", asset::IShader::ESS_VERTEX);
	auto fs = createSpecShader("../particles.frag", asset::IShader::ESS_FRAGMENT);

	asset::ICPUSpecializedShader* shaders[2] = { vs.get(),fs.get() };
	// auto pipeline = introspector.createApproximateRenderpassIndependentPipelineFromIntrospection(shaders,shaders+2,glslExts->begin(),glslExts->end());
	auto pipeline = introspector.createApproximateRenderpassIndependentPipelineFromIntrospection(shaders,shaders+2,nullptr,nullptr);
	{
		auto& vtxParams = pipeline->getVertexInputParams();
		vtxParams.attributes[0].binding = 0u;
		vtxParams.attributes[0].format = asset::EF_R32G32B32_SFLOAT;
		vtxParams.attributes[0].relativeOffset = 0u;
		vtxParams.bindings[0].inputRate = asset::EVIR_PER_VERTEX;
		vtxParams.bindings[0].stride = 4u * sizeof(float);

		pipeline->getPrimitiveAssemblyParams().primitiveType = asset::EPT_POINT_LIST;

		auto& blendParams = pipeline->getBlendParams();
		blendParams.logicOpEnable = false;
		blendParams.logicOp = nbl::asset::ELO_NO_OP;
		for (size_t i = 0ull; i < nbl::asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
			blendParams.blendParams[i].attachmentEnabled = (i == 0ull);
	}
	auto gfxLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(nullptr,nullptr,core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(pipeline->getLayout()->getDescriptorSetLayout(0)));
	pipeline->setLayout(core::smart_refctd_ptr(gfxLayout));

	core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> rpIndependentPipeline = CPU2GPU.getGPUObjectsFromAssets(&pipeline.get(),&pipeline.get()+1,cpu2gpuParams)->front();
	auto* ds0layoutGraphics = gfxLayout->getDescriptorSetLayout(0);
	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuDs0layoutGraphics = CPU2GPU.getGPUObjectsFromAssets(&ds0layoutGraphics, &ds0layoutGraphics + 1, cpu2gpuParams)->front();
	auto gpuds0Graphics = device->createGPUDescriptorSet(dscPool.get(), std::move(gpuDs0layoutGraphics));
	
	video::IGPUGraphicsPipeline::SCreationParams gp_params;
	gp_params.rasterizationSamplesHint = asset::IImage::ESCF_1_BIT;
	gp_params.renderpass = core::smart_refctd_ptr<video::IGPURenderpass>(renderpass);
	gp_params.renderpassIndependent = core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>(rpIndependentPipeline);
	gp_params.subpassIx = 0u;

	auto graphicsPipeline = device->createGPUGraphicsPipeline(nullptr, std::move(gp_params));

	constexpr uint32_t GRAPHICS_SET = 0u;
	constexpr uint32_t GRAPHICS_DATA_UBO_BINDING = 0u;
	asset::SBasicViewParameters viewParams;
	devLocalReqs.vulkanReqs.size = sizeof(viewParams);
	video::IGPUBuffer::SCreationParams gfxUboCreationParams = {};
	gfxUboCreationParams.usage = static_cast<asset::IBuffer::E_USAGE_FLAGS>(asset::IBuffer::EUF_UNIFORM_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT);
	auto gpuUboGraphics = device->createGPUBufferOnDedMem(gfxUboCreationParams, devLocalReqs, true);
	{
		video::IGPUDescriptorSet::SWriteDescriptorSet w;
		video::IGPUDescriptorSet::SDescriptorInfo i;
		w.arrayElement = 0u;
		w.binding = GRAPHICS_DATA_UBO_BINDING;
		w.count = 1u;
		w.descriptorType = asset::EDT_UNIFORM_BUFFER;
		w.dstSet = gpuds0Graphics.get();
		w.info = &i;
		i.desc = gpuUboGraphics;
		i.buffer.offset = 0u;
		i.buffer.size = gpuUboGraphics->getBufferSize();

		device->updateDescriptorSets(1u, &w, 0u, nullptr);
	}

	auto lastTime = std::chrono::high_resolution_clock::now();
	constexpr uint32_t FRAME_COUNT = 500000u;
	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	asset::SBufferRange<video::IGPUBuffer> computeUBORange{ 0, gpuUboCompute->getSize(), gpuUboCompute };
	asset::SBufferRange<video::IGPUBuffer> graphicsUBORange{ 0, gpuUboGraphics->getSize(), gpuUboGraphics };
	
	core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf[FRAMES_IN_FLIGHT];
	device->createCommandBuffers(cmdpool.get(),video::IGPUCommandBuffer::EL_PRIMARY,FRAMES_IN_FLIGHT,cmdbuf);

	core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	for (uint32_t i=0u; i<FRAMES_IN_FLIGHT; i++)
	{
		imageAcquire[i] = device->createSemaphore();
		renderFinished[i] = device->createSemaphore();
	}

	// render loop
	uint32_t i = 0u;
	while (windowCb->isWindowOpen())
	{
		const auto resourceIx = i++%FRAMES_IN_FLIGHT;
		auto& cb = cmdbuf[resourceIx];
		auto& fence = frameComplete[resourceIx];
		if (fence)
		{
			auto retval = device->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT);
			assert(retval == video::IGPUFence::ES_TIMEOUT || retval == video::IGPUFence::ES_SUCCESS);
			device->resetFences(1u, &fence.get());
		}
		else
		{
			fence = device->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
		}

		// safe to proceed
		cb->begin(0);
		
		{
			auto time = std::chrono::high_resolution_clock::now();
			core::vector3df_SIMD gravPoint = cameraPosition+camFront*250.f;
			uboComputeData.gravPointAndDt = gravPoint;
			uboComputeData.gravPointAndDt.w = std::chrono::duration_cast<std::chrono::milliseconds>(time-lastTime).count()*1e-4;

			lastTime = time;
			cb->updateBuffer(computeUBORange.buffer.get(),computeUBORange.offset,computeUBORange.size,&uboComputeData);
		}
		cb->bindComputePipeline(gpuComputePipeline.get());
		cb->bindDescriptorSets(asset::EPBP_COMPUTE,
			gpuComputePipeline->getLayout(),
			COMPUTE_SET,
			1u,
			&gpuds0Compute.get(),
			nullptr);
		cb->dispatch(PARTICLE_COUNT/WORKGROUP_SIZE,1u,1u);

		asset::SMemoryBarrier memBarrier;
		memBarrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
		memBarrier.dstAccessMask = asset::EAF_VERTEX_ATTRIBUTE_READ_BIT;
		cb->pipelineBarrier(
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EPSF_VERTEX_INPUT_BIT,
			static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
			1, &memBarrier,
			0, nullptr,
			0, nullptr);

		{
			memcpy(viewParams.MVP,&viewProj,sizeof(viewProj));
			cb->updateBuffer(graphicsUBORange.buffer.get(),graphicsUBORange.offset,graphicsUBORange.size,&viewParams);
		}
		{
			asset::SViewport vp;
			vp.minDepth = 1.f;
			vp.maxDepth = 0.f;
			vp.x = 0u;
			vp.y = 0u;
			vp.width = WIN_W;
			vp.height = WIN_H;
			cb->setViewport(0u, 1u, &vp);

			VkRect2D scissor;
			scissor.offset = { 0, 0 };
			scissor.extent = { WIN_W, WIN_H };
			cb->setScissor(0u, 1u, &scissor);
		}
		// renderpass 
		uint32_t imgnum = 0u;
		sc->acquireNextImage(MAX_TIMEOUT,imageAcquire[resourceIx].get(),nullptr,&imgnum);
		{
			video::IGPUCommandBuffer::SRenderpassBeginInfo info;
			asset::SClearValue clear;
			clear.color.float32[0] = 0.f;
			clear.color.float32[1] = 0.f;
			clear.color.float32[2] = 0.f;
			clear.color.float32[3] = 1.f;
			info.renderpass = renderpass;
			info.framebuffer = fbo[imgnum];
			info.clearValueCount = 1u;
			info.clearValues = &clear;
			info.renderArea.offset = { 0, 0 };
			info.renderArea.extent = { WIN_W, WIN_H };
			cb->beginRenderPass(&info,asset::ESC_INLINE);
		}
		// individual draw
		{
			cb->bindGraphicsPipeline(graphicsPipeline.get());
			size_t vbOffset = 0;
			cb->bindVertexBuffers(0, 1, &gpuParticleBuf.get(), &vbOffset);
			cb->bindDescriptorSets(asset::EPBP_GRAPHICS,rpIndependentPipeline->getLayout(),GRAPHICS_SET,1u,&gpuds0Graphics.get(),nullptr);
			cb->draw(PARTICLE_COUNT, 1, 0, 0);
		}
		cb->endRenderPass();
		cb->end();

		CommonAPI::Submit(
			device.get(),
			sc.get(),
			cb.get(),
			queue,
			imageAcquire[resourceIx].get(),
			renderFinished[resourceIx].get(),
			fence.get());

		CommonAPI::Present(
			device.get(),
			sc.get(),
			queue,
			renderFinished[resourceIx].get(),
			imgnum);
	}

	device->waitIdle();

	return 0;
}


// If you see this line of code, i forgot to remove it
// It forces the usage of NVIDIA GPU by OpenGL
extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }