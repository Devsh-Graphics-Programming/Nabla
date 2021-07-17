// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "CFileSystem.h"
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

	auto initOutp = CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(video::EAT_OPENGL, "Specialization constants");
	auto win = std::move(initOutp.window);
	auto gl = std::move(initOutp.apiConnection);
	auto surface = std::move(initOutp.surface);
	auto device = std::move(initOutp.logicalDevice);
	auto gpu = std::move(initOutp.physicalDevice);
	auto queue = std::move(initOutp.queue);
	auto sc = std::move(initOutp.swapchain);
	auto renderpass = std::move(initOutp.renderpass);
	auto fbo = std::move(initOutp.fbo);
	auto cmdpool = std::move(initOutp.commandPool);
	{
		video::IDriverMemoryBacked::SDriverMemoryRequirements mreq;
		core::smart_refctd_ptr<video::IGPUCommandBuffer> cb;
		device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cb);
		assert(cb);

		cb->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

		asset::SViewport vp;
		vp.minDepth = 1.f;
		vp.maxDepth = 0.f;
		vp.x = 0u;
		vp.y = 0u;
		vp.width = WIN_W;
		vp.height = WIN_H;
		cb->setViewport(0u, 1u, &vp);

		cb->end();

		video::IGPUQueue::SSubmitInfo info;
		auto* cb_ = cb.get();
		info.commandBufferCount = 1u;
		info.commandBuffers = &cb_;
		info.pSignalSemaphores = nullptr;
		info.signalSemaphoreCount = 0u;
		info.pWaitSemaphores = nullptr;
		info.waitSemaphoreCount = 0u;
		info.pWaitDstStageMask = nullptr;
		queue->submit(1u, &info, nullptr);
	}
	video::IDescriptorPool::SDescriptorPoolSize poolSize[2];
	poolSize[0].count = 1;
	poolSize[0].type = asset::EDT_STORAGE_BUFFER;
	poolSize[1].count = 1;
	poolSize[1].type = asset::EDT_UNIFORM_BUFFER;

	auto dscPool = device->createDescriptorPool(video::IDescriptorPool::ECF_FREE_DESCRIPTOR_SET_BIT, 2, 2, poolSize);

	auto filesystem = core::make_smart_refctd_ptr<io::CFileSystem>("");
	auto am = core::make_smart_refctd_ptr<asset::IAssetManager>(std::move(filesystem));
	video::IGPUObjectFromAssetConverter CPU2GPU;
	core::vectorSIMDf cameraPosition(0, 0, -10);
	matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(90), float(WIN_W) / WIN_H, 0.01, 100);
	matrix3x4SIMD view = matrix3x4SIMD::buildCameraLookAtMatrixRH(cameraPosition, core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 1, 0));
	auto viewProj = matrix4SIMD::concatenateBFollowedByA(proj, matrix4SIMD(view));
	auto camFront = view[2];

	auto glslExts = device->getSupportedGLSLExtensions();
	asset::CShaderIntrospector introspector(am->getGLSLCompiler());

	core::smart_refctd_ptr<asset::ICPUShader> computeUnspec;
	{
		auto file = am->getFileSystem()->createAndOpenFile("../particles.comp");
		computeUnspec = am->getGLSLCompiler()->resolveIncludeDirectives(file, asset::ISpecializedShader::ESS_COMPUTE, file->getFileName().c_str());
		file->drop();
	}
	const asset::CIntrospectionData* introspection = nullptr;
	{
		asset::CShaderIntrospector::SIntrospectionParams params;
		params.entryPoint = "main";
		params.filePathHint = "../particles.comp";
		params.GLSLextensions = glslExts;
		params.stage = asset::ISpecializedShader::ESS_COMPUTE;
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

		specInfo = asset::ISpecializedShader::SInfo(std::move(entries), std::move(backbuf), "main", asset::ISpecializedShader::ESS_COMPUTE, "../particles.comp");
	}
	auto compute = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(computeUnspec), std::move(specInfo));

	auto computePipeline = introspector.createApproximateComputePipelineFromIntrospection(compute.get(), glslExts->begin(), glslExts->end());
	auto computeLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(nullptr,nullptr,core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(computePipeline->getLayout()->getDescriptorSetLayout(0)));
	computePipeline->setLayout(core::smart_refctd_ptr(computeLayout));

	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	cpu2gpuParams.assetManager = am.get();
	cpu2gpuParams.device = device.get();
	cpu2gpuParams.perQueue[video::IGPUObjectFromAssetConverter::EQU_TRANSFER].queue = queue;
	cpu2gpuParams.perQueue[video::IGPUObjectFromAssetConverter::EQU_COMPUTE].queue = queue;
	cpu2gpuParams.finalQueueFamIx = queue->getFamilyIndex();
	cpu2gpuParams.sharingMode = asset::ESM_CONCURRENT;
	cpu2gpuParams.limits = gpu->getLimits();
	cpu2gpuParams.assetManager = am.get();
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
	auto gpuParticleBuf = device->createDeviceLocalGPUBufferOnDedMem(2ull * BUF_SZ);
	asset::SBufferRange<video::IGPUBuffer> range;
	range.buffer = gpuParticleBuf;
	range.offset = POS_BUF_IX * BUF_SZ;
	range.size = BUF_SZ;
	device->updateBufferRangeViaStagingBuffer(queue, range, particlePos.data());
	particlePos.clear();

	auto gpuUboCompute = device->createDeviceLocalGPUBufferOnDedMem(core::roundUp(sizeof(UBOCompute), 64ull));
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
		i[2].buffer.size = gpuUboCompute->getSize();

		device->updateDescriptorSets(2u, w, 0u, nullptr);
	}

	asset::SBufferBinding<video::IGPUBuffer> vtxBindings[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
	vtxBindings[0].buffer = gpuParticleBuf;
	vtxBindings[0].offset = 0u;
	//auto meshbuffer = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(nullptr, nullptr, vtxBindings, asset::SBufferBinding<video::IGPUBuffer>{});
	//meshbuffer->setIndexCount(PARTICLE_COUNT);
	//meshbuffer->setIndexType(asset::EIT_UNKNOWN);

	auto createSpecShader = [&](const char* filepath, asset::ISpecializedShader::E_SHADER_STAGE stage) {
		auto file = am->getFileSystem()->createAndOpenFile(filepath);
		auto unspec = am->getGLSLCompiler()->resolveIncludeDirectives(file, stage, file->getFileName().c_str());

		asset::ISpecializedShader::SInfo info(nullptr, nullptr, "main", stage, file->getFileName().c_str());
		file->drop();
		return core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspec), std::move(info));
	};
	auto vs = createSpecShader("../particles.vert", asset::ISpecializedShader::ESS_VERTEX);
	auto fs = createSpecShader("../particles.frag", asset::ISpecializedShader::ESS_FRAGMENT);
	asset::ICPUSpecializedShader* shaders[2]{ vs.get(),fs.get() };
	auto pipeline = introspector.createApproximateRenderpassIndependentPipelineFromIntrospection(shaders,shaders+2,glslExts->begin(),glslExts->end());
	{
		auto& vtxParams = pipeline->getVertexInputParams();
		vtxParams.attributes[0].binding = 0u;
		vtxParams.attributes[0].format = asset::EF_R32G32B32_SFLOAT;
		vtxParams.attributes[0].relativeOffset = 0u;
		vtxParams.bindings[0].inputRate = asset::EVIR_PER_VERTEX;
		vtxParams.bindings[0].stride = 4u * sizeof(float);

		pipeline->getPrimitiveAssemblyParams().primitiveType = asset::EPT_POINT_LIST;
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
	auto gpuUboGraphics = device->createDeviceLocalGPUBufferOnDedMem(sizeof(viewParams));
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
		i.buffer.size = gpuUboGraphics->getSize();

		device->updateDescriptorSets(1u, &w, 0u, nullptr);
	}

	auto lastTime = std::chrono::high_resolution_clock::now();
	constexpr uint32_t FRAME_COUNT = 500000u;
	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	asset::SBufferRange<video::IGPUBuffer> computeUBORange{ 0, gpuUboCompute->getSize(), gpuUboCompute };
	asset::SBufferRange<video::IGPUBuffer> graphicsUBORange{ 0, gpuUboGraphics->getSize(), gpuUboGraphics };
	for (uint32_t i = 0u; i < FRAME_COUNT; ++i)
	{
		memcpy(viewParams.MVP,&viewProj,sizeof(viewProj));
		device->updateBufferRangeViaStagingBuffer(queue,graphicsUBORange,&viewParams);
		core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf[SC_IMG_COUNT];
		device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, SC_IMG_COUNT, cmdbuf);
		for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
		{
			auto& cb = cmdbuf[i];
			auto& fb = fbo[i];

			cb->begin(0);

			size_t offset = 0u;
			video::IGPUCommandBuffer::SRenderpassBeginInfo info;
			asset::SClearValue clear;
			asset::VkRect2D area;
			area.offset = { 0, 0 };
			area.extent = { WIN_W, WIN_H };
			clear.color.float32[0] = 0.f;
			clear.color.float32[1] = 0.f;
			clear.color.float32[2] = 0.f;
			clear.color.float32[3] = 1.f;
			info.renderpass = renderpass;
			info.framebuffer = fb;
			info.clearValueCount = 1u;
			info.clearValues = &clear;
			info.renderArea = area;

			//TODO: make those functions take const pointers
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
			cb->pipelineBarrier(asset::EPSF_COMPUTE_SHADER_BIT, asset::EPSF_VERTEX_INPUT_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

			cb->bindGraphicsPipeline(graphicsPipeline.get());
			size_t vbOffset = 0;
			cb->bindVertexBuffers(0, 1, &gpuParticleBuf.get(), &vbOffset);
			cb->bindDescriptorSets(asset::EPBP_GRAPHICS,rpIndependentPipeline->getLayout(),GRAPHICS_SET,1u,&gpuds0Graphics.get(),nullptr);
			cb->beginRenderPass(&info, asset::ESC_INLINE);
			cb->draw(PARTICLE_COUNT, 1, 0, 0);
			cb->endRenderPass();

			cb->end();
		}

		auto img_acq_sem = device->createSemaphore();
		auto render1_finished_sem = device->createSemaphore();

		uint32_t imgnum = 0u;
		sc->acquireNextImage(MAX_TIMEOUT, img_acq_sem.get(), nullptr, &imgnum);

		core::vector3df_SIMD gravPoint = cameraPosition + camFront * 250.f;
		auto time = std::chrono::high_resolution_clock::now();
		uboComputeData.gravPointAndDt = gravPoint;
		uboComputeData.gravPointAndDt.w = std::chrono::duration_cast<std::chrono::milliseconds>((time - lastTime)).count() * 1e-4;
		device->updateBufferRangeViaStagingBuffer(queue,computeUBORange,&uboComputeData);
		lastTime = time;
		CommonAPI::Submit(device.get(), sc.get(), cmdbuf, queue, img_acq_sem.get(), render1_finished_sem.get(), SC_IMG_COUNT, imgnum);

		CommonAPI::Present(device.get(), sc.get(), queue, render1_finished_sem.get(), imgnum);
	}

	return 0;
}


// If you see this line of code, i forgot to remove it
// It forces the usage of NVIDIA GPU by OpenGL
extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }