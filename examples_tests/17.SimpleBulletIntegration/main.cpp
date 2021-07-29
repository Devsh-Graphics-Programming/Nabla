// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "CFileSystem.h"
using namespace nbl;
using namespace core;

const char* vertexSource = R"===(
#version 430 core
layout(location = 0) in vec4 vPos;
layout(location = 3) in vec3 vNormal;
layout(location = 4) in vec4 vCol;

#include <nbl/builtin/glsl/utils/common.glsl>
#include <nbl/builtin/glsl/utils/transform.glsl>

layout( push_constant, row_major ) uniform Block {
	mat4 modelViewProj;
} PushConstants;

layout(location = 0) out vec3 Color; //per vertex output color, will be interpolated across the triangle

void main()
{
    vec4 pos = PushConstants.modelViewProj*vPos;
	pos += vec4(5.0f * gl_InstanceIndex, 0.0f, 0.0f, 0.0f);
	gl_Position = pos;
    // Color = vNormal*0.5+vec3(0.5);
	Color = vCol.xyz;
}
)===";

const char* fragmentSource = R"===(
#version 430 core

layout(location = 0) in vec3 Color;

layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = vec4(Color,1.0);
}
)===";


static core::smart_refctd_ptr<asset::ICPUMeshBuffer> createMeshBufferFromGeomCreatorReturnType(
	asset::IGeometryCreator::return_type& _data,
	asset::IAssetManager* _manager,
	asset::ICPUSpecializedShader** shadersBegin, asset::ICPUSpecializedShader** shadersEnd)
{
	//creating pipeline just to forward vtx and primitive params
	auto pipeline = core::make_smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>(
		nullptr, shadersBegin, shadersEnd, 
		_data.inputParams, 
		asset::SBlendParams(),
		_data.assemblyParams,
		asset::SRasterizationParams()
		);

	auto mb = core::make_smart_refctd_ptr<asset::ICPUMeshBuffer>(
		nullptr, nullptr,
		_data.bindings, std::move(_data.indexBuffer)
	);

	mb->setIndexCount(_data.indexCount);
	mb->setIndexType(_data.indexType);
	mb->setBoundingBox(_data.bbox);
	mb->setPipeline(std::move(pipeline));
	constexpr auto NORMAL_ATTRIBUTE = 3;
	mb->setNormalAttributeIx(NORMAL_ATTRIBUTE);

	return mb;

}

int main()
{
	constexpr uint32_t WIN_W = 1280;
	constexpr uint32_t WIN_H = 720;
	constexpr uint32_t SC_IMG_COUNT = 3u;
	constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT>SC_IMG_COUNT);

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

	// Instance Data
	constexpr uint32_t NumInstances = 2;

	struct InstanceData {
		core::vector3df_SIMD col;
	};


	// Asset Manager

	auto filesystem = core::make_smart_refctd_ptr<io::CFileSystem>("");
	auto am = core::make_smart_refctd_ptr<asset::IAssetManager>(std::move(filesystem));


	// CPU2GPU

	video::IGPUObjectFromAssetConverter CPU2GPU;

	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	cpu2gpuParams.assetManager = am.get();
	cpu2gpuParams.device = device.get();
	cpu2gpuParams.perQueue[video::IGPUObjectFromAssetConverter::EQU_TRANSFER].queue = queue;
	cpu2gpuParams.perQueue[video::IGPUObjectFromAssetConverter::EQU_COMPUTE].queue = queue;
	cpu2gpuParams.finalQueueFamIx = queue->getFamilyIndex();
	cpu2gpuParams.sharingMode = asset::ESM_CONCURRENT;
	cpu2gpuParams.limits = gpu->getLimits();
	cpu2gpuParams.assetManager = am.get();

	// weird fix -> do not read the next 6 lines (It doesn't affect the program logically) -> waiting for access_violation_repro branch to fix and merge
	core::smart_refctd_ptr<asset::ICPUShader> computeUnspec;
	{
		 auto file = am->getFileSystem()->createAndOpenFile("../../29.SpecializationConstants/particles.comp");
		 computeUnspec = am->getGLSLCompiler()->resolveIncludeDirectives(file, asset::ISpecializedShader::ESS_COMPUTE, file->getFileName().c_str());
		 file->drop();
	}

	// Geom Create

	auto geometryCreator = am->getGeometryCreator();
	auto cubeMesh = geometryCreator->createCubeMesh(core::vector3df(2.0f,2.0f,2.0f));

	// Camera Stuff
	core::vectorSIMDf cameraPosition(0, 0, -10);
	matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(60), float(WIN_W) / WIN_H, 0.01, 100);
	matrix3x4SIMD view = matrix3x4SIMD::buildCameraLookAtMatrixRH(cameraPosition, core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 1, 0));
	auto viewProj = matrix4SIMD::concatenateBFollowedByA(proj, matrix4SIMD(view));

	// Creating CPU Shaders 

	auto createCPUSpecializedShaderFromSource = [=](const char* source, asset::ISpecializedShader::E_SHADER_STAGE stage) -> core::smart_refctd_ptr<asset::ICPUSpecializedShader>
	{
		auto unspec = am->getGLSLCompiler()->createSPIRVFromGLSL(source, stage, "main", "runtimeID");
		if (!unspec)
			return nullptr;

		asset::ISpecializedShader::SInfo info(nullptr, nullptr, "main", stage, "");
		return core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspec), std::move(info));
	};

	auto createCPUSpecializedShaderFromSourceWithIncludes = [&](const char* source, asset::ISpecializedShader::E_SHADER_STAGE stage, const char* origFilepath)
	{
		auto resolved_includes = am->getGLSLCompiler()->resolveIncludeDirectives(source, stage, origFilepath);
		return createCPUSpecializedShaderFromSource(reinterpret_cast<const char*>(resolved_includes->getSPVorGLSL()->getPointer()), stage);
	};

	auto vs = createCPUSpecializedShaderFromSourceWithIncludes(vertexSource,asset::ISpecializedShader::ESS_VERTEX, "shader.vert");
	auto fs = createCPUSpecializedShaderFromSource(fragmentSource,asset::ISpecializedShader::ESS_FRAGMENT);
	asset::ICPUSpecializedShader* shaders[2]{ vs.get(),fs.get() };

	auto cpuMesh = createMeshBufferFromGeomCreatorReturnType(cubeMesh, am.get(), shaders, shaders+2);
	
	// Instances Buffer

	core::vector<InstanceData> instanceData;
	// for(uint32_t i = 0; i < NumInstances; ++i) {
	// }
	instanceData.push_back(InstanceData{core::vector3df_SIMD(1.0f, 0.3f, 1.0f)});
	instanceData.push_back(InstanceData{core::vector3df_SIMD(0.1f, 0.0f, 0.4f)});
	
	constexpr size_t BUF_SZ = sizeof(InstanceData) * NumInstances;
	auto gpuInstancesBuffer = device->createDeviceLocalGPUBufferOnDedMem(BUF_SZ);
	{
		asset::SBufferRange<video::IGPUBuffer> range;
		range.buffer = gpuInstancesBuffer;
		range.offset = 0;
		range.size = BUF_SZ;
		device->updateBufferRangeViaStagingBuffer(queue, range, instanceData.data());
	}

	auto pipeline = cpuMesh->getPipeline();
	{
		// we're working with RH coordinate system(view proj) and in that case the cubeMesh frontFace is NOT CCW.
		pipeline->getRasterizationParams().frontFaceIsCCW = 0;

		auto & vtxinputParams = pipeline->getVertexInputParams();
		vtxinputParams.bindings[1].inputRate = asset::EVIR_PER_INSTANCE;
		vtxinputParams.bindings[1].stride = sizeof(InstanceData);
		vtxinputParams.attributes[4].binding = 1;
		vtxinputParams.attributes[4].relativeOffset = 0;
		vtxinputParams.attributes[4].format = asset::E_FORMAT::EF_R32G32B32A32_SFLOAT;
		vtxinputParams.enabledAttribFlags |= 0x1u << 4;
		vtxinputParams.enabledBindingFlags |= 0x1u << 1;
// for wireframe rendering
#if 0
		pipeline->getRasterizationParams().polygonMode = asset::EPM_LINE; 
#endif
	}
	
	asset::SPushConstantRange range[1] = { asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD) };
	auto gfxLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(range,range+1u);
	pipeline->setLayout(core::smart_refctd_ptr(gfxLayout));

	core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> rpIndependentPipeline = CPU2GPU.getGPUObjectsFromAssets(&pipeline,&pipeline+1,cpu2gpuParams)->front();
	
	auto gpuMesh = CPU2GPU.getGPUObjectsFromAssets(&cpuMesh.get(), &cpuMesh.get() + 1,cpu2gpuParams)->front();
	
	video::IGPUGraphicsPipeline::SCreationParams gp_params;
	gp_params.rasterizationSamplesHint = asset::IImage::ESCF_1_BIT;
	gp_params.renderpass = core::smart_refctd_ptr<video::IGPURenderpass>(renderpass);
	gp_params.renderpassIndependent = rpIndependentPipeline; // TODO: fix use gpuMesh->getPipeline instead
	gp_params.subpassIx = 0u;

	auto graphicsPipeline = device->createGPUGraphicsPipeline(nullptr, std::move(gp_params));

	auto lastTime = std::chrono::high_resolution_clock::now();
	constexpr uint32_t FRAME_COUNT = 500000u;
	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;

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
	for (uint32_t i = 0u; i < FRAME_COUNT; ++i)
	{
		const auto resourceIx = i%FRAMES_IN_FLIGHT;
		auto& cb = cmdbuf[resourceIx];
		auto& fence = frameComplete[resourceIx];
		if (fence)
		while (device->waitForFences(1u,&fence.get(),false,MAX_TIMEOUT)==video::IGPUFence::ES_TIMEOUT)
		{
		}
		else
			fence = device->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		auto time = std::chrono::high_resolution_clock::now();
		auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(time-lastTime).count();
		lastTime = time;

		// safe to proceed
		cb->begin(0);

		{
			asset::SViewport vp;
			vp.minDepth = 1.f;
			vp.maxDepth = 0.f;
			vp.x = 0u;
			vp.y = 0u;
			vp.width = WIN_W;
			vp.height = WIN_H;
			cb->setViewport(0u, 1u, &vp);
		}
		// renderpass 
		uint32_t imgnum = 0u;
		sc->acquireNextImage(MAX_TIMEOUT,imageAcquire[resourceIx].get(),nullptr,&imgnum);
		{
			video::IGPUCommandBuffer::SRenderpassBeginInfo info;
			asset::SClearValue clear;
			asset::VkRect2D area;
			clear.color.float32[0] = 0.1f;
			clear.color.float32[1] = 0.1f;
			clear.color.float32[2] = 0.1f;
			clear.color.float32[3] = 1.f;
			info.renderpass = renderpass;
			info.framebuffer = fbo[imgnum];
			info.clearValueCount = 1u;
			info.clearValues = &clear;
			info.renderArea.offset = { 0, 0 };
			info.renderArea.extent = { WIN_W, WIN_H };
			cb->beginRenderPass(&info,asset::ESC_INLINE);
		}
		// draw
		{
			// Animate Stuff
			
			core::matrix3x4SIMD modelMatrix;
			static double rot = 0;
			rot += dt * 0.0005f;
			
			modelMatrix.setRotation(nbl::core::quaternion(0, rot, 0));
			core::matrix4SIMD mvp = core::concatenateBFollowedByA(viewProj, modelMatrix);

			// Draw Stuff 

			cb->bindGraphicsPipeline(graphicsPipeline.get());
			
			// TODO:  for the fututre: cb->drawMeshBuffer(gpuMesh); instead of binding vertex/index buffer explicitly
			
			video::IGPUBuffer const * vbuffers[2] = {
				gpuMesh->getVertexBufferBindings()[0].buffer.get(), // vertex buffer
				gpuInstancesBuffer.get(), // instance buffer
			};

			uint64_t vbuf_offsets[2] = {
				gpuMesh->getVertexBufferBindings()[0].offset,
				0
			};

			auto ibuf = gpuMesh->getIndexBufferBinding().buffer.get();
			auto ibuf_offset = gpuMesh->getIndexBufferBinding().offset;

			cb->bindVertexBuffers(0, 2, vbuffers, vbuf_offsets);
			cb->bindIndexBuffer(ibuf, ibuf_offset, gpuMesh->getIndexType());

			cb->pushConstants(rpIndependentPipeline->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), mvp.pointer());
			cb->drawIndexed(gpuMesh->getIndexCount(), NumInstances, 0, 0, 0);
		}
		cb->endRenderPass();
		cb->end();

		CommonAPI::Submit(device.get(), sc.get(), cb.get(), queue, imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
		CommonAPI::Present(device.get(), sc.get(), queue, renderFinished[resourceIx].get(), imgnum);
	}

	return 0;
}


// If you see this line of code, i forgot to remove it
// It forces the usage of NVIDIA GPU by OpenGL
extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }