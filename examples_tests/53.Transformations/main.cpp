// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"

using namespace nbl;
using namespace core;

const char* vertexSource = R"===(
#version 430 core
layout(location = 0) in vec4 vPos;
layout(location = 3) in vec3 vNormal;
layout(location = 4) in vec4 vCol;

layout(location = 5) in vec4 vWorldMatRow0;
layout(location = 6) in vec4 vWorldMatRow1;
layout(location = 7) in vec4 vWorldMatRow2;

layout( push_constant, row_major ) uniform Block {
	mat4 viewProj;
} PushConstants;

layout(location = 0) out vec3 Color;
layout(location = 1) out vec3 Normal;

void main()
{
	vec4 worldPos = vec4(dot(vWorldMatRow0, vPos), dot(vWorldMatRow1, vPos), dot(vWorldMatRow2, vPos), 1);
    vec4 pos = PushConstants.viewProj*worldPos;
	gl_Position = pos;
	Color = vCol.xyz;

	mat3x4 transposeWorldMat = mat3x4(vWorldMatRow0, vWorldMatRow1, vWorldMatRow2);
	mat3 inverseTransposeWorld = inverse(mat3(transposeWorldMat));
	Normal = inverseTransposeWorld * normalize(vNormal);
}
)===";

const char* fragmentSource = R"===(
#version 430 core

layout(location = 0) in vec3 Color;
layout(location = 1) in vec3 Normal;

layout(location = 0) out vec4 pixelColor;

void main()
{
    vec3 normal = normalize(Normal);

    float ambient = 0.2;
    float diffuse = 0.8;
    float cos_theta_term = max(dot(normal,vec3(3.0,5.0,-4.0)),0.0);

    float fresnel = 0.0; //not going to implement yet, not important
    float specular = 0.0;///pow(max(dot(halfVector,normal),0.0),shininess);

    const float sunPower = 3.14156*0.3;

    pixelColor = vec4(Color, 1)*sunPower*(ambient+mix(diffuse,specular,fresnel)*cos_theta_term/3.14159);
}
)===";

struct InstanceData {
	core::vector3df_SIMD color;
	core::matrix3x4SIMD modelMatrix; // = 3 x vector3df_SIMD
};

// I was tempted to name this PlanetData but Sun and Moon are not planets xD
struct SolarSystemObject {
	uint32_t parentIndex = 0u;
	float yRotationSpeed = 0.0f;
	float zRotationSpeed = 0.0f;
	float scale = 1.0f;
	core::vector3df_SIMD initialRelativePosition;
	core::matrix3x4SIMD matForChildren;
};

struct GPUObject {
	core::smart_refctd_ptr<video::IGPUMeshBuffer> gpuMesh;
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> graphicsPipeline;
};

static core::smart_refctd_ptr<asset::ICPUMeshBuffer> createMeshBufferFromGeomCreatorReturnType(
	asset::IGeometryCreator::return_type& _data,
	asset::IAssetManager* _manager,
	asset::ICPUSpecializedShader** _shadersBegin, asset::ICPUSpecializedShader** _shadersEnd)
{
	//creating pipeline just to forward vtx and primitive params
	auto pipeline = core::make_smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>(
		nullptr, _shadersBegin, _shadersEnd, 
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
    constexpr uint32_t FBO_COUNT = 1u;
	constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT>FBO_COUNT);

	auto initOutput = CommonAPI::Init<WIN_W, WIN_H, FBO_COUNT>(video::EAT_OPENGL, "Solar System Transformations", asset::EF_D32_SFLOAT);
	auto system = std::move(initOutput.system);
    auto window = std::move(initOutput.window);
    auto gl = std::move(initOutput.apiConnection);
    auto surface = std::move(initOutput.surface);
    auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
    auto device = std::move(initOutput.logicalDevice);
    auto queues = std::move(initOutput.queues);
    auto graphicsQueue = queues[decltype(initOutput)::EQT_GRAPHICS];
    auto transferUpQueue = queues[decltype(initOutput)::EQT_TRANSFER_UP];
    auto swapchain = std::move(initOutput.swapchain);
    auto renderpass = std::move(initOutput.renderpass);
    auto fbo = std::move(initOutput.fbo[0]);
    auto commandPool = std::move(initOutput.commandPool);
    auto assetManager = std::move(initOutput.assetManager);
    auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);

    nbl::video::IGPUObjectFromAssetConverter CPU2GPU;
	
    core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf[FRAMES_IN_FLIGHT];
    device->createCommandBuffers(commandPool.get(), nbl::video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, cmdbuf);

	core::vector<GPUObject> gpuObjects; 

	// Instance Data
	constexpr float SimulationSpeedScale = 0.03f;
	constexpr uint32_t NumSolarSystemObjects = 11;
	constexpr uint32_t NumInstances = NumSolarSystemObjects;
	
	// SolarSystemObject and InstanceData have 1-to-1 relationship
	core::vector<InstanceData> instancesData;
	core::vector<SolarSystemObject> solarSystemObjectsData;
	instancesData.resize(NumInstances);
	solarSystemObjectsData.resize(NumInstances);
	
	// Sun
	uint32_t constexpr sunIndex = 0u;
	instancesData[sunIndex].color = core::vector3df_SIMD(0.8f, 1.0f, 0.1f);
	solarSystemObjectsData[sunIndex].parentIndex = 0u;
	solarSystemObjectsData[sunIndex].yRotationSpeed = 0.0f;
	solarSystemObjectsData[sunIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[sunIndex].scale = 5.0f;
	solarSystemObjectsData[sunIndex].initialRelativePosition = core::vector3df_SIMD(0.0f, 0.0f, 0.0f);
	
	// Mercury
	uint32_t constexpr mercuryIndex = 1u;
	instancesData[mercuryIndex].color = core::vector3df_SIMD(0.7f, 0.3f, 0.1f);
	solarSystemObjectsData[mercuryIndex].parentIndex = sunIndex;
	solarSystemObjectsData[mercuryIndex].yRotationSpeed = 0.5f;
	solarSystemObjectsData[mercuryIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[mercuryIndex].scale = 0.5f;
	solarSystemObjectsData[mercuryIndex].initialRelativePosition = core::vector3df_SIMD(4.0f, 0.0f, 0.0f);
	
	// Venus
	uint32_t constexpr venusIndex = 2u;
	instancesData[venusIndex].color = core::vector3df_SIMD(0.8f, 0.6f, 0.1f);
	solarSystemObjectsData[venusIndex].parentIndex = sunIndex;
	solarSystemObjectsData[venusIndex].yRotationSpeed = 0.8f;
	solarSystemObjectsData[venusIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[venusIndex].scale = 1.0f;
	solarSystemObjectsData[venusIndex].initialRelativePosition = core::vector3df_SIMD(8.0f, 0.0f, 0.0f);

	// Earth
	uint32_t constexpr earthIndex = 3u;
	instancesData[earthIndex].color = core::vector3df_SIMD(0.1f, 0.4f, 0.8f);
	solarSystemObjectsData[earthIndex].parentIndex = sunIndex;
	solarSystemObjectsData[earthIndex].yRotationSpeed = 1.0f;
	solarSystemObjectsData[earthIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[earthIndex].scale = 2.0f;
	solarSystemObjectsData[earthIndex].initialRelativePosition = core::vector3df_SIMD(12.0f, 0.0f, 0.0f);
	
	// Mars
	uint32_t constexpr marsIndex = 4u;
	instancesData[marsIndex].color = core::vector3df_SIMD(0.9f, 0.3f, 0.1f);
	solarSystemObjectsData[marsIndex].parentIndex = sunIndex;
	solarSystemObjectsData[marsIndex].yRotationSpeed = 2.0f;
	solarSystemObjectsData[marsIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[marsIndex].scale = 1.5f;
	solarSystemObjectsData[marsIndex].initialRelativePosition = core::vector3df_SIMD(16.0f, 0.0f, 0.0f);
	
	// Jupiter
	uint32_t constexpr jupiterIndex = 5u;
	instancesData[jupiterIndex].color = core::vector3df_SIMD(0.6f, 0.4f, 0.4f);
	solarSystemObjectsData[jupiterIndex].parentIndex = sunIndex;
	solarSystemObjectsData[jupiterIndex].yRotationSpeed = 11.0f;
	solarSystemObjectsData[jupiterIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[jupiterIndex].scale = 4.0f;
	solarSystemObjectsData[jupiterIndex].initialRelativePosition = core::vector3df_SIMD(20.0f, 0.0f, 0.0f);
	
	// Saturn
	uint32_t constexpr saturnIndex = 6u;
	instancesData[saturnIndex].color = core::vector3df_SIMD(0.7f, 0.7f, 0.5f);
	solarSystemObjectsData[saturnIndex].parentIndex = sunIndex;
	solarSystemObjectsData[saturnIndex].yRotationSpeed = 30.0f;
	solarSystemObjectsData[saturnIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[saturnIndex].scale = 3.0f;
	solarSystemObjectsData[saturnIndex].initialRelativePosition = core::vector3df_SIMD(24.0f, 0.0f, 0.0f);
	
	// Uranus
	uint32_t constexpr uranusIndex = 7u;
	instancesData[uranusIndex].color = core::vector3df_SIMD(0.4f, 0.4f, 0.6f);
	solarSystemObjectsData[uranusIndex].parentIndex = sunIndex;
	solarSystemObjectsData[uranusIndex].yRotationSpeed = 40.0f;
	solarSystemObjectsData[uranusIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[uranusIndex].scale = 3.5f;
	solarSystemObjectsData[uranusIndex].initialRelativePosition = core::vector3df_SIMD(28.0f, 0.0f, 0.0f);
	
	// Neptune
	uint32_t constexpr neptuneIndex = 8u;
	instancesData[neptuneIndex].color = core::vector3df_SIMD(0.5f, 0.2f, 0.9f);
	solarSystemObjectsData[neptuneIndex].parentIndex = sunIndex;
	solarSystemObjectsData[neptuneIndex].yRotationSpeed = 50.0f;
	solarSystemObjectsData[neptuneIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[neptuneIndex].scale = 4.0f;
	solarSystemObjectsData[neptuneIndex].initialRelativePosition = core::vector3df_SIMD(32.0f, 0.0f, 0.0f);
	
	// Pluto 
	uint32_t constexpr plutoIndex = 9u;
	instancesData[plutoIndex].color = core::vector3df_SIMD(0.7f, 0.5f, 0.5f);
	solarSystemObjectsData[plutoIndex].parentIndex = sunIndex;
	solarSystemObjectsData[plutoIndex].yRotationSpeed = 1.0f;
	solarSystemObjectsData[plutoIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[plutoIndex].scale = 0.5f;
	solarSystemObjectsData[plutoIndex].initialRelativePosition = core::vector3df_SIMD(36.0f, 0.0f, 0.0f);
	
	// Moon
	uint32_t constexpr moonIndex = 10u;
	instancesData[moonIndex].color = core::vector3df_SIMD(0.3f, 0.2f, 0.25f);
	solarSystemObjectsData[moonIndex].parentIndex = earthIndex;
	solarSystemObjectsData[moonIndex].yRotationSpeed = 0.2f;
	solarSystemObjectsData[moonIndex].zRotationSpeed = 0.4f;
	solarSystemObjectsData[moonIndex].scale = 0.4f;
	solarSystemObjectsData[moonIndex].initialRelativePosition = core::vector3df_SIMD(2.5f, 0.0f, 0.0f);

	// weird fix -> do not read the next 6 lines (It doesn't affect the program logically) -> waiting for access_violation_repro branch to fix and merge
	core::smart_refctd_ptr<asset::ICPUShader> computeUnspec;
	{
		system::ISystem::future_t<smart_refctd_ptr<system::IFile>> future;
		system->createFile(future, "../../29.SpecializationConstants/particles.comp", nbl::system::IFile::ECF_READ_WRITE);
		auto file = future.get();
		auto sname = file->getFileName().string();
		char const* shaderName = sname.c_str();//yep, makes sense
		computeUnspec = assetManager->getGLSLCompiler()->resolveIncludeDirectives(file.get(), asset::ISpecializedShader::ESS_COMPUTE, shaderName);
	}

	// Geom Create
	auto geometryCreator = assetManager->getGeometryCreator();
	auto sphereGeom = geometryCreator->createSphereMesh(0.5f);

	// Camera Stuff
	core::vectorSIMDf cameraPosition(0, 20, -50);
	matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(60), float(WIN_W) / WIN_H, 0.01, 100);
	matrix3x4SIMD view = matrix3x4SIMD::buildCameraLookAtMatrixRH(cameraPosition, core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 1, 0));
	auto viewProj = matrix4SIMD::concatenateBFollowedByA(proj, matrix4SIMD(view));

	// Creating CPU Shaders 

	auto createCPUSpecializedShaderFromSource = [=](const char* source, asset::ISpecializedShader::E_SHADER_STAGE stage) -> core::smart_refctd_ptr<asset::ICPUSpecializedShader>
	{
		auto unspec = assetManager->getGLSLCompiler()->createSPIRVFromGLSL(source, stage, "main", "runtimeID");
		if (!unspec)
			return nullptr;

		asset::ISpecializedShader::SInfo info(nullptr, nullptr, "main", stage, "");
		return core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspec), std::move(info));
	};

	auto vs = createCPUSpecializedShaderFromSource(vertexSource,asset::ISpecializedShader::ESS_VERTEX);
	auto fs = createCPUSpecializedShaderFromSource(fragmentSource,asset::ISpecializedShader::ESS_FRAGMENT);
	asset::ICPUSpecializedShader* shaders[2]{ vs.get(), fs.get() };
	
	auto cpuMeshPlanets = createMeshBufferFromGeomCreatorReturnType(sphereGeom, assetManager.get(), shaders, shaders+2);
	
	// Instances Buffer
	
	constexpr size_t BUF_SZ = sizeof(InstanceData) * NumInstances;
	auto gpuInstancesBuffer = device->createDeviceLocalGPUBufferOnDedMem(BUF_SZ);

	// Create GPU Objects (IGPUMeshBuffer + GraphicsPipeline)
	auto createGPUObject = [&](
		asset::ICPUMeshBuffer * cpuMesh,
		core::smart_refctd_ptr<video::IGPUBuffer> instancesBuffer,
		uint64_t numInstances, uint64_t instanceBufferOffset,
		asset::E_FACE_CULL_MODE faceCullingMode = asset::EFCM_BACK_BIT) -> GPUObject {
		GPUObject ret = {};
		
		uint32_t instanceBufferBinding = asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT - 1u;

		auto pipeline = cpuMesh->getPipeline();
		{
			// we're working with RH coordinate system(view proj) and in that case the cubeGeom frontFace is NOT CCW.
			auto & rasterParams = pipeline->getRasterizationParams();
			rasterParams.frontFaceIsCCW = 0;
			rasterParams.faceCullingMode = faceCullingMode;

			auto & vtxinputParams = pipeline->getVertexInputParams();
			vtxinputParams.bindings[instanceBufferBinding].inputRate = asset::EVIR_PER_INSTANCE;
			vtxinputParams.bindings[instanceBufferBinding].stride = sizeof(InstanceData);
			// Color
			vtxinputParams.attributes[4].binding = instanceBufferBinding;
			vtxinputParams.attributes[4].relativeOffset = 0;
			vtxinputParams.attributes[4].format = asset::E_FORMAT::EF_R32G32B32A32_SFLOAT;
			// World Row 0
			vtxinputParams.attributes[5].binding = instanceBufferBinding;
			vtxinputParams.attributes[5].relativeOffset = sizeof(core::vector3df_SIMD);
			vtxinputParams.attributes[5].format = asset::E_FORMAT::EF_R32G32B32A32_SFLOAT;
			// World Row 1
			vtxinputParams.attributes[6].binding = instanceBufferBinding;
			vtxinputParams.attributes[6].relativeOffset = sizeof(core::vector3df_SIMD) * 2;
			vtxinputParams.attributes[6].format = asset::E_FORMAT::EF_R32G32B32A32_SFLOAT;
			// World Row 2
			vtxinputParams.attributes[7].binding = instanceBufferBinding;
			vtxinputParams.attributes[7].relativeOffset = sizeof(core::vector3df_SIMD) * 3;
			vtxinputParams.attributes[7].format = asset::E_FORMAT::EF_R32G32B32A32_SFLOAT;


			vtxinputParams.enabledAttribFlags |= 0x1u << 4 | 0x1u << 5 | 0x1u << 6 | 0x1u << 7;
			vtxinputParams.enabledBindingFlags |= 0x1u << instanceBufferBinding;

			// for wireframe rendering
			#if 0
			pipeline->getRasterizationParams().polygonMode = asset::EPM_LINE; 
			#endif
		}

		asset::SPushConstantRange range[1] = { asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD) };
		auto gfxLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(range,range+1u);
		pipeline->setLayout(core::smart_refctd_ptr(gfxLayout));

		core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> rpIndependentPipeline = CPU2GPU.getGPUObjectsFromAssets(&pipeline,&pipeline+1,cpu2gpuParams)->front();
	
		ret.gpuMesh = CPU2GPU.getGPUObjectsFromAssets(&cpuMesh, &cpuMesh + 1,cpu2gpuParams)->front();
	
		asset::SBufferBinding<video::IGPUBuffer> vtxInstanceBufBnd;
		vtxInstanceBufBnd.offset = instanceBufferOffset;
		vtxInstanceBufBnd.buffer = instancesBuffer;
		ret.gpuMesh->setVertexBufferBinding(std::move(vtxInstanceBufBnd), instanceBufferBinding);
		ret.gpuMesh->setInstanceCount(numInstances);

		video::IGPUGraphicsPipeline::SCreationParams gp_params;
		gp_params.rasterizationSamplesHint = asset::IImage::ESCF_1_BIT;
		gp_params.renderpass = core::smart_refctd_ptr<video::IGPURenderpass>(renderpass);
		gp_params.renderpassIndependent = rpIndependentPipeline; // TODO: fix use gpuMesh->getPipeline instead
		gp_params.subpassIx = 0u;

		ret.graphicsPipeline = device->createGPUGraphicsPipeline(nullptr, std::move(gp_params));

		return ret;
	};

	gpuObjects.push_back(createGPUObject(cpuMeshPlanets.get(), gpuInstancesBuffer, NumSolarSystemObjects, 0));

	auto lastTime = std::chrono::high_resolution_clock::now();
	constexpr uint32_t FRAME_COUNT = 500000u;
	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;

	core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	for (uint32_t i=0u; i<FRAMES_IN_FLIGHT; i++)
	{
		imageAcquire[i] = device->createSemaphore();
		renderFinished[i] = device->createSemaphore();
	}

	// render loop
	double dt = 0;

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

		auto now = std::chrono::high_resolution_clock::now();
		dt = std::chrono::duration_cast<std::chrono::milliseconds>(now-lastTime).count();
		lastTime = now;
		
		// safe to proceed
		cb->begin(0);

		{
			asset::SViewport vp;
			vp.minDepth = 0.f;
			vp.maxDepth = 1.f;
			vp.x = 0u;
			vp.y = 0u;
			vp.width = WIN_W;
			vp.height = WIN_H;
			cb->setViewport(0u, 1u, &vp);
		}
		// renderpass 
		uint32_t imgnum = 0u;
		swapchain->acquireNextImage(MAX_TIMEOUT,imageAcquire[resourceIx].get(),nullptr,&imgnum);
		{
			video::IGPUCommandBuffer::SRenderpassBeginInfo info;
			asset::SClearValue clearValues[2] ={};
			asset::VkRect2D area;
			clearValues[0].color.float32[0] = 0.1f;
			clearValues[0].color.float32[1] = 0.1f;
			clearValues[0].color.float32[2] = 0.1f;
			clearValues[0].color.float32[3] = 1.f;

			clearValues[1].depthStencil.depth = 0.0f;
			clearValues[1].depthStencil.stencil = 0.0f;

			info.renderpass = renderpass;
			info.framebuffer = fbo;
			info.clearValueCount = 2u;
			info.clearValues = clearValues;
			info.renderArea.offset = { 0, 0 };
			info.renderArea.extent = { WIN_W, WIN_H };
			cb->beginRenderPass(&info,asset::ESC_INLINE);
		}
		// Update instances buffer 
		{
			static float current_rotation = 0.0f;
			current_rotation += dt * 0.005f * SimulationSpeedScale;

			// Update Planets Transformations
			for(uint32_t i = 0; i < NumInstances; ++i) {
				auto & solarSystemObj = solarSystemObjectsData[i];

				core::matrix3x4SIMD translationMat;
				core::matrix3x4SIMD scaleMat;
				core::matrix3x4SIMD rotationMat;
				core::matrix3x4SIMD parentMat;
				
				translationMat.setTranslation(solarSystemObj.initialRelativePosition);
				scaleMat.setScale(core::vectorSIMDf(solarSystemObj.scale));
				
				{
					auto rot = current_rotation + 300; // just offset in time for beauty
					rotationMat.setRotation(core::quaternion(0.0f, rot * solarSystemObj.yRotationSpeed, rot * solarSystemObj.zRotationSpeed));
				}

				if(solarSystemObj.parentIndex > 0u) {
					auto parentObj = solarSystemObjectsData[solarSystemObj.parentIndex];
					parentMat = parentObj.matForChildren;
				}

				solarSystemObj.matForChildren = matrix3x4SIMD::concatenateBFollowedByA(matrix3x4SIMD::concatenateBFollowedByA(parentMat, rotationMat), translationMat); // parentMat * rotationMat * translationMat
				instancesData[i].modelMatrix = matrix3x4SIMD::concatenateBFollowedByA(solarSystemObj.matForChildren, scaleMat); // solarSystemObj.matForChildren * scaleMat
			}

			asset::SBufferRange<video::IGPUBuffer> range;
			range.buffer = gpuInstancesBuffer;
			range.offset = 0;
			range.size = BUF_SZ;
			device->updateBufferRangeViaStagingBuffer(graphicsQueue, range, instancesData.data());
		}
		// draw
		{
			// Draw Stuff 
			for(uint32_t i = 0; i < gpuObjects.size(); ++i) {
				auto & gpuObject = gpuObjects[i];

				cb->bindGraphicsPipeline(gpuObject.graphicsPipeline.get());
				cb->pushConstants(gpuObject.graphicsPipeline->getRenderpassIndependentPipeline()->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), viewProj.pointer());
				cb->drawMeshBuffer(gpuObject.gpuMesh.get());
			}
		}
		cb->endRenderPass();
		cb->end();
		
		CommonAPI::Submit(device.get(), swapchain.get(), cb.get(), graphicsQueue, imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
		CommonAPI::Present(device.get(), swapchain.get(), graphicsQueue, renderFinished[resourceIx].get(), imgnum);
		
	}

	return 0;
}