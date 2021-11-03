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

layout (set = 0, binding = 0, row_major) readonly buffer GlobalTforms
{
	mat4x3 data[];
} globalTform;

layout( push_constant, row_major ) uniform Block {
	mat4 viewProj;
} PushConstants;

layout(location = 0) out vec3 Color;
layout(location = 1) out vec3 Normal;

void main()
{
	mat4x3 tform = globalTform.data[gl_InstanceIndex];
	mat3x4 tpose = transpose(tform);

	vec4 lcpos = vPos;
	lcpos.xyz *= vCol.a; // color's alpha has encoded scale
	vec4 worldPos = vec4(dot(tpose[0], lcpos), dot(tpose[1], lcpos), dot(tpose[2], lcpos), 1.0);
    vec4 pos = PushConstants.viewProj*worldPos;
	gl_Position = pos;
	Color = vCol.xyz;

	mat3x4 transposeWorldMat = tpose;
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
	scene::ITransformTree::node_t node;
	scene::ITransformTree::node_t parentIndex;
	float yRotationSpeed = 0.0f;
	float zRotationSpeed = 0.0f;
	float scale = 1.0f;
	core::vector3df_SIMD initialRelativePosition;
	core::matrix3x4SIMD matForChildren;

	core::matrix3x4SIMD getTform() const
	{
		core::matrix3x4SIMD t;
		core::quaternion q;
		q.makeIdentity();
		t.setScaleRotationAndTranslation(core::vectorSIMDf(scale, scale, scale), q, initialRelativePosition);
		return t;
	}
};

class CEventReceiver
{
public:
	CEventReceiver() : debugDrawRequestFlag(false) {}

	void process(const nbl::ui::IKeyboardEventChannel::range_t& events)
	{
		for (auto eventIterator = events.begin(); eventIterator != events.end(); eventIterator++)
		{
			auto event = *eventIterator;

			if (event.keyCode == nbl::ui::EKC_D)
				debugDrawRequestFlag = true;
			if (event.keyCode == nbl::ui::EKC_C)
				debugDrawRequestFlag = false;
		}
	}

	inline bool isDebugRequested() const { return debugDrawRequestFlag; }

private:
	bool debugDrawRequestFlag;
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
    auto windowCb = std::move(initOutput.windowCb);
	auto inputSystem = std::move(initOutput.inputSystem);
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
	auto utils = std::move(initOutput.utilities);

    nbl::video::IGPUObjectFromAssetConverter CPU2GPU;
	
    core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf[FRAMES_IN_FLIGHT];
    device->createCommandBuffers(commandPool.get(), nbl::video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, cmdbuf);

	constexpr uint32_t ObjectCount = 11u;
	constexpr uint32_t PropertyCount = 5u;


	//scene::ITransformTree* tt0; 
	//assert(tt0->getNodePropertyPool()->getPropertyCount() == PropertyCount);
	const size_t parentPropSz = sizeof(uint32_t);//tt0->getNodePropertyPool()->getPropertySize(scene::ITransformTree::parent_prop_ix);
	const size_t relTformPropSz = sizeof(core::matrix3x4SIMD);//tt0->getNodePropertyPool()->getPropertySize(scene::ITransformTree::relative_transform_prop_ix);
	const size_t modifStampPropSz = sizeof(uint32_t);//tt0->getNodePropertyPool()->getPropertySize(scene::ITransformTree::modified_stamp_prop_ix);
	const size_t globalTformPropSz = sizeof(core::matrix3x4SIMD);//tt0->getNodePropertyPool()->getPropertySize(scene::ITransformTree::global_transform_prop_ix);
	const size_t recompStampPropSz = sizeof(uint32_t);//tt0->getNodePropertyPool()->getPropertySize(scene::ITransformTree::recomputed_stamp_prop_ix);

	constexpr uint32_t GlobalTformPropNum = 3u;

	const size_t SSBOAlignment = gpuPhysicalDevice->getLimits().SSBOAlignment;
	const size_t offset_parent = 0u;
	const size_t offset_relTform = core::alignUp(offset_parent + parentPropSz*ObjectCount, SSBOAlignment);
	const size_t offset_modifStamp = core::alignUp(offset_relTform + relTformPropSz*ObjectCount, SSBOAlignment);
	const size_t offset_globalTform = core::alignUp(offset_modifStamp + modifStampPropSz*ObjectCount, SSBOAlignment);
	const size_t offset_recompStamp = core::alignUp(offset_globalTform + globalTformPropSz*ObjectCount, SSBOAlignment);

	const size_t ssboSz = offset_recompStamp + recompStampPropSz * ObjectCount;

	video::IGPUBuffer::SCreationParams ssboCreationParams;
	ssboCreationParams.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
	ssboCreationParams.sharingMode = asset::E_SHARING_MODE::ESM_CONCURRENT;
	ssboCreationParams.queueFamilyIndexCount = 0u;
	ssboCreationParams.queueFamilyIndices = nullptr;

	auto ssbo_buf = device->createDeviceLocalGPUBufferOnDedMem(ssboCreationParams,ssboSz);

	asset::SBufferRange<video::IGPUBuffer> propBufs[PropertyCount];
	for (uint32_t i = 0u; i < PropertyCount; ++i)
		propBufs[i].buffer = ssbo_buf;
	propBufs[0].offset = offset_parent;
	propBufs[0].size = parentPropSz*ObjectCount;
	propBufs[1].offset = offset_relTform;
	propBufs[1].size = relTformPropSz*ObjectCount;
	propBufs[2].offset = offset_modifStamp;
	propBufs[2].size = modifStampPropSz*ObjectCount;
	propBufs[3].offset = offset_globalTform;
	propBufs[3].size = globalTformPropSz*ObjectCount;
	propBufs[4].offset = offset_recompStamp;
	propBufs[4].size = recompStampPropSz*ObjectCount;

	auto tt = scene::ITransformTree::create(device.get(), renderpass, propBufs, ObjectCount, true);
	auto ttm = scene::ITransformTreeManager::create(core::smart_refctd_ptr(device));

	auto ppHandler = core::make_smart_refctd_ptr<video::CPropertyPoolHandler>(core::smart_refctd_ptr(device));

	core::vector<GPUObject> gpuObjects; 

	// Instance Data
	constexpr float SimulationSpeedScale = 0.03f;
	constexpr uint32_t NumSolarSystemObjects = ObjectCount;
	constexpr uint32_t NumInstances = NumSolarSystemObjects;
	
	// GPU data pool
	//auto propertyPool = video::CPropertyPool<core::allocator,InstanceData,SolarSystemObject>::create(device.get(),blocks,NumSolarSystemObjects);

	// SolarSystemObject and InstanceData have 1-to-1 relationship
	core::vector<InstanceData> instancesData;
	core::vector<SolarSystemObject> solarSystemObjectsData;
	instancesData.resize(NumInstances);
	solarSystemObjectsData.resize(NumInstances);

	// allocate node handles from the transform tree
	core::vector<scene::ITransformTree::node_t> tmp_nodes(NumInstances, scene::ITransformTree::invalid_node);
	{
		bool success = tt->allocateNodes({tmp_nodes.data(),tmp_nodes.data()+tmp_nodes.size()});
		if (!success)
			exit(-1);
		auto objectIt = solarSystemObjectsData.begin();
		for (auto nodeID : tmp_nodes)
			(objectIt++)->node = nodeID;
	}

	// Sun
	uint32_t constexpr sunIndex = 0u;
	instancesData[sunIndex].color = core::vector3df_SIMD(0.8f, 1.0f, 0.1f);
	solarSystemObjectsData[sunIndex].parentIndex = scene::ITransformTree::invalid_node;
	solarSystemObjectsData[sunIndex].yRotationSpeed = 0.0f;
	solarSystemObjectsData[sunIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[sunIndex].scale = 5.0f;
	solarSystemObjectsData[sunIndex].initialRelativePosition = core::vector3df_SIMD(0.0f, 0.0f, 0.0f);
	const auto sun_node = solarSystemObjectsData[sunIndex].node;
	
	// Mercury
	uint32_t constexpr mercuryIndex = 1u;
	instancesData[mercuryIndex].color = core::vector3df_SIMD(0.7f, 0.3f, 0.1f);
	solarSystemObjectsData[mercuryIndex].parentIndex = sun_node;
	solarSystemObjectsData[mercuryIndex].yRotationSpeed = 0.5f;
	solarSystemObjectsData[mercuryIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[mercuryIndex].scale = 0.5f;
	solarSystemObjectsData[mercuryIndex].initialRelativePosition = core::vector3df_SIMD(4.0f, 0.0f, 0.0f);
	
	// Venus
	uint32_t constexpr venusIndex = 2u;
	instancesData[venusIndex].color = core::vector3df_SIMD(0.8f, 0.6f, 0.1f);
	solarSystemObjectsData[venusIndex].parentIndex = sun_node;
	solarSystemObjectsData[venusIndex].yRotationSpeed = 0.8f;
	solarSystemObjectsData[venusIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[venusIndex].scale = 1.0f;
	solarSystemObjectsData[venusIndex].initialRelativePosition = core::vector3df_SIMD(8.0f, 0.0f, 0.0f);

	// Earth
	uint32_t constexpr earthIndex = 3u;
	instancesData[earthIndex].color = core::vector3df_SIMD(0.1f, 0.4f, 0.8f);
	solarSystemObjectsData[earthIndex].parentIndex = sun_node;
	solarSystemObjectsData[earthIndex].yRotationSpeed = 1.0f;
	solarSystemObjectsData[earthIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[earthIndex].scale = 2.0f;
	solarSystemObjectsData[earthIndex].initialRelativePosition = core::vector3df_SIMD(12.0f, 0.0f, 0.0f);

	// Moon
	uint32_t constexpr moonIndex = 10u;
	instancesData[moonIndex].color = core::vector3df_SIMD(0.3f, 0.2f, 0.25f);
	solarSystemObjectsData[moonIndex].parentIndex = solarSystemObjectsData[earthIndex].node;
	solarSystemObjectsData[moonIndex].yRotationSpeed = 2.2f;
	solarSystemObjectsData[moonIndex].zRotationSpeed = 0.f;
	solarSystemObjectsData[moonIndex].scale = 0.4f;
	solarSystemObjectsData[moonIndex].initialRelativePosition = core::vector3df_SIMD(2.5f, 0.0f, 0.0f);
	
	// Mars
	uint32_t constexpr marsIndex = 4u;
	instancesData[marsIndex].color = core::vector3df_SIMD(0.9f, 0.3f, 0.1f);
	solarSystemObjectsData[marsIndex].parentIndex = sun_node;
	solarSystemObjectsData[marsIndex].yRotationSpeed = 2.0f;
	solarSystemObjectsData[marsIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[marsIndex].scale = 1.5f;
	solarSystemObjectsData[marsIndex].initialRelativePosition = core::vector3df_SIMD(16.0f, 0.0f, 0.0f);
	
	// Jupiter
	uint32_t constexpr jupiterIndex = 5u;
	instancesData[jupiterIndex].color = core::vector3df_SIMD(0.6f, 0.4f, 0.4f);
	solarSystemObjectsData[jupiterIndex].parentIndex = sun_node;
	solarSystemObjectsData[jupiterIndex].yRotationSpeed = 11.0f;
	solarSystemObjectsData[jupiterIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[jupiterIndex].scale = 4.0f;
	solarSystemObjectsData[jupiterIndex].initialRelativePosition = core::vector3df_SIMD(20.0f, 0.0f, 0.0f);
	
	// Saturn
	uint32_t constexpr saturnIndex = 6u;
	instancesData[saturnIndex].color = core::vector3df_SIMD(0.7f, 0.7f, 0.5f);
	solarSystemObjectsData[saturnIndex].parentIndex = sun_node;
	solarSystemObjectsData[saturnIndex].yRotationSpeed = 30.0f;
	solarSystemObjectsData[saturnIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[saturnIndex].scale = 3.0f;
	solarSystemObjectsData[saturnIndex].initialRelativePosition = core::vector3df_SIMD(24.0f, 0.0f, 0.0f);
	
	// Uranus
	uint32_t constexpr uranusIndex = 7u;
	instancesData[uranusIndex].color = core::vector3df_SIMD(0.4f, 0.4f, 0.6f);
	solarSystemObjectsData[uranusIndex].parentIndex = sun_node;
	solarSystemObjectsData[uranusIndex].yRotationSpeed = 40.0f;
	solarSystemObjectsData[uranusIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[uranusIndex].scale = 3.5f;
	solarSystemObjectsData[uranusIndex].initialRelativePosition = core::vector3df_SIMD(28.0f, 0.0f, 0.0f);
	
	// Neptune
	uint32_t constexpr neptuneIndex = 8u;
	instancesData[neptuneIndex].color = core::vector3df_SIMD(0.5f, 0.2f, 0.9f);
	solarSystemObjectsData[neptuneIndex].parentIndex = sun_node;
	solarSystemObjectsData[neptuneIndex].yRotationSpeed = 50.0f;
	solarSystemObjectsData[neptuneIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[neptuneIndex].scale = 4.0f;
	solarSystemObjectsData[neptuneIndex].initialRelativePosition = core::vector3df_SIMD(32.0f, 0.0f, 0.0f);
	
	// Pluto 
	uint32_t constexpr plutoIndex = 9u;
	instancesData[plutoIndex].color = core::vector3df_SIMD(0.7f, 0.5f, 0.5f);
	solarSystemObjectsData[plutoIndex].parentIndex = sun_node;
	solarSystemObjectsData[plutoIndex].yRotationSpeed = 1.0f;
	solarSystemObjectsData[plutoIndex].zRotationSpeed = 0.0f;
	solarSystemObjectsData[plutoIndex].scale = 0.5f;
	solarSystemObjectsData[plutoIndex].initialRelativePosition = core::vector3df_SIMD(36.0f, 0.0f, 0.0f);

	// upload data
	{
		auto* q = device->getQueue(commandPool->getQueueFamilyIndex(),0u);

		nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf_nodes;
		device->createCommandBuffers(commandPool.get(),nbl::video::IGPUCommandBuffer::EL_PRIMARY,1u,&cmdbuf_nodes);

		auto fence_nodes = device->createFence(static_cast<nbl::video::IGPUFence::E_CREATE_FLAGS>(0));

		core::vector<scene::ITransformTree::parent_t> tmp_parents(NumInstances);
		core::vector<scene::ITransformTree::relative_transform_t> tmp_transforms(NumInstances);
		for (auto i=0u; i<NumInstances; i++)
		{
			tmp_parents[i] = solarSystemObjectsData[i].parentIndex;
			tmp_transforms[i] = solarSystemObjectsData[i].getTform();
		}
		auto tmp_node_buf = utils->createFilledDeviceLocalGPUBufferOnDedMem(q,sizeof(scene::ITransformTree::node_t)*NumInstances,tmp_nodes.data());
		tmp_node_buf->setObjectDebugName("Temporary Nodes");
		auto tmp_parent_buf = utils->createFilledDeviceLocalGPUBufferOnDedMem(q,sizeof(scene::ITransformTree::parent_t)*NumInstances,tmp_parents.data());
		tmp_parent_buf->setObjectDebugName("Temporary Parents");
		auto tmp_transform_buf = utils->createFilledDeviceLocalGPUBufferOnDedMem(q,sizeof(scene::ITransformTree::relative_transform_t)*NumInstances,tmp_transforms.data());
		tmp_transform_buf->setObjectDebugName("Temporary Transforms");

		//
		video::IGPUBuffer::SCreationParams scratchParams = {};
		scratchParams.canUpdateSubRange = true;
		scratchParams.usage = core::bitflag(video::IGPUBuffer::EUF_TRANSFER_DST_BIT)|video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
		asset::SBufferBinding<video::IGPUBuffer> scratch = {0ull,device->createDeviceLocalGPUBufferOnDedMem(scratchParams,utils->getDefaultPropertyPoolHandler()->getMaxScratchSize())};
		scratch.buffer->setObjectDebugName("Scratch Buffer");
		{
			video::CPropertyPoolHandler::TransferRequest transfers[scene::ITransformTreeManager::TransferCount];

			{
				scene::ITransformTreeManager::TransferRequest req;
				req.tree = tt.get();
				req.parents = {0ull,tmp_parent_buf};
				req.relativeTransforms = {0ull,tmp_transform_buf};
				req.nodes = {0ull,tmp_node_buf->getSize(),tmp_node_buf};
				ttm->setupTransfers(req,transfers);
			}

			cmdbuf_nodes->begin(0);
			utils->getDefaultPropertyPoolHandler()->transferProperties(
				cmdbuf_nodes.get(),fence_nodes.get(),scratch,{0ull,tmp_node_buf},
				transfers,transfers+scene::ITransformTreeManager::TransferCount,initOutput.logger.get()
			);
			cmdbuf_nodes->end();
		}

	scene::ITransformTree::node_t moon_node = scene::ITransformTree::invalid_node;
	{
		scene::ITransformTreeManager::AllocationRequest req;
		req.cmdbuf = cmdbuf_nodes.get();
		req.fence = fence_nodes.get();
		auto tform = solarSystemObjectsData[sunIndex].getTform();
		req.relativeTransforms = &tform;
		req.outNodes = { &moon_node, &moon_node + 1 };
		req.parents = &earth_node;
		req.poolHandler = ppHandler.get();
		req.tree = tt.get();
		req.upBuff = utils->getDefaultUpStreamingBuffer();
		req.logger = initOutput.logger.get();
		ttm->addNodes(req);
		cmdbuf_nodes->end();

		auto* q = device->getQueue(0u, 0u);
		video::IGPUQueue::SSubmitInfo submit;
		submit.commandBufferCount = 1u;
		submit.commandBuffers = &cmdbuf_nodes.get();
		q->submit(1u, &submit, fence_nodes.get());
	}
	
	waitres = device->waitForFences(1u, &fence_nodes.get(), false, 999999999ull);
	assert(waitres == video::IGPUFence::ES_SUCCESS);

	cmdbuf_nodes->reset(video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
	device->resetFences(1u, &fence_nodes.get());

	solarSystemObjectsData[moonIndex].node = moon_node;

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
		auto unspec = assetManager->getGLSLCompiler()->createSPIRVFromGLSL(source, stage, "main", "runtimeID", nullptr, true, nullptr, initOutput.logger.get());
		if (!unspec)
			return nullptr;

		asset::ISpecializedShader::SInfo info(nullptr, nullptr, "main", stage, "");
		return core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspec), std::move(info));
	};

	auto vs = createCPUSpecializedShaderFromSource(vertexSource,asset::ISpecializedShader::ESS_VERTEX);
	auto fs = createCPUSpecializedShaderFromSource(fragmentSource,asset::ISpecializedShader::ESS_FRAGMENT);
	asset::ICPUSpecializedShader* shaders[2]{ vs.get(), fs.get() };
	
	auto cpuMeshPlanets = createMeshBufferFromGeomCreatorReturnType(sphereGeom, assetManager.get(), shaders, shaders+2);

	core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> cpu_gfxDsl0;
	{
		asset::ICPUDescriptorSetLayout::SBinding bnd;
		bnd.binding = 0u;
		bnd.count = 1u;
		bnd.samplers = nullptr;
		bnd.stageFlags = video::IGPUSpecializedShader::ESS_VERTEX;
		bnd.type = asset::EDT_STORAGE_BUFFER;

		cpu_gfxDsl0 = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(&bnd,&bnd+1);
	}

	constexpr size_t ColorBufSz = sizeof(core::vectorSIMDf) * ObjectCount;
	video::IGPUBuffer::SCreationParams colorBufCreationParams;
	colorBufCreationParams.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
	colorBufCreationParams.sharingMode = asset::E_SHARING_MODE::ESM_CONCURRENT;
	colorBufCreationParams.queueFamilyIndexCount = 0u;
	colorBufCreationParams.queueFamilyIndices = nullptr;

	auto gpuColorBuf = device->createDeviceLocalGPUBufferOnDedMem(colorBufCreationParams, ColorBufSz);
	core::vectorSIMDf colors[ObjectCount]{
		core::vectorSIMDf(0.f, 0.f, 1.f),
		core::vectorSIMDf(0.f, 1.f, 0.f),
		core::vectorSIMDf(0.f, 1.f, 1.f),
		core::vectorSIMDf(1.f, 0.f, 0.f),
		core::vectorSIMDf(1.f, 0.f, 1.f),
		core::vectorSIMDf(1.f, 1.f, 0.f),
		core::vectorSIMDf(1.f, 1.f, 1.f),
		core::vectorSIMDf(0.5f, 1.f, 0.5f),
		core::vectorSIMDf(0.f, 1.f, 0.f),
		core::vectorSIMDf(0.5f, 0.7f, 0.2f),
		core::vectorSIMDf(0.6f, 0.8f, 0.1f)
	};
	for (uint32_t i = 0u; i < ObjectCount; ++i)
		colors[i].w = solarSystemObjectsData[i].scale;
	utils->updateBufferRangeViaStagingBuffer(device->getQueue(0, 0), { 0ull, ColorBufSz, gpuColorBuf }, colors);

	// Create GPU Objects (IGPUMeshBuffer + GraphicsPipeline)
	auto createGPUObject = [&](
		asset::ICPUMeshBuffer * cpuMesh,
		uint64_t numInstances, uint64_t colorBufferOffset, core::smart_refctd_ptr<video::IGPUBuffer> colorBuffer,
		asset::E_FACE_CULL_MODE faceCullingMode = asset::EFCM_BACK_BIT) -> GPUObject {
		GPUObject ret = {};
		
		constexpr auto ColorBindingNum = 15u;
		constexpr auto ColorAttribNum = 4u;

		auto pipeline = cpuMesh->getPipeline();
		{
			// we're working with RH coordinate system(view proj) and in that case the cubeGeom frontFace is NOT CCW.
			auto& rasterParams = pipeline->getRasterizationParams();
			rasterParams.frontFaceIsCCW = 0;
			rasterParams.faceCullingMode = faceCullingMode;


			auto& vtxinputParams = pipeline->getVertexInputParams();
			vtxinputParams.bindings[ColorBindingNum].inputRate = asset::EVIR_PER_INSTANCE;
			vtxinputParams.bindings[ColorBindingNum].stride = sizeof(core::vectorSIMDf);
			vtxinputParams.attributes[ColorAttribNum].binding = ColorBindingNum;
			vtxinputParams.attributes[ColorAttribNum].format = asset::EF_R32G32B32A32_SFLOAT;
			vtxinputParams.attributes[ColorAttribNum].relativeOffset = 0u;

			vtxinputParams.enabledAttribFlags |= 0x1u << ColorAttribNum;
			vtxinputParams.enabledBindingFlags |= 0x1u << ColorBindingNum;
		}

		asset::SPushConstantRange range[1] = { asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD) };
		auto gfxLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(range,range+1u,core::smart_refctd_ptr(cpu_gfxDsl0));
		pipeline->setLayout(core::smart_refctd_ptr(gfxLayout));

		core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> rpIndependentPipeline = CPU2GPU.getGPUObjectsFromAssets(&pipeline,&pipeline+1,cpu2gpuParams)->front();
	   
		asset::SBufferBinding<video::IGPUBuffer> colorBufBinding;
		colorBufBinding.offset = colorBufferOffset;
		colorBufBinding.buffer = colorBuffer;

		ret.gpuMesh = CPU2GPU.getGPUObjectsFromAssets(&cpuMesh, &cpuMesh + 1,cpu2gpuParams)->front();
		ret.gpuMesh->setVertexBufferBinding(std::move(colorBufBinding), ColorBindingNum);
		ret.gpuMesh->setInstanceCount(numInstances);


		video::IGPUGraphicsPipeline::SCreationParams gp_params;
		gp_params.rasterizationSamplesHint = asset::IImage::ESCF_1_BIT;
		gp_params.renderpass = core::smart_refctd_ptr<video::IGPURenderpass>(renderpass);
		gp_params.renderpassIndependent = rpIndependentPipeline; // TODO: fix use gpuMesh->getPipeline instead
		gp_params.subpassIx = 0u;

		ret.graphicsPipeline = device->createGPUGraphicsPipeline(nullptr, std::move(gp_params));

		return ret;
	};

	gpuObjects.push_back(createGPUObject(cpuMeshPlanets.get(), NumSolarSystemObjects, 0ull, gpuColorBuf));

	auto* gfxDsl0 = gpuObjects.back().gpuMesh->getPipeline()->getLayout()->getDescriptorSetLayout(0);
	auto gfxDescPool = device->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &gfxDsl0, &gfxDsl0 + 1);
	auto gfxDs0 = device->createGPUDescriptorSet(gfxDescPool.get(), core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>(const_cast<video::IGPUDescriptorSetLayout*>(gfxDsl0)));
	{
		video::IGPUDescriptorSet::SDescriptorInfo info;
		info.desc = propBufs[GlobalTformPropNum].buffer;
		info.buffer.offset = propBufs[GlobalTformPropNum].offset;
		info.buffer.size = propBufs[GlobalTformPropNum].size;
		video::IGPUDescriptorSet::SWriteDescriptorSet w;
		w.arrayElement = 0;
		w.binding = 0;
		w.count = 1;
		w.descriptorType = asset::EDT_STORAGE_BUFFER;
		w.dstSet = gfxDs0.get();
		w.info = &info;

		device->updateDescriptorSets(1u, &w, 0u, nullptr);
	}

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
	
	uint32_t resourceIx = 0;

	constexpr size_t ModsRangesBufSz = 2u*sizeof(uint32_t) + sizeof(scene::nbl_glsl_transform_tree_modification_request_range_t)*ObjectCount;
	video::IGPUBuffer::SCreationParams creationParams;
	creationParams.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
	creationParams.sharingMode = asset::E_SHARING_MODE::ESM_CONCURRENT;
	creationParams.queueFamilyIndexCount = 0u;
	creationParams.queueFamilyIndices = nullptr;

	auto modRangesBuf = device->createDeviceLocalGPUBufferOnDedMem(creationParams, ModsRangesBufSz);
	auto relTformModsBuf = device->createDeviceLocalGPUBufferOnDedMem(creationParams, sizeof(scene::nbl_glsl_transform_tree_relative_transform_modification_t) * ObjectCount);
	auto nodeIdsBuf = device->createDeviceLocalGPUBufferOnDedMem(creationParams, std::max(sizeof(uint32_t) + sizeof(scene::ITransformTree::node_t) * ObjectCount, 128ull));
	{
		//update `nodeIdsBuf`
		uint32_t countAndIds[1u + ObjectCount];
		countAndIds[0] = ObjectCount;
		for (uint32_t i = 0u; i < ObjectCount; ++i)
			countAndIds[1u+i] = solarSystemObjectsData[i].node;

		asset::SBufferRange<video::IGPUBuffer> bufrng;
		bufrng.buffer = nodeIdsBuf;
		bufrng.offset = 0;
		bufrng.size = nodeIdsBuf->getSize();
		utils->updateBufferRangeViaStagingBuffer(device->getQueue(0, 0), bufrng, countAndIds);

		core::vector<scene::ITransformTree::DebugNodeVtxInput> liveDebugNodeVtxInputs;

		for (const auto& solarSystemObject : solarSystemObjectsData)
		{
			scene::ITransformTree::DebugNodeVtxInput debugVtxInput;
			debugVtxInput.node = solarSystemObject.node;
			debugVtxInput.scale = solarSystemObject.scale;

			liveDebugNodeVtxInputs.push_back(debugVtxInput);
		}
			
		tt->setDebugLiveAllocations(liveDebugNodeVtxInputs);
	}

	scene::ITransformTree::DebugPushConstants debugPushConstants;
	debugPushConstants.lineColor = core::vector4df_SIMD(1, 0, 0, 0);
	debugPushConstants.aabbColor = core::vector4df_SIMD(0, 0, 1, 0);

	CEventReceiver eventReceiver;
	CommonAPI::InputSystem::ChannelReader<nbl::ui::IKeyboardEventChannel> keyboard;

	uint32_t timestamp = 1u;
	while(windowCb->isWindowOpen())
	{
		resourceIx++;
		if(resourceIx >= FRAMES_IN_FLIGHT)
			resourceIx = 0;

		auto& cb = cmdbuf[resourceIx];
		auto& fence = frameComplete[resourceIx];
		if (fence)
			device->blockForFences(1u,&fence.get());
		else
			fence = device->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		auto now = std::chrono::high_resolution_clock::now();
		dt = std::chrono::duration_cast<std::chrono::milliseconds>(now-lastTime).count();
		lastTime = now;
		
		timestamp++;

		// safe to proceed
		cb->begin(0);

		// we don't wait on anything because we do everything on the same queue
		uint32_t waitSemaphoreCount = 0u;
		const asset::E_PIPELINE_STAGE_FLAGS* waitStages = nullptr;
		video::IGPUSemaphore* const* waitSems = nullptr;
			
		// queue update to `modRangesBuf`
		{
			struct SSBO
			{
				uint32_t rangeCount;
				// OpenGL drivers (even Nvidia) have some bugs and can sporadically completely
				// forget about memory or execution barriers between `glCopyBufferSubData` and compute dispatches
				// as well as modify memory to bogus values even though you're always copying the same data to the same location
				uint32_t maxRangeLength;
				scene::ITransformTreeManager::ModificationRequestRange ranges[ObjectCount];
			};
			static_assert(offsetof(SSBO, ranges) == sizeof(uint32_t)*2ull);
			SSBO requestRanges;
			requestRanges.rangeCount = ObjectCount;
			requestRanges.maxRangeLength = 1u;
			for (uint32_t i = 0u; i < ObjectCount; ++i)
			{
				auto& obj = solarSystemObjectsData[i];
				requestRanges.ranges[i].nodeID = obj.node;
				requestRanges.ranges[i].requestsBegin = i;
				requestRanges.ranges[i].requestsEnd = i+1u;
				requestRanges.ranges[i].newTimestamp = timestamp;
			}

			asset::SBufferRange<video::IGPUBuffer> bufrng;
			bufrng.buffer = modRangesBuf;
			bufrng.offset = 0;
			bufrng.size = modRangesBuf->getSize();
			utils->updateBufferRangeViaStagingBuffer(cb.get(),fence.get(),graphicsQueue,bufrng,&requestRanges,waitSemaphoreCount,waitSems,waitStages);
		}

		// update `relTformModsBuf`
		{
			static float current_rotation = 0.0f;
			current_rotation += dt * 0.005f * SimulationSpeedScale;

			std::array<scene::ITransformTreeManager::RelativeTransformModificationRequest, ObjectCount> reqs;
			for (uint32_t i = 0u; i < reqs.size(); ++i)
			{
				core::matrix3x4SIMD translationMat;
				core::matrix3x4SIMD rotationMat;
				core::matrix3x4SIMD scaleMat;

				translationMat.setTranslation(solarSystemObjectsData[i].initialRelativePosition);
				{
					auto rot = current_rotation + 300; // just offset in time for beauty
					rotationMat.setRotation(core::quaternion(0.0f, rot * solarSystemObjectsData[i].yRotationSpeed, rot * solarSystemObjectsData[i].zRotationSpeed));
				}
				scaleMat.setScale(core::vectorSIMDf(solarSystemObjectsData[i].scale)); //?

				auto tform = core::matrix3x4SIMD::concatenateBFollowedByA(rotationMat, translationMat);

				reqs[i] = scene::ITransformTreeManager::RelativeTransformModificationRequest(scene::ITransformTreeManager::RelativeTransformModificationRequest::ET_OVERWRITE, tform);
			}

			asset::SBufferRange<video::IGPUBuffer> bufrng;
			bufrng.buffer = relTformModsBuf;
			bufrng.offset = 0;
			bufrng.size = relTformModsBuf->getSize();

			utils->updateBufferRangeViaStagingBuffer(cb.get(),fence.get(),graphicsQueue,bufrng,reqs.data(),waitSemaphoreCount,waitSems,waitStages);
		}

		// Update instances transforms 
		{
			// buffers to barrier w.r.t. updates
			video::IGPUCommandBuffer::SBufferMemoryBarrier barriers[scene::ITransformTreeManager::SBarrierSuggestion::MaxBufferCount];
			auto setBufferBarrier = [&barriers,cb](const uint32_t ix, const asset::SBufferRange<video::IGPUBuffer>& range, const asset::SMemoryBarrier& barrier)
			{
				barriers[ix].barrier = barrier;
				barriers[ix].dstQueueFamilyIndex = barriers[ix].srcQueueFamilyIndex = cb->getQueueFamilyIndex();
				barriers[ix].buffer = range.buffer;
				barriers[ix].offset = range.offset;
				barriers[ix].size = range.size;
			};
			
			const core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> renderingStages = asset::EPSF_VERTEX_SHADER_BIT;
			const core::bitflag<asset::E_ACCESS_FLAGS> renderingAccesses = asset::EAF_SHADER_READ_BIT;

			scene::ITransformTreeManager::ParamsBase baseParams;
			baseParams.cmdbuf = cb.get();
			baseParams.tree = tt.get();
			baseParams.fence = fence.get();
			baseParams.dispatchIndirect.buffer = nullptr;
			baseParams.dispatchDirect.nodeCount = ObjectCount;
			baseParams.logger = initOutput.logger.get();

			// compilers are too dumb to figure out const correctness (there's also a TODO in `core::smart_refctd_ptr`)
			const scene::ITransformTree* ptt = tt.get();
			const video::IPropertyPool* node_pp = ptt->getNodePropertyPool();
			//
			{
				auto sugg = scene::ITransformTreeManager::barrierHelper(scene::ITransformTreeManager::SBarrierSuggestion::EF_PRE_RELATIVE_TFORM_UPDATE);
				sugg.srcStageMask |= asset::EPSF_TRANSFER_BIT; // barrier after buffer upload, before TTM updates (so TTM update CS gets properly written data)
				sugg.requestRanges.srcAccessMask |= asset::EAF_TRANSFER_WRITE_BIT;
				sugg.modificationRequests.srcAccessMask |= asset::EAF_TRANSFER_WRITE_BIT;
				uint32_t barrierCount = 0u;
				setBufferBarrier(barrierCount++,{0ull,modRangesBuf->getSize(),modRangesBuf},sugg.requestRanges);
				setBufferBarrier(barrierCount++,{0ull,relTformModsBuf->getSize(),relTformModsBuf},sugg.modificationRequests);
				cb->pipelineBarrier(sugg.srcStageMask,sugg.dstStageMask,asset::EDF_NONE,0u,nullptr,barrierCount,barriers,0u,nullptr);
			}
			//
			{
				scene::ITransformTreeManager::LocalTransformUpdateParams lcparams;
				static_cast<scene::ITransformTreeManager::ParamsBase&>(lcparams) = baseParams;
				lcparams.modificationRequests = asset::SBufferBinding<video::IGPUBuffer>{ 0ull, relTformModsBuf };
				lcparams.requestRanges = asset::SBufferBinding<video::IGPUBuffer>{ 0ull, modRangesBuf };
				ttm->updateLocalTransforms(lcparams);
			}
			// barrier between TTM update and TTM recompute
			{
				auto sugg = scene::ITransformTreeManager::barrierHelper(scene::ITransformTreeManager::SBarrierSuggestion::EF_INBETWEEN_RLEATIVE_UPDATE_AND_GLOBAL_RECOMPUTE);
				sugg.srcStageMask |= renderingStages; // also Rendering and TTM recompute
				sugg.globalTransforms.srcAccessMask |= asset::EAF_SHADER_READ_BIT;
				sugg.dstStageMask |= asset::EPSF_TRANSFER_BIT; // as well as TTM update and Transfer
				sugg.requestRanges.dstAccessMask |= asset::EAF_TRANSFER_WRITE_BIT;
				sugg.modificationRequests.dstAccessMask |= asset::EAF_TRANSFER_WRITE_BIT;
				uint32_t barrierCount = 0u;
				setBufferBarrier(barrierCount++,{0ull,modRangesBuf->getSize(),modRangesBuf},sugg.requestRanges);
				setBufferBarrier(barrierCount++,{0ull,relTformModsBuf->getSize(),relTformModsBuf},sugg.modificationRequests);
				setBufferBarrier(barrierCount++,node_pp->getPropertyMemoryBlock(scene::ITransformTree::relative_transform_prop_ix),sugg.relativeTransforms);
				setBufferBarrier(barrierCount++,node_pp->getPropertyMemoryBlock(scene::ITransformTree::modified_stamp_prop_ix),sugg.modifiedTimestamps);
				cb->pipelineBarrier(sugg.srcStageMask,sugg.dstStageMask,asset::EDF_NONE,0u,nullptr,barrierCount,barriers,0u,nullptr);
			}
			//
			{
				scene::ITransformTreeManager::GlobalTransformUpdateParams gparams;
				static_cast<scene::ITransformTreeManager::ParamsBase&>(gparams) = baseParams;
				gparams.nodeIDs = { 0ull,nodeIdsBuf };
				ttm->recomputeGlobalTransforms(gparams);
			}
			// barrier between TTM recompute and TTM recompute+update 
			{
				auto sugg = scene::ITransformTreeManager::barrierHelper(scene::ITransformTreeManager::SBarrierSuggestion::EF_POST_GLOBAL_TFORM_RECOMPUTE);
				sugg.dstStageMask |= renderingStages; // also also TTM recompute and rendering shader (to read the global transforms)
				uint32_t barrierCount = 0u;
				setBufferBarrier(barrierCount++,node_pp->getPropertyMemoryBlock(scene::ITransformTree::relative_transform_prop_ix),sugg.relativeTransforms);
				setBufferBarrier(barrierCount++,node_pp->getPropertyMemoryBlock(scene::ITransformTree::modified_stamp_prop_ix),sugg.modifiedTimestamps);
				setBufferBarrier(barrierCount++,node_pp->getPropertyMemoryBlock(scene::ITransformTree::global_transform_prop_ix),sugg.globalTransforms);
				setBufferBarrier(barrierCount++,node_pp->getPropertyMemoryBlock(scene::ITransformTree::recomputed_stamp_prop_ix),sugg.recomputedTimestamps);
				cb->pipelineBarrier(sugg.srcStageMask,sugg.dstStageMask,asset::EDF_NONE,0u,nullptr,barrierCount,barriers,0u,nullptr);
			}
		}

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
		// begin renderpass
		{
			video::IGPUCommandBuffer::SRenderpassBeginInfo info;
			asset::SClearValue clearValues[2] ={};
			VkRect2D area;
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
		// draw
		{
			assert(gpuObjects.size() == 1ull);
			// Draw Stuff 
			for(uint32_t i = 0; i < gpuObjects.size(); ++i)
			{
				auto & gpuObject = gpuObjects[i];

				cb->bindGraphicsPipeline(gpuObject.graphicsPipeline.get());
				cb->pushConstants(gpuObject.graphicsPipeline->getRenderpassIndependentPipeline()->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), viewProj.pointer());
				cb->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuObject.graphicsPipeline->getRenderpassIndependentPipeline()->getLayout(), 0u, 1u, &gfxDs0.get());
				cb->drawMeshBuffer(gpuObject.gpuMesh.get());

				//gpuObject.gpuMesh->getBoundingBox().MaxEdge
			}
		}

		inputSystem->getDefaultKeyboard(&keyboard);
		keyboard.consumeEvents([&](const nbl::ui::IKeyboardEventChannel::range_t& events) -> void { eventReceiver.process(events); }, initOutput.logger.get());
		{ 
			debugPushConstants.viewProjectionMatrix = viewProj;
			const auto& boundingBox = gpuObjects[0].gpuMesh->getBoundingBox();
			debugPushConstants.minEdge = core::vector4df_SIMD(boundingBox.MinEdge.X, boundingBox.MinEdge.Y, boundingBox.MinEdge.Z);
			debugPushConstants.maxEdge = core::vector4df_SIMD(boundingBox.MaxEdge.X, boundingBox.MaxEdge.Y, boundingBox.MaxEdge.Z);

			tt->setDebugEnabledFlag(eventReceiver.isDebugRequested());
			tt->debugDraw(device.get(), cb.get(), debugPushConstants);
		}

		cb->endRenderPass();
		cb->end();
		
		// acquires and presents 
		uint32_t imgnum = 0u;
		swapchain->acquireNextImage(MAX_TIMEOUT,imageAcquire[resourceIx].get(),nullptr,&imgnum);
		CommonAPI::Submit(device.get(), swapchain.get(), cb.get(), graphicsQueue, imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
		CommonAPI::Present(device.get(), swapchain.get(), graphicsQueue, renderFinished[resourceIx].get(), imgnum);
		
	}

	return 0;
}