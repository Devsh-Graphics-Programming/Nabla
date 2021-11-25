// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/Camera.hpp"
#include "../common/CommonAPI.h"

using namespace nbl;
using namespace core;

const char* vertexSource = R"===(
#version 430 core

#include "nbl/builtin/glsl/transform_tree/render_descriptor_set.glsl"

layout(location = 0) in vec3 vPos;
layout(location = 3) in vec3 vNormal;
layout(location = 4) in vec4 vCol;

layout( push_constant, row_major ) uniform Block {
	mat4 viewProj;
} PushConstants;

layout(location = 0) out vec3 Color;
layout(location = 1) out vec3 Normal;

#include "nbl/builtin/glsl/utils/transform.glsl"
#include "nbl/builtin/glsl/utils/normal_encode.glsl"
#include "nbl/builtin/glsl/utils/normal_decode.glsl"
void main()
{
	const vec3 lcpos = vPos*vCol.a; // color's alpha has encoded scale
	const vec3 worldPos = nbl_glsl_pseudoMul3x4with3x1(nodeGlobalTransforms.data[gl_InstanceIndex],lcpos);

	gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,worldPos);
	Color = vCol.xyz;

	Normal = normalize(nbl_glsl_CompressedNormalMatrix_t_decode(nodeNormalMatrix.data[gl_InstanceIndex])*vNormal);
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

class TransformationApp : public ApplicationBase
{
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t WIN_W = 1280;
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t WIN_H = 720;
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t FBO_COUNT = 1u;
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t FRAMES_IN_FLIGHT = 5u;
		static_assert(FRAMES_IN_FLIGHT > FBO_COUNT);

		_NBL_STATIC_INLINE_CONSTEXPR uint32_t ObjectCount = 11u;

		_NBL_STATIC_INLINE_CONSTEXPR uint32_t FRAME_COUNT = 500000u;
		_NBL_STATIC_INLINE_CONSTEXPR uint64_t MAX_TIMEOUT = 99999999999999ull;

		_NBL_STATIC_INLINE_CONSTEXPR float SimulationSpeedScale = 0.03f; //! Instance Data

	public:
		void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
		{
			window = std::move(wnd);
		}
		void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& s) override
		{
			system = std::move(s);
		}
		nbl::ui::IWindow* getWindow() override
		{
			return window.get();
		}
		video::IAPIConnection* getAPIConnection() override
		{
			return gl.get();
		}
		video::ILogicalDevice* getLogicalDevice()  override
		{
			return device.get();
		}
		video::IGPURenderpass* getRenderpass() override
		{
			return renderpass.get();
		}
		void setSurface(core::smart_refctd_ptr<video::ISurface>&& s) override
		{
			surface = std::move(s);
		}
		void setFBOs(std::vector<core::smart_refctd_ptr<video::IGPUFramebuffer>>& f) override
		{
			for (int i = 0; i < f.size(); i++)
			{
				fbos[i] = core::smart_refctd_ptr(f[i]);
			}
		}
		void setSwapchain(core::smart_refctd_ptr<video::ISwapchain>&& s) override
		{
			swapchain = std::move(s);
		}
		uint32_t getSwapchainImageCount() override
		{
			return FBO_COUNT;
		}
		virtual nbl::asset::E_FORMAT getDepthFormat() override
		{
			return nbl::asset::EF_D32_SFLOAT;
		}
		APP_CONSTRUCTOR(TransformationApp)
		void onAppInitialized_impl() override
		{
			initOutput.window = core::smart_refctd_ptr(window);

			CommonAPI::Init<WIN_W, WIN_H, FBO_COUNT>(initOutput, video::EAT_OPENGL, "Solar System Transformations", asset::EF_D32_SFLOAT);
			system = std::move(initOutput.system);
			window = std::move(initOutput.window);
			windowCb = std::move(initOutput.windowCb);
			gl = std::move(initOutput.apiConnection);
			surface = std::move(initOutput.surface);
			gpuPhysicalDevice = std::move(initOutput.physicalDevice);
			device = std::move(initOutput.logicalDevice);
			queues = std::move(initOutput.queues);
			auto* transferUpQueue = queues[decltype(initOutput)::EQT_TRANSFER_UP];
			swapchain = std::move(initOutput.swapchain);
			renderpass = std::move(initOutput.renderpass);
			fbos = std::move(initOutput.fbo);
			auto fbo = fbos[0];
			commandPool = std::move(initOutput.commandPool);
			assetManager = std::move(initOutput.assetManager);
			cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
			utils = std::move(initOutput.utilities);

			device->createCommandBuffers(commandPool.get(), nbl::video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, cmdbuf);

			nbl::video::IGPUObjectFromAssetConverter CPU2GPU;

			using transform_tree_t = scene::ITransformTreeWithNormalMatrices;
			const size_t parentPropSz = sizeof(transform_tree_t::parent_t);
			const size_t relTformPropSz = sizeof(transform_tree_t::relative_transform_t);
			const size_t modifStampPropSz = sizeof(transform_tree_t::modified_stamp_t);
			const size_t globalTformPropSz = sizeof(transform_tree_t::global_transform_t);
			const size_t recompStampPropSz = sizeof(transform_tree_t::recomputed_stamp_t);
			const size_t normalMatrixPropSz = sizeof(transform_tree_t::normal_matrix_t);

			constexpr uint32_t GlobalTformPropNum = 3u;

			const size_t SSBOAlignment = gpuPhysicalDevice->getLimits().SSBOAlignment;
			const size_t offset_parent = 0u;
			const size_t offset_relTform = core::alignUp(offset_parent + parentPropSz * ObjectCount, SSBOAlignment);
			const size_t offset_modifStamp = core::alignUp(offset_relTform + relTformPropSz * ObjectCount, SSBOAlignment);
			const size_t offset_globalTform = core::alignUp(offset_modifStamp + modifStampPropSz * ObjectCount, SSBOAlignment);
			const size_t offset_recompStamp = core::alignUp(offset_globalTform + globalTformPropSz * ObjectCount, SSBOAlignment);
			const size_t offset_normalMatrix = core::alignUp(offset_recompStamp + recompStampPropSz * ObjectCount, SSBOAlignment);

			const size_t ssboSz = offset_normalMatrix + normalMatrixPropSz * ObjectCount;

			video::IGPUBuffer::SCreationParams ssboCreationParams;
			ssboCreationParams.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
			ssboCreationParams.sharingMode = asset::E_SHARING_MODE::ESM_CONCURRENT;
			ssboCreationParams.queueFamilyIndexCount = 0u;
			ssboCreationParams.queueFamilyIndices = nullptr;

			auto ssbo_buf = device->createDeviceLocalGPUBufferOnDedMem(ssboCreationParams, ssboSz);

			asset::SBufferRange<video::IGPUBuffer> propBufs[transform_tree_t::property_pool_t::PropertyCount];
			for (uint32_t i=0u; i<transform_tree_t::property_pool_t::PropertyCount; ++i)
				propBufs[i].buffer = ssbo_buf;
			propBufs[0].offset = offset_parent;
			propBufs[0].size = parentPropSz * ObjectCount;
			propBufs[1].offset = offset_relTform;
			propBufs[1].size = relTformPropSz * ObjectCount;
			propBufs[2].offset = offset_modifStamp;
			propBufs[2].size = modifStampPropSz * ObjectCount;
			propBufs[3].offset = offset_globalTform;
			propBufs[3].size = globalTformPropSz * ObjectCount;
			propBufs[4].offset = offset_recompStamp;
			propBufs[4].size = recompStampPropSz * ObjectCount;
			propBufs[5].offset = offset_normalMatrix;
			propBufs[5].size = normalMatrixPropSz * ObjectCount;

			tt = transform_tree_t::create(device.get(),propBufs,ObjectCount,true); // A contiguous Pool for a TT is unusually used because we index into with with `gl_InstanceIndex`.
			ttm = scene::ITransformTreeManager::create(utils.get(), transferUpQueue);

			if (!ttm.get())
				return;

			debugDrawPipeline = ttm->createDebugPipeline<transform_tree_t>(renderpass);
			ttDS = tt->getRenderDescriptorSet();

			auto ppHandler = core::make_smart_refctd_ptr<video::CPropertyPoolHandler>(core::smart_refctd_ptr(device));

			constexpr uint32_t NumSolarSystemObjects = ObjectCount;
			constexpr uint32_t NumInstances = NumSolarSystemObjects;

			// GPU data pool 
			//auto propertyPool = video::CPropertyPool<core::allocator,InstanceData,SolarSystemObject>::create(device.get(),blocks,NumSolarSystemObjects);

			// SolarSystemObject and InstanceData have 1-to-1 relationship
			core::vector<InstanceData> instancesData;
			instancesData.resize(NumInstances);
			solarSystemObjectsData.resize(NumInstances);

			// allocate node handles from the transform tree
			core::vector<scene::ITransformTree::node_t> tmp_nodes(NumInstances, scene::ITransformTree::invalid_node);
			{
				bool success = tt->allocateNodes({ tmp_nodes.data(),tmp_nodes.data() + tmp_nodes.size() });
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
				auto* q = device->getQueue(commandPool->getQueueFamilyIndex(), 0u);

				nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf_nodes;
				device->createCommandBuffers(commandPool.get(), nbl::video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf_nodes);

				auto fence_nodes = device->createFence(static_cast<nbl::video::IGPUFence::E_CREATE_FLAGS>(0));

				core::vector<scene::ITransformTree::parent_t> tmp_parents(NumInstances);
				core::vector<scene::ITransformTree::relative_transform_t> tmp_transforms(NumInstances);
				for (auto i = 0u; i < NumInstances; i++)
				{
					tmp_parents[i] = solarSystemObjectsData[i].parentIndex;
					tmp_transforms[i] = solarSystemObjectsData[i].getTform();
				}
				auto tmp_node_buf = utils->createFilledDeviceLocalGPUBufferOnDedMem(q, sizeof(scene::ITransformTree::node_t) * NumInstances, tmp_nodes.data());
				tmp_node_buf->setObjectDebugName("Temporary Nodes");
				auto tmp_parent_buf = utils->createFilledDeviceLocalGPUBufferOnDedMem(q, sizeof(scene::ITransformTree::parent_t) * NumInstances, tmp_parents.data());
				tmp_parent_buf->setObjectDebugName("Temporary Parents");
				auto tmp_transform_buf = utils->createFilledDeviceLocalGPUBufferOnDedMem(q, sizeof(scene::ITransformTree::relative_transform_t) * NumInstances, tmp_transforms.data());
				tmp_transform_buf->setObjectDebugName("Temporary Transforms");

				//
				video::IGPUBuffer::SCreationParams scratchParams = {};
				scratchParams.canUpdateSubRange = true;
				scratchParams.usage = core::bitflag(video::IGPUBuffer::EUF_TRANSFER_DST_BIT) | video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
				asset::SBufferBinding<video::IGPUBuffer> scratch = { 0ull,device->createDeviceLocalGPUBufferOnDedMem(scratchParams,utils->getDefaultPropertyPoolHandler()->getMaxScratchSize()) };
				scratch.buffer->setObjectDebugName("Scratch Buffer");
				{
					video::CPropertyPoolHandler::TransferRequest transfers[scene::ITransformTreeManager::TransferCount];

					{
						scene::ITransformTreeManager::TransferRequest req;
						req.tree = tt.get();
						req.parents = { 0ull,tmp_parent_buf };
						req.relativeTransforms = { 0ull,tmp_transform_buf };
						req.nodes = { 0ull,tmp_node_buf->getSize(),tmp_node_buf };
						ttm->setupTransfers(req, transfers);
					}

					cmdbuf_nodes->begin(0);
					utils->getDefaultPropertyPoolHandler()->transferProperties(
						cmdbuf_nodes.get(), fence_nodes.get(), scratch, { 0ull,tmp_node_buf },
						transfers, transfers + scene::ITransformTreeManager::TransferCount, initOutput.logger.get()
					);
					cmdbuf_nodes->end();
				}

				// submit
				{
					video::IGPUQueue::SSubmitInfo submit;
					submit.commandBufferCount = 1u;
					submit.commandBuffers = &cmdbuf_nodes.get();
					q->submit(1u, &submit, fence_nodes.get());
				}

				// wait
				const bool success = device->blockForFences(1u, &fence_nodes.get());
				if (!success)
					exit(-3);
			}

			// Geom Create
			auto geometryCreator = assetManager->getGeometryCreator();
			auto sphereGeom = geometryCreator->createSphereMesh(0.5f);

			// Camera Stuff
			core::vectorSIMDf cameraPosition(0, 20, -50);
			matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(60), float(WIN_W) / WIN_H, 0.01, 100);
			matrix3x4SIMD view = matrix3x4SIMD::buildCameraLookAtMatrixRH(cameraPosition, core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 1, 0));
			viewProj = matrix4SIMD::concatenateBFollowedByA(proj, matrix4SIMD(view));

			// Creating CPU Shaders 

			auto createCPUSpecializedShaderFromSource = [=](std::string&& source, asset::ISpecializedShader::E_SHADER_STAGE stage) -> core::smart_refctd_ptr<asset::ICPUSpecializedShader>
			{
				const std::string path = localInputCWD.string(); // TODO: make GLSL Compiler take `const system::path&` instead of cstrings
				auto unspec = assetManager->getGLSLCompiler()->resolveIncludeDirectives(std::move(source),stage,path.c_str(),1u,initOutput.logger.get());
				if (!unspec)
					return nullptr;

				asset::ISpecializedShader::SInfo info(nullptr, nullptr, "main", stage, "");
				return core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspec), std::move(info));
			};

			auto vs = createCPUSpecializedShaderFromSource(vertexSource, asset::ISpecializedShader::ESS_VERTEX);
			auto fs = createCPUSpecializedShaderFromSource(fragmentSource, asset::ISpecializedShader::ESS_FRAGMENT);
			asset::ICPUSpecializedShader* shaders[2]{ vs.get(), fs.get() };

			auto cpuMeshPlanets = createMeshBufferFromGeomCreatorReturnType(sphereGeom, assetManager.get(), shaders, shaders + 2);

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
				asset::ICPUMeshBuffer* cpuMesh,
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
					auto gfxLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(range,range+1u,scene::ITransformTreeWithNormalMatrices::createRenderDescriptorSetLayout());
					pipeline->setLayout(core::smart_refctd_ptr(gfxLayout));

					core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> rpIndependentPipeline = CPU2GPU.getGPUObjectsFromAssets(&pipeline, &pipeline + 1, cpu2gpuParams)->front();

					asset::SBufferBinding<video::IGPUBuffer> colorBufBinding;
					colorBufBinding.offset = colorBufferOffset;
					colorBufBinding.buffer = colorBuffer;

					ret.gpuMesh = CPU2GPU.getGPUObjectsFromAssets(&cpuMesh, &cpuMesh + 1, cpu2gpuParams)->front();
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

			lastTime = std::chrono::high_resolution_clock::now();

			for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
			{
				imageAcquire[i] = device->createSemaphore();
				renderFinished[i] = device->createSemaphore();
			}

			constexpr size_t ModsRangesBufSz = 2u * sizeof(uint32_t) + sizeof(scene::nbl_glsl_transform_tree_modification_request_range_t) * ObjectCount;
			video::IGPUBuffer::SCreationParams creationParams;
			creationParams.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
			creationParams.sharingMode = asset::E_SHARING_MODE::ESM_CONCURRENT;
			creationParams.queueFamilyIndexCount = 0u;
			creationParams.queueFamilyIndices = nullptr;



			modRangesBuf = device->createDeviceLocalGPUBufferOnDedMem(creationParams, ModsRangesBufSz);
			relTformModsBuf = device->createDeviceLocalGPUBufferOnDedMem(creationParams, sizeof(scene::nbl_glsl_transform_tree_relative_transform_modification_t) * ObjectCount);
			nodeIdsBuf = device->createDeviceLocalGPUBufferOnDedMem(creationParams, std::max(sizeof(uint32_t) + sizeof(scene::ITransformTree::node_t) * ObjectCount, 128ull));
			{
				//update `nodeIdsBuf`
				uint32_t countAndIds[1u + ObjectCount];
				countAndIds[0] = ObjectCount;
				for (uint32_t i = 0u; i < ObjectCount; ++i)
					countAndIds[1u + i] = solarSystemObjectsData[i].node;

				asset::SBufferRange<video::IGPUBuffer> bufrng;
				bufrng.buffer = nodeIdsBuf;
				bufrng.offset = 0;
				bufrng.size = nodeIdsBuf->getSize();
				utils->updateBufferRangeViaStagingBuffer(device->getQueue(0, 0), bufrng, countAndIds);
			}

			ttmDescriptorSets = ttm->createAllDescriptorSets(device.get());
			ttm->updateUpdateLocalTransformsDescriptorSet(device.get(),ttmDescriptorSets.updateLocal.get(),{0ull,modRangesBuf},{0ull,relTformModsBuf});
			ttm->updateRecomputeGlobalTransformsDescriptorSet(device.get(),ttmDescriptorSets.recomputeGlobal.get(),{0ull,nodeIdsBuf});
			core::vector<CompressedAABB> aabbs;
			for (auto obj : solarSystemObjectsData)
			{
				auto aabb = cpuMeshPlanets->getBoundingBox();
				aabb.MinEdge *= obj.scale;
				aabb.MaxEdge *= obj.scale;
				aabbs.emplace_back() = aabb;
			}
			ttm->updateDebugDrawDescriptorSet(device.get(),ttmDescriptorSets.debugDraw.get(),{0ull,utils->createFilledDeviceLocalGPUBufferOnDedMem(device->getQueue(0,0),sizeof(core::CompressedAABB)*aabbs.size(),aabbs.data())});
		}

		void onAppTerminated_impl() override
		{
			
		}

		void workLoopBody() override
		{
			resourceIx++;
			if (resourceIx >= FRAMES_IN_FLIGHT)
				resourceIx = 0;

			auto& cb = cmdbuf[resourceIx];
			auto& fence = frameComplete[resourceIx];
			if (fence)
				device->blockForFences(1u, &fence.get());
			else
				fence = device->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

			auto now = std::chrono::high_resolution_clock::now();
			dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime).count();
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
				static_assert(offsetof(SSBO, ranges) == sizeof(uint32_t) * 2ull);
				SSBO requestRanges;
				requestRanges.rangeCount = ObjectCount;
				requestRanges.maxRangeLength = 1u;
				for (uint32_t i = 0u; i < ObjectCount; ++i)
				{
					auto& obj = solarSystemObjectsData[i];
					requestRanges.ranges[i].nodeID = obj.node;
					requestRanges.ranges[i].requestsBegin = i;
					requestRanges.ranges[i].requestsEnd = i + 1u;
					requestRanges.ranges[i].newTimestamp = timestamp;
				}

				asset::SBufferRange<video::IGPUBuffer> bufrng;
				bufrng.buffer = modRangesBuf;
				bufrng.offset = 0;
				bufrng.size = modRangesBuf->getSize();
				utils->updateBufferRangeViaStagingBuffer(cb.get(), fence.get(), queues[decltype(initOutput)::EQT_GRAPHICS], bufrng, &requestRanges, waitSemaphoreCount, waitSems, waitStages);
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
					scaleMat.setScale(core::vectorSIMDf(solarSystemObjectsData[i].scale));

					auto tform = core::matrix3x4SIMD::concatenateBFollowedByA(rotationMat, translationMat);

					reqs[i] = scene::ITransformTreeManager::RelativeTransformModificationRequest(scene::ITransformTreeManager::RelativeTransformModificationRequest::ET_OVERWRITE, tform);
				}

				asset::SBufferRange<video::IGPUBuffer> bufrng;
				bufrng.buffer = relTformModsBuf;
				bufrng.offset = 0;
				bufrng.size = relTformModsBuf->getSize();

				utils->updateBufferRangeViaStagingBuffer(cb.get(), fence.get(), queues[decltype(initOutput)::EQT_GRAPHICS], bufrng, reqs.data(), waitSemaphoreCount, waitSems, waitStages);
			}

			// Update instances transforms 
			{
				// buffers to barrier w.r.t. updates
				video::IGPUCommandBuffer::SBufferMemoryBarrier barriers[scene::ITransformTreeManager::SBarrierSuggestion::MaxBufferCount];
				auto setBufferBarrier = [&barriers, cb](const uint32_t ix, const asset::SBufferRange<video::IGPUBuffer>& range, const asset::SMemoryBarrier& barrier)
				{
					barriers[ix].barrier = barrier;
					barriers[ix].dstQueueFamilyIndex = barriers[ix].srcQueueFamilyIndex = cb->getQueueFamilyIndex();
					barriers[ix].buffer = range.buffer;
					barriers[ix].offset = range.offset;
					barriers[ix].size = range.size;
				};

				const core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> renderingStages = asset::EPSF_VERTEX_SHADER_BIT;
				const core::bitflag<asset::E_ACCESS_FLAGS> renderingAccesses = asset::EAF_SHADER_READ_BIT;
				 
				scene::ITransformTreeManager::BaseParams baseParams;
				baseParams.cmdbuf = cb.get();
				baseParams.tree = tt.get();
				baseParams.logger = initOutput.logger.get();
				scene::ITransformTreeManager::DispatchParams dispatchParams;
				dispatchParams.indirect.buffer = nullptr;
				dispatchParams.direct.nodeCount = ObjectCount;

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
					setBufferBarrier(barrierCount++, { 0ull,modRangesBuf->getSize(),modRangesBuf }, sugg.requestRanges);
					setBufferBarrier(barrierCount++, { 0ull,relTformModsBuf->getSize(),relTformModsBuf }, sugg.modificationRequests);
					cb->pipelineBarrier(sugg.srcStageMask, sugg.dstStageMask, asset::EDF_NONE, 0u, nullptr, barrierCount, barriers, 0u, nullptr);
				}
				ttm->updateLocalTransforms(baseParams,dispatchParams,ttmDescriptorSets.updateLocal.get());
				// barrier between TTM update and TTM recompute
				{
					auto sugg = scene::ITransformTreeManager::barrierHelper(scene::ITransformTreeManager::SBarrierSuggestion::EF_INBETWEEN_RLEATIVE_UPDATE_AND_GLOBAL_RECOMPUTE);
					sugg.srcStageMask |= renderingStages; // also Rendering and TTM recompute
					sugg.globalTransforms.srcAccessMask |= asset::EAF_SHADER_READ_BIT;
					sugg.dstStageMask |= asset::EPSF_TRANSFER_BIT; // as well as TTM update and Transfer
					sugg.requestRanges.dstAccessMask |= asset::EAF_TRANSFER_WRITE_BIT;
					sugg.modificationRequests.dstAccessMask |= asset::EAF_TRANSFER_WRITE_BIT;
					uint32_t barrierCount = 0u;
					setBufferBarrier(barrierCount++, { 0ull,modRangesBuf->getSize(),modRangesBuf }, sugg.requestRanges);
					setBufferBarrier(barrierCount++, { 0ull,relTformModsBuf->getSize(),relTformModsBuf }, sugg.modificationRequests);
					setBufferBarrier(barrierCount++, node_pp->getPropertyMemoryBlock(scene::ITransformTree::relative_transform_prop_ix), sugg.relativeTransforms);
					setBufferBarrier(barrierCount++, node_pp->getPropertyMemoryBlock(scene::ITransformTree::modified_stamp_prop_ix), sugg.modifiedTimestamps);
					cb->pipelineBarrier(sugg.srcStageMask, sugg.dstStageMask, asset::EDF_NONE, 0u, nullptr, barrierCount, barriers, 0u, nullptr);
				}
				ttm->recomputeGlobalTransforms(baseParams,dispatchParams,ttmDescriptorSets.recomputeGlobal.get());
				// barrier between TTM recompute and TTM recompute+update 
				{
					auto sugg = scene::ITransformTreeManager::barrierHelper(scene::ITransformTreeManager::SBarrierSuggestion::EF_POST_GLOBAL_TFORM_RECOMPUTE);
					sugg.dstStageMask |= renderingStages; // also also TTM recompute and rendering shader (to read the global transforms)
					uint32_t barrierCount = 0u;
					setBufferBarrier(barrierCount++, node_pp->getPropertyMemoryBlock(scene::ITransformTree::relative_transform_prop_ix), sugg.relativeTransforms);
					setBufferBarrier(barrierCount++, node_pp->getPropertyMemoryBlock(scene::ITransformTree::modified_stamp_prop_ix), sugg.modifiedTimestamps);
					setBufferBarrier(barrierCount++, node_pp->getPropertyMemoryBlock(scene::ITransformTree::global_transform_prop_ix), sugg.globalTransforms);
					setBufferBarrier(barrierCount++, node_pp->getPropertyMemoryBlock(scene::ITransformTree::recomputed_stamp_prop_ix), sugg.recomputedTimestamps);
					setBufferBarrier(barrierCount++, node_pp->getPropertyMemoryBlock(scene::ITransformTreeWithNormalMatrices::normal_matrix_prop_ix), sugg.normalMatrices);
					cb->pipelineBarrier(sugg.srcStageMask, sugg.dstStageMask, asset::EDF_NONE, 0u, nullptr, barrierCount, barriers, 0u, nullptr);
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
				asset::SClearValue clearValues[2] = {};
				VkRect2D area;
				clearValues[0].color.float32[0] = 0.1f;
				clearValues[0].color.float32[1] = 0.1f;
				clearValues[0].color.float32[2] = 0.1f;
				clearValues[0].color.float32[3] = 1.f;

				clearValues[1].depthStencil.depth = 0.0f;
				clearValues[1].depthStencil.stencil = 0.0f;

				info.renderpass = renderpass;
				info.framebuffer = fbos[0];
				info.clearValueCount = 2u;
				info.clearValues = clearValues;
				info.renderArea.offset = { 0, 0 };
				info.renderArea.extent = { WIN_W, WIN_H };
				cb->beginRenderPass(&info, asset::ESC_INLINE);
			}
			// draw
			{
				assert(gpuObjects.size() == 1ull);
				// Draw Stuff 
				for (uint32_t i = 0; i < gpuObjects.size(); ++i)
				{
					auto& gpuObject = gpuObjects[i];

					cb->bindGraphicsPipeline(gpuObject.graphicsPipeline.get());
					cb->pushConstants(gpuObject.graphicsPipeline->getRenderpassIndependentPipeline()->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), viewProj.pointer());
					cb->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuObject.graphicsPipeline->getRenderpassIndependentPipeline()->getLayout(), 0u, 1u, &ttDS);
					cb->drawMeshBuffer(gpuObject.gpuMesh.get());
				}
				{
					asset::SBufferBinding<video::IGPUBuffer> nodeAndAABBList = {sizeof(uint32_t),nodeIdsBuf};
					scene::ITransformTreeManager::DebugPushConstants pushConstants;
					pushConstants.viewProjectionMatrix = viewProj;
					pushConstants.lineColor = core::vectorSIMDf(1.f,0.f,0.f,1.f);
					pushConstants.aabbColor = core::vectorSIMDf(0.f,1.f,0.f,1.f);
					ttm->debugDraw(cb.get(),debugDrawPipeline.get(),tt.get(),ttmDescriptorSets.debugDraw.get(),nodeAndAABBList,nodeAndAABBList,pushConstants,ObjectCount);
				}
			}
			// TODO: debug draw
			cb->endRenderPass();
			cb->end();

			// acquires and presents 
			uint32_t imgnum = 0u;
			swapchain->acquireNextImage(MAX_TIMEOUT, imageAcquire[resourceIx].get(), nullptr, &imgnum);
			CommonAPI::Submit(device.get(), swapchain.get(), cb.get(), queues[decltype(initOutput)::EQT_GRAPHICS], imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
			CommonAPI::Present(device.get(), swapchain.get(), queues[decltype(initOutput)::EQT_GRAPHICS], renderFinished[resourceIx].get(), imgnum);
		}

		bool keepRunning() override
		{
			return windowCb->isWindowOpen();
		}

	private:
		CommonAPI::InitOutput<FBO_COUNT> initOutput;
		nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
		nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> gl;
		nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
		nbl::video::IPhysicalDevice* gpuPhysicalDevice;
		nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> device;
		std::array<nbl::video::IGPUQueue*, CommonAPI::InitOutput<FBO_COUNT>::EQT_COUNT> queues = { nullptr, nullptr, nullptr, nullptr };
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
		std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, FBO_COUNT> fbos;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool> commandPool;
		nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
		nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
		nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
		nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
		nbl::core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
		nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
		nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utils;

		nbl::core::smart_refctd_ptr<nbl::scene::ITransformTreeManager> ttm;
		nbl::core::smart_refctd_ptr<nbl::scene::ITransformTree> tt;

		core::smart_refctd_ptr<video::IGPUBuffer> modRangesBuf;
		core::smart_refctd_ptr<video::IGPUBuffer> relTformModsBuf;
		core::smart_refctd_ptr<video::IGPUBuffer> nodeIdsBuf;
		const video::IGPUDescriptorSet* ttDS;
		scene::ITransformTreeManager::DescriptorSets ttmDescriptorSets;
		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> debugDrawPipeline;

		core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf[FRAMES_IN_FLIGHT];
		core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
		core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
		core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };

		core::vector<GPUObject> gpuObjects;
		core::vector<SolarSystemObject> solarSystemObjectsData;

		uint32_t timestamp = 1u;
		uint32_t resourceIx = 0;
		double dt = 0; //! render loop
		std::chrono::steady_clock::time_point lastTime;

		core::matrix4SIMD viewProj;
};

NBL_COMMON_API_MAIN(TransformationApp)