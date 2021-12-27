#define _NBL_STATIC_LIB
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "../common/Camera.hpp"
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/ui/ICursorControl.h"

#include <fstream>

using namespace nbl;

// #define DEBUG_VIZ

struct vec3
{
	float x, y, z;
};

struct alignas(16) vec3_aligned
{
	float x, y, z;
};

struct uvec2
{
	uint32_t x, y;
};

struct nbl_glsl_ext_ClusteredLighting_SpotLight
{
	vec3 position;
	float outerCosineOverCosineRange; // `cos(outerHalfAngle) / (cos(innerHalfAngle) - cos(outerHalfAngle))`
	uvec2 intensity; // rgb19e7 encoded
	uvec2 direction; // xyz encoded as packSNorm2x16, w used for storing `cosineRange`
};

struct nbl_glsl_shapes_AABB_t
{
	vec3_aligned minVx;
	vec3_aligned maxVx;
};

#define LOG(fmt, ...) logger->log(fmt, system::ILogger::ELL_PERFORMANCE, __VA_ARGS__)

class ClusteredRenderingSampleApp : public ApplicationBase
{
	constexpr static uint32_t WIN_W = 1280u;
	constexpr static uint32_t WIN_H = 720u;
	constexpr static uint32_t SC_IMG_COUNT = 3u;
	constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	static constexpr uint32_t LIGHT_COUNT = 1680u;

	// Todo(achal): It could be a good idea to make LOD_COUNT dynamic based on double/float precision?
	//
	// Level 9 is the outermost/coarsest, Level 0 is the innermost/finest
	static constexpr uint32_t LOD_COUNT = 10u;

	constexpr static uint32_t CAMERA_DS_NUMBER = 1u;
	constexpr static uint32_t LIGHT_DS_NUMBER = 2u;

	constexpr static uint32_t Z_PREPASS_INDEX = 0u;
	constexpr static uint32_t LIGHTING_PASS_INDEX = 1u;
	constexpr static uint32_t DEBUG_DRAW_PASS_INDEX = 2u;

	constexpr static float LIGHT_CONTRIBUTION_THRESHOLD = 2.f;
	constexpr static float LIGHT_RADIUS = 25.f;

	// These are only clipmap specific
	constexpr static uint32_t VOXEL_COUNT_PER_DIM = 4u;
	constexpr static uint32_t VOXEL_COUNT_PER_LEVEL = VOXEL_COUNT_PER_DIM * VOXEL_COUNT_PER_DIM * VOXEL_COUNT_PER_DIM;

	constexpr static float DEBUG_CONE_RADIUS = 10.f;
	constexpr static float DEBUG_CONE_LENGTH = 25.f;
	constexpr static vec3 DEBUG_CONE_DIRECTION = { 0.f, -1.f, 0.f };

	struct cone_t
	{
		core::vectorSIMDf tip;
		core::vectorSIMDf direction; // always normalized
		float height;
		float cosHalfAngle;
	};

#ifdef DEBUG_VIZ
	void debugCreateLightVolumeGPUResources()
	{
		// Todo(achal): Depending upon the light type I can change the light volume shape here
		const float radius = 10.f;
		const float length = 25.f;
		asset::IGeometryCreator::return_type lightVolume = assetManager->getGeometryCreator()->createConeMesh(radius, length, 10);

		asset::SPushConstantRange pcRange = {};
		pcRange.offset = 0u;
		pcRange.size = sizeof(core::matrix4SIMD);
		pcRange.stageFlags = asset::IShader::ESS_VERTEX;

		auto vertShader = createShader("../debug_draw_light_volume.vert");
		auto fragShader = createShader("nbl/builtin/material/debug/vertex_normal/specialized_shader.frag");

		video::IGPUSpecializedShader* shaders[2] = { vertShader.get(), fragShader.get() };

		asset::SVertexInputParams& vertexInputParams = lightVolume.inputParams;
		constexpr uint32_t POS_ATTRIBUTE_LOCATION = 0u;
		vertexInputParams.enabledAttribFlags = (1u << POS_ATTRIBUTE_LOCATION); // disable all other unused attributes to not get validation perf warning

		asset::SBlendParams blendParams = {};
		for (size_t i = 0ull; i < nbl::asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
			blendParams.blendParams[i].attachmentEnabled = (i == 0ull);

		asset::SRasterizationParams rasterizationParams = {};

		auto renderpassIndep = logicalDevice->createGPURenderpassIndependentPipeline
		(
			nullptr,
			logicalDevice->createGPUPipelineLayout(&pcRange, &pcRange + 1, nullptr, nullptr, nullptr, nullptr),
			shaders,
			shaders + 2,
			lightVolume.inputParams,
			blendParams,
			lightVolume.assemblyParams,
			rasterizationParams
		);

		constexpr uint32_t MAX_DATA_BUFFER_COUNT = video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT + 1u;

		uint32_t cpuBufferCount = 0u;
		core::vector<asset::ICPUBuffer*> cpuBuffers(MAX_DATA_BUFFER_COUNT);
		for (size_t i = 0ull; i < video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; ++i)
		{
			const bool bufferBindingEnabled = (lightVolume.inputParams.enabledBindingFlags & (1 << i));
			if (bufferBindingEnabled)
				cpuBuffers[cpuBufferCount++] = lightVolume.bindings[i].buffer.get();
		}
		asset::ICPUBuffer* cpuIndexBuffer = lightVolume.indexBuffer.buffer.get();
		if (cpuIndexBuffer)
			cpuBuffers[cpuBufferCount++] = cpuIndexBuffer;

		cpu2gpuParams.beginCommandBuffers();
		auto gpuArray = cpu2gpu.getGPUObjectsFromAssets(cpuBuffers.data(), cpuBuffers.data() + cpuBufferCount, cpu2gpuParams);
		if (!gpuArray || gpuArray->size() < 1u || !(*gpuArray)[0])
		{
			logger->log("Failed to convert debug light volume's vertex and index buffers from CPU to GPU!\n", system::ILogger::ELL_ERROR);
			exit(-1);
		}
		cpu2gpuParams.waitForCreationToComplete();

		asset::SBufferBinding<video::IGPUBuffer> gpuBufferBindings[MAX_DATA_BUFFER_COUNT] = {};
		for (auto i = 0, j = 0; i < video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; i++)
		{
			const bool bufferBindingEnabled = (lightVolume.inputParams.enabledBindingFlags & (1 << i));
			if (!bufferBindingEnabled)
				continue;

			auto buffPair = gpuArray->operator[](j++);
			gpuBufferBindings[i].offset = buffPair->getOffset();
			gpuBufferBindings[i].buffer = core::smart_refctd_ptr<video::IGPUBuffer>(buffPair->getBuffer());
		}
		if (cpuIndexBuffer)
		{
			auto buffPair = gpuArray->back();
			gpuBufferBindings[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT].offset = buffPair->getOffset();
			gpuBufferBindings[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT].buffer = core::smart_refctd_ptr<video::IGPUBuffer>(buffPair->getBuffer());
		}

		debugLightVolumeMeshBuffer = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(
			core::smart_refctd_ptr(renderpassIndep),
			nullptr,
			gpuBufferBindings,
			std::move(gpuBufferBindings[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT]));
		if (!debugLightVolumeMeshBuffer)
			exit(-1);
		{
			debugLightVolumeMeshBuffer->setIndexType(lightVolume.indexType);
			debugLightVolumeMeshBuffer->setIndexCount(lightVolume.indexCount);
			debugLightVolumeMeshBuffer->setBoundingBox(lightVolume.bbox);
		}

		video::IGPUGraphicsPipeline::SCreationParams creationParams = {};
		creationParams.renderpass = renderpass;
		creationParams.renderpassIndependent = renderpassIndep;
		creationParams.subpassIx = DEBUG_DRAW_PASS_INDEX;
		debugLightVolumeGraphicsPipeline = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(creationParams));
		if (!debugLightVolumeGraphicsPipeline)
			exit(-1);
	}

	void debugCreateAABBGPUResources()
	{
		constexpr size_t INDEX_COUNT = 24ull;
		uint16_t indices[INDEX_COUNT];
		{
			indices[0] = 0b000;
			indices[1] = 0b001;
			indices[2] = 0b001;
			indices[3] = 0b011;
			indices[4] = 0b011;
			indices[5] = 0b010;
			indices[6] = 0b010;
			indices[7] = 0b000;
			indices[8] = 0b000;
			indices[9] = 0b100;
			indices[10] = 0b001;
			indices[11] = 0b101;
			indices[12] = 0b010;
			indices[13] = 0b110;
			indices[14] = 0b011;
			indices[15] = 0b111;
			indices[16] = 0b100;
			indices[17] = 0b101;
			indices[18] = 0b101;
			indices[19] = 0b111;
			indices[20] = 0b100;
			indices[21] = 0b110;
			indices[22] = 0b110;
			indices[23] = 0b111;
		}
		const size_t indexBufferSize = INDEX_COUNT * sizeof(uint16_t);

		video::IGPUBuffer::SCreationParams creationParams = {};
		creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_INDEX_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT);
		debugAABBIndexBuffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, indexBufferSize);

		core::smart_refctd_ptr<video::IGPUFence> fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);
		asset::SBufferRange<video::IGPUBuffer> bufferRange;
		bufferRange.offset = 0ull;
		bufferRange.size = debugAABBIndexBuffer->getCachedCreationParams().declaredSize;
		bufferRange.buffer = debugAABBIndexBuffer;
		utilities->updateBufferRangeViaStagingBuffer(
			fence.get(),
			queues[CommonAPI::InitOutput::EQT_TRANSFER_UP],
			bufferRange,
			indices);
		logicalDevice->blockForFences(1u, &fence.get());

		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> dsLayout = nullptr;
		{
			video::IGPUDescriptorSetLayout::SBinding binding[2];
			binding[0].binding = 0u;
			binding[0].type = asset::EDT_STORAGE_BUFFER;
			binding[0].count = 1u;
			binding[0].stageFlags = asset::IShader::ESS_VERTEX;
			binding[0].samplers = nullptr;

			binding[1].binding = 1u;
			binding[1].type = asset::EDT_UNIFORM_BUFFER;
			binding[1].count = 1u;
			binding[1].stageFlags = asset::IShader::ESS_VERTEX;
			binding[1].samplers = nullptr;

			dsLayout = logicalDevice->createGPUDescriptorSetLayout(binding, binding + 2);

			if (!dsLayout)
			{
				logger->log("Failed to create GPU DS layout for debug draw AABB!\n", system::ILogger::ELL_ERROR);
				exit(-1);
			}
		}

		auto pipelineLayout = logicalDevice->createGPUPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr(dsLayout));

		auto vertShader = createShader("../debug_draw_aabb.vert");
		auto fragShader = createShader("nbl/builtin/material/debug/vertex_normal/specialized_shader.frag");

		core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> renderpassIndep = nullptr;
		{
			asset::SVertexInputParams vertexInputParams = {};

			asset::SBlendParams blendParams = {};
			for (size_t i = 0ull; i < nbl::asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
				blendParams.blendParams[i].attachmentEnabled = (i == 0ull);

			asset::SPrimitiveAssemblyParams primitiveAssemblyParams = {};
			primitiveAssemblyParams.primitiveType = asset::EPT_LINE_LIST;

			asset::SRasterizationParams rasterizationParams = {};

			video::IGPUSpecializedShader* const shaders[2] = { vertShader.get(), fragShader.get() };
			renderpassIndep = logicalDevice->createGPURenderpassIndependentPipeline(
				nullptr,
				std::move(pipelineLayout),
				shaders, shaders + 2,
				vertexInputParams,
				blendParams,
				primitiveAssemblyParams,
				rasterizationParams);
		}

		// graphics pipeline
		{
			video::IGPUGraphicsPipeline::SCreationParams creationParams = {};
			creationParams.renderpassIndependent = renderpassIndep;
			creationParams.renderpass = renderpass;
			creationParams.subpassIx = DEBUG_DRAW_PASS_INDEX;
			debugAABBGraphicsPipeline = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(creationParams));
		}

		// create debug draw AABB buffers
		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		{
			const size_t neededSSBOSize = VOXEL_COUNT_PER_LEVEL * LOD_COUNT * sizeof(nbl_glsl_shapes_AABB_t);

			video::IGPUBuffer::SCreationParams creationParams = {};
			creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_TRANSFER_DST_BIT | video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT);
			debugClustersForLightGPU[i] = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, neededSSBOSize);
		}

		// create and update descriptor sets
		{
			// Todo(achal): Is creating 5 DS the only solution for:
			// 1. Having a GPU buffer per command buffer? or,
			// 2. Somehow sharing a GPU buffer amongst 5 different command buffers without
			// race conditions
			// Is using a property pool a solution?
			const uint32_t setCount = FRAMES_IN_FLIGHT;
			auto dsPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &dsLayout.get(), &dsLayout.get() + 1ull, &setCount);

			for (uint32_t i = 0u; i < setCount; ++i)
			{
				debugAABBDescriptorSets[i] = logicalDevice->createGPUDescriptorSet(dsPool.get(), core::smart_refctd_ptr(dsLayout));

				video::IGPUDescriptorSet::SWriteDescriptorSet writes[2];

				writes[0].dstSet = debugAABBDescriptorSets[i].get();
				writes[0].binding = 0u;
				writes[0].count = 1u;
				writes[0].arrayElement = 0u;
				writes[0].descriptorType = asset::EDT_STORAGE_BUFFER;
				video::IGPUDescriptorSet::SDescriptorInfo infos[2];
				{
					infos[0].desc = debugClustersForLightGPU[i];
					infos[0].buffer.offset = 0u;
					infos[0].buffer.size = debugClustersForLightGPU[i]->getCachedCreationParams().declaredSize;
				}
				writes[0].info = &infos[0];

				writes[1].dstSet = debugAABBDescriptorSets[i].get();
				writes[1].binding = 1u;
				writes[1].count = 1u;
				writes[1].arrayElement = 0u;
				writes[1].descriptorType = asset::EDT_UNIFORM_BUFFER;
				{
					infos[1].desc = cameraUbo;
					infos[1].buffer.offset = 0u;
					infos[1].buffer.size = cameraUbo->getCachedCreationParams().declaredSize;
				}
				writes[1].info = &infos[1];
				logicalDevice->updateDescriptorSets(2u, writes, 0u, nullptr);
			}
		}
	}

	void debugRecordLightIDToClustersMap(
		core::unordered_map<uint32_t, core::unordered_set<uint32_t>>& map,
		const core::vector<uint32_t>& lightIndexList)
	{
		if (!lightGridCPUBuffer || (lightGridCPUBuffer->getSize() == 0ull))
		{
			logger->log("lightGridCPUBuffer does not exist or is empty!\n", system::ILogger::ELL_ERROR);
			return;
		}

		if (lightIndexList.empty())
			logger->log("lightIndexList is empty!\n");

		uint32_t* lightGridEntry = static_cast<uint32_t*>(lightGridCPUBuffer->getPointer());
		for (int32_t level = LOD_COUNT - 1; level >= 0; --level)
		{
			for (uint32_t clusterID = 0u; clusterID < VOXEL_COUNT_PER_LEVEL; ++clusterID)
			{
				const uint32_t packedValue = *lightGridEntry++;
				const uint32_t offset = packedValue & 0xFFFF;
				const uint32_t count = (packedValue >> 16) & 0xFFFF;

				const uint32_t globalClusterID = (LOD_COUNT - 1 - level) * VOXEL_COUNT_PER_LEVEL + clusterID;

				for (uint32_t i = 0u; i < count; ++i)
				{
					const uint32_t globalLightID = lightIndexList[offset + i];

					if (map.find(globalLightID) != map.end())
						map[globalLightID].insert(globalClusterID);
					else
						map[globalLightID] = { globalClusterID };
				}
			}
		}
	}

	void debugUpdateLightClusterAssignment(
		video::IGPUCommandBuffer* commandBuffer,
		const uint32_t lightIndex,
		core::vector<nbl_glsl_shapes_AABB_t>& clustersForLight,
		core::unordered_map<uint32_t, core::unordered_set<uint32_t>>& debugDrawLightIDToClustersMap,
		const nbl_glsl_shapes_AABB_t* clipmap)
	{
		// figure out assigned clusters
		const auto& clusterIndices = debugDrawLightIDToClustersMap[lightIndex];
		if (!clusterIndices.empty())
		{
			// Todo(achal): It might not be very efficient to create a vector (allocate memory) every frame,
			// would reusing vectors be better? This is debug code anyway..
			clustersForLight.resize(clusterIndices.size());

			uint32_t i = 0u;
			for (uint32_t clusterIdx : clusterIndices)
				clustersForLight[i++] = clipmap[clusterIdx];

			// update buffer needs to go outside of a render pass!!!
			commandBuffer->updateBuffer(debugClustersForLightGPU[resourceIx].get(), 0ull, clusterIndices.size() * sizeof(nbl_glsl_shapes_AABB_t), clustersForLight.data());
		}
	}

	void debugDrawLightClusterAssignment(
		video::IGPUCommandBuffer* commandBuffer,
		const uint32_t lightIndex,
		core::vector<nbl_glsl_shapes_AABB_t>& clustersForLight)
	{
		if (lightIndex != -1)
		{
			debugUpdateModelMatrixForLightVolume(getLightVolume(lights[lightIndex]));

			core::matrix4SIMD mvp = core::concatenateBFollowedByA(camera.getConcatenatedMatrix(), debugLightVolumeModelMatrix);
			commandBuffer->pushConstants(
				debugLightVolumeGraphicsPipeline->getRenderpassIndependentPipeline()->getLayout(),
				asset::IShader::ESS_VERTEX,
				0u, sizeof(matrix4SIMD), &mvp);
			commandBuffer->bindGraphicsPipeline(debugLightVolumeGraphicsPipeline.get());
			commandBuffer->drawMeshBuffer(debugLightVolumeMeshBuffer.get());
		}

		if (!clustersForLight.empty())
		{
			// draw assigned clusters
			video::IGPUDescriptorSet const* descriptorSet[1] = { debugAABBDescriptorSets[resourceIx].get() };
			commandBuffer->bindDescriptorSets(
				asset::EPBP_GRAPHICS,
				debugAABBGraphicsPipeline->getRenderpassIndependentPipeline()->getLayout(),
				0u, 1u, descriptorSet);
			commandBuffer->bindGraphicsPipeline(debugAABBGraphicsPipeline.get());
			commandBuffer->bindIndexBuffer(debugAABBIndexBuffer.get(), 0u, asset::EIT_16BIT);
			commandBuffer->drawIndexed(24u, clustersForLight.size(), 0u, 0u, 0u);
		}
	}

	void debugUpdateModelMatrixForLightVolume(const cone_t& cone)
	{
		// Scale
		core::vectorSIMDf scaleFactor;
		{
			const float tanOuterHalfAngle = core::sqrt(core::max(1.f - (cone.cosHalfAngle * cone.cosHalfAngle), 0.f)) / cone.cosHalfAngle;
			scaleFactor.X = (cone.height * tanOuterHalfAngle) / DEBUG_CONE_RADIUS;
			scaleFactor.Y = cone.height / DEBUG_CONE_LENGTH;
			scaleFactor.Z = scaleFactor.X;
		}

		// Rotation
		const core::vectorSIMDf modelSpaceConeDirection(DEBUG_CONE_DIRECTION.x, DEBUG_CONE_DIRECTION.y, DEBUG_CONE_DIRECTION.z);
		const float angle = std::acosf(core::dot(cone.direction, modelSpaceConeDirection).x);
		core::vectorSIMDf axis = core::normalize(core::cross(modelSpaceConeDirection, cone.direction));

		// Axis of rotation of the cone doesn't pass through its tip, hence
		// the order of transformations is a bit tricky here
		// First apply no translation to get the cone tip after scaling and rotation
		debugLightVolumeModelMatrix.setScaleRotationAndTranslation(
			scaleFactor,
			quaternion::fromAngleAxis(angle, axis),
			core::vectorSIMDf(0.f));
		const core::vectorSIMDf modelSpaceConeTip(0.f, DEBUG_CONE_LENGTH, 0.f);
		core::vectorSIMDf scaledAndRotatedConeTip = modelSpaceConeTip;
		debugLightVolumeModelMatrix.transformVect(scaledAndRotatedConeTip);
		// Now we can apply the correct translation to the model matrix
		// Translation
		debugLightVolumeModelMatrix.setTranslation(cone.tip - scaledAndRotatedConeTip);
	}

#else
#define debugCreateLightVolumeGPUResources(...)
#define debugCreateAABBGPUResources(...)
#define debugRecordLightIDToClustersMap(...)
#define debugUpdateLightClusterAssignment(...)
#define debugDrawLightClusterAssignment(...)
#define debugIncrementActiveLightIndex(...)
#define debugUpdateModelMatrixForLightVolume(...)
#endif

public:
	// Todo(achal): Remove this, annoying as fuck
	auto createDescriptorPool(const uint32_t textureCount)
	{
		constexpr uint32_t maxItemCount = 256u;
		{
			nbl::video::IDescriptorPool::SDescriptorPoolSize poolSize;
			poolSize.count = textureCount;
			poolSize.type = nbl::asset::EDT_COMBINED_IMAGE_SAMPLER;
			return logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxItemCount, 1u, &poolSize);
		}
	}
	void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
	{
		window = std::move(wnd);
	}
	nbl::ui::IWindow* getWindow() override
	{
		return window.get();
	}
	void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& system) override
	{
		system = std::move(system);
	}

	APP_CONSTRUCTOR(ClusteredRenderingSampleApp);

	void onAppInitialized_impl() override
	{
		// Input set of lights in world position
		constexpr uint32_t MAX_LIGHT_COUNT = (1u << 22);
		generateLights(LIGHT_COUNT);

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

		CommonAPI::InitOutput initOutput;
		initOutput.window = core::smart_refctd_ptr(window);
		CommonAPI::Init(
			initOutput,
			video::EAT_VULKAN,
			"ClusteredLighting",
			requiredInstanceFeatures,
			optionalInstanceFeatures,
			requiredDeviceFeatures,
			optionalDeviceFeatures,
			WIN_W, WIN_H, SC_IMG_COUNT,
			swapchainImageUsage,
			surfaceFormat);

		system = std::move(initOutput.system);
		window = std::move(initOutput.window);
		windowCb = std::move(initOutput.windowCb);
		apiConnection = std::move(initOutput.apiConnection);
		surface = std::move(initOutput.surface);
		physicalDevice = std::move(initOutput.physicalDevice);
		logicalDevice = std::move(initOutput.logicalDevice);
		utilities = std::move(initOutput.utilities);
		queues = std::move(initOutput.queues);
		swapchain = std::move(initOutput.swapchain);
		commandPools = std::move(initOutput.commandPools);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);

		renderpass = createRenderpass();
		if (!renderpass)
		{
			logger->log("Failed to create the render pass!\n", system::ILogger::ELL_ERROR);
			exit(-1);
		}

		video::CPropertyPoolHandler* propertyPoolHandler = utilities->getDefaultPropertyPoolHandler();

		const uint32_t capacity = MAX_LIGHT_COUNT;
		const bool contiguous = false;

		video::IGPUBuffer::SCreationParams creationParams = {};
		creationParams.usage = static_cast<asset::IBuffer::E_USAGE_FLAGS>(asset::IBuffer::EUF_STORAGE_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT);

		asset::SBufferRange<video::IGPUBuffer> blocks[PropertyPoolType::PropertyCount];
		for (uint32_t i = 0u; i < PropertyPoolType::PropertyCount; ++i)
		{
			asset::SBufferRange<video::IGPUBuffer>& block = blocks[i];
			block.offset = 0u;
			block.size = PropertyPoolType::PropertySizes[i] * capacity;
			block.buffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, block.size);
		}

		propertyPool = PropertyPoolType::create(logicalDevice.get(), blocks, capacity, contiguous);

		video::CPropertyPoolHandler::UpStreamingRequest upstreamingRequest;
		upstreamingRequest.setFromPool(propertyPool.get(), 0);
		upstreamingRequest.fill = false; // Don't know what this is used for
		upstreamingRequest.elementCount = LIGHT_COUNT;
		upstreamingRequest.source.data = lights.data();
		upstreamingRequest.source.device2device = false;
		// Don't know what are these for
		upstreamingRequest.srcAddresses = nullptr;
		upstreamingRequest.dstAddresses = nullptr;

		const auto& computeCommandPool = commandPools[CommonAPI::InitOutput::EQT_COMPUTE];
		core::smart_refctd_ptr<video::IGPUCommandBuffer> propertyTransferCommandBuffer;
		logicalDevice->createCommandBuffers(
			computeCommandPool.get(),
			video::IGPUCommandBuffer::EL_PRIMARY,
			1u,
			&propertyTransferCommandBuffer);

		auto propertyTransferFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
		auto computeQueue = queues[CommonAPI::InitOutput::EQT_COMPUTE];

		asset::SBufferBinding<video::IGPUBuffer> scratchBufferBinding;
		{
			video::IGPUBuffer::SCreationParams scratchParams = {};
			scratchParams.canUpdateSubRange = true;
			scratchParams.usage = core::bitflag(video::IGPUBuffer::EUF_TRANSFER_DST_BIT) | video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
			scratchBufferBinding = { 0ull,logicalDevice->createDeviceLocalGPUBufferOnDedMem(scratchParams,propertyPoolHandler->getMaxScratchSize()) };
			scratchBufferBinding.buffer->setObjectDebugName("Scratch Buffer");
		}

		propertyTransferCommandBuffer->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
		{
			uint32_t waitSemaphoreCount = 0u;
			video::IGPUSemaphore* const* semaphoresToWaitBeforeOverwrite = nullptr;
			const asset::E_PIPELINE_STAGE_FLAGS* stagesToWaitForPerSemaphore = nullptr;

			auto* pRequest = &upstreamingRequest;

			// Todo(achal): Discarding return value produces nodiscard warning
			propertyPoolHandler->transferProperties(
				utilities->getDefaultUpStreamingBuffer(),
				propertyTransferCommandBuffer.get(),
				propertyTransferFence.get(),
				computeQueue,
				scratchBufferBinding,
				pRequest, 1u,
				waitSemaphoreCount, semaphoresToWaitBeforeOverwrite, stagesToWaitForPerSemaphore,
				system::logger_opt_ptr(logger.get()),
				std::chrono::high_resolution_clock::time_point::max());
		}
		propertyTransferCommandBuffer->end();

		video::IGPUQueue::SSubmitInfo submit;
		{
			submit.commandBufferCount = 1u;
			submit.commandBuffers = &propertyTransferCommandBuffer.get();
			submit.signalSemaphoreCount = 0u;
			submit.waitSemaphoreCount = 0u;

			computeQueue->submit(1u, &submit, propertyTransferFence.get());

			logicalDevice->blockForFences(1u, &propertyTransferFence.get());
		}

		constexpr uint32_t CULL_DESCRIPTOR_COUNT = 6u;
		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> cullDsLayout = nullptr;
		{
			video::IGPUDescriptorSetLayout::SBinding bindings[CULL_DESCRIPTOR_COUNT];

			// property pool of lights
			bindings[0].binding = 0u;
			bindings[0].type = asset::EDT_STORAGE_BUFFER;
			bindings[0].count = 1u;
			bindings[0].stageFlags = asset::IShader::ESS_COMPUTE;
			bindings[0].samplers = nullptr;

			// active light indices
			bindings[1].binding = 1u;
			bindings[1].type = asset::EDT_STORAGE_BUFFER;
			bindings[1].count = 1u;
			bindings[1].stageFlags = asset::IShader::ESS_COMPUTE;
			bindings[1].samplers = nullptr;

			// intersection records
			bindings[2].binding = 2u;
			bindings[2].type = asset::EDT_STORAGE_BUFFER;
			bindings[2].count = 1u;
			bindings[2].stageFlags = asset::IShader::ESS_COMPUTE;
			bindings[2].samplers = nullptr;

			// intersection record count (atomic counter)
			bindings[3].binding = 3u;
			bindings[3].type = asset::EDT_STORAGE_BUFFER;
			bindings[3].count = 1u;
			bindings[3].stageFlags = asset::IShader::ESS_COMPUTE;
			bindings[3].samplers = nullptr;

			// light grid 3D image
			bindings[4].binding = 4u;
			bindings[4].type = asset::EDT_STORAGE_IMAGE;
			bindings[4].count = 1u;
			bindings[4].stageFlags = asset::IShader::ESS_COMPUTE;
			bindings[4].samplers = nullptr;

			// Todo(achal): This is optional if user doesn't want me to preserve activeLightIndices
			// after the culling occurs
			// scratchBuffer
			bindings[5].binding = 5u;
			bindings[5].type = asset::EDT_STORAGE_BUFFER;
			bindings[5].count = 1u;
			bindings[5].stageFlags = asset::IShader::ESS_COMPUTE;
			bindings[5].samplers = nullptr;

			cullDsLayout = logicalDevice->createGPUDescriptorSetLayout(bindings, bindings + CULL_DESCRIPTOR_COUNT);
		}
		const uint32_t cullDsCount = 1u;
		core::smart_refctd_ptr<video::IDescriptorPool> cullDescriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &cullDsLayout.get(), &cullDsLayout.get() + 1ull, &cullDsCount);

		asset::SPushConstantRange pcRange = {};
		pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
		pcRange.offset = 0u;
		pcRange.size = 4 * sizeof(float);
		core::smart_refctd_ptr<video::IGPUPipelineLayout> cullPipelineLayout = logicalDevice->createGPUPipelineLayout(&pcRange, &pcRange+1ull, core::smart_refctd_ptr(cullDsLayout));

		// active light indices
		{
			const size_t neededSize = LIGHT_COUNT * sizeof(uint32_t);
			video::IGPUBuffer::SCreationParams creationParams = {};
			creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT);
			activeLightIndicesGPUBuffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, neededSize);

			// Todo(achal): Should this be changeable per frame????
			core::vector<uint32_t> activeLightIndices(LIGHT_COUNT);
			for (uint32_t i = 0u; i < activeLightIndices.size(); ++i)
				activeLightIndices[i] = i;

			asset::SBufferRange<video::IGPUBuffer> bufferRange = {};
			bufferRange.buffer = activeLightIndicesGPUBuffer;
			bufferRange.offset = 0ull;
			bufferRange.size = activeLightIndicesGPUBuffer->getCachedCreationParams().declaredSize;
			utilities->updateBufferRangeViaStagingBuffer(
				queues[CommonAPI::InitOutput::EQT_TRANSFER_UP],
				bufferRange,
				activeLightIndices.data());
		}
		// intersection records
		constexpr uint32_t MAX_INTERSECTION_COUNT = VOXEL_COUNT_PER_LEVEL * LOD_COUNT * LIGHT_COUNT;
		{
			const size_t neededSize = MAX_INTERSECTION_COUNT * sizeof(uint64_t);
			video::IGPUBuffer::SCreationParams creationParams = {};
			creationParams.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
			intersectionRecordsGPUBuffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, neededSize);
		}
		// intersection record count (atomic counter)
		{
			// This will double as a indirect dispatch command 
			const size_t neededSize = 3*sizeof(uint32_t); // or sizeof(VkDispatchIndirectCommand)
			video::IGPUBuffer::SCreationParams creationParams = {};
			creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT | video::IGPUBuffer::EUF_INDIRECT_BUFFER_BIT);
			intersectionRecordCountGPUBuffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, neededSize);

			uint32_t clearValues[3] = { 1u, 1u, 1u};

			asset::SBufferRange<video::IGPUBuffer> bufferRange = {};
			bufferRange.buffer = intersectionRecordCountGPUBuffer;
			bufferRange.offset = 0ull;
			bufferRange.size = intersectionRecordCountGPUBuffer->getCachedCreationParams().declaredSize;
			utilities->updateBufferRangeViaStagingBuffer(
				queues[CommonAPI::InitOutput::EQT_TRANSFER_UP],
				bufferRange,
				clearValues);
		}
		// lightGridTexture2
		{
			// Appending one level grid after another in the Z direction
			const uint32_t width = VOXEL_COUNT_PER_DIM;
			const uint32_t height = VOXEL_COUNT_PER_DIM;
			const uint32_t depth = VOXEL_COUNT_PER_DIM * LOD_COUNT;
			video::IGPUImage::SCreationParams creationParams = {};
			creationParams.type = video::IGPUImage::ET_3D;
			creationParams.format = asset::EF_R32_UINT;
			creationParams.extent = { width, height, depth };
			creationParams.mipLevels = 1u;
			creationParams.arrayLayers = 1u;
			creationParams.samples = asset::IImage::ESCF_1_BIT;
			creationParams.tiling = asset::IImage::ET_OPTIMAL;
			creationParams.usage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_STORAGE_BIT | asset::IImage::EUF_TRANSFER_DST_BIT | asset::IImage::EUF_SAMPLED_BIT);
			creationParams.sharingMode = asset::ESM_EXCLUSIVE;
			creationParams.queueFamilyIndexCount = 0u;
			creationParams.queueFamilyIndices = nullptr;
			creationParams.initialLayout = asset::EIL_UNDEFINED;
			lightGridTexture2 = logicalDevice->createDeviceLocalGPUImageOnDedMem(std::move(creationParams));

			if (!lightGridTexture2)
			{
				logger->log("Failed to create the light grid 3D texture!\n", system::ILogger::ELL_ERROR);
				exit(-1);
			}
		}
		// lightGridTextureView2
		{
			video::IGPUImageView::SCreationParams creationParams = {};
			creationParams.image = lightGridTexture2;
			creationParams.viewType = video::IGPUImageView::ET_3D;
			creationParams.format = asset::EF_R32_UINT;
			creationParams.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			creationParams.subresourceRange.levelCount = 1u;
			creationParams.subresourceRange.layerCount = 1u;

			lightGridTextureView2 = logicalDevice->createGPUImageView(std::move(creationParams));
			if (!lightGridTextureView2)
			{
				logger->log("Failed to create image view for light grid 3D texture!\n", system::ILogger::ELL_ERROR);
				exit(-1);
			}
		}
		// scratchBuffer
		{
			const size_t neededSize = LIGHT_COUNT * sizeof(uint32_t);
			video::IGPUBuffer::SCreationParams creationParams = {};
			creationParams.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
			cullScratchGPUBuffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, neededSize);
		}

		core::smart_refctd_ptr<video::IGPUSpecializedShader> cullCompShader = createShader("../comp.comp");
		cullPipeline = logicalDevice->createGPUComputePipeline(nullptr, std::move(cullPipelineLayout), std::move(cullCompShader));

		cullDs = logicalDevice->createGPUDescriptorSet(cullDescriptorPool.get(), std::move(cullDsLayout));
		{
			video::IGPUDescriptorSet::SWriteDescriptorSet writes[CULL_DESCRIPTOR_COUNT] = {};
			writes[0].dstSet = cullDs.get();
			writes[0].binding = 0u;
			writes[0].arrayElement = 0u;
			writes[0].count = 1u;
			writes[0].descriptorType = asset::EDT_STORAGE_BUFFER;

			writes[1].dstSet = cullDs.get();
			writes[1].binding = 1u;
			writes[1].arrayElement = 0u;
			writes[1].count = 1u;
			writes[1].descriptorType = asset::EDT_STORAGE_BUFFER;

			writes[2].dstSet = cullDs.get();
			writes[2].binding = 2u;
			writes[2].arrayElement = 0u;
			writes[2].count = 1u;
			writes[2].descriptorType = asset::EDT_STORAGE_BUFFER;

			writes[3].dstSet = cullDs.get();
			writes[3].binding = 3u;
			writes[3].arrayElement = 0u;
			writes[3].count = 1u;
			writes[3].descriptorType = asset::EDT_STORAGE_BUFFER;

			writes[4].dstSet = cullDs.get();
			writes[4].binding = 4u;
			writes[4].arrayElement = 0u;
			writes[4].count = 1u;
			writes[4].descriptorType = asset::EDT_STORAGE_IMAGE;

			writes[5].dstSet = cullDs.get();
			writes[5].binding = 5u;
			writes[5].arrayElement = 0u;
			writes[5].count = 1u;
			writes[5].descriptorType = asset::EDT_STORAGE_BUFFER;

			video::IGPUDescriptorSet::SDescriptorInfo infos[CULL_DESCRIPTOR_COUNT] = {};
			const auto& propertyMemoryBlock = propertyPool->getPropertyMemoryBlock(0u);
			infos[0].desc = propertyMemoryBlock.buffer;
			infos[0].buffer.offset = propertyMemoryBlock.offset;
			infos[0].buffer.size = propertyMemoryBlock.size;

			infos[1].desc = activeLightIndicesGPUBuffer;
			infos[1].buffer.offset = 0ull;
			infos[1].buffer.size = activeLightIndicesGPUBuffer->getCachedCreationParams().declaredSize;

			infos[2].desc = intersectionRecordsGPUBuffer;
			infos[2].buffer.offset = 0ull;
			infos[2].buffer.size = intersectionRecordsGPUBuffer->getCachedCreationParams().declaredSize;

			infos[3].desc = intersectionRecordCountGPUBuffer;
			infos[3].buffer.offset = 0ull;
			infos[3].buffer.size = intersectionRecordCountGPUBuffer->getCachedCreationParams().declaredSize;

			infos[4].desc = lightGridTextureView2;
			infos[4].image.imageLayout = asset::EIL_GENERAL;
			infos[4].image.sampler = nullptr;

			infos[5].desc = cullScratchGPUBuffer;
			infos[5].buffer.offset = 0ull;
			infos[5].buffer.size = cullScratchGPUBuffer->getCachedCreationParams().declaredSize;

			writes[0].info = &infos[0];
			writes[1].info = &infos[1];
			writes[2].info = &infos[2];
			writes[3].info = &infos[3];
			writes[4].info = &infos[4];
			writes[5].info = &infos[5];
			logicalDevice->updateDescriptorSets(CULL_DESCRIPTOR_COUNT, writes, 0u, nullptr);
		}

		{
			video::CScanner* scanner = utilities->getDefaultScanner();

			constexpr uint32_t SCAN_DESCRIPTOR_COUNT = 2u;
			video::IGPUDescriptorSetLayout::SBinding binding[SCAN_DESCRIPTOR_COUNT];
			{
				// light grid
				binding[0].binding = 0u;
				binding[0].type = asset::EDT_STORAGE_IMAGE;
				binding[0].count = 1u;
				binding[0].stageFlags = video::IGPUShader::ESS_COMPUTE;
				binding[0].samplers = nullptr;

				// scratch
				binding[1].binding = 1u;
				binding[1].type = asset::EDT_STORAGE_BUFFER;
				binding[1].count = 1u;
				binding[1].stageFlags = video::IGPUShader::ESS_COMPUTE;
				binding[1].samplers = nullptr;
			}
			auto scanDSLayout = logicalDevice->createGPUDescriptorSetLayout(binding, binding + SCAN_DESCRIPTOR_COUNT);

			asset::SPushConstantRange pcRange;
			pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
			pcRange.offset = 0u;
			pcRange.size = sizeof(video::CScanner::DefaultPushConstants);

			auto scanPipelineLayout = logicalDevice->createGPUPipelineLayout(&pcRange, &pcRange+1ull, core::smart_refctd_ptr(scanDSLayout));
			auto shader = createShader("../scan.comp");
			scanPipeline = logicalDevice->createGPUComputePipeline(nullptr, std::move(scanPipelineLayout), std::move(shader));

			const uint32_t setCount = 1u;
			auto scanDSPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &scanDSLayout.get(), &scanDSLayout.get() + 1ull, &setCount);
			scanDS = logicalDevice->createGPUDescriptorSet(scanDSPool.get(), core::smart_refctd_ptr(scanDSLayout));
			scanner->buildParameters(VOXEL_COUNT_PER_LEVEL * LOD_COUNT, scanPushConstants, scanDispatchInfo);
			
			// Todo(achal): This scratch buffer can be "merged" with the one I need for the first
			// compute pass. Required scratch memory could be: max(required_for_scan, required_for_cull)
			// scan scratch buffer
			{
				const size_t neededSize = scanPushConstants.scanParams.getScratchSize();
				video::IGPUBuffer::SCreationParams creationParams = {};
				creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT);
				scanScratchGPUBuffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, neededSize);
			}

			{
				video::IGPUDescriptorSet::SWriteDescriptorSet writes[SCAN_DESCRIPTOR_COUNT];

				// light grid
				writes[0].dstSet = scanDS.get();
				writes[0].binding = 0u;
				writes[0].arrayElement = 0u;
				writes[0].count = 1u;
				writes[0].descriptorType = asset::EDT_STORAGE_IMAGE;

				// scratch
				writes[1].dstSet = scanDS.get();
				writes[1].binding = 1u;
				writes[1].arrayElement = 0u;
				writes[1].count = 1u;
				writes[1].descriptorType = asset::EDT_STORAGE_BUFFER;

				video::IGPUDescriptorSet::SDescriptorInfo infos[SCAN_DESCRIPTOR_COUNT];

				infos[0].image.imageLayout = asset::EIL_GENERAL;
				infos[0].image.sampler = nullptr;
				infos[0].desc = lightGridTextureView2;

				infos[1].buffer.offset = 0ull;
				infos[1].buffer.size = scanScratchGPUBuffer->getCachedCreationParams().declaredSize;
				infos[1].desc = scanScratchGPUBuffer;

				writes[0].info = &infos[0];
				writes[1].info = &infos[1];
				logicalDevice->updateDescriptorSets(SCAN_DESCRIPTOR_COUNT, writes, 0u, nullptr);
			}
		}

		{
			core::smart_refctd_ptr<video::IGPUSpecializedShader> scatterShader = createShader("../scatter.comp");

			constexpr uint32_t SCATTER_DESCRIPTOR_COUNT = 3u;
			core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> scatterDSLayout = nullptr;
			{
				video::IGPUDescriptorSetLayout::SBinding bindings[SCATTER_DESCRIPTOR_COUNT];

				// intersection records
				bindings[0].binding = 0u;
				bindings[0].type = asset::EDT_STORAGE_BUFFER;
				bindings[0].count = 1u;
				bindings[0].stageFlags = asset::IShader::ESS_COMPUTE;
				bindings[0].samplers = nullptr;

				// light index list offsets
				bindings[1].binding = 1u;
				bindings[1].type = asset::EDT_STORAGE_IMAGE;
				bindings[1].count = 1u;
				bindings[1].stageFlags = asset::IShader::ESS_COMPUTE;
				bindings[1].samplers = nullptr;

				// light index list
				bindings[2].binding = 2u;
				bindings[2].type = asset::EDT_STORAGE_BUFFER;
				bindings[2].count = 1u;
				bindings[2].stageFlags = asset::IShader::ESS_COMPUTE;
				bindings[2].samplers = nullptr;

				scatterDSLayout = logicalDevice->createGPUDescriptorSetLayout(bindings, bindings + SCATTER_DESCRIPTOR_COUNT);
			}

			const uint32_t scatterDSCount = 1u;
			core::smart_refctd_ptr<video::IDescriptorPool> scatterDescriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &scatterDSLayout.get(), &scatterDSLayout.get() + 1ull, &scatterDSCount);

			core::smart_refctd_ptr<video::IGPUPipelineLayout> scatterPipelineLayout = logicalDevice->createGPUPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr(scatterDSLayout));

			scatterPipeline = logicalDevice->createGPUComputePipeline(nullptr, std::move(scatterPipelineLayout), std::move(scatterShader));
			scatterDS = logicalDevice->createGPUDescriptorSet(scatterDescriptorPool.get(), std::move(scatterDSLayout));

			// light index list gpu buffer (result of this pass)
			{
				constexpr uint32_t MAX_INTERSECTION_COUNT = VOXEL_COUNT_PER_LEVEL * LOD_COUNT * LIGHT_COUNT;
				const size_t neededSize = MAX_INTERSECTION_COUNT * sizeof(uint32_t);
				video::IGPUBuffer::SCreationParams creationParams = {};
				creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT);
				lightIndexListGPUBuffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, neededSize);
			}
			{
				video::IGPUDescriptorSet::SWriteDescriptorSet writes[SCATTER_DESCRIPTOR_COUNT] = {};
				writes[0].dstSet = scatterDS.get();
				writes[0].binding = 0u;
				writes[0].arrayElement = 0u;
				writes[0].count = 1u;
				writes[0].descriptorType = asset::EDT_STORAGE_BUFFER;

				writes[1].dstSet = scatterDS.get();
				writes[1].binding = 1u;
				writes[1].arrayElement = 0u;
				writes[1].count = 1u;
				writes[1].descriptorType = asset::EDT_STORAGE_IMAGE;

				writes[2].dstSet = scatterDS.get();
				writes[2].binding = 2u;
				writes[2].arrayElement = 0u;
				writes[2].count = 1u;
				writes[2].descriptorType = asset::EDT_STORAGE_BUFFER;

				video::IGPUDescriptorSet::SDescriptorInfo infos[SCATTER_DESCRIPTOR_COUNT] = {};
				infos[0].desc = intersectionRecordsGPUBuffer;
				infos[0].buffer.offset = 0ull;
				infos[0].buffer.size = intersectionRecordsGPUBuffer->getCachedCreationParams().declaredSize;

				infos[1].desc = lightGridTextureView2;
				infos[1].image.imageLayout = asset::EIL_GENERAL;
				infos[1].image.sampler = nullptr;

				infos[2].desc = lightIndexListGPUBuffer;
				infos[2].buffer.offset = 0ull;
				infos[2].buffer.size = lightIndexListGPUBuffer->getCachedCreationParams().declaredSize;

				writes[0].info = &infos[0];
				writes[1].info = &infos[1];
				writes[2].info = &infos[2];
				logicalDevice->updateDescriptorSets(SCATTER_DESCRIPTOR_COUNT, writes, 0u, nullptr);
			}
		}

		core::vectorSIMDf cameraPosition(-157.229813, 369.800446, -19.696722, 0.000000);
		core::vectorSIMDf cameraTarget(-387.548462, 198.927414, -26.500174, 1.000000);

		const float cameraFar = 4000.f;
		matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 2.f, cameraFar);
		camera = Camera(cameraPosition, cameraTarget, projectionMatrix, 12.5f, 2.f);

	// This is more of an octree traversal code than a clipmap
#if 0
		auto processNode = [](auto& liveNodeList, uint32_t x, y, z, level) -> void
		{
			const auto node = buckets[x][y][z][level];
			for (const auto lightID : node.lights)
			{
				for (auto child : node.children) // this step could be done with SIMD/subgroup/workgroup partition
				{
					if (lights[lightID].intersectsAABB(child.getAABB()))
					{
						const auto posInList = child.lights.append(lightID);
						if (posInList == 0u)
							liveNodeList.push(child);
					}
				}
			}
		};

		stack<node_t> liveNodeList[2];
		processNode(liveNodeList[0], 0, 0, 0, 0);
		uint32_t readList = 0u, writeList = 1u;
		for (auto level = 1; level < LevelMax; level++)
		{
			assert(liveNodeList[readList].empty());
			while (!liveNodeList[readList].empty()) // this bit is parallel if you prefix sum light counts across live cells
				processNode(liveNodeList[writeList], liveNodeList[readList].pop());
			std::swap(readList, writeList);
		}
#endif

		lightGridCPUBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(VOXEL_COUNT_PER_LEVEL * LOD_COUNT * sizeof(uint32_t));

		// create the light grid 3D texture
		{
			// Appending one level grid after another in the Z direction
			const uint32_t width = VOXEL_COUNT_PER_DIM;
			const uint32_t height = VOXEL_COUNT_PER_DIM;
			const uint32_t depth = VOXEL_COUNT_PER_DIM * LOD_COUNT;
			video::IGPUImage::SCreationParams creationParams = {};
			creationParams.type = video::IGPUImage::ET_3D;
			creationParams.format = asset::EF_R32_UINT;
			creationParams.extent = { width, height, depth };
			creationParams.mipLevels = 1u;
			creationParams.arrayLayers = 1u;
			creationParams.samples = asset::IImage::ESCF_1_BIT;
			creationParams.tiling = asset::IImage::ET_OPTIMAL;
			creationParams.usage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_SAMPLED_BIT | asset::IImage::EUF_TRANSFER_DST_BIT);
			creationParams.sharingMode = asset::ESM_EXCLUSIVE;
			creationParams.queueFamilyIndexCount = 0u;
			creationParams.queueFamilyIndices = nullptr;
			creationParams.initialLayout = asset::EIL_UNDEFINED;

			lightGridTexture = logicalDevice->createDeviceLocalGPUImageOnDedMem(std::move(creationParams));

			if (!lightGridTexture)
			{
				logger->log("Failed to create the light grid 3d texture!\n", system::ILogger::ELL_ERROR);
				exit(-1);
			}
		}
		
		// lightGridTextureView
		{
			video::IGPUImageView::SCreationParams creationParams = {};
			creationParams.image = lightGridTexture;
			creationParams.viewType = video::IGPUImageView::ET_3D;
			creationParams.format = asset::EF_R32_UINT;
			creationParams.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			creationParams.subresourceRange.levelCount = 1u;
			creationParams.subresourceRange.layerCount = 1u;

			lightGridTextureView = logicalDevice->createGPUImageView(std::move(creationParams));
			if (!lightGridTextureView)
			{
				logger->log("Failed to create image view for light grid 3D texture!\n", system::ILogger::ELL_ERROR);
				exit(-1);
			}
		}
		
		{
			constexpr uint32_t MAX_LIGHT_CLUSTER_INTERSECTION_COUNT = LIGHT_COUNT * VOXEL_COUNT_PER_LEVEL * LOD_COUNT;
			const size_t neededSSBOSize = MAX_LIGHT_CLUSTER_INTERSECTION_COUNT * sizeof(uint32_t);
			video::IGPUBuffer::SCreationParams creationParams = {};
			creationParams.usage = static_cast<asset::IBuffer::E_USAGE_FLAGS>(asset::IBuffer::EUF_UNIFORM_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT);

			lightIndexListUbo = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, neededSSBOSize);
			if (!lightIndexListUbo)
			{
				logger->log("Failed to create SSBO for light index list!\n", system::ILogger::ELL_ERROR);
				exit(-1);
			}

			lightIndexListUboView = logicalDevice->createGPUBufferView(lightIndexListUbo.get(), asset::EF_R32_UINT, 0ull, lightIndexListUbo->getCachedCreationParams().declaredSize);
			if (!lightIndexListUboView)
			{
				logger->log("Failed to create a buffer view for light index list!\n", system::ILogger::ELL_ERROR);
				exit(-1);
			}
		}

		// lightIndexList storage texel buffer
		{
			lightIndexListSSBOView = logicalDevice->createGPUBufferView(lightIndexListGPUBuffer.get(), asset::EF_R32_UINT, 0ull, lightIndexListGPUBuffer->getCachedCreationParams().declaredSize);
		}
		
		// Todo(achal): This should probably need the active light indices as well
		constexpr uint32_t LIGHTING_DESCRIPTOR_COUNT = 3u;
		core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> lightDSLayout_cpu = nullptr;
		{
			core::smart_refctd_ptr<asset::ICPUSampler> cpuSampler = nullptr;
			asset::ICPUDescriptorSetLayout::SBinding binding[LIGHTING_DESCRIPTOR_COUNT];
			{
				{
					asset::ICPUSampler::SParams params = {};
					params.TextureWrapU = asset::ISampler::ETC_CLAMP_TO_EDGE;
					params.TextureWrapV = asset::ISampler::ETC_CLAMP_TO_EDGE;
					params.TextureWrapW = asset::ISampler::ETC_CLAMP_TO_EDGE;
					params.BorderColor = asset::ISampler::ETBC_FLOAT_OPAQUE_BLACK;
					params.MinFilter = asset::ISampler::ETF_NEAREST;
					params.MaxFilter = asset::ISampler::ETF_NEAREST;
					params.MipmapMode = asset::ISampler::ESMM_NEAREST;
					params.AnisotropicFilter = 0u;
					params.CompareEnable = 0u;
					params.CompareFunc = asset::ISampler::ECO_ALWAYS;

					cpuSampler = core::make_smart_refctd_ptr<asset::ICPUSampler>(params);
				}

				// property pool of lights
				binding[0].binding = 0u;
				binding[0].type = asset::EDT_STORAGE_BUFFER;
				binding[0].count = 1u;
				binding[0].stageFlags = asset::IShader::ESS_FRAGMENT;

				// light grid
				binding[1].binding = 1;
				binding[1].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
				binding[1].count = 1u;
				binding[1].stageFlags = asset::IShader::ESS_FRAGMENT;
				binding[1].samplers = &cpuSampler;

				// light index list
				binding[2].binding = 2u;
				binding[2].type = asset::EDT_UNIFORM_TEXEL_BUFFER;
				binding[2].count = 1u;
				binding[2].stageFlags = asset::IShader::ESS_FRAGMENT;
			}

			lightDSLayout_cpu = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(binding, binding + LIGHTING_DESCRIPTOR_COUNT);
			if (!lightDSLayout_cpu)
			{
				logger->log("Failed to create CPU DS Layout for light resources!\n", system::ILogger::ELL_ERROR);
				exit(-1);
			}

			cpu2gpuParams.beginCommandBuffers();
			auto gpuArray = cpu2gpu.getGPUObjectsFromAssets(&lightDSLayout_cpu, &lightDSLayout_cpu + 1, cpu2gpuParams);
			if (!gpuArray || gpuArray->size() < 1u || !(*gpuArray)[0])
			{
				logger->log("Failed to convert Light CPU DS layout to GPU DS layout!\n", system::ILogger::ELL_ERROR);
				exit(-1);
			}
			cpu2gpuParams.waitForCreationToComplete();
			core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> dsLayout_gpu = (*gpuArray)[0];

			const uint32_t setCount = 1u;
			core::smart_refctd_ptr<video::IDescriptorPool> pool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &dsLayout_gpu.get(), &dsLayout_gpu.get() + 1ull, &setCount);
			lightDS = logicalDevice->createGPUDescriptorSet(pool.get(), core::smart_refctd_ptr(dsLayout_gpu));
			if (!lightDS)
			{
				logger->log("Failed to create Light GPU DS!\n", system::ILogger::ELL_ERROR);
				exit(-1);
			}
			video::IGPUDescriptorSet::SWriteDescriptorSet write[LIGHTING_DESCRIPTOR_COUNT];

			// property pool of lights
			write[0].dstSet = lightDS.get();
			write[0].binding = 0u;
			write[0].count = 1u;
			write[0].arrayElement = 0u;
			write[0].descriptorType = asset::EDT_STORAGE_BUFFER;

			// light grid
			write[1].dstSet = lightDS.get();
			write[1].binding = 1u;
			write[1].count = 1u;
			write[1].arrayElement = 0u;
			write[1].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;

			write[2].dstSet = lightDS.get();
			write[2].binding = 2u;
			write[2].count = 1u;
			write[2].arrayElement = 0u;
			write[2].descriptorType = asset::EDT_UNIFORM_TEXEL_BUFFER;

			video::IGPUDescriptorSet::SDescriptorInfo info[LIGHTING_DESCRIPTOR_COUNT];
			{
				const auto& propertyMemoryBlock = propertyPool->getPropertyMemoryBlock(0u);
				info[0].desc = propertyMemoryBlock.buffer;
				info[0].buffer.offset = propertyMemoryBlock.offset;
				info[0].buffer.size = propertyMemoryBlock.size;

				info[1].image.imageLayout = asset::EIL_GENERAL;// asset::EIL_SHADER_READ_ONLY_OPTIMAL;
				info[1].image.sampler = dsLayout_gpu->getBindings()[1].samplers[0];
				info[1].desc = lightGridTextureView2;

				info[2].desc = lightIndexListSSBOView;// lightIndexListUboView;
				info[2].buffer.offset = 0ull;
				info[2].buffer.size = lightIndexListSSBOView->getByteSize();// lightIndexListUboView->getByteSize();
			}
			write[0].info = info;
			write[1].info = info + 1;
			write[2].info = info + 2;
			logicalDevice->updateDescriptorSets(LIGHTING_DESCRIPTOR_COUNT, write, 0u, nullptr);
		}

		const asset::E_FORMAT depthFormat = asset::EF_D32_SFLOAT;
		core::smart_refctd_ptr<video::IGPUImage> depthImages[CommonAPI::InitOutput::MaxSwapChainImageCount] = { nullptr };
		for (uint32_t i = 0u; i < swapchain->getImageCount(); ++i)
		{
			video::IGPUImage::SCreationParams imgParams;
			imgParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
			imgParams.type = asset::IImage::ET_2D;
			imgParams.format = depthFormat;
			imgParams.extent = { WIN_W, WIN_H, 1 };
			imgParams.usage = asset::IImage::E_USAGE_FLAGS::EUF_DEPTH_STENCIL_ATTACHMENT_BIT;
			imgParams.mipLevels = 1u;
			imgParams.arrayLayers = 1u;
			imgParams.samples = asset::IImage::ESCF_1_BIT;
			depthImages[i] = logicalDevice->createDeviceLocalGPUImageOnDedMem(std::move(imgParams));
		}

		for (uint32_t i = 0u; i < swapchain->getImageCount(); ++i)
		{
			constexpr uint32_t fboAttachmentCount = 2u;
			core::smart_refctd_ptr<video::IGPUImageView> views[fboAttachmentCount] = { nullptr };

			auto swapchainImage = swapchain->getImages().begin()[i];
			{
				video::IGPUImageView::SCreationParams viewParams;
				viewParams.format = swapchainImage->getCreationParameters().format;
				viewParams.viewType = asset::IImageView<nbl::video::IGPUImage>::ET_2D;
				viewParams.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
				viewParams.subresourceRange.baseMipLevel = 0u;
				viewParams.subresourceRange.levelCount = 1u;
				viewParams.subresourceRange.baseArrayLayer = 0u;
				viewParams.subresourceRange.layerCount = 1u;
				viewParams.image = std::move(swapchainImage);

				views[0] = logicalDevice->createGPUImageView(std::move(viewParams));
				if (!views[0])
					logger->log("Failed to create swapchain image view %d\n", system::ILogger::ELL_ERROR, i);
			}

			auto depthImage = depthImages[i];
			{
				video::IGPUImageView::SCreationParams viewParams;
				viewParams.format = depthFormat;
				viewParams.viewType = asset::IImageView<nbl::video::IGPUImage>::ET_2D;
				viewParams.subresourceRange.aspectMask = asset::IImage::EAF_DEPTH_BIT;
				viewParams.subresourceRange.baseMipLevel = 0u;
				viewParams.subresourceRange.levelCount = 1u;
				viewParams.subresourceRange.baseArrayLayer = 0u;
				viewParams.subresourceRange.layerCount = 1u;
				viewParams.image = std::move(depthImage);

				views[1] = logicalDevice->createGPUImageView(std::move(viewParams));
				if (!views[1])
					logger->log("Failed to create depth image view %d\n", system::ILogger::ELL_ERROR, i);
			}

			video::IGPUFramebuffer::SCreationParams creationParams = {};
			creationParams.width = WIN_W;
			creationParams.height = WIN_H;
			creationParams.layers = 1u;
			creationParams.renderpass = renderpass;
			creationParams.flags = static_cast<nbl::video::IGPUFramebuffer::E_CREATE_FLAGS>(0);
			creationParams.attachmentCount = fboAttachmentCount;
			creationParams.attachments = views;
			fbos[i] = logicalDevice->createGPUFramebuffer(std::move(creationParams));

			if (!fbos[i])
				logger->log("Failed to create fbo %d\n", system::ILogger::ELL_ERROR, i);
		}

		// Load in the mesh
		const system::path& archiveFilePath = sharedInputCWD / "sponza.zip";
		const system::path& modelFilePath = sharedInputCWD / "sponza.zip/sponza.obj";
		core::smart_refctd_ptr<asset::ICPUMesh> cpuMesh = nullptr;
		const asset::COBJMetadata* metadataOBJ = nullptr;
		{
			auto* quantNormalCache = assetManager->getMeshManipulator()->getQuantNormalCache();
			quantNormalCache->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), sharedOutputCWD / "normalCache101010.sse");

			auto fileArchive = system->openFileArchive(archiveFilePath);
			// test no alias loading (TODO: fix loading from absolute paths)
			system->mount(std::move(fileArchive));

			asset::IAssetLoader::SAssetLoadParams loadParams;
			loadParams.workingDirectory = sharedInputCWD;
			loadParams.logger = logger.get();
			auto meshesBundle = assetManager->getAsset(modelFilePath.string(), loadParams);
			if (meshesBundle.getContents().empty())
			{
				logger->log("Failed to load the model!\n", system::ILogger::ELL_ERROR);
				exit(-1);
			}

			cpuMesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(meshesBundle.getContents().begin()[0]);
			metadataOBJ = meshesBundle.getMetadata()->selfCast<const asset::COBJMetadata>();

			quantNormalCache->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), sharedOutputCWD / "normalCache101010.sse");
		}
		
		// Setup Camera UBO
		{
			// we can safely assume that all meshbuffers within mesh loaded from OBJ has
			// the same DS1 layout (used for camera-specific data), so we can create just one DS
			const asset::ICPUMeshBuffer* firstMeshBuffer = *cpuMesh->getMeshBuffers().begin();

			core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuDSLayout = nullptr;
			{
				auto cpuDSLayout = firstMeshBuffer->getPipeline()->getLayout()->getDescriptorSetLayout(CAMERA_DS_NUMBER);

				cpu2gpuParams.beginCommandBuffers();
				auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&cpuDSLayout, &cpuDSLayout + 1, cpu2gpuParams);
				if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
				{
					logger->log("Failed to convert Camera CPU DS to GPU DS!\n", system::ILogger::ELL_ERROR);
					exit(-1);
				}
				cpu2gpuParams.waitForCreationToComplete();
				gpuDSLayout = (*gpu_array)[0];
			}

			auto dsBindings = gpuDSLayout->getBindings();
			cameraUboBindingNumber = 0u;
			for (const auto& bnd : dsBindings)
			{
				if (bnd.type == asset::EDT_UNIFORM_BUFFER)
				{
					cameraUboBindingNumber = bnd.binding;
					break;
				}
			}

			pipelineMetadata = metadataOBJ->getAssetSpecificMetadata(firstMeshBuffer->getPipeline());

			size_t uboSize = 0ull;
			for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
				if (shdrIn.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set == 1u && shdrIn.descriptorSection.uniformBufferObject.binding == cameraUboBindingNumber)
					uboSize = std::max<size_t>(uboSize, shdrIn.descriptorSection.uniformBufferObject.relByteoffset + shdrIn.descriptorSection.uniformBufferObject.bytesize);

			core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool = createDescriptorPool(1u);

			video::IDriverMemoryBacked::SDriverMemoryRequirements ubomemreq = logicalDevice->getDeviceLocalGPUMemoryReqs();
			ubomemreq.vulkanReqs.size = uboSize;
			video::IGPUBuffer::SCreationParams gpuuboCreationParams;
			gpuuboCreationParams.usage = static_cast<asset::IBuffer::E_USAGE_FLAGS>(asset::IBuffer::EUF_UNIFORM_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT);
			gpuuboCreationParams.sharingMode = asset::E_SHARING_MODE::ESM_EXCLUSIVE;
			gpuuboCreationParams.queueFamilyIndexCount = 0u;
			gpuuboCreationParams.queueFamilyIndices = nullptr;
			cameraUbo = logicalDevice->createGPUBufferOnDedMem(gpuuboCreationParams, ubomemreq);
			cameraDS = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), std::move(gpuDSLayout));

			video::IGPUDescriptorSet::SWriteDescriptorSet write;
			write.dstSet = cameraDS.get();
			write.binding = cameraUboBindingNumber;
			write.count = 1u;
			write.arrayElement = 0u;
			write.descriptorType = asset::EDT_UNIFORM_BUFFER;
			video::IGPUDescriptorSet::SDescriptorInfo info;
			{
				info.desc = cameraUbo;
				info.buffer.offset = 0ull;
				info.buffer.size = uboSize;
			}
			write.info = &info;
			logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
		}

		// Convert mesh to GPU objects
		{
			const char* vertShaderPath = "../vert.vert";
			core::smart_refctd_ptr<asset::ICPUSpecializedShader> vertSpecShader_cpu = nullptr;
			{
				asset::IAssetLoader::SAssetLoadParams params = {};
				params.logger = logger.get();
				vertSpecShader_cpu = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(vertShaderPath, params).getContents().begin());
			}

			const char* fragShaderPath = "../frag.frag";
			core::smart_refctd_ptr<asset::ICPUSpecializedShader> fragSpecShader_cpu = nullptr;
			{
				asset::IAssetLoader::SAssetLoadParams params = {};
				params.logger = logger.get();
				auto loadedShader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(fragShaderPath, params).getContents().begin());
				auto unspecOverridenShader = asset::IGLSLCompiler::createOverridenCopy(loadedShader->getUnspecialized(), "#define LIGHT_COUNT %d\n", LIGHT_COUNT);
				fragSpecShader_cpu = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecOverridenShader), asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));
			}

			for (size_t i = 0ull; i < cpuMesh->getMeshBuffers().size(); ++i)
			{ 
				auto& meshBuffer = cpuMesh->getMeshBuffers().begin()[i];
				
				// Adding the DS layout here is solely for correct creation of pipeline layout
				// it shouldn't have any effect on the actual DS created
				meshBuffer->getPipeline()->getLayout()->setDescriptorSetLayout(LIGHT_DS_NUMBER, core::smart_refctd_ptr(lightDSLayout_cpu));
				meshBuffer->getPipeline()->setShaderAtStage(asset::IShader::ESS_VERTEX, vertSpecShader_cpu.get());
				meshBuffer->getPipeline()->setShaderAtStage(asset::IShader::ESS_FRAGMENT, fragSpecShader_cpu.get());

				// Todo(achal): Can get rid of this probably after
				// https://github.com/Devsh-Graphics-Programming/Nabla/pull/160#discussion_r747185441
				for (size_t i = 0ull; i < nbl::asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
					meshBuffer->getPipeline()->getBlendParams().blendParams[i].attachmentEnabled = (i == 0ull);

				meshBuffer->getPipeline()->getRasterizationParams().frontFaceIsCCW = false;
			}

			cpu2gpuParams.beginCommandBuffers();
			asset::ICPUMesh* meshRaw = static_cast<asset::ICPUMesh*>(cpuMesh.get());
			auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&meshRaw, &meshRaw + 1, cpu2gpuParams);
			if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
			{
				logger->log("Failed to convert mesh to GPU objects!\n", system::ILogger::ELL_ERROR);
				exit(-1);
			}
			cpu2gpuParams.waitForCreationToComplete();
			gpuMesh = (*gpu_array)[0];
		}

		logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_SECONDARY, 1u, &bakedCommandBuffer);
		{
			core::unordered_map<const void*, core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> graphicsPipelineCache;
			core::vector<core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> graphicsPipelines(gpuMesh->getMeshBuffers().size());
			for (size_t i = 0ull; i < graphicsPipelines.size(); ++i)
			{
				const video::IGPURenderpassIndependentPipeline* renderpassIndep = gpuMesh->getMeshBuffers()[i]->getPipeline();
				video::IGPURenderpassIndependentPipeline* renderpassIndep_mutable = const_cast<video::IGPURenderpassIndependentPipeline*>(renderpassIndep);
				auto& rasterizationParams_mutable = const_cast<asset::SRasterizationParams&>(renderpassIndep_mutable->getRasterizationParams());
				rasterizationParams_mutable.depthCompareOp = asset::ECO_GREATER_OR_EQUAL;

				auto foundPpln = graphicsPipelineCache.find(renderpassIndep);
				if (foundPpln == graphicsPipelineCache.end())
				{
					video::IGPUGraphicsPipeline::SCreationParams params;
					params.renderpassIndependent = core::smart_refctd_ptr<const video::IGPURenderpassIndependentPipeline>(renderpassIndep);
					params.renderpass = core::smart_refctd_ptr(renderpass);
					params.subpassIx = LIGHTING_PASS_INDEX;
					foundPpln = graphicsPipelineCache.emplace_hint(foundPpln, renderpassIndep, logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(params)));
				}
				graphicsPipelines[i] = foundPpln->second;
			}
			if (!bakeSecondaryCommandBufferForSubpass(LIGHTING_PASS_INDEX, bakedCommandBuffer.get(), graphicsPipelines))
			{
				logger->log("Failed to create lighting pass command buffer!", system::ILogger::ELL_ERROR);
				exit(-1);
			}
		}

		logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_SECONDARY, 1u, &zPrepassCommandBuffer);
		{
			core::unordered_map<const void*, core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> graphicsPipelineCache;
			core::vector<core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> graphicsPipelines(gpuMesh->getMeshBuffers().size());
			for (size_t i = 0ull; i < graphicsPipelines.size(); ++i)
			{
				const video::IGPURenderpassIndependentPipeline* renderpassIndep = gpuMesh->getMeshBuffers()[i]->getPipeline();
				video::IGPURenderpassIndependentPipeline* renderpassIndep_mutable = const_cast<video::IGPURenderpassIndependentPipeline*>(renderpassIndep);
				auto foundPpln = graphicsPipelineCache.find(renderpassIndep);
				if (foundPpln == graphicsPipelineCache.end())
				{
					// There is no other way for me currently to "disable" a shader stage
					// from an already existing renderpass independent pipeline. This
					// needs to be done for the z pre pass. The problem with doing this
					// in ICPURenderpassIndependentPipeline is that, because of the caching
					// of converted assets that happens in the asset converter even if I set
					// in ICPURenderpassIndependentPipeline just before the conversion I get
					// the old cached value!
					renderpassIndep_mutable->setShaderAtStage(asset::IShader::ESS_FRAGMENT, nullptr);

					video::IGPUGraphicsPipeline::SCreationParams params;
					params.renderpassIndependent = core::smart_refctd_ptr<const video::IGPURenderpassIndependentPipeline>(renderpassIndep);
					params.renderpass = core::smart_refctd_ptr(renderpass);
					params.subpassIx = Z_PREPASS_INDEX;
					foundPpln = graphicsPipelineCache.emplace_hint(foundPpln, renderpassIndep, logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(params)));
				}
				graphicsPipelines[i] = foundPpln->second;
			}
			if (!bakeSecondaryCommandBufferForSubpass(Z_PREPASS_INDEX, zPrepassCommandBuffer.get(), graphicsPipelines))
			{
				logger->log("Failed to create depth pre-pass command buffer!", system::ILogger::ELL_ERROR);
				exit(-1);
			}
		}

		logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			imageAcquire[i] = logicalDevice->createSemaphore();
			renderFinished[i] = logicalDevice->createSemaphore();
		}

		{
			matrix4SIMD invProj;
			if (!projectionMatrix.getInverseTransform(invProj))
			{
				logger->log("Camera projection matrix is not invertible!\n", system::ILogger::ELL_ERROR);
				exit(-1);
			}

			vector4df_SIMD topRight(1.f, 1.f, 1.f, 1.f);
			invProj.transformVect(topRight);
			topRight /= topRight.w;
			clipmapExtent = 2.f * core::sqrt(topRight.x * topRight.x + topRight.y * topRight.y + topRight.z * topRight.z);
			assert(clipmapExtent > 2.f * cameraFar);
		}

		debugCreateLightVolumeGPUResources();
		debugCreateAABBGPUResources();

		oracle.reportBeginFrameRecord();
	}

	void onAppTerminated_impl() override
	{
		logicalDevice->waitIdle();

		const auto& fboCreationParams = fbos[acquiredNextFBO]->getCreationParameters();
		auto gpuSourceImageView = fboCreationParams.attachments[0];

		bool status = ext::ScreenShot::createScreenShot(
			logicalDevice.get(),
			queues[CommonAPI::InitOutput::EQT_TRANSFER_DOWN],
			renderFinished[resourceIx].get(),
			gpuSourceImageView.get(),
			assetManager.get(),
			"ScreenShot.png",
			asset::EIL_PRESENT_SRC,
			static_cast<asset::E_ACCESS_FLAGS>(0u));

		assert(status);
	}

	// Todo(achal): Overflowing stack, move stuff to heap
	void workLoopBody() override
	{
		++resourceIx;
		if (resourceIx >= FRAMES_IN_FLIGHT)
			resourceIx = 0;

		auto& commandBuffer = commandBuffers[resourceIx];
		auto& fence = frameComplete[resourceIx];
		if (fence)
		{
			logicalDevice->blockForFences(1u, &fence.get());
			logicalDevice->resetFences(1u, &fence.get());
		}
		else
			fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		//
		commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
		commandBuffer->begin(0);

		// late latch input
		const auto nextPresentationTimestamp = oracle.acquireNextImage(swapchain.get(), imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);

		// input
		{
			inputSystem->getDefaultMouse(&mouse);
			inputSystem->getDefaultKeyboard(&keyboard);

			camera.beginInputProcessing(nextPresentationTimestamp);
			mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
			keyboard.consumeEvents(
				[&](const IKeyboardEventChannel::range_t& events) -> void
				{
					camera.keyboardProcess(events);

					for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
					{
						auto ev = *eventIt;
						if ((ev.keyCode == ui::EKC_B) && (ev.action == ui::SKeyboardEvent::ECA_RELEASED))
						{
							const core::vectorSIMDf& camPos = camera.getPosition();
							logger->log("Position (world space): (%f, %f, %f, %f)\n",
								system::ILogger::ELL_DEBUG,
								camPos.x, camPos.y, camPos.z, camPos.w);
						}

						if ((ev.keyCode == ui::EKC_1) && (ev.action == ui::SKeyboardEvent::ECA_RELEASED))
						{
							// Todo(achal): Don't know how I'm gonna deal with this when I add
							// the ability to change active lights at runtime

							++debugActiveLightIndex;
							debugActiveLightIndex %= LIGHT_COUNT;
						}

						if ((ev.keyCode == ui::EKC_2) && (ev.action == ui::SKeyboardEvent::ECA_RELEASED))
						{
							--debugActiveLightIndex;
							if (debugActiveLightIndex < 0)
								debugActiveLightIndex += LIGHT_COUNT;
						}

#if 0
						if ((ev.keyCode == ui::EKC_G) && (ev.action == ui::SKeyboardEvent::ECA_RELEASED))
						{
							const uint32_t lightCount = 840u;
							logger->log("Generating %d lights..\n", system::ILogger::ELL_DEBUG, lightCount);

							generateLights(lightCount);

							video::CPropertyPoolHandler::UpStreamingRequest upstreamingRequest;
							upstreamingRequest.setFromPool(propertyPool.get(), 0);
							upstreamingRequest.fill = false; // Don't know what this is used for
							upstreamingRequest.elementCount = lightCount;
							upstreamingRequest.source.data = lights.data();
							upstreamingRequest.source.device2device = false;
							// Don't know what are these for
							upstreamingRequest.srcAddresses = nullptr;
							upstreamingRequest.dstAddresses = nullptr;

							core::smart_refctd_ptr<video::IGPUCommandBuffer> propertyTransferCommandBuffer;
							logicalDevice->createCommandBuffers(
								commandPools[CommonAPI::InitOutput::EQT_COMPUTE].get(),
								video::IGPUCommandBuffer::EL_PRIMARY,
								1u,
								&propertyTransferCommandBuffer);

							auto propertyTransferFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
							auto computeQueue = queues[CommonAPI::InitOutput::EQT_COMPUTE];

							// Todo(achal): This could probably be reused
							asset::SBufferBinding<video::IGPUBuffer> scratchBufferBinding;
							{
								video::IGPUBuffer::SCreationParams scratchParams = {};
								scratchParams.canUpdateSubRange = true;
								scratchParams.usage = core::bitflag(video::IGPUBuffer::EUF_TRANSFER_DST_BIT) | video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
								scratchBufferBinding = { 0ull,logicalDevice->createDeviceLocalGPUBufferOnDedMem(scratchParams,utilities->getDefaultPropertyPoolHandler()->getMaxScratchSize()) };
								scratchBufferBinding.buffer->setObjectDebugName("Scratch Buffer");
							}

							propertyTransferCommandBuffer->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
							{
								// Todo(achal): I should probably wait for FRAGMENT_STAGE (where
								// this light SSBO is used) before overwriting with new data
								uint32_t waitSemaphoreCount = 0u;
								video::IGPUSemaphore* const* semaphoresToWaitBeforeOverwrite = nullptr;
								const asset::E_PIPELINE_STAGE_FLAGS* stagesToWaitForPerSemaphore = nullptr;

								auto* pRequest = &upstreamingRequest;

								utilities->getDefaultPropertyPoolHandler()->transferProperties(
									utilities->getDefaultUpStreamingBuffer(),
									propertyTransferCommandBuffer.get(),
									propertyTransferFence.get(),
									computeQueue,
									scratchBufferBinding,
									pRequest, 1u,
									waitSemaphoreCount, semaphoresToWaitBeforeOverwrite, stagesToWaitForPerSemaphore,
									system::logger_opt_ptr(logger.get()),
									std::chrono::high_resolution_clock::time_point::max());
							}
							propertyTransferCommandBuffer->end();

							video::IGPUQueue::SSubmitInfo submit;
							{
								submit.commandBufferCount = 1u;
								submit.commandBuffers = &propertyTransferCommandBuffer.get();
								submit.signalSemaphoreCount = 0u;
								submit.waitSemaphoreCount = 0u;

								computeQueue->submit(1u, &submit, propertyTransferFence.get());

								// Todo(achal): Can I do better (more fine-grained) sync here?
								logicalDevice->blockForFences(1u, &propertyTransferFence.get());
							}
						}
#endif
					}

				}, logger.get());
			camera.endInputProcessing(nextPresentationTimestamp);
		}

		// update camera
		{
			const auto& viewMatrix = camera.getViewMatrix();
			const auto& viewProjectionMatrix = camera.getConcatenatedMatrix();

			core::vector<uint8_t> uboData(cameraUbo->getSize());
			for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
			{
				if (shdrIn.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set == 1u && shdrIn.descriptorSection.uniformBufferObject.binding == cameraUboBindingNumber)
				{
					switch (shdrIn.type)
					{
					case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_PROJ:
					{
						memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, viewProjectionMatrix.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
					} break;

					case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW:
					{
						memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, viewMatrix.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
					} break;

					case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE:
					{
						core::matrix3x4SIMD invertedTransposedViewMatrix;
						bool invertible = viewMatrix.getSub3x3InverseTranspose(invertedTransposedViewMatrix);
						assert(invertible);
						invertedTransposedViewMatrix.setTranslation(camera.getPosition());

						memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, invertedTransposedViewMatrix.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
					} break;
					}
				}
			}
			commandBuffer->updateBuffer(cameraUbo.get(), 0ull, cameraUbo->getSize(), uboData.data());
		}

		// Todo(achal): I probably don't need to build the clipmaps every frame
		// but only when the camera position changes, but building a clipmap doesn't
		// seem like an expensive operation so I'm gonna leave it as is right now..
		const core::vectorSIMDf& cameraPosition = camera.getPosition();
		nbl_glsl_shapes_AABB_t rootAABB;
		rootAABB.minVx = { cameraPosition.x - (clipmapExtent / 2.f), cameraPosition.y -(clipmapExtent / 2.f), cameraPosition.z - (clipmapExtent / 2.f) };
		rootAABB.maxVx = { cameraPosition.x + (clipmapExtent / 2.f), cameraPosition.y + (clipmapExtent / 2.f), cameraPosition.z + (clipmapExtent / 2.f) };

		// Todo(achal): Probably I don't have to store voxels at all, probably I can generate them procedurally on the fly!
		// ..and probably it doesn't matter much because this isn't a hot path!
		nbl_glsl_shapes_AABB_t clipmap[VOXEL_COUNT_PER_LEVEL * LOD_COUNT];
		buildClipmapForRegion(rootAABB, clipmap);

		// Todo(achal): Find a way to get core::ceil(std::log2(LIGHT_COUNT)) `constexpr`-ly
		constexpr uint32_t LOG2_LIGHT_COUNT = 11u;
		struct intersection_record_t
		{
			// Todo(achal): Would it be better to split this in clusterX, clusterY,
			// clusterZ and mipLevel?
			uint64_t globalClusterID : 24;
			uint64_t localLightID : LOG2_LIGHT_COUNT;
			uint64_t globalLightID : LOG2_LIGHT_COUNT;
		};

		video::IGPUCommandBuffer::SImageMemoryBarrier toGeneral = {};
		toGeneral.barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0u);
		toGeneral.barrier.dstAccessMask = asset::EAF_SHADER_WRITE_BIT;
		toGeneral.oldLayout = asset::EIL_UNDEFINED;
		toGeneral.newLayout = asset::EIL_GENERAL;
		toGeneral.srcQueueFamilyIndex = ~0u;
		toGeneral.dstQueueFamilyIndex = ~0u;
		toGeneral.image = lightGridTexture2;
		toGeneral.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
		toGeneral.subresourceRange.baseMipLevel = 0u;
		toGeneral.subresourceRange.levelCount = 1u;
		toGeneral.subresourceRange.baseArrayLayer = 0u;
		toGeneral.subresourceRange.layerCount = 1u;

		// Todo(achal): This pipeline barrier for light grid is necessary because we have to
		// transition the layout before actually clearing the image BUT this
		// can be done only once outside this loop. Move outside the loop
		commandBuffer->pipelineBarrier(
			asset::EPSF_TOP_OF_PIPE_BIT,
			asset::EPSF_COMPUTE_SHADER_BIT,
			static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
			0u, nullptr,
			0u, nullptr,
			1u, &toGeneral);

		// Todo(achal): I would need a different set of command buffers allocated from a pool which utilizes
		// the compute queue, then I would also need to do queue ownership transfers, most
		// likely with a pipelineBarrier
		commandBuffer->bindComputePipeline(cullPipeline.get());
		float pushConstants[4] = { cameraPosition.x, cameraPosition.y, cameraPosition.z, clipmapExtent };
		commandBuffer->pushConstants(cullPipeline->getLayout(), asset::IShader::ESS_COMPUTE, 0u, 4 * sizeof(float), pushConstants);
		commandBuffer->bindDescriptorSets(asset::EPBP_COMPUTE, cullPipeline->getLayout(), 0u, 1u, &cullDs.get());
		{
			asset::SClearColorValue lightGridClearValue = { 0 };
			asset::IImage::SSubresourceRange range;
			range.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
			range.baseMipLevel = 0u;
			range.levelCount = 1u;
			range.baseArrayLayer = 0u;
			range.layerCount = 1u;
			commandBuffer->clearColorImage(lightGridTexture2.get(), asset::EIL_GENERAL, &lightGridClearValue, 1u, &range);
		}

		// memory dependency to ensure the light grid texture is cleared via clearColorImage
		video::IGPUCommandBuffer::SImageMemoryBarrier lightGridUpdated = {};
		lightGridUpdated.barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
		lightGridUpdated.barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_READ_BIT | asset::EAF_SHADER_WRITE_BIT);
		lightGridUpdated.oldLayout = asset::EIL_GENERAL;
		lightGridUpdated.newLayout = asset::EIL_GENERAL;
		lightGridUpdated.srcQueueFamilyIndex = ~0u;
		lightGridUpdated.dstQueueFamilyIndex = ~0u;
		lightGridUpdated.image = lightGridTexture2;
		lightGridUpdated.subresourceRange.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
		lightGridUpdated.subresourceRange.baseArrayLayer = 0u;
		lightGridUpdated.subresourceRange.layerCount = 1u;
		lightGridUpdated.subresourceRange.baseMipLevel = 0u;
		lightGridUpdated.subresourceRange.levelCount = 1u;
		commandBuffer->pipelineBarrier(
			asset::EPSF_TRANSFER_BIT,
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EDF_NONE,
			0u, nullptr,
			0u, nullptr,
			1u, &lightGridUpdated);
		
		// Todo(achal): Not have weird size workgroups
		commandBuffer->dispatch(2u, 1u, 1u);

		video::CScanner* scanner = utilities->getDefaultScanner();

		commandBuffer->fillBuffer(scanScratchGPUBuffer.get(), 0u, sizeof(uint32_t) + scanScratchGPUBuffer->getCachedCreationParams().declaredSize / 2u, 0u);
		commandBuffer->bindComputePipeline(scanPipeline.get());
		commandBuffer->bindDescriptorSets(asset::EPBP_COMPUTE, scanPipeline->getLayout(), 0u, 1u, &scanDS.get());

		// buffer memory depdency to ensure part of scratch buffer for scan is cleared
		video::IGPUCommandBuffer::SBufferMemoryBarrier scanScratchUpdated = {};
		scanScratchUpdated.barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
		scanScratchUpdated.barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_READ_BIT | asset::EAF_SHADER_WRITE_BIT);
		scanScratchUpdated.srcQueueFamilyIndex = ~0u;
		scanScratchUpdated.dstQueueFamilyIndex = ~0u;
		scanScratchUpdated.buffer = scanScratchGPUBuffer;
		scanScratchUpdated.offset = 0ull;
		scanScratchUpdated.size = scanScratchGPUBuffer->getCachedCreationParams().declaredSize;
		
		// image memory dependency to ensure that the first pass has finished writing to the
		// light grid before the second pass (scan) can read from it, we only need this dependency
		// for the light grid so not using a global memory barrier here
		lightGridUpdated.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
		lightGridUpdated.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
		commandBuffer->pipelineBarrier(
			static_cast<asset::E_PIPELINE_STAGE_FLAGS>(asset::EPSF_COMPUTE_SHADER_BIT | asset::EPSF_TRANSFER_BIT),
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EDF_NONE,
			0u, nullptr,
			1u, &scanScratchUpdated,
			1u, &lightGridUpdated);

		scanner->dispatchHelper(
			commandBuffer.get(),
			scanPipeline->getLayout(),
			scanPushConstants,
			scanDispatchInfo,
			asset::EPSF_TOP_OF_PIPE_BIT,
			0u, nullptr,
			asset::EPSF_BOTTOM_OF_PIPE_BIT,
			0u, nullptr);
		
		// buffer memory dependency to ensure that the first compute dispatch has finished writing
		// all the intersection record data
		video::IGPUCommandBuffer::SBufferMemoryBarrier intersectionDataUpdated[2u] = { };
		{
			// intersection record count
			intersectionDataUpdated[0].barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
			intersectionDataUpdated[0].barrier.dstAccessMask = asset::EAF_INDIRECT_COMMAND_READ_BIT;
			intersectionDataUpdated[0].srcQueueFamilyIndex = ~0u;
			intersectionDataUpdated[0].dstQueueFamilyIndex = ~0u;
			intersectionDataUpdated[0].buffer = intersectionRecordCountGPUBuffer;
			intersectionDataUpdated[0].offset = 0ull;
			intersectionDataUpdated[0].size = intersectionRecordCountGPUBuffer->getCachedCreationParams().declaredSize;

			// intersection records
			intersectionDataUpdated[1].barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
			intersectionDataUpdated[1].barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
			intersectionDataUpdated[1].srcQueueFamilyIndex = ~0u;
			intersectionDataUpdated[1].dstQueueFamilyIndex = ~0u;
			intersectionDataUpdated[1].buffer = intersectionRecordsGPUBuffer;
			intersectionDataUpdated[1].offset = 0ull;
			intersectionDataUpdated[1].size = intersectionRecordsGPUBuffer->getCachedCreationParams().declaredSize;
		}

		// image memory dependency to ensure that scan has finished writing to the light grid
		lightGridUpdated.barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_READ_BIT | asset::EAF_SHADER_WRITE_BIT);
		lightGridUpdated.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;

		commandBuffer->pipelineBarrier(
			asset::EPSF_COMPUTE_SHADER_BIT,
			static_cast<asset::E_PIPELINE_STAGE_FLAGS>(asset::EPSF_COMPUTE_SHADER_BIT | asset::EPSF_DRAW_INDIRECT_BIT),
			asset::EDF_NONE,
			0u, nullptr,
			2u, intersectionDataUpdated,
			1u, &lightGridUpdated);

		commandBuffer->bindComputePipeline(scatterPipeline.get());
		commandBuffer->bindDescriptorSets(asset::EPBP_COMPUTE, scatterPipeline->getLayout(), 0u, 1u, &scatterDS.get());
		commandBuffer->dispatchIndirect(intersectionRecordCountGPUBuffer.get(), 0ull);

		// Todo(achal): Do I need to externally synchronize the end of compute and start
		// of a renderpass???







		auto isNodeInMidRegion = [](const uint32_t nodeIdx) -> bool
		{
			return (nodeIdx == 21u) || (nodeIdx == 22u) || (nodeIdx == 25u) || (nodeIdx == 26u)
				|| (nodeIdx == 37u) || (nodeIdx == 38u) || (nodeIdx == 41u) || (nodeIdx == 42u);
		};

		// will be an input to the system
		// 
		// Todo(achal): Need to have FRAMES_IN_FLIGHT copies of this as well because
		// it can be change per frame
#if 0
		core::vector<uint32_t> activeLightIndices(LIGHT_COUNT);
		for (uint32_t i = 0u; i < activeLightIndices.size(); ++i)
			activeLightIndices[i] = i;

		core::unordered_set<uint32_t> testSets[2];

		uint32_t readSet = 0u, writeSet = 1u;
		testSets[readSet].reserve(LIGHT_COUNT);
		testSets[writeSet].reserve(LIGHT_COUNT);

		for (const uint32_t globalLightID : activeLightIndices)
			testSets[readSet].insert(globalLightID);

		// Todo(achal): Probably should make these class members
		uint32_t lightIndexCount = 0u;
		core::vector<uint32_t> lightIndexList(LIGHT_COUNT* VOXEL_COUNT_PER_LEVEL* LOD_COUNT, ~0u);

		uint32_t intersectionCount = 0u; // max value for clipmap: 640*2^22
		core::vector<intersection_record_t> intersectionRecords(VOXEL_COUNT_PER_LEVEL* LOD_COUNT* LIGHT_COUNT);
		core::vector<uint32_t> lightCounts(VOXEL_COUNT_PER_LEVEL* LOD_COUNT, 0u);

		for (int32_t level = LOD_COUNT - 1; level >= 0; --level)
		{
			// logger->log("Number of lights to test this level: %d\n", system::ILogger::ELL_DEBUG, testSets[readSet].size());

			// record intersections (pass1)
			for (const uint32_t globalLightID : testSets[readSet])
			{
				const cone_t& lightVolume = getLightVolume(lights[globalLightID]);

				for (uint32_t clusterID = 0u; clusterID < VOXEL_COUNT_PER_LEVEL; ++clusterID)
				{
					const uint32_t globalClusterID = (LOD_COUNT - 1 - level) * VOXEL_COUNT_PER_LEVEL + clusterID;

					if (doesConeIntersectAABB(lightVolume, clipmap[globalClusterID]))
					{
						if ((level != 0u) && (isNodeInMidRegion(clusterID)))
						{
							testSets[writeSet].insert(globalLightID);
						}
						else
						{
							intersection_record_t& record = intersectionRecords[intersectionCount++];
							record.globalClusterID = globalClusterID;
							record.localLightID = lightCounts[record.globalClusterID];
							record.globalLightID = globalLightID;

							lightCounts[record.globalClusterID]++;
						}
					}
				}
			}

			std::swap(readSet, writeSet);
			testSets[writeSet].clear(); // clear the slate so you can write to it again
		}
		lightIndexCount += intersectionCount;
#endif

#if 0
		{
			std::ifstream file("C:/Users/pande/Desktop/test.bin", std::ios::ate | std::ios::binary);

			if (!file.is_open())
			{
				throw std::runtime_error("failed to open file!");
			}

			size_t fileSize = (size_t)file.tellg();
			std::vector<char> buffer(fileSize);

			file.seekg(0);
			file.read(buffer.data(), fileSize);

			const uint32_t gpuIntersectionCount = 2u * 5352u;
			uint32_t* data = reinterpret_cast<uint32_t*>(buffer.data());
			int32_t lightCount = 204;
			for (uint32_t i = 0u; i < gpuIntersectionCount; ++i)
			{
				const uint32_t x = *data++;
				const uint32_t y = *data++;

				const uint32_t localClusterIDx = x&0x7F;
				const uint32_t localClusterIDy = (x>>7)&0x7F;
				const uint32_t localClusterIDz = (x>>14)&0x7F;
				const uint32_t level = (x>>21)&0x7F;
					
				const uint32_t localLightIndex = y & 0xFFF;
				const uint32_t globalLightIndex = (y >> 12) & 0xFFFFF;

				const uint32_t globalClusterIndex = (LOD_COUNT - 1 - level) * VOXEL_COUNT_PER_LEVEL + localClusterIDz * 16 + localClusterIDy * 4 + localClusterIDx;
				// logger->log("globalClusterIndex: %d\n", system::ILogger::ELL_DEBUG, globalClusterIndex);


				if (globalClusterIndex == 197)
				{
					logger->log("localLightIndex: %d\n", system::ILogger::ELL_DEBUG, localLightIndex);
					--lightCount;
				}

#if 0
				logger->log(
					"Unpacked intersection record:\n"
					"localClusterID: (%d, %d, %d)\n"
					"level: %d\n"
					"localLightIndex: %d\n"
					"globalLightIndex: %d\n",
					system::ILogger::ELL_DEBUG,
					localClusterIDx, localClusterIDy, localClusterIDz,
					level,
					localLightIndex,
					globalLightIndex);
				__debugbreak();
#endif

			}
			if (lightCount == 0)
			{
				logger->log("Everything seems fine!\n");
				__debugbreak();
			}
			logger->log("lightCount: %d\n", system::ILogger::ELL_DEBUG, lightCount);
			__debugbreak();

			file.close();
		}
#endif

#if 0
		// combined prefix sum at the end
		uint32_t lightIndexListOffsets[VOXEL_COUNT_PER_LEVEL * LOD_COUNT] = { 0u };
		for (uint32_t i = 0u; i < VOXEL_COUNT_PER_LEVEL * LOD_COUNT; ++i)
			lightIndexListOffsets[i] = (i == 0) ? 0u : lightIndexListOffsets[i - 1] + lightCounts[i - 1];

		// scatter
		for (uint32_t i = 0u; i < intersectionCount; ++i)
		{
			const intersection_record_t& record = intersectionRecords[i];

			const uint32_t baseAddress = lightIndexListOffsets[record.globalClusterID];
			const uint32_t scatterAddress = baseAddress + record.localLightID;
			lightIndexList[scatterAddress] = record.globalLightID;
		}

		// set the light grid buffer values
		uint32_t* lightGridEntry = static_cast<uint32_t*>(lightGridCPUBuffer->getPointer());
		for (int32_t level = LOD_COUNT - 1; level >= 0; --level)
		{
			for (uint32_t clusterID = 0u; clusterID < VOXEL_COUNT_PER_LEVEL; ++clusterID)
			{
				const uint32_t globalClusterID = (LOD_COUNT - 1 - level) * VOXEL_COUNT_PER_LEVEL + clusterID;
				const uint32_t offset = lightIndexListOffsets[globalClusterID];
				const uint32_t count = lightCounts[globalClusterID];
				// Todo(achal): Convert this into an assert
				if (offset >= 65536 || count >= 65536)
					__debugbreak();

				*lightGridEntry++ = (count << 16) | offset;
			}
		}

		// upload light grid cpu buffer to the light grid 3d texture
		asset::IImage::SBufferCopy region = {};
		region.bufferOffset = 0ull;
		region.bufferRowLength = 0u; // tightly packed according to image extent
		region.bufferImageHeight = 0u;
		region.imageSubresource.aspectMask = asset::IImage::EAF_COLOR_BIT;
		region.imageSubresource.baseArrayLayer = 0u;
		region.imageSubresource.layerCount = 1u;
		region.imageSubresource.mipLevel = 0u;
		region.imageOffset = { 0u, 0u, 0u };
		region.imageExtent = { VOXEL_COUNT_PER_DIM, VOXEL_COUNT_PER_DIM, VOXEL_COUNT_PER_DIM * LOD_COUNT };

		
		{
			video::IGPUQueue* queue = queues[CommonAPI::InitOutput::EQT_GRAPHICS];
			core::smart_refctd_ptr<video::IGPUFence> updateLightGridFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

			core::smart_refctd_ptr<video::IGPUCommandPool> pool = logicalDevice->createCommandPool(queue->getFamilyIndex(), video::IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
			core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf = nullptr;
			logicalDevice->createCommandBuffers(pool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);
			assert(cmdbuf);
			cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

			video::IGPUCommandBuffer::SImageMemoryBarrier toTransferDst = {};
			toTransferDst.barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0u);
			toTransferDst.barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
			toTransferDst.oldLayout = asset::EIL_UNDEFINED;
			toTransferDst.newLayout = asset::EIL_TRANSFER_DST_OPTIMAL;
			toTransferDst.srcQueueFamilyIndex = ~0u;
			toTransferDst.dstQueueFamilyIndex = ~0u;
			toTransferDst.image = lightGridTexture;
			toTransferDst.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			toTransferDst.subresourceRange.baseMipLevel = 0u;
			toTransferDst.subresourceRange.levelCount = 1u;
			toTransferDst.subresourceRange.baseArrayLayer = 0u;
			toTransferDst.subresourceRange.layerCount = 1u;

			cmdbuf->pipelineBarrier(
				asset::EPSF_TOP_OF_PIPE_BIT,
				asset::EPSF_TRANSFER_BIT,
				static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
				0u, nullptr,
				0u, nullptr,
				1u, &toTransferDst);

			uint32_t waitSemaphoreCount = 0u;
			video::IGPUSemaphore* const* semaphoresToWaitBeforeOverwrite = nullptr;
			const asset::E_PIPELINE_STAGE_FLAGS* stagesToWaitForPerSemaphore = nullptr;
			// Todo(achal): Handle nodiscard
			utilities->updateImageViaStagingBuffer(
				cmdbuf.get(),
				updateLightGridFence.get(),
				queue,
				lightGridCPUBuffer.get(),
				{&region, &region+1},
				lightGridTexture.get(),
				asset::EIL_TRANSFER_DST_OPTIMAL,
				waitSemaphoreCount, semaphoresToWaitBeforeOverwrite, stagesToWaitForPerSemaphore);

			video::IGPUCommandBuffer::SImageMemoryBarrier toShaderReadOnlyOptimal = {};
			toShaderReadOnlyOptimal.barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
			toShaderReadOnlyOptimal.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
			toShaderReadOnlyOptimal.oldLayout = asset::EIL_TRANSFER_DST_OPTIMAL;
			toShaderReadOnlyOptimal.newLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;
			toShaderReadOnlyOptimal.srcQueueFamilyIndex = ~0u;
			toShaderReadOnlyOptimal.dstQueueFamilyIndex = ~0u;
			toShaderReadOnlyOptimal.image = lightGridTexture;
			toShaderReadOnlyOptimal.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			toShaderReadOnlyOptimal.subresourceRange.baseMipLevel = 0u;
			toShaderReadOnlyOptimal.subresourceRange.levelCount = 1u;
			toShaderReadOnlyOptimal.subresourceRange.baseArrayLayer = 0u;
			toShaderReadOnlyOptimal.subresourceRange.layerCount = 1u;

			cmdbuf->pipelineBarrier(
				asset::EPSF_TRANSFER_BIT,
				asset::EPSF_FRAGMENT_SHADER_BIT,
				static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
				0u, nullptr,
				0u, nullptr,
				1u, &toShaderReadOnlyOptimal);

			cmdbuf->end();

			video::IGPUQueue::SSubmitInfo submit = {};
			submit.commandBufferCount = 1u;
			submit.commandBuffers = &cmdbuf.get();
			queue->submit(1u, &submit, updateLightGridFence.get());

			logicalDevice->blockForFences(1u, &updateLightGridFence.get());
		}

		// upload/update lightIndexList
		// Todo(achal): Better sync?
		{
			asset::SBufferRange<video::IGPUBuffer> bufferRange;
			bufferRange.offset = 0ull;
			bufferRange.size = lightIndexCount * sizeof(uint32_t);
			bufferRange.buffer = lightIndexListUbo;
			// Todo(achal): Handle return value (nodiscard)
			utilities->updateBufferRangeViaStagingBuffer(
				queues[CommonAPI::InitOutput::EQT_GRAPHICS],
				bufferRange,
				lightIndexList.data());
		}
		
		// Todo(achal): It might not be very efficient to create an unordred_map (allocate memory) every frame,
		// would reusing them be better? This is debug code anyway..
		core::unordered_map<uint32_t, core::unordered_set<uint32_t>> debugDrawLightIDToClustersMap;
		debugRecordLightIDToClustersMap(debugDrawLightIDToClustersMap, lightIndexList);

		// Todo(achal): debugClustersForLight might go out of scope and die before the command
		// buffer gets the chance to execute and hence actually transfer the data from CPU to GPU,
		// I think I can use the property pool to handle this stuff
		core::vector<nbl_glsl_shapes_AABB_t> debugClustersForLight;
		debugUpdateLightClusterAssignment(commandBuffer.get(), debugActiveLightIndex, debugClustersForLight, debugDrawLightIDToClustersMap, clipmap);
#endif
		
		// renderpass
		{
			asset::SViewport viewport;
			viewport.minDepth = 1.f;
			viewport.maxDepth = 0.f;
			viewport.x = 0u;
			viewport.y = 0u;
			viewport.width = WIN_W;
			viewport.height = WIN_H;
			commandBuffer->setViewport(0u, 1u, &viewport);

			VkRect2D scissor = { {0, 0}, {WIN_W, WIN_H} };
			commandBuffer->setScissor(0u, 1u, &scissor);

			video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
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

			commandBuffer->beginRenderPass(&beginInfo, asset::ESC_SECONDARY_COMMAND_BUFFERS);
			commandBuffer->executeCommands(1u, &zPrepassCommandBuffer.get());
			commandBuffer->nextSubpass(asset::ESC_SECONDARY_COMMAND_BUFFERS);
			commandBuffer->executeCommands(1u, &bakedCommandBuffer.get());
			commandBuffer->nextSubpass(asset::ESC_INLINE);
			debugDrawLightClusterAssignment(commandBuffer.get(), debugActiveLightIndex, debugClustersForLight);
			commandBuffer->endRenderPass();
			commandBuffer->end();
		}

		CommonAPI::Submit(
			logicalDevice.get(),
			swapchain.get(),
			commandBuffer.get(),
			queues[CommonAPI::InitOutput::EQT_GRAPHICS],
			imageAcquire[resourceIx].get(),
			renderFinished[resourceIx].get(),
			fence.get());

		CommonAPI::Present(
			logicalDevice.get(),
			swapchain.get(),
			queues[CommonAPI::InitOutput::EQT_GRAPHICS],
			renderFinished[resourceIx].get(),
			acquiredNextFBO);
	}

	bool keepRunning() override
	{
		return windowCb->isWindowOpen();
	}

private:
	// Returns true if the point lies in negative-half-space (space in which the normal to the plane isn't present) of the plane
	// Point on the plane returns true.
	// In other words, if you hold the plane such that its normal points towards your face, then this
	// will return true if the point is "behind" the plane (or farther from your face than the plane's surface)
	bool isPointBehindPlane(const core::vector3df_SIMD& point, const core::plane3dSIMDf& plane)
	{
		// As an optimization we can add an epsilon to 0, to ignore cones which have a
		// very very small intersecting region with the AABB, could help with FP precision
		// too when the point is on the plane
		return (core::dot(point, plane.getNormal()).x + plane.getDistance()) <= 0.f /* + EPSILON*/;
	};

	// Imagine yourself to be at the center of the bounding box. Using CCW winding order
	// in RH ensures that the normals to the box's planes point towards you.
	bool doesConeIntersectAABB(const cone_t& cone, const nbl_glsl_shapes_AABB_t& aabb)
	{
		constexpr uint32_t PLANE_COUNT = 6u;
		core::plane3dSIMDf planes[PLANE_COUNT];
		{
			// 0 -> (minVx.x, minVx.y, minVx.z)
			// 1 -> (maxVx.x, minVx.y, minVx.z)
			// 2 -> (minVx.x, maxVx.y, minVx.z)
			// 3 -> (maxVx.x, maxVx.y, minVx.z)
			// 4 -> (minVx.x, minVx.y, maxVx.z)
			// 5 -> (maxVx.x, minVx.y, maxVx.z)
			// 6 -> (minVx.x, maxVx.y, maxVx.z)
			// 7 -> (maxVx.x, maxVx.y, maxVx.z)

			planes[0].setPlane(
				core::vector3df_SIMD(aabb.maxVx.x, aabb.minVx.y, aabb.minVx.z),
				core::vector3df_SIMD(aabb.maxVx.x, aabb.minVx.y, aabb.maxVx.z),
				core::vector3df_SIMD(aabb.maxVx.x, aabb.maxVx.y, aabb.maxVx.z)); // 157
			planes[1].setPlane(
				core::vector3df_SIMD(aabb.minVx.x, aabb.minVx.y, aabb.minVx.z),
				core::vector3df_SIMD(aabb.maxVx.x, aabb.minVx.y, aabb.minVx.z),
				core::vector3df_SIMD(aabb.maxVx.x, aabb.maxVx.y, aabb.minVx.z)); // 013
			planes[2].setPlane(
				core::vector3df_SIMD(aabb.minVx.x, aabb.minVx.y, aabb.maxVx.z),
				core::vector3df_SIMD(aabb.minVx.x, aabb.minVx.y, aabb.minVx.z),
				core::vector3df_SIMD(aabb.minVx.x, aabb.maxVx.y, aabb.minVx.z)); // 402
			planes[3].setPlane(
				core::vector3df_SIMD(aabb.maxVx.x, aabb.minVx.y, aabb.maxVx.z),
				core::vector3df_SIMD(aabb.minVx.x, aabb.minVx.y, aabb.maxVx.z),
				core::vector3df_SIMD(aabb.minVx.x, aabb.maxVx.y, aabb.maxVx.z)); // 546
			planes[4].setPlane(
				core::vector3df_SIMD(aabb.minVx.x, aabb.minVx.y, aabb.maxVx.z),
				core::vector3df_SIMD(aabb.maxVx.x, aabb.minVx.y, aabb.maxVx.z),
				core::vector3df_SIMD(aabb.maxVx.x, aabb.minVx.y, aabb.minVx.z)); // 451
			planes[5].setPlane(
				core::vector3df_SIMD(aabb.maxVx.x, aabb.maxVx.y, aabb.maxVx.z),
				core::vector3df_SIMD(aabb.minVx.x, aabb.maxVx.y, aabb.maxVx.z),
				core::vector3df_SIMD(aabb.minVx.x, aabb.maxVx.y, aabb.minVx.z)); // 762
		}

		// Todo(achal): Cannot handle half angle > 90 degrees right now
		assert(cone.cosHalfAngle > 0.f);
		const float tanOuterHalfAngle = core::sqrt(core::max(1.f - (cone.cosHalfAngle * cone.cosHalfAngle), 0.f)) / cone.cosHalfAngle;
		const float coneRadius = cone.height * tanOuterHalfAngle;

		for (uint32_t i = 0u; i < PLANE_COUNT; ++i)
		{
			// Calling setPlane above ensures normalized normal

			const core::vectorSIMDf m = core::cross(core::cross(planes[i].getNormal(), cone.direction), cone.direction);
			const core::vectorSIMDf farthestBasePoint = cone.tip + (cone.direction * cone.height) - (m * coneRadius); // farthest to plane's surface away from positive half-space

			// There are two edge cases here:
			// 1. When cone's direction and plane's normal are anti-parallel
			//		There is no reason to check farthestBasePoint in this case, because cone's tip is the farthest point!
			//		But there is no harm in doing so.
			// 2. When cone's direction and plane's normal are parallel
			//		This edge case will get handled nicely by the farthestBasePoint coming as center of the base of the cone itself
			if (isPointBehindPlane(cone.tip, planes[i]) && isPointBehindPlane(farthestBasePoint, planes[i]))
				return false;
		}

		return true;
	};

	// Todo(achal): Get this working for point lights(spheres) as an edge case of spot lights
	cone_t getLightVolume(const nbl_glsl_ext_ClusteredLighting_SpotLight& light)
	{
		cone_t cone = {};
		cone.tip = core::vectorSIMDf(light.position.x, light.position.y, light.position.z);

		// get cone's height based on its contribution to the scene (intensity and attenuation)
		{
			core::vectorSIMDf intensity;
			{
				uint64_t lsb = light.intensity.x;
				uint64_t msb = light.intensity.y;
				const uint64_t packedIntensity = (msb << 32) | lsb;
				const core::rgb32f rgb = core::rgb19e7_to_rgb32f(packedIntensity);
				intensity = core::vectorSIMDf(rgb.x, rgb.y, rgb.z);
			}

			const float radiusSq = LIGHT_RADIUS * LIGHT_RADIUS;
			// Taking max intensity among all the RGB components will give the largest
			// light volume enclosing light volumes due to other components
			const float maxIntensityComponent = core::max(core::max(intensity.r, intensity.g), intensity.b);
			const float determinant = 1.f - ((2.f * LIGHT_CONTRIBUTION_THRESHOLD) / (maxIntensityComponent * radiusSq));
			if ((determinant > 1.f) || (determinant < -1.f))
			{
				logger->log("This should never happen!\n", system::ILogger::ELL_ERROR);
				exit(-1);
			}

			cone.height = LIGHT_RADIUS * core::inversesqrt((1.f / (determinant * determinant)) - 1.f);
		}

		// get cone's direction and half angle (outer half angle of spot light)
		{
			const core::vectorSIMDf directionXY = unpackSnorm2x16(light.direction.x);
			const core::vectorSIMDf directionZW = unpackSnorm2x16(light.direction.y);

			cone.direction = core::vectorSIMDf(directionXY.x, directionXY.y, directionZW.x);

			const float cosineRange = directionZW.y;
			// Todo(achal): I cannot handle spotlights/cone intersection against AABB
			// if it has outerHalfAngle > 90.f
			cone.cosHalfAngle = light.outerCosineOverCosineRange * cosineRange;
			assert(std::acosf(cone.cosHalfAngle) <= core::radians(90.f));
		}

		return cone;
	};

	bool bakeSecondaryCommandBufferForSubpass(const uint32_t subpass, video::IGPUCommandBuffer* cmdbuf, const core::vector<core::smart_refctd_ptr<video::IGPUGraphicsPipeline>>& graphicsPipelines)
	{
		assert(gpuMesh->getMeshBuffers().size() == graphicsPipelines.size());

		video::IGPUCommandBuffer::SInheritanceInfo inheritanceInfo = {};
		inheritanceInfo.renderpass = renderpass;
		inheritanceInfo.subpass = subpass;
		// inheritanceInfo.framebuffer = ; // might be good to have it

		cmdbuf->begin(video::IGPUCommandBuffer::EU_RENDER_PASS_CONTINUE_BIT | video::IGPUCommandBuffer::EU_SIMULTANEOUS_USE_BIT, &inheritanceInfo);

		asset::SViewport viewport;
		viewport.minDepth = 1.f;
		viewport.maxDepth = 0.f;
		viewport.x = 0u;
		viewport.y = 0u;
		viewport.width = WIN_W;
		viewport.height = WIN_H;
		// Todo(achal): Probably move this out to its primary command buffer
		cmdbuf->setViewport(0u, 1u, &viewport);

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = { WIN_W, WIN_H };
		cmdbuf->setScissor(0u, 1u, &scissor);

		{
			const uint32_t drawCallCount = gpuMesh->getMeshBuffers().size();

			core::smart_refctd_ptr<video::CDrawIndirectAllocator<>> drawAllocator;
			{
				video::IDrawIndirectAllocator::ImplicitBufferCreationParameters params;
				params.device = logicalDevice.get();
				params.maxDrawCommandStride = sizeof(asset::DrawElementsIndirectCommand_t);
				params.drawCommandCapacity = drawCallCount;
				params.drawCountCapacity = 0u;
				drawAllocator = video::CDrawIndirectAllocator<>::create(std::move(params));
			}

			video::IDrawIndirectAllocator::Allocation allocation;
			{
				allocation.count = drawCallCount;
				{
					allocation.multiDrawCommandRangeByteOffsets = new uint32_t[allocation.count];
					// you absolutely must do this
					std::fill_n(allocation.multiDrawCommandRangeByteOffsets, allocation.count, video::IDrawIndirectAllocator::invalid_draw_range_begin);
				}
				{
					auto drawCounts = new uint32_t[allocation.count];
					std::fill_n(drawCounts, allocation.count, 1u);
					allocation.multiDrawCommandMaxCounts = drawCounts;
				}
				allocation.setAllCommandStructSizesConstant(sizeof(asset::DrawElementsIndirectCommand_t));
				drawAllocator->allocateMultiDraws(allocation);
				delete[] allocation.multiDrawCommandMaxCounts;
			}

			video::CSubpassKiln subpassKiln;

			auto drawCallData = new asset::DrawElementsIndirectCommand_t[drawCallCount];
			{
				auto drawIndexIt = allocation.multiDrawCommandRangeByteOffsets;
				auto drawCallDataIt = drawCallData;

				for (size_t i = 0ull; i < gpuMesh->getMeshBuffers().size(); ++i)
				{
					const auto& mb = gpuMesh->getMeshBuffers().begin()[i];
					auto& drawcall = subpassKiln.getDrawcallMetadataVector().emplace_back();

					// push constants
					memcpy(drawcall.pushConstantData, mb->getPushConstantsDataPtr(), video::IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE);
					// graphics pipeline
					drawcall.pipeline = graphicsPipelines[i];
					// descriptor sets
					drawcall.descriptorSets[1] = cameraDS;
					drawcall.descriptorSets[2] = lightDS;
					drawcall.descriptorSets[3] = core::smart_refctd_ptr<const video::IGPUDescriptorSet>(mb->getAttachedDescriptorSet());
					// vertex buffers
					std::copy_n(mb->getVertexBufferBindings(), video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT, drawcall.vertexBufferBindings);
					// index buffer
					drawcall.indexBufferBinding = mb->getIndexBufferBinding().buffer;
					drawcall.drawCommandStride = sizeof(asset::DrawElementsIndirectCommand_t);
					drawcall.indexType = mb->getIndexType();
					//drawcall.drawCountOffset // leave as invalid
					drawcall.drawCallOffset = *(drawIndexIt++);
					drawcall.drawMaxCount = 1u;

					// TODO: in the far future, just make IMeshBuffer hold a union of `DrawArraysIndirectCommand_t` `DrawElementsIndirectCommand_t`
					drawCallDataIt->count = mb->getIndexCount();
					drawCallDataIt->instanceCount = mb->getInstanceCount();
					switch (drawcall.indexType)
					{
					case asset::EIT_32BIT:
						drawCallDataIt->firstIndex = mb->getIndexBufferBinding().offset / sizeof(uint32_t);
						break;
					case asset::EIT_16BIT:
						drawCallDataIt->firstIndex = mb->getIndexBufferBinding().offset / sizeof(uint16_t);
						break;
					default:
						assert(false);
						break;
					}
					drawCallDataIt->baseVertex = mb->getBaseVertex();
					drawCallDataIt->baseInstance = mb->getBaseInstance();

					drawCallDataIt++;
				}
			}

			// do the transfer of drawcall structs
			{
				video::CPropertyPoolHandler::UpStreamingRequest request;
				request.destination = drawAllocator->getDrawCommandMemoryBlock();
				request.fill = false;
				request.elementSize = sizeof(asset::DrawElementsIndirectCommand_t);
				request.elementCount = drawCallCount;
				request.source.device2device = false;
				request.source.data = drawCallData;
				request.srcAddresses = nullptr; // iota 0,1,2,3,4,etc.
				request.dstAddresses = allocation.multiDrawCommandRangeByteOffsets;
				std::for_each_n(allocation.multiDrawCommandRangeByteOffsets, request.elementCount, [&](auto& handle) {handle /= request.elementSize; });

				auto upQueue = queues[CommonAPI::InitOutput::EQT_TRANSFER_UP];
				auto fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);
				core::smart_refctd_ptr<video::IGPUCommandBuffer> tferCmdBuf;
				logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_TRANSFER_UP].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &tferCmdBuf);

				tferCmdBuf->begin(0u); // TODO some one time submit bit or something
				{
					auto* ppHandler = utilities->getDefaultPropertyPoolHandler();
					// if we did multiple transfers, we'd reuse the scratch
					asset::SBufferBinding<video::IGPUBuffer> scratch;
					{
						video::IGPUBuffer::SCreationParams scratchParams = {};
						scratchParams.canUpdateSubRange = true;
						scratchParams.usage = core::bitflag(video::IGPUBuffer::EUF_TRANSFER_DST_BIT) | video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
						scratch = { 0ull,logicalDevice->createDeviceLocalGPUBufferOnDedMem(scratchParams,ppHandler->getMaxScratchSize()) };
						scratch.buffer->setObjectDebugName("Scratch Buffer");
					}
					auto* pRequest = &request;
					uint32_t waitSemaphoreCount = 0u;
					video::IGPUSemaphore* const* waitSemaphores = nullptr;
					const asset::E_PIPELINE_STAGE_FLAGS* waitStages = nullptr;
					if (ppHandler->transferProperties(
						utilities->getDefaultUpStreamingBuffer(), tferCmdBuf.get(), fence.get(), upQueue,
						scratch, pRequest, 1u, waitSemaphoreCount, waitSemaphores, waitStages,
						logger.get(), std::chrono::high_resolution_clock::time_point::max() // wait forever if necessary, need initialization to finish
					))
						return false;
				}
				tferCmdBuf->end();
				{
					video::IGPUQueue::SSubmitInfo submit = {}; // intializes all semaphore stuff to 0 and nullptr
					submit.commandBufferCount = 1u;
					submit.commandBuffers = &tferCmdBuf.get();
					upQueue->submit(1u, &submit, fence.get());
				}
				logicalDevice->blockForFences(1u, &fence.get());
			}
			delete[] drawCallData;
			// free the draw command index list
			delete[] allocation.multiDrawCommandRangeByteOffsets;

			subpassKiln.bake(cmdbuf, renderpass.get(), subpass, drawAllocator->getDrawCommandMemoryBlock().buffer.get(), nullptr);
		}
		cmdbuf->end();

		return true;
	}

	inline void logAABB(const nbl_glsl_shapes_AABB_t& aabb)
	{
		logger->log("\nMin Point: [%f, %f, %f]\nMax Point: [%f, %f %f]\n",
			system::ILogger::ELL_DEBUG,
			aabb.minVx.x, aabb.minVx.y, aabb.minVx.z,
			aabb.maxVx.x, aabb.maxVx.y, aabb.maxVx.z);
	}

	// As per: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/packUnorm.xhtml
	inline uint32_t packSnorm2x16(const float x, const float y)
	{
		uint32_t x_snorm16 = (uint32_t)((uint16_t)core::round(core::clamp(x, -1.f, 1.f) * 32767.f));
		uint32_t y_snorm16 = (uint32_t)((uint16_t)core::round(core::clamp(y, -1.f, 1.f) * 32767.f));
		return ((y_snorm16 << 16) | x_snorm16);
	};

	// As per: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/unpackUnorm.xhtml
	inline core::vectorSIMDf unpackSnorm2x16(uint32_t encoded)
	{
		core::vectorSIMDf result;

		int16_t firstComp(encoded & 0xFFFFu);
		int16_t secondComp((encoded>>16)&0xFFFFu);
		result.x = core::clamp(firstComp / 32727.f, -1.f, 1.f);
		result.y = core::clamp(secondComp / 32727.f, -1.f, 1.f);

		return result;
	};

	void buildClipmapForRegion(const nbl_glsl_shapes_AABB_t& rootRegion, nbl_glsl_shapes_AABB_t* outClipmap)
	{
		nbl_glsl_shapes_AABB_t aabbRegion = rootRegion;
		vec3_aligned center = { (aabbRegion.minVx.x + aabbRegion.maxVx.x) / 2.f, (aabbRegion.minVx.y + aabbRegion.maxVx.y) / 2.f, (aabbRegion.minVx.z + aabbRegion.maxVx.z) / 2.f };
		for (int32_t level = LOD_COUNT - 1; level >= 0; --level)
		{
			nbl_glsl_shapes_AABB_t* begin = outClipmap + ((LOD_COUNT - 1ull - level) * VOXEL_COUNT_PER_LEVEL);
			voxelizeRegion(aabbRegion, begin, begin + VOXEL_COUNT_PER_LEVEL);

			aabbRegion.minVx.x = ((aabbRegion.minVx.x - center.x)/2.f)+center.x;
			aabbRegion.minVx.y = ((aabbRegion.minVx.y - center.y)/2.f)+center.y;
			aabbRegion.minVx.z = ((aabbRegion.minVx.z - center.z)/2.f)+center.z;

			aabbRegion.maxVx.x = ((aabbRegion.maxVx.x - center.x)/2.f)+center.x;
			aabbRegion.maxVx.y = ((aabbRegion.maxVx.y - center.y)/2.f)+center.y;
			aabbRegion.maxVx.z = ((aabbRegion.maxVx.z - center.z)/2.f)+center.z;
		}
	}

	inline void voxelizeRegion(const nbl_glsl_shapes_AABB_t& region, nbl_glsl_shapes_AABB_t* outVoxelsBegin, nbl_glsl_shapes_AABB_t* outVoxelsEnd)
	{
		const core::vectorSIMDf extent(region.maxVx.x - region.minVx.x, region.maxVx.y - region.minVx.y, region.maxVx.z - region.minVx.z);
		const core::vector3df voxelSideLength(extent.X / VOXEL_COUNT_PER_DIM, extent.Y / VOXEL_COUNT_PER_DIM, extent.Z / VOXEL_COUNT_PER_DIM);

		uint32_t k = 0u;
		for (uint32_t z = 0u; z < VOXEL_COUNT_PER_DIM; ++z)
		{
			for (uint32_t y = 0u; y < VOXEL_COUNT_PER_DIM; ++y)
			{
				for (uint32_t x = 0u; x < VOXEL_COUNT_PER_DIM; ++x)
				{
					nbl_glsl_shapes_AABB_t* voxel = outVoxelsBegin + k++;
					voxel->minVx = { region.minVx.x + x * voxelSideLength.X, region.minVx.y + y * voxelSideLength.Y, region.minVx.z + z * voxelSideLength.Z };
					voxel->maxVx = { voxel->minVx.x + voxelSideLength.X, voxel->minVx.y + voxelSideLength.Y, voxel->minVx.z + voxelSideLength.Z };
				}
			}
		}
		assert(k == VOXEL_COUNT_PER_LEVEL);
	}

	inline void generateLights(const uint32_t lightCount)
	{
		const vectorSIMDf displacementBetweenLights(25.f, 0.f, 0.f);

		const float cosOuterHalfAngle = core::cos(core::radians(2.38f));
		const float cosInnerHalfAngle = core::cos(core::radians(0.5f));
		const float cosineRange = (cosInnerHalfAngle - cosOuterHalfAngle);

		const float startHeight = 257.f;
		vectorSIMDf startPoint(-809.f, startHeight, -34.f, 1.f);
		vectorSIMDf endPoint(964.f, startHeight, -34.f, 1.f);
		uvec2 lightDirection_encoded;
		{
			vectorSIMDf lightDirection(0.045677, 0.032760, -0.998440, 0.001499); // normalized!

			lightDirection_encoded.x = packSnorm2x16(lightDirection.x, lightDirection.y);
			// 1. cosineRange could technically have values in the range [-2.f,2.f] so it might
			// no be good idea to encode it as a normalized integer
			// 2. If we encode rcpCosineRange instead of cosineRange we can handle divison
			// by zero edge cases on the CPU only (by probably discarding those lights) instead
			// of doing that in the shader.
			lightDirection_encoded.y = packSnorm2x16(lightDirection.z, cosineRange);
		}
		uvec2 lightIntensity_encoded;
		{
			const core::vectorSIMDf lightIntensity = core::vectorSIMDf(5.f, 5.f, 5.f, 1.f);
			uint64_t retval = core::rgb32f_to_rgb19e7(lightIntensity.x, lightIntensity.y, lightIntensity.z);
			lightIntensity_encoded.x = uint32_t(retval);
			lightIntensity_encoded.y = uint32_t(retval >> 32);
		}

		const uint32_t grid2dCount = 2u;
		const uint32_t rowsPerGridCount = 12u;
		const uint32_t totalRowCount = rowsPerGridCount*grid2dCount;
		// Todo(achal): Do I need to assert lightCount to be divisible by 24 for this?
		const uint32_t lightsPerRow = lightCount / totalRowCount;

		const uint32_t maxLightCount = totalRowCount*core::floor(core::abs(endPoint.x - startPoint.x)/displacementBetweenLights.x);
		if (lightCount > maxLightCount)
		{
			logger->log("More than %d lights are not supported!\n", system::ILogger::ELL_ERROR, maxLightCount);
			exit(-1);
		}

		lights.resize(lightCount);

		// Grid #0
		for (uint32_t row = 0u; row < rowsPerGridCount; ++row)
		{
			const uint32_t absRowIdx = row;
			const float rowHeight = startHeight - (row * 25.f);
			const vectorSIMDf startPoint(-809.f, rowHeight, -34.f, 1.f);
			for (uint32_t lightIdx = 0u; lightIdx < lightsPerRow; ++lightIdx)
			{
				const uint32_t absLightIdx = absRowIdx*lightsPerRow + lightIdx;
				nbl_glsl_ext_ClusteredLighting_SpotLight& light = lights[absLightIdx];

				const vectorSIMDf pos = startPoint + displacementBetweenLights * lightIdx;
				light.position = { pos.x, pos.y, pos.z };
				light.direction = lightDirection_encoded;
				light.outerCosineOverCosineRange = cosOuterHalfAngle / cosineRange;
				light.intensity = lightIntensity_encoded;
			}
		}

		// Move to the next grid with same number of rows, but flipped light direction and
		// a different startPoint
		lightDirection_encoded = {};
		{
			const core::vectorSIMDf lightDirection = core::vectorSIMDf(0.045677, 0.032760, 0.998440, 0.001499); // normalized
			lightDirection_encoded.x = packSnorm2x16(lightDirection.x, lightDirection.y);
			lightDirection_encoded.y = packSnorm2x16(lightDirection.z, cosineRange);
		}

		// Grid #1
		for (uint32_t row = 0u; row < rowsPerGridCount; ++row)
		{
			const uint32_t absRowIdx = rowsPerGridCount + row;
			const float rowHeight = startHeight - (row * 25.f);
			const vectorSIMDf startPoint(-809.f, rowHeight, 3.f, 1.f);
			for (uint32_t lightIdx = 0u; lightIdx < lightsPerRow; ++lightIdx)
			{
				const uint32_t absLightIdx = absRowIdx*lightsPerRow + lightIdx;
				nbl_glsl_ext_ClusteredLighting_SpotLight& light = lights[absLightIdx];

				const vectorSIMDf pos = startPoint + displacementBetweenLights * lightIdx;
				light.position = { pos.x, pos.y, pos.z };
				light.direction = lightDirection_encoded;
				light.outerCosineOverCosineRange = cosOuterHalfAngle / cosineRange;
				light.intensity = lightIntensity_encoded;
			}
		}
	}

	inline core::smart_refctd_ptr<video::IGPURenderpass> createRenderpass()
	{
		constexpr uint32_t ATTACHMENT_COUNT = 2u;

		video::IGPURenderpass::SCreationParams::SAttachmentDescription attachments[ATTACHMENT_COUNT];
		// swapchain image color attachment
		attachments[0].initialLayout = asset::EIL_UNDEFINED;
		attachments[0].finalLayout = asset::EIL_PRESENT_SRC;
		attachments[0].format = swapchain->getCreationParameters().surfaceFormat.format;
		attachments[0].samples = asset::IImage::ESCF_1_BIT;
		attachments[0].loadOp = nbl::video::IGPURenderpass::ELO_CLEAR;
		attachments[0].storeOp = nbl::video::IGPURenderpass::ESO_STORE;
		// depth attachment to be written to in the first subpass and read from (as a depthStencilAttachment) in the next
		attachments[1].initialLayout = asset::EIL_UNDEFINED;
		attachments[1].finalLayout = asset::EIL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		attachments[1].format = asset::EF_D32_SFLOAT;
		attachments[1].samples = asset::IImage::ESCF_1_BIT;
		attachments[1].loadOp = nbl::video::IGPURenderpass::ELO_CLEAR;
		attachments[1].storeOp = nbl::video::IGPURenderpass::ESO_DONT_CARE; // after the last usage of this attachment we can throw away its contents

		video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef swapchainColorAttRef;
		swapchainColorAttRef.attachment = 0u;
		swapchainColorAttRef.layout = asset::EIL_COLOR_ATTACHMENT_OPTIMAL;

		video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef depthStencilAttRef;
		depthStencilAttRef.attachment = 1u;
		depthStencilAttRef.layout = asset::EIL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		constexpr uint32_t SUBPASS_COUNT = 3u;
		video::IGPURenderpass::SCreationParams::SSubpassDescription subpasses[SUBPASS_COUNT] = {};

		// The Z Pre pass subpass
		subpasses[Z_PREPASS_INDEX].pipelineBindPoint = asset::EPBP_GRAPHICS;
		subpasses[Z_PREPASS_INDEX].depthStencilAttachment = &depthStencilAttRef;

		// The lighting subpass
		subpasses[LIGHTING_PASS_INDEX].pipelineBindPoint = asset::EPBP_GRAPHICS;
		subpasses[LIGHTING_PASS_INDEX].depthStencilAttachment = &depthStencilAttRef;
		subpasses[LIGHTING_PASS_INDEX].colorAttachmentCount = 1u;
		subpasses[LIGHTING_PASS_INDEX].colorAttachments = &swapchainColorAttRef;

		// The debug draw subpass
		subpasses[DEBUG_DRAW_PASS_INDEX].pipelineBindPoint = asset::EPBP_GRAPHICS;
		subpasses[DEBUG_DRAW_PASS_INDEX].colorAttachmentCount = 1u;
		subpasses[DEBUG_DRAW_PASS_INDEX].colorAttachments = &swapchainColorAttRef;
		subpasses[DEBUG_DRAW_PASS_INDEX].depthStencilAttachment = &depthStencilAttRef;

		constexpr uint32_t SUBPASS_DEPS_COUNT = 4u;
		video::IGPURenderpass::SCreationParams::SSubpassDependency subpassDeps[SUBPASS_DEPS_COUNT];

		subpassDeps[0].srcSubpass = video::IGPURenderpass::SCreationParams::SSubpassDependency::SUBPASS_EXTERNAL;
		subpassDeps[0].dstSubpass = Z_PREPASS_INDEX;
		subpassDeps[0].srcStageMask = asset::EPSF_BOTTOM_OF_PIPE_BIT;
		subpassDeps[0].srcAccessMask = asset::EAF_MEMORY_READ_BIT;
		subpassDeps[0].dstStageMask = asset::EPSF_LATE_FRAGMENT_TESTS_BIT;
		subpassDeps[0].dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_DEPTH_STENCIL_ATTACHMENT_READ_BIT | asset::EAF_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
		subpassDeps[0].dependencyFlags = asset::EDF_BY_REGION_BIT; // Todo(achal): Not sure

		subpassDeps[1].srcSubpass = Z_PREPASS_INDEX;
		subpassDeps[1].dstSubpass = LIGHTING_PASS_INDEX;
		subpassDeps[1].srcStageMask = asset::EPSF_LATE_FRAGMENT_TESTS_BIT;
		subpassDeps[1].srcAccessMask = asset::EAF_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		subpassDeps[1].dstStageMask = asset::EPSF_EARLY_FRAGMENT_TESTS_BIT;
		subpassDeps[1].dstAccessMask = asset::EAF_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
		subpassDeps[1].dependencyFlags = asset::EDF_BY_REGION_BIT; // Todo(achal): Not sure

		// Todo(achal): Not 100% sure why would I need this
		subpassDeps[2].srcSubpass = Z_PREPASS_INDEX;
		subpassDeps[2].dstSubpass = video::IGPURenderpass::SCreationParams::SSubpassDependency::SUBPASS_EXTERNAL;
		subpassDeps[2].srcStageMask = asset::EPSF_LATE_FRAGMENT_TESTS_BIT;
		subpassDeps[2].srcAccessMask = asset::EAF_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		subpassDeps[2].dstStageMask = asset::EPSF_BOTTOM_OF_PIPE_BIT;
		subpassDeps[2].dstAccessMask = asset::EAF_MEMORY_READ_BIT;
		subpassDeps[2].dependencyFlags = asset::EDF_BY_REGION_BIT; // Todo(achal): Not sure

		subpassDeps[3].srcSubpass = LIGHTING_PASS_INDEX;
		subpassDeps[3].dstSubpass = DEBUG_DRAW_PASS_INDEX;
		subpassDeps[3].srcStageMask = static_cast<asset::E_PIPELINE_STAGE_FLAGS>(asset::EPSF_COLOR_ATTACHMENT_OUTPUT_BIT | asset::EPSF_LATE_FRAGMENT_TESTS_BIT);
		subpassDeps[3].srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_COLOR_ATTACHMENT_WRITE_BIT | asset::EAF_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
		subpassDeps[3].dstStageMask = static_cast<asset::E_PIPELINE_STAGE_FLAGS>(asset::EPSF_COLOR_ATTACHMENT_OUTPUT_BIT | asset::EPSF_LATE_FRAGMENT_TESTS_BIT);
		subpassDeps[3].dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_COLOR_ATTACHMENT_WRITE_BIT | asset::EAF_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
		subpassDeps[3].dependencyFlags = asset::EDF_BY_REGION_BIT;  // Todo(achal): Not sure

		video::IGPURenderpass::SCreationParams creationParams = {};
		creationParams.attachmentCount = ATTACHMENT_COUNT;
		creationParams.attachments = attachments;
		creationParams.dependencies = subpassDeps;
		creationParams.dependencyCount = SUBPASS_DEPS_COUNT;
		creationParams.subpasses = subpasses;
		creationParams.subpassCount = SUBPASS_COUNT;

		return logicalDevice->createGPURenderpass(creationParams);
	}

	inline core::smart_refctd_ptr<video::IGPUSpecializedShader> createShader(const char* path)
	{
		asset::IAssetLoader::SAssetLoadParams params = {};
		params.logger = logger.get();
		auto cpuSpecShader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(path, params).getContents().begin());
		if (!cpuSpecShader)
		{
			logger->log("Failed to load shader at path %s!\n", system::ILogger::ELL_ERROR, path);
			return nullptr;
		}

		auto gpuArray = cpu2gpu.getGPUObjectsFromAssets(&cpuSpecShader.get(), &cpuSpecShader.get() + 1, cpu2gpuParams);
		if (!gpuArray || gpuArray->size() < 1u || !(*gpuArray)[0])
		{
			logger->log("Failed to convert debug draw CPU specialized shader to GPU specialized shaders!\n", system::ILogger::ELL_ERROR);
			return nullptr;
		}

		return gpuArray->begin()[0];
	};

	core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
	core::smart_refctd_ptr<nbl::ui::IWindow> window;
	core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
	core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
	core::smart_refctd_ptr<nbl::video::ISurface> surface;
	core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
	core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
	video::IPhysicalDevice* physicalDevice;
	std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
	core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
	core::smart_refctd_ptr<video::IGPURenderpass> renderpass = nullptr;
	core::smart_refctd_ptr<video::IGPUFramebuffer> fbos[CommonAPI::InitOutput::MaxSwapChainImageCount] = { nullptr };
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
	core::smart_refctd_ptr<nbl::system::ISystem> system;
	core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	core::smart_refctd_ptr<nbl::system::ILogger> logger;
	core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
	video::IGPUObjectFromAssetConverter cpu2gpu;

	int32_t resourceIx = -1;
	uint32_t acquiredNextFBO = {};

	core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };

	core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUCommandBuffer> bakedCommandBuffer;
	core::smart_refctd_ptr<video::IGPUCommandBuffer> zPrepassCommandBuffer;

	const asset::COBJMetadata::CRenderpassIndependentPipeline* pipelineMetadata = nullptr;

	float clipmapExtent;

	core::smart_refctd_ptr<video::IGPUComputePipeline> cullPipeline = nullptr;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> cullDs = nullptr;
	core::smart_refctd_ptr<video::IGPUBuffer> activeLightIndicesGPUBuffer = nullptr; // should be an input to the system
	core::smart_refctd_ptr<video::IGPUBuffer> intersectionRecordsGPUBuffer = nullptr;
	core::smart_refctd_ptr<video::IGPUBuffer> intersectionRecordCountGPUBuffer = nullptr;
	core::smart_refctd_ptr<video::IGPUBuffer> cullScratchGPUBuffer = nullptr;
	core::smart_refctd_ptr<video::IGPUComputePipeline> scatterPipeline = nullptr;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> scatterDS = nullptr;

	core::smart_refctd_ptr<video::IGPUImage> lightGridTexture2 = nullptr;
	core::smart_refctd_ptr<video::IGPUImageView> lightGridTextureView2 = nullptr;
	core::smart_refctd_ptr<video::IGPUBuffer> lightIndexListGPUBuffer = nullptr;
	core::smart_refctd_ptr<video::IGPUBufferView> lightIndexListSSBOView = nullptr;

	core::smart_refctd_ptr<video::IGPUDescriptorSet> scanDS = nullptr;
	video::CScanner::DefaultPushConstants scanPushConstants;
	video::CScanner::DispatchInfo scanDispatchInfo;
	core::smart_refctd_ptr<video::IGPUBuffer> scanScratchGPUBuffer = nullptr;
	core::smart_refctd_ptr<video::IGPUComputePipeline> scanPipeline = nullptr;
	// asset::SBufferRange<video::IGPUBuffer> scanScratchBufferRange = {};

	core::vector<nbl_glsl_ext_ClusteredLighting_SpotLight> lights;
	core::smart_refctd_ptr<asset::ICPUBuffer> lightGridCPUBuffer;
	core::smart_refctd_ptr<video::IGPUImage> lightGridTexture = nullptr;
	core::smart_refctd_ptr<video::IGPUImageView> lightGridTextureView = nullptr;
	// Todo(achal): This shouldn't probably be a ubo????
	core::smart_refctd_ptr<video::IGPUBuffer> lightIndexListUbo = nullptr;
	core::smart_refctd_ptr<video::IGPUBufferView> lightIndexListUboView = nullptr;

	// I need descriptor lifetime tracking!
	core::smart_refctd_ptr<video::IGPUBuffer> globalLightListSSBO = nullptr;
	using PropertyPoolType = video::CPropertyPool<core::allocator, nbl_glsl_ext_ClusteredLighting_SpotLight>;
	core::smart_refctd_ptr<PropertyPoolType> propertyPool = nullptr;

	core::smart_refctd_ptr<video::IGPUBuffer> cameraUbo;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> cameraDS;
	uint32_t cameraUboBindingNumber;

	core::smart_refctd_ptr<video::IGPUDescriptorSet> lightDS = nullptr;

	core::smart_refctd_ptr<video::IGPUMesh> gpuMesh;

	video::CDumbPresentationOracle oracle;

	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

	Camera camera = Camera(vectorSIMDf(0, 0, 0), vectorSIMDf(0, 0, 0), matrix4SIMD());

	int32_t debugActiveLightIndex = -1;
	core::smart_refctd_ptr<video::IGPUBuffer> debugClustersForLightGPU[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUBuffer> debugAABBIndexBuffer = nullptr;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> debugAABBDescriptorSets[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> debugAABBGraphicsPipeline = nullptr;
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> debugLightVolumeGraphicsPipeline = nullptr;
	core::smart_refctd_ptr<video::IGPUMeshBuffer> debugLightVolumeMeshBuffer = nullptr;
	// Todo(achal): In theory it is possible that the model matrix gets updated but
	// the following commandbuffer doesn't get the chance to actually execute and consequently
	// send the model matrix (via push constants), before the next while loop comes
	// and updates it. To remedy this, I either need FRAMES_IN_FLIGHT model matrices
	// or some sorta CPU-GPU sync (fence?)
	core::matrix3x4SIMD debugLightVolumeModelMatrix;
};

NBL_COMMON_API_MAIN(ClusteredRenderingSampleApp)

extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }