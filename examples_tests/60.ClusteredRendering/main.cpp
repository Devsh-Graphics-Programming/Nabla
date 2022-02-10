#define _NBL_STATIC_LIB
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "../common/Camera.hpp"
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/ui/ICursorControl.h"

using namespace nbl;

// #define DEBUG_VIZ

// #define CLIPMAP
#define OCTREE

#define WG_DIM 256

#define FATAL_LOG(x, ...) {logger->log(##x, system::ILogger::ELL_ERROR, __VA_ARGS__); exit(-1);}

struct vec3
{
	float x, y, z;
};

// #ifdef DEBUG_VIZ
struct alignas(16) vec3_aligned
{
	float x, y, z;
};
// #endif

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

// #ifdef DEBUG_VIZ
struct nbl_glsl_shapes_AABB_t
{
	vec3_aligned minVx;
	vec3_aligned maxVx;
};
// #endif

class ClusteredRenderingSampleApp : public ApplicationBase
{
	constexpr static uint32_t WIN_W = 1280u;
	constexpr static uint32_t WIN_H = 720u;
	constexpr static uint32_t SC_IMG_COUNT = 3u;
	constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
	constexpr static uint64_t MAX_TIMEOUT = 99999999999999ull;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	constexpr static uint32_t LIGHT_COUNT = 6860u;

	// Todo(achal): It could be a good idea to make LOD_COUNT dynamic based on double/float precision?
	//
	// Level 9 is the outermost/coarsest, Level 0 is the innermost/finest
#ifdef CLIPMAP
	static constexpr uint32_t LOD_COUNT = 10u;
#else OCTREE
	static constexpr uint32_t LOD_COUNT = 7u;
#endif

	constexpr static uint32_t MEMORY_BUDGET = 15ull * 1024ull * 1024ull;
	// Todo(achal): Try making this 2048
	// Todo(achal): ..something something atomic contention..
	constexpr static uint32_t BUDGETING_HISTOGRAM_BIN_COUNT = 1024u;
	constexpr static float MIN_HISTOGRAM_IMPORTANCE = 1 / 2048.f;
	constexpr static float MAX_HISTOGRAM_IMPORTANCE = float(1 << 16);
	constexpr static float BUDGETING_MARGIN = 0.9f;

	constexpr static uint32_t CAMERA_DS_NUMBER = 1u;
	constexpr static uint32_t LIGHT_DS_NUMBER = 2u;

	constexpr static uint32_t Z_PREPASS_INDEX = 0u;
	constexpr static uint32_t LIGHTING_PASS_INDEX = 1u;
	constexpr static uint32_t DEBUG_DRAW_PASS_INDEX = 2u;

	constexpr static float LIGHT_CONTRIBUTION_THRESHOLD = 2.f;
	constexpr static float LIGHT_RADIUS = 25.f;

#ifdef CLIPMAP
	constexpr static uint32_t VOXEL_COUNT_PER_DIM = 4u;
	constexpr static uint32_t VOXEL_COUNT_PER_LEVEL = VOXEL_COUNT_PER_DIM * VOXEL_COUNT_PER_DIM * VOXEL_COUNT_PER_DIM;
#endif

#ifdef DEBUG_VIZ
	constexpr static float DEBUG_CONE_RADIUS = 10.f;
	constexpr static float DEBUG_CONE_LENGTH = 25.f;
	constexpr static vec3 DEBUG_CONE_DIRECTION = { 0.f, -1.f, 0.f };
#endif

	struct cone_t
	{
		core::vectorSIMDf tip;
		core::vectorSIMDf direction; // always normalized
		float height;
		float cosHalfAngle;
	};

	// Todo(achal): Probably merge the following two into one, idk..
	struct octree_first_cull_push_constants_t
	{
		float camPosClipmapExtent[4];
		uint32_t lightCount;
		uint32_t buildHistogramID;
	};

	struct octree_intermediate_cull_push_constants_t
	{
		float camPosClipmapExtent[4];
		uint32_t hierarchyLevel;
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
			FATAL_LOG("Failed to convert debug light volume's vertex and index buffers from CPU to GPU!\n");
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
				FATAL_LOG("Failed to create GPU DS layout for debug draw AABB!\n");
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
	void onAppInitialized_impl() override
	{
		const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT);
		const video::ISurface::SFormat surfaceFormat(asset::EF_B8G8R8A8_UNORM, asset::ECP_COUNT, asset::EOTF_UNKNOWN);

		CommonAPI::InitOutput initOutput;
		initOutput.window = core::smart_refctd_ptr(window);
		CommonAPI::InitWithDefaultExt(
			initOutput,
			video::EAT_VULKAN,
			"ClusteredLighting",
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
			FATAL_LOG("Failed to create the render pass!\n");

		fbos = CommonAPI::createFBOWithSwapchainImages(
			swapchain->getImageCount(), WIN_W, WIN_H,
			logicalDevice,
			swapchain,
			renderpass,
			asset::EF_D32_SFLOAT);

		// Input set of lights in world position
		constexpr uint32_t MAX_LIGHT_COUNT = (1u << 22);
		generateLights(LIGHT_COUNT);

		core::vectorSIMDf cameraPosition(-157.229813, 369.800446, -19.696722, 0.000000);
		core::vectorSIMDf cameraTarget(-387.548462, 198.927414, -26.500174, 1.000000);

		const float cameraFar = 4000.f;
		core::matrix4SIMD projectionMatrix = core::matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 2.f, cameraFar);
		camera = Camera(cameraPosition, cameraTarget, projectionMatrix, 12.5f, 2.f);
		{
			core::matrix4SIMD invProj;
			if (!projectionMatrix.getInverseTransform(invProj))
				FATAL_LOG("Camera projection matrix is not invertible!\n")

			core::vector4df_SIMD topRight(1.f, 1.f, 1.f, 1.f);
			invProj.transformVect(topRight);
			topRight /= topRight.w;
			genesisVoxelExtent = 2.f * core::sqrt(topRight.x * topRight.x + topRight.y * topRight.y + topRight.z * topRight.z);
			assert(genesisVoxelExtent > 2.f * cameraFar);
		}

		video::CPropertyPoolHandler* propertyPoolHandler = utilities->getDefaultPropertyPoolHandler();

		const uint32_t capacity = MAX_LIGHT_COUNT;
		const bool contiguous = false;

		video::IGPUBuffer::SCreationParams creationParams = {};
		creationParams.usage = static_cast<asset::IBuffer::E_USAGE_FLAGS>(asset::IBuffer::EUF_STORAGE_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_TRANSFER_SRC_BIT);

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

		// active light indices
		{
			// Todo(achal): Should be changeable per frame
			activeLightIndices.resize(LIGHT_COUNT);

			for (uint32_t i = 0u; i < activeLightIndices.size(); ++i)
				activeLightIndices[i] = i;

			const size_t neededSize = LIGHT_COUNT * sizeof(uint32_t);
			video::IGPUBuffer::SCreationParams creationParams = {};
			creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT);
			activeLightIndicesGPUBuffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, neededSize);

			asset::SBufferRange<video::IGPUBuffer> bufferRange = {};
			bufferRange.buffer = activeLightIndicesGPUBuffer;
			bufferRange.offset = 0ull;
			bufferRange.size = activeLightIndicesGPUBuffer->getCachedCreationParams().declaredSize;
			utilities->updateBufferRangeViaStagingBuffer(
				queues[CommonAPI::InitOutput::EQT_TRANSFER_UP],
				bufferRange,
				activeLightIndices.data());
		}

		constexpr uint32_t OCTREE_FIRST_CULL_DESCRIPTOR_COUNT = 4u;
		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> octreeFirstCullDSLayout = nullptr;
		{
			video::IGPUDescriptorSetLayout::SBinding bindings[OCTREE_FIRST_CULL_DESCRIPTOR_COUNT];

			// 0. property pool of lights
			// 1. active light indices
			// 2. out scratch buffer
			// 3. importance histogram
			for (uint32_t i = 0u; i < OCTREE_FIRST_CULL_DESCRIPTOR_COUNT; ++i)
			{
				bindings[i].binding = i;
				bindings[i].type = asset::EDT_STORAGE_BUFFER;
				bindings[i].count = 1u;
				bindings[i].stageFlags = asset::IShader::ESS_COMPUTE;
				bindings[i].samplers = nullptr;
			}

			octreeFirstCullDSLayout = logicalDevice->createGPUDescriptorSetLayout(bindings, bindings + OCTREE_FIRST_CULL_DESCRIPTOR_COUNT);
		}
		const uint32_t octreeFirstCullDSCount = SC_IMG_COUNT + 2u*SC_IMG_COUNT; // includes both for first and intermediate cull
		core::smart_refctd_ptr<video::IDescriptorPool> octreeFirstCullDescriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &octreeFirstCullDSLayout.get(), &octreeFirstCullDSLayout.get() + 1ull, &octreeFirstCullDSCount);

		core::smart_refctd_ptr<video::IGPUPipelineLayout> octreeFirstCullPipelineLayout = nullptr;
		{
			asset::SPushConstantRange pcRange = {};
			pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
			pcRange.offset = 0u;
			pcRange.size = sizeof(octree_first_cull_push_constants_t);
			octreeFirstCullPipelineLayout = logicalDevice->createGPUPipelineLayout(&pcRange, &pcRange + 1ull, core::smart_refctd_ptr(octreeFirstCullDSLayout));
		}

		// create octree scratch buffers
		for (uint32_t scImageIndex = 0u; scImageIndex < SC_IMG_COUNT; ++scImageIndex)
		{
			for (uint32_t i = 0u; i < 2u; ++i)
			{
				video::IGPUBuffer::SCreationParams creationParams = {};
				creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT | video::IGPUBuffer::EUF_TRANSFER_SRC_BIT);

				// Todo(achal): Do I have to abide by something something SSBO alignment?
				const size_t neededSize = sizeof(uint32_t) + sizeof(uint32_t) + MEMORY_BUDGET;

				octreeScratchBuffers[scImageIndex][i] = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, neededSize);

				uint32_t clearValue = 0u;

				asset::SBufferRange<video::IGPUBuffer> bufferRange = {};
				bufferRange.buffer = octreeScratchBuffers[scImageIndex][i];
				bufferRange.offset = 0ull;
				bufferRange.size = sizeof(uint32_t);
				utilities->updateBufferRangeViaStagingBuffer(
					queues[CommonAPI::InitOutput::EQT_TRANSFER_UP],
					bufferRange,
					&clearValue);
			}
		}

		// create histogram buffer
		for (uint32_t scImageIndex = 0u; scImageIndex < SC_IMG_COUNT; ++scImageIndex)
		{
			// currently used to debug the histogram and its cumulative histogram
			const size_t neededSize = 2ull * BUDGETING_HISTOGRAM_BIN_COUNT * sizeof(uint32_t);

			importanceHistogramBuffer[scImageIndex] = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, neededSize);

			core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf = nullptr;
			logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_COMPUTE].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);

			cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
			cmdbuf->fillBuffer(importanceHistogramBuffer[scImageIndex].get(), 0u, neededSize, 0u);
			cmdbuf->end();

			auto fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);

			video::IGPUQueue::SSubmitInfo submitInfo = {};
			submitInfo.commandBufferCount = 1u;
			submitInfo.commandBuffers = &cmdbuf.get();
			computeQueue->submit(1u, &submitInfo, fence.get());

			logicalDevice->blockForFences(1u, &fence.get());
		}

		const char* octreeFirstCullShaderPath = "../octree_first_cull.comp";
		{
			asset::IAssetLoader::SAssetLoadParams params = {};
			params.logger = logger.get();
			auto loadedShader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(octreeFirstCullShaderPath, params).getContents().begin());

			auto unspecOverridenShader = asset::IGLSLCompiler::createOverridenCopy(loadedShader->getUnspecialized(),
				"#define BIN_COUNT %d\n"
				"#define MIN_HISTOGRAM_IMPORTANCE %f\n"
				"#define MAX_HISTOGRAM_IMPORTANCE %f\n"
				"#define MEMORY_BUDGET %d\n"
				"#define BUDGETING_MARGIN %f\n",
				BUDGETING_HISTOGRAM_BIN_COUNT, MIN_HISTOGRAM_IMPORTANCE, MAX_HISTOGRAM_IMPORTANCE, MEMORY_BUDGET, BUDGETING_MARGIN);

			auto specShader_cpu = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecOverridenShader), asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));

			auto gpuArray = cpu2gpu.getGPUObjectsFromAssets(&specShader_cpu.get(), &specShader_cpu.get() + 1, cpu2gpuParams);
			if (!gpuArray || gpuArray->size() < 1u || !(*gpuArray)[0])
				FATAL_LOG("Failed to convert CPU specialized shader to GPU specialized shaders!\n");

			core::smart_refctd_ptr<video::IGPUSpecializedShader> specShader = gpuArray->begin()[0];
			octreeFirstCullPipeline = logicalDevice->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(octreeFirstCullPipelineLayout), std::move(specShader));
		}

		for (uint32_t scImageIndex = 0u; scImageIndex < SC_IMG_COUNT; ++scImageIndex)
		{
			octreeFirstCullDS[scImageIndex] = logicalDevice->createGPUDescriptorSet(octreeFirstCullDescriptorPool.get(), core::smart_refctd_ptr(octreeFirstCullDSLayout));

			video::IGPUDescriptorSet::SWriteDescriptorSet writes[OCTREE_FIRST_CULL_DESCRIPTOR_COUNT] = {};
			video::IGPUDescriptorSet::SDescriptorInfo infos[OCTREE_FIRST_CULL_DESCRIPTOR_COUNT] = {};

			for (uint32_t i = 0u; i < OCTREE_FIRST_CULL_DESCRIPTOR_COUNT; ++i)
			{
				writes[i].dstSet = octreeFirstCullDS[scImageIndex].get();
				writes[i].binding = i;
				writes[i].arrayElement = 0u;
				writes[i].count = 1u;
				writes[i].descriptorType = asset::EDT_STORAGE_BUFFER;
				writes[i].info = &infos[i];
			}

			const auto& propertyMemoryBlock = propertyPool->getPropertyMemoryBlock(0u);
			infos[0].desc = propertyMemoryBlock.buffer;
			infos[0].buffer.offset = propertyMemoryBlock.offset;
			infos[0].buffer.size = propertyMemoryBlock.size;

			infos[1].desc = activeLightIndicesGPUBuffer;
			infos[1].buffer.offset = 0ull;
			infos[1].buffer.size = activeLightIndicesGPUBuffer->getCachedCreationParams().declaredSize;

			infos[2].desc = octreeScratchBuffers[scImageIndex][0];
			infos[2].buffer.offset = 0ull;
			infos[2].buffer.size = octreeScratchBuffers[scImageIndex][0]->getCachedCreationParams().declaredSize;

			infos[3].desc = importanceHistogramBuffer[scImageIndex];
			infos[3].buffer.offset = 0ull;
			infos[3].buffer.size = importanceHistogramBuffer[scImageIndex]->getCachedCreationParams().declaredSize;

			logicalDevice->updateDescriptorSets(OCTREE_FIRST_CULL_DESCRIPTOR_COUNT, writes, 0u, nullptr);
		}

		// octreeIntermediateCullDSLayout would be the same as the octreeFirstCullDSLayout
		// octreeIntermediatePipelineLayout would be the same as octreeFirstPipelineLayout
		
		const char* octreeIntermediateCullShaderPath = "../octree_intermediate_cull.comp";
		{
			asset::IAssetLoader::SAssetLoadParams params = {};
			params.logger = logger.get();
			auto loadedShader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(octreeIntermediateCullShaderPath, params).getContents().begin());

			auto unspecOverridenShader = asset::IGLSLCompiler::createOverridenCopy(loadedShader->getUnspecialized(),
				// Todo(achal): Most of this stuff can be removed now
				"#define BIN_COUNT %d\n"
				"#define MIN_HISTOGRAM_IMPORTANCE %f\n"
				"#define MAX_HISTOGRAM_IMPORTANCE %f\n"
				"#define MEMORY_BUDGET %d\n"
				"#define BUDGETING_MARGIN %f\n",
				BUDGETING_HISTOGRAM_BIN_COUNT, MIN_HISTOGRAM_IMPORTANCE, MAX_HISTOGRAM_IMPORTANCE, MEMORY_BUDGET, BUDGETING_MARGIN);

			auto specShader_cpu = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecOverridenShader), asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));

			auto gpuArray = cpu2gpu.getGPUObjectsFromAssets(&specShader_cpu.get(), &specShader_cpu.get() + 1, cpu2gpuParams);
			if (!gpuArray || gpuArray->size() < 1u || !(*gpuArray)[0])
				FATAL_LOG("Failed to convert CPU specialized shader to GPU specialized shaders!\n");

			core::smart_refctd_ptr<video::IGPUSpecializedShader> specShader = gpuArray->begin()[0];
			octreeIntermediateCullPipeline = logicalDevice->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(octreeFirstCullPipelineLayout), std::move(specShader));
		}

		// octree intermediate cull ds
		for (uint32_t scImageIndex = 0u; scImageIndex < SC_IMG_COUNT; ++scImageIndex)
		{
			for (uint32_t dsIndex = 0u; dsIndex < 2u; ++dsIndex)
			{
				octreeIntermediateCullDS[scImageIndex][dsIndex] = logicalDevice->createGPUDescriptorSet(octreeFirstCullDescriptorPool.get(), core::smart_refctd_ptr(octreeFirstCullDSLayout));

				video::IGPUDescriptorSet::SWriteDescriptorSet writes[OCTREE_FIRST_CULL_DESCRIPTOR_COUNT] = {};
				video::IGPUDescriptorSet::SDescriptorInfo infos[OCTREE_FIRST_CULL_DESCRIPTOR_COUNT] = {};

				for (uint32_t i = 0u; i < OCTREE_FIRST_CULL_DESCRIPTOR_COUNT; ++i)
				{
					writes[i].dstSet = octreeIntermediateCullDS[scImageIndex][dsIndex].get();
					writes[i].binding = i;
					writes[i].arrayElement = 0u;
					writes[i].count = 1u;
					writes[i].descriptorType = asset::EDT_STORAGE_BUFFER;
					writes[i].info = &infos[i];
				}

				const auto& propertyMemoryBlock = propertyPool->getPropertyMemoryBlock(0u);
				infos[0].desc = propertyMemoryBlock.buffer;
				infos[0].buffer.offset = propertyMemoryBlock.offset;
				infos[0].buffer.size = propertyMemoryBlock.size;

				// inScratch
				infos[1].desc = octreeScratchBuffers[scImageIndex][dsIndex];
				infos[1].buffer.offset = 0ull;
				infos[1].buffer.size = octreeScratchBuffers[scImageIndex][dsIndex]->getCachedCreationParams().declaredSize;

				// outScratch
				infos[2].desc = octreeScratchBuffers[scImageIndex][1u-dsIndex];
				infos[2].buffer.offset = 0ull;
				infos[2].buffer.size = octreeScratchBuffers[scImageIndex][1u-dsIndex]->getCachedCreationParams().declaredSize;

				infos[3].desc = importanceHistogramBuffer[scImageIndex];
				infos[3].buffer.offset = 0ull;
				infos[3].buffer.size = importanceHistogramBuffer[scImageIndex]->getCachedCreationParams().declaredSize;

				logicalDevice->updateDescriptorSets(OCTREE_FIRST_CULL_DESCRIPTOR_COUNT, writes, 0u, nullptr);
			}
		}

		// lightGridTexture
		{
#ifdef CLIPMAP
			// Appending one level grid after another in the Z direction
			const uint32_t width = VOXEL_COUNT_PER_DIM;
			const uint32_t height = VOXEL_COUNT_PER_DIM;
			const uint32_t depth = VOXEL_COUNT_PER_DIM * LOD_COUNT;
#else OCTREE
			const uint32_t width = 1 << (LOD_COUNT - 1);
			const uint32_t height = 1 << (LOD_COUNT - 1);
			const uint32_t depth = 1 << (LOD_COUNT - 1);
#endif
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
			lightGridTexture = logicalDevice->createDeviceLocalGPUImageOnDedMem(std::move(creationParams));

			if (!lightGridTexture)
				FATAL_LOG("Failed to create the light grid 3D texture!\n");
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
				FATAL_LOG("Failed to create image view for light grid 3D texture!\n");
		}
		// light index list gpu buffer and view
		{
			video::IGPUBuffer::SCreationParams creationParams = {};
			creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT);
			lightIndexListGPUBuffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, MEMORY_BUDGET);
			lightIndexListSSBOView = logicalDevice->createGPUBufferView(lightIndexListGPUBuffer.get(), asset::EF_R32_UINT, 0ull, lightIndexListGPUBuffer->getCachedCreationParams().declaredSize);
		}

		constexpr uint32_t OCTREE_LAST_CULL_DESCRIPTOR_COUNT = 4u;
		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> octreeLastCullDSLayout = nullptr;
		{
			video::IGPUDescriptorSetLayout::SBinding binding[OCTREE_LAST_CULL_DESCRIPTOR_COUNT];
			{
				// property pool of lights
				binding[0].binding = 0u;
				binding[0].type = asset::EDT_STORAGE_BUFFER;
				binding[0].count = 1u;
				binding[0].stageFlags = asset::IShader::ESS_COMPUTE;
				binding[0].samplers = nullptr;

				// inScratch
				binding[1].binding = 1u;
				binding[1].type = asset::EDT_STORAGE_BUFFER;
				binding[1].count = 1u;
				binding[1].stageFlags = asset::IShader::ESS_COMPUTE;
				binding[1].samplers = nullptr;

				// intersection records
				binding[2].binding = 2u;
				binding[2].type = asset::EDT_STORAGE_BUFFER;
				binding[2].count = 1u;
				binding[2].stageFlags = asset::IShader::ESS_COMPUTE;
				binding[2].samplers = nullptr;

				// light grid (storage image)
				binding[3].binding = 3u;
				binding[3].type = asset::EDT_STORAGE_IMAGE;
				binding[3].count = 1u;
				binding[3].stageFlags = asset::IShader::ESS_COMPUTE;
				binding[3].samplers = nullptr;
			}

			octreeLastCullDSLayout = logicalDevice->createGPUDescriptorSetLayout(binding, binding + OCTREE_LAST_CULL_DESCRIPTOR_COUNT);
			if (!octreeLastCullDSLayout)
				FATAL_LOG("Failed to create DS Layout for the last octree cull pass resources!\n");
		}

		core::smart_refctd_ptr<video::IGPUPipelineLayout> octreeLastCullPipelineLayout = nullptr;
		{
			asset::SPushConstantRange pcRange = {};
			pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
			pcRange.offset = 0u;
			pcRange.size = sizeof(octree_intermediate_cull_push_constants_t);
			octreeLastCullPipelineLayout = logicalDevice->createGPUPipelineLayout(&pcRange, &pcRange + 1ull, core::smart_refctd_ptr(octreeLastCullDSLayout));
		}

		const char* octreeLastCullShaderPath = "../octree_last_cull.comp";
		{
			asset::IAssetLoader::SAssetLoadParams params = {};
			params.logger = logger.get();
			auto loadedShader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(octreeLastCullShaderPath, params).getContents().begin());

			auto unspecOverridenShader = asset::IGLSLCompiler::createOverridenCopy(loadedShader->getUnspecialized(), "#define MEMORY_BUDGET %d\n", MEMORY_BUDGET);

			auto specShader_cpu = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecOverridenShader), asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));
			auto gpuArray = cpu2gpu.getGPUObjectsFromAssets(&specShader_cpu.get(), &specShader_cpu.get() + 1, cpu2gpuParams);
			if (!gpuArray || gpuArray->size() < 1u || !(*gpuArray)[0])
				FATAL_LOG("Failed to convert CPU specialized shader to GPU specialized shaders!\n");

			core::smart_refctd_ptr<video::IGPUSpecializedShader> specShader = gpuArray->begin()[0];
			octreeLastCullPipeline = logicalDevice->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(octreeLastCullPipelineLayout), std::move(specShader));
		}

		// octree last cull ds
		const uint32_t octreeLastCullDSCount = SC_IMG_COUNT;
		core::smart_refctd_ptr<video::IDescriptorPool> octreeLastCullDescriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &octreeLastCullDSLayout.get(), &octreeLastCullDSLayout.get() + 1ull, &octreeLastCullDSCount);
		for (uint32_t scImageIndex = 0u; scImageIndex < SC_IMG_COUNT; ++scImageIndex)
		{
			octreeLastCullDS[scImageIndex] = logicalDevice->createGPUDescriptorSet(octreeLastCullDescriptorPool.get(), core::smart_refctd_ptr(octreeLastCullDSLayout));

			video::IGPUDescriptorSet::SWriteDescriptorSet writes[OCTREE_LAST_CULL_DESCRIPTOR_COUNT] = {};
			video::IGPUDescriptorSet::SDescriptorInfo infos[OCTREE_LAST_CULL_DESCRIPTOR_COUNT] = {};

			for (uint32_t i = 0u; i < OCTREE_LAST_CULL_DESCRIPTOR_COUNT; ++i)
			{
				writes[i].dstSet = octreeLastCullDS[scImageIndex].get();
				writes[i].binding = i;
				writes[i].arrayElement = 0u;
				writes[i].count = 1u;
				writes[i].descriptorType = asset::EDT_STORAGE_BUFFER;
				writes[i].info = &infos[i];
			}

			writes[OCTREE_LAST_CULL_DESCRIPTOR_COUNT - 1u].descriptorType = asset::EDT_STORAGE_IMAGE;

			// light pool
			const auto& propertyMemoryBlock = propertyPool->getPropertyMemoryBlock(0u);
			infos[0].desc = propertyMemoryBlock.buffer;
			infos[0].buffer.offset = propertyMemoryBlock.offset;
			infos[0].buffer.size = propertyMemoryBlock.size;

			// inScratch
			infos[1].desc = octreeScratchBuffers[scImageIndex][0];
			infos[1].buffer.offset = 0ull;
			infos[1].buffer.size = octreeScratchBuffers[scImageIndex][1]->getCachedCreationParams().declaredSize;

			// intersection records
			infos[2].desc = octreeScratchBuffers[scImageIndex][1u];
			infos[2].buffer.offset = 0ull;
			infos[2].buffer.size = octreeScratchBuffers[scImageIndex][1u]->getCachedCreationParams().declaredSize;

			// light grid
			infos[3].desc = lightGridTextureView;
			infos[3].image.imageLayout = asset::EIL_GENERAL;
			infos[3].image.sampler = nullptr;

			logicalDevice->updateDescriptorSets(OCTREE_LAST_CULL_DESCRIPTOR_COUNT, writes, 0u, nullptr);
		}

#ifdef CLIPMAP
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

		core::smart_refctd_ptr<video::IGPUPipelineLayout> cullPipelineLayout = nullptr;
		{
			asset::SPushConstantRange pcRange = {};
			pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
			pcRange.offset = 0u;
			pcRange.size = 4 * sizeof(float);

			cullPipelineLayout = logicalDevice->createGPUPipelineLayout(&pcRange, &pcRange + 1ull, core::smart_refctd_ptr(cullDsLayout));
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
		// scratchBuffer
		{
			const size_t neededSize = LIGHT_COUNT * sizeof(uint32_t);
			video::IGPUBuffer::SCreationParams creationParams = {};
			creationParams.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
			cullScratchGPUBuffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, neededSize);
		}

		const char* cullCompShaderPath = "../cull.comp";
		{
			asset::IAssetLoader::SAssetLoadParams params = {};
			params.logger = logger.get();
			auto loadedShader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(cullCompShaderPath, params).getContents().begin());
			auto unspecOverridenShader = asset::IGLSLCompiler::createOverridenCopy(loadedShader->getUnspecialized(), "#define WG_DIM %d\n#define LIGHT_COUNT %d\n", WG_DIM, LIGHT_COUNT);
			auto cullSpecShader_cpu = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecOverridenShader), asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));

			auto gpuArray = cpu2gpu.getGPUObjectsFromAssets(&cullSpecShader_cpu.get(), &cullSpecShader_cpu.get() + 1, cpu2gpuParams);
			if (!gpuArray || gpuArray->size() < 1u || !(*gpuArray)[0])
				FATAL_LOG("Failed to convert CPU specialized shader to GPU specialized shaders!\n");

			core::smart_refctd_ptr<video::IGPUSpecializedShader> cullSpecShader = gpuArray->begin()[0];
			cullPipeline = logicalDevice->createGPUComputePipeline(nullptr, std::move(cullPipelineLayout), std::move(cullSpecShader));
		}

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

			infos[4].desc = lightGridTextureView;
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
#endif

		// Todo(achal): I'm sure this scan scratch buffer can be reused from some other scratch we allocate
		{
			video::CScanner* scanner = utilities->getDefaultScanner();
			const auto& lightGridExtent = lightGridTexture->getCreationParameters().extent;
			scanner->buildParameters(lightGridExtent.width * lightGridExtent.height * lightGridExtent.depth, scanPushConstants, scanDispatchInfo);

			const size_t neededSize = scanPushConstants.scanParams.getScratchSize();
			video::IGPUBuffer::SCreationParams creationParams = {};
			creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT);
			scanScratchGPUBuffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, neededSize);
		}

		constexpr uint32_t SCAN_DESCRIPTOR_COUNT = 2u;
		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> scanDSLayout = nullptr;
		{
			video::IGPUDescriptorSetLayout::SBinding binding[SCAN_DESCRIPTOR_COUNT];

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

			scanDSLayout = logicalDevice->createGPUDescriptorSetLayout(binding, binding + SCAN_DESCRIPTOR_COUNT);
		}

		core::smart_refctd_ptr<video::IGPUPipelineLayout> scanPipelineLayout = nullptr;
		{
			asset::SPushConstantRange pcRange;
			pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
			pcRange.offset = 0u;
			pcRange.size = sizeof(video::CScanner::DefaultPushConstants);
			scanPipelineLayout = logicalDevice->createGPUPipelineLayout(&pcRange, &pcRange+1ull, core::smart_refctd_ptr(scanDSLayout));
		}

		const char* scanShaderPath = "../scan.comp";
		{
			auto specShader = createShader(scanShaderPath);
			scanPipeline = logicalDevice->createGPUComputePipeline(nullptr, std::move(scanPipelineLayout), std::move(specShader));
		}

		// create & update scan ds
		const uint32_t scanDSCount = SC_IMG_COUNT;
		auto scanDSPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &scanDSLayout.get(), &scanDSLayout.get() + 1ull, &scanDSCount);
		for (uint32_t scImageIndex = 0u; scImageIndex < SC_IMG_COUNT; ++scImageIndex)
		{
			scanDS[scImageIndex] = logicalDevice->createGPUDescriptorSet(scanDSPool.get(), core::smart_refctd_ptr(scanDSLayout));

			video::IGPUDescriptorSet::SWriteDescriptorSet writes[SCAN_DESCRIPTOR_COUNT];

			for (uint32_t i = 0u; i < SCAN_DESCRIPTOR_COUNT; ++i)
			{
				writes[i].dstSet = scanDS[scImageIndex].get();
				writes[i].binding = i;
				writes[i].arrayElement = 0u;
				writes[i].count = 1u;
			}

			// light grid
			writes[0].descriptorType = asset::EDT_STORAGE_IMAGE;
			// scratch
			writes[1].descriptorType = asset::EDT_STORAGE_BUFFER;

			video::IGPUDescriptorSet::SDescriptorInfo infos[SCAN_DESCRIPTOR_COUNT];

			infos[0].image.imageLayout = asset::EIL_GENERAL;
			infos[0].image.sampler = nullptr;
			infos[0].desc = lightGridTextureView;

			infos[1].buffer.offset = 0ull;
			infos[1].buffer.size = scanScratchGPUBuffer->getCachedCreationParams().declaredSize;
			infos[1].desc = scanScratchGPUBuffer;

			writes[0].info = &infos[0];
			writes[1].info = &infos[1];
			logicalDevice->updateDescriptorSets(SCAN_DESCRIPTOR_COUNT, writes, 0u, nullptr);
		}

#ifdef CLIPMAP
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

				infos[1].desc = lightGridTextureView;
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

#endif
		
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
				FATAL_LOG("Failed to create CPU DS Layout for light resources!\n");

			cpu2gpuParams.beginCommandBuffers();
			auto gpuArray = cpu2gpu.getGPUObjectsFromAssets(&lightDSLayout_cpu, &lightDSLayout_cpu + 1, cpu2gpuParams);
			if (!gpuArray || gpuArray->size() < 1u || !(*gpuArray)[0])
				FATAL_LOG("Failed to convert Light CPU DS layout to GPU DS layout!\n");
			cpu2gpuParams.waitForCreationToComplete();
			core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> dsLayout_gpu = (*gpuArray)[0];

			const uint32_t setCount = 1u;
			core::smart_refctd_ptr<video::IDescriptorPool> pool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &dsLayout_gpu.get(), &dsLayout_gpu.get() + 1ull, &setCount);
			lightDS = logicalDevice->createGPUDescriptorSet(pool.get(), core::smart_refctd_ptr(dsLayout_gpu));
			if (!lightDS)
				FATAL_LOG("Failed to create Light GPU DS!\n");

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

			// light index list
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

				info[1].image.imageLayout = asset::EIL_GENERAL;// Todo(achal): asset::EIL_SHADER_READ_ONLY_OPTIMAL;
				info[1].image.sampler = nullptr;
				info[1].desc = lightGridTextureView;

				info[2].desc = lightIndexListSSBOView;
				info[2].buffer.offset = 0ull;
				info[2].buffer.size = lightIndexListSSBOView->getByteSize();
			}
			write[0].info = info;
			write[1].info = info + 1;
			write[2].info = info + 2;
			logicalDevice->updateDescriptorSets(LIGHTING_DESCRIPTOR_COUNT, write, 0u, nullptr);
		}

		// load in the mesh
		asset::SAssetBundle meshesBundle;
		loadModel(sharedInputCWD / "sponza.zip", sharedInputCWD / "sponza.zip/sponza.obj", meshesBundle);

		core::smart_refctd_ptr<asset::ICPUMesh> cpuMesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(meshesBundle.getContents().begin()[0]);
		const asset::COBJMetadata* metadataOBJ = meshesBundle.getMetadata()->selfCast<const asset::COBJMetadata>();
		setupCameraUBO(cpuMesh, metadataOBJ);

		core::smart_refctd_ptr<video::IGPUMesh> gpuMesh = convertCPUMeshToGPU(cpuMesh.get(), lightDSLayout_cpu.get());
		if (!gpuMesh)
			FATAL_LOG("Failed to convert mesh to GPU objects!\n");

		logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_SECONDARY, 1u, &lightingCommandBuffer);
		{
			core::vector<core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> graphicsPipelines(gpuMesh->getMeshBuffers().size());
			createLightingGraphicsPipelines(graphicsPipelines, gpuMesh.get());
			if (!bakeSecondaryCommandBufferForSubpass(LIGHTING_PASS_INDEX, lightingCommandBuffer.get(), graphicsPipelines, gpuMesh.get()))
				FATAL_LOG("Failed to create lighting pass command buffer!");
		}

		logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_SECONDARY, 1u, &zPrepassCommandBuffer);
		{
			core::vector<core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> graphicsPipelines(gpuMesh->getMeshBuffers().size());
			createZPrePassGraphicsPipelines(graphicsPipelines, gpuMesh.get());
			if (!bakeSecondaryCommandBufferForSubpass(Z_PREPASS_INDEX, zPrepassCommandBuffer.get(), graphicsPipelines, gpuMesh.get()))
				FATAL_LOG("Failed to create depth pre-pass command buffer!");
		}

		logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			imageAcquire[i] = logicalDevice->createSemaphore();
			renderFinished[i] = logicalDevice->createSemaphore();
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
			mouse.consumeEvents([&](const ui::IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
			keyboard.consumeEvents(
				[&](const ui::IKeyboardEventChannel::range_t& events) -> void
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

		video::IGPUCommandBuffer::SImageMemoryBarrier toGeneral = {};
		toGeneral.barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0u);
		toGeneral.barrier.dstAccessMask = asset::EAF_SHADER_WRITE_BIT;
		toGeneral.oldLayout = asset::EIL_UNDEFINED;
		toGeneral.newLayout = asset::EIL_GENERAL;
		toGeneral.srcQueueFamilyIndex = ~0u;
		toGeneral.dstQueueFamilyIndex = ~0u;
		toGeneral.image = lightGridTexture;
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

		// first pass, against level 1
		commandBuffer->bindDescriptorSets(asset::EPBP_COMPUTE, octreeFirstCullPipeline->getLayout(), 0u, 1u, &octreeFirstCullDS[acquiredNextFBO].get());
		const core::vectorSIMDf& cameraPosition = camera.getPosition();
		{
			octree_first_cull_push_constants_t pushConstants = {};
			pushConstants.camPosClipmapExtent[0] = cameraPosition.x;
			pushConstants.camPosClipmapExtent[1] = cameraPosition.y;
			pushConstants.camPosClipmapExtent[2] = cameraPosition.z;
			pushConstants.camPosClipmapExtent[3] = genesisVoxelExtent;
			pushConstants.lightCount = LIGHT_COUNT;
			pushConstants.buildHistogramID = buildHistogramID; buildHistogramID ^= 0x1u;
			commandBuffer->pushConstants(octreeFirstCullPipeline->getLayout(), video::IGPUShader::ESS_COMPUTE, 0u, sizeof(octree_first_cull_push_constants_t), &pushConstants);
		}
		commandBuffer->bindComputePipeline(octreeFirstCullPipeline.get());
		commandBuffer->dispatch((LIGHT_COUNT + WG_DIM - 1) / WG_DIM, 1u, 1u);

		// auto firstPassOutScratchBuffer = recordDownloadGPUBufferCommands(octreeScratchBuffers[acquiredNextFBO][0], commandBuffer.get(), logicalDevice.get());

		// Todo(achal): Wrap the following two barriers into 1 pipeline barrier call
		video::IGPUCommandBuffer::SBufferMemoryBarrier scratchUpdatedBarrier = {};
		scratchUpdatedBarrier.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
		scratchUpdatedBarrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
		scratchUpdatedBarrier.srcQueueFamilyIndex = ~0u;
		scratchUpdatedBarrier.dstQueueFamilyIndex = ~0u;
		scratchUpdatedBarrier.buffer = octreeScratchBuffers[acquiredNextFBO][0];
		scratchUpdatedBarrier.offset = 0ull;
		scratchUpdatedBarrier.size = scratchUpdatedBarrier.buffer->getCachedCreationParams().declaredSize;
		commandBuffer->pipelineBarrier(
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EDF_BY_REGION_BIT,
			0u, nullptr,
			1u, &scratchUpdatedBarrier,
			0u, nullptr);

		// memory dependency for histogram
		video::IGPUCommandBuffer::SBufferMemoryBarrier histogramUpdatedBarrier = {};
		histogramUpdatedBarrier.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
		histogramUpdatedBarrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
		histogramUpdatedBarrier.srcQueueFamilyIndex = ~0u;
		histogramUpdatedBarrier.dstQueueFamilyIndex = ~0u;
		histogramUpdatedBarrier.buffer = importanceHistogramBuffer[acquiredNextFBO];
		histogramUpdatedBarrier.offset = 0ull;
		histogramUpdatedBarrier.size = histogramUpdatedBarrier.buffer->getCachedCreationParams().declaredSize;
		commandBuffer->pipelineBarrier(
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EDF_BY_REGION_BIT,
			0u, nullptr,
			1u, &histogramUpdatedBarrier,
			0u, nullptr);

		// next pass (against level 2)
		commandBuffer->bindDescriptorSets(asset::EPBP_COMPUTE, octreeIntermediateCullPipeline->getLayout(), 0u, 1u, &octreeIntermediateCullDS[acquiredNextFBO][0].get());
		{
			octree_intermediate_cull_push_constants_t pc = {};
			pc.camPosClipmapExtent[0] = cameraPosition.x;
			pc.camPosClipmapExtent[1] = cameraPosition.y;
			pc.camPosClipmapExtent[2] = cameraPosition.z;
			pc.camPosClipmapExtent[3] = genesisVoxelExtent;
			pc.hierarchyLevel = 2u;
			commandBuffer->pushConstants(octreeIntermediateCullPipeline->getLayout(), video::IGPUShader::ESS_COMPUTE, 0u, sizeof(octree_intermediate_cull_push_constants_t), &pc);
		}
		commandBuffer->bindComputePipeline(octreeIntermediateCullPipeline.get());
		{
			// In cases of severe underflow this might reduce performance since, in that case, we might have way less
			// intersection records to process than the number of invocations we're launching. Other alternatives include:
			//		1. Download the previous pass's outScratch and get its count
			//		2. Somehow setup an indirect dispatch in the previous pass --doesn't seem possible with single-buffering,
			//		might be possible with double or triple-buffering.
			constexpr uint32_t MAX_INVOCATIONS = MEMORY_BUDGET / sizeof(uint64_t);
			commandBuffer->dispatch((MAX_INVOCATIONS + WG_DIM - 1)/WG_DIM, 1u, 1u);
		}

		// auto secondPassOutScratchBuffer = recordDownloadGPUBufferCommands(octreeScratchBuffers[acquiredNextFBO][1], commandBuffer.get(), logicalDevice.get());

		// next pass (against level 3)
		scratchUpdatedBarrier.buffer = octreeScratchBuffers[acquiredNextFBO][1];
		commandBuffer->pipelineBarrier(
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EDF_BY_REGION_BIT,
			0u, nullptr,
			1u, &scratchUpdatedBarrier,
			0u, nullptr);

		commandBuffer->bindDescriptorSets(asset::EPBP_COMPUTE, octreeIntermediateCullPipeline->getLayout(), 0u, 1u, &octreeIntermediateCullDS[acquiredNextFBO][1].get());
		{
			octree_intermediate_cull_push_constants_t pc = {};
			pc.camPosClipmapExtent[0] = cameraPosition.x;
			pc.camPosClipmapExtent[1] = cameraPosition.y;
			pc.camPosClipmapExtent[2] = cameraPosition.z;
			pc.camPosClipmapExtent[3] = genesisVoxelExtent;
			pc.hierarchyLevel = 3u;
			commandBuffer->pushConstants(octreeIntermediateCullPipeline->getLayout(), video::IGPUShader::ESS_COMPUTE, 0u, sizeof(octree_intermediate_cull_push_constants_t), &pc);
		}
		commandBuffer->bindComputePipeline(octreeIntermediateCullPipeline.get());
		{
			constexpr uint32_t MAX_INVOCATIONS = MEMORY_BUDGET / sizeof(uint64_t);
			commandBuffer->dispatch((MAX_INVOCATIONS + WG_DIM - 1) / WG_DIM, 1u, 1u);
		}

		// next pass (against level 4)
		scratchUpdatedBarrier.buffer = octreeScratchBuffers[acquiredNextFBO][0];
		commandBuffer->pipelineBarrier(
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EDF_BY_REGION_BIT,
			0u, nullptr,
			1u, &scratchUpdatedBarrier,
			0u, nullptr);
		commandBuffer->bindDescriptorSets(asset::EPBP_COMPUTE, octreeIntermediateCullPipeline->getLayout(), 0u, 1u, &octreeIntermediateCullDS[acquiredNextFBO][0].get());
		{
			octree_intermediate_cull_push_constants_t pc = {};
			pc.camPosClipmapExtent[0] = cameraPosition.x;
			pc.camPosClipmapExtent[1] = cameraPosition.y;
			pc.camPosClipmapExtent[2] = cameraPosition.z;
			pc.camPosClipmapExtent[3] = genesisVoxelExtent;
			pc.hierarchyLevel = 4u;
			commandBuffer->pushConstants(octreeIntermediateCullPipeline->getLayout(), video::IGPUShader::ESS_COMPUTE, 0u, sizeof(octree_intermediate_cull_push_constants_t), &pc);
		}
		commandBuffer->bindComputePipeline(octreeIntermediateCullPipeline.get());
		{
			constexpr uint32_t MAX_INVOCATIONS = MEMORY_BUDGET / sizeof(uint64_t);
			commandBuffer->dispatch((MAX_INVOCATIONS + WG_DIM - 1) / WG_DIM, 1u, 1u);
		}

		// next pass (against level 5)
		scratchUpdatedBarrier.buffer = octreeScratchBuffers[acquiredNextFBO][1];
		commandBuffer->pipelineBarrier(
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EDF_BY_REGION_BIT,
			0u, nullptr,
			1u, &scratchUpdatedBarrier,
			0u, nullptr);
		commandBuffer->bindDescriptorSets(asset::EPBP_COMPUTE, octreeIntermediateCullPipeline->getLayout(), 0u, 1u, &octreeIntermediateCullDS[acquiredNextFBO][1].get());
		{
			octree_intermediate_cull_push_constants_t pc = {};
			pc.camPosClipmapExtent[0] = cameraPosition.x;
			pc.camPosClipmapExtent[1] = cameraPosition.y;
			pc.camPosClipmapExtent[2] = cameraPosition.z;
			pc.camPosClipmapExtent[3] = genesisVoxelExtent;
			pc.hierarchyLevel = 5u;
			commandBuffer->pushConstants(octreeIntermediateCullPipeline->getLayout(), video::IGPUShader::ESS_COMPUTE, 0u, sizeof(octree_intermediate_cull_push_constants_t), &pc);
		}
		commandBuffer->bindComputePipeline(octreeIntermediateCullPipeline.get());
		{
			constexpr uint32_t MAX_INVOCATIONS = MEMORY_BUDGET / sizeof(uint64_t);
			commandBuffer->dispatch((MAX_INVOCATIONS + WG_DIM - 1) / WG_DIM, 1u, 1u);
		}

		// Todo(achal): I would need a different set of command buffers allocated from a pool which utilizes
		// the compute queue, then I would also need to do queue ownership transfers, most
		// likely with a pipelineBarrier
		{
			asset::SClearColorValue lightGridClearValue = { 0 };
			asset::IImage::SSubresourceRange range;
			range.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
			range.baseMipLevel = 0u;
			range.levelCount = 1u;
			range.baseArrayLayer = 0u;
			range.layerCount = 1u;
			commandBuffer->clearColorImage(lightGridTexture.get(), asset::EIL_GENERAL, &lightGridClearValue, 1u, &range);
		}

		// memory dependency to ensure the light grid texture is cleared via clearColorImage
		video::IGPUCommandBuffer::SImageMemoryBarrier lightGridUpdated = {};
		lightGridUpdated.barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
		lightGridUpdated.barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_READ_BIT | asset::EAF_SHADER_WRITE_BIT);
		lightGridUpdated.oldLayout = asset::EIL_GENERAL;
		lightGridUpdated.newLayout = asset::EIL_GENERAL;
		lightGridUpdated.srcQueueFamilyIndex = ~0u;
		lightGridUpdated.dstQueueFamilyIndex = ~0u;
		lightGridUpdated.image = lightGridTexture;
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
		
#ifdef CLIPMAP
		commandBuffer->bindComputePipeline(cullPipeline.get());
		float pushConstants[4] = { cameraPosition.x, cameraPosition.y, cameraPosition.z, clipmapExtent };
		commandBuffer->pushConstants(cullPipeline->getLayout(), asset::IShader::ESS_COMPUTE, 0u, 4 * sizeof(float), pushConstants);
		commandBuffer->bindDescriptorSets(asset::EPBP_COMPUTE, cullPipeline->getLayout(), 0u, 1u, &cullDs.get());
		const uint32_t wgCount = (LIGHT_COUNT + WG_DIM - 1) / WG_DIM;
		commandBuffer->dispatch(wgCount, 1u, 1u);
#endif

#ifdef OCTREE
		// final pass (against level 6)
		scratchUpdatedBarrier.buffer = octreeScratchBuffers[acquiredNextFBO][0];
		commandBuffer->pipelineBarrier(
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EDF_BY_REGION_BIT,
			0u, nullptr,
			1u, &scratchUpdatedBarrier,
			0u, nullptr);
		commandBuffer->bindDescriptorSets(asset::EPBP_COMPUTE, octreeLastCullPipeline->getLayout(), 0u, 1u, &octreeLastCullDS[acquiredNextFBO].get());
		{
			octree_intermediate_cull_push_constants_t pc = {};
			pc.camPosClipmapExtent[0] = cameraPosition.x;
			pc.camPosClipmapExtent[1] = cameraPosition.y;
			pc.camPosClipmapExtent[2] = cameraPosition.z;
			pc.camPosClipmapExtent[3] = genesisVoxelExtent;
			pc.hierarchyLevel = 6u;
			commandBuffer->pushConstants(octreeLastCullPipeline->getLayout(), video::IGPUShader::ESS_COMPUTE, 0u, sizeof(octree_intermediate_cull_push_constants_t), &pc);
		}
		commandBuffer->bindComputePipeline(octreeLastCullPipeline.get());
		{
			constexpr uint32_t MAX_INVOCATIONS = MEMORY_BUDGET / sizeof(uint64_t);
			commandBuffer->dispatch((MAX_INVOCATIONS + WG_DIM - 1) / WG_DIM, 1u, 1u);
		}
#endif

		video::CScanner* scanner = utilities->getDefaultScanner();

		commandBuffer->fillBuffer(scanScratchGPUBuffer.get(), 0u, sizeof(uint32_t) + scanScratchGPUBuffer->getCachedCreationParams().declaredSize / 2u, 0u);
		commandBuffer->bindComputePipeline(scanPipeline.get());
		commandBuffer->bindDescriptorSets(asset::EPBP_COMPUTE, scanPipeline->getLayout(), 0u, 1u, &scanDS[acquiredNextFBO].get());

		// buffer memory dependency to ensure part of scratch buffer for scan is cleared
		video::IGPUCommandBuffer::SBufferMemoryBarrier scanScratchUpdated = {};
		scanScratchUpdated.barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
		scanScratchUpdated.barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_READ_BIT | asset::EAF_SHADER_WRITE_BIT);
		scanScratchUpdated.srcQueueFamilyIndex = ~0u;
		scanScratchUpdated.dstQueueFamilyIndex = ~0u;
		scanScratchUpdated.buffer = scanScratchGPUBuffer;
		scanScratchUpdated.offset = 0ull;
		scanScratchUpdated.size = scanScratchGPUBuffer->getCachedCreationParams().declaredSize;

		commandBuffer->pipelineBarrier(
			asset::EPSF_TRANSFER_BIT,
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EDF_NONE,
			0u, nullptr,
			1u, &scanScratchUpdated,
			0u, nullptr);
		
		// image memory dependency to ensure that the previous pass has finished writing to the
		// light grid before the scan pass can read from it, we only need this dependency
		// for the light grid so not using a global memory barrier here
		lightGridUpdated.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
		lightGridUpdated.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;

		commandBuffer->pipelineBarrier(
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EDF_NONE,
			0u, nullptr,
			0u, nullptr,
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
		
#ifdef CLIPMAP
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
		// YES
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
			commandBuffer->executeCommands(1u, &lightingCommandBuffer.get());
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

		core::vectorSIMDf levelMinVertex(-genesisVoxelExtent / 2.f, -genesisVoxelExtent / 2.f, -genesisVoxelExtent / 2.f);
		const float voxelSideLength = genesisVoxelExtent / 2.f;

		auto getAABBFromLocalClusterID = [&cameraPosition, &levelMinVertex](uint32_t x, uint32_t y, uint32_t z, float voxelSideLength) -> nbl_glsl_shapes_AABB_t
		{
			core::vectorSIMDf minVertex = cameraPosition + levelMinVertex + core::vectorSIMDf(x, y, z) * voxelSideLength;
			core::vectorSIMDf maxVertex = cameraPosition + levelMinVertex + core::vectorSIMDf(x + 1, y + 1, z + 1) * voxelSideLength;

			nbl_glsl_shapes_AABB_t result;
			result.minVx = { minVertex.x, minVertex.y, minVertex.z };
			result.maxVx = { maxVertex.x, maxVertex.y, maxVertex.z };

			return result;
		};

#if 0
		struct intersection_record_t
		{
			uint32_t globalLightIndex;
			uint32_t localClusterID[3];
		};

		logicalDevice->blockForFences(1u, &fence.get());

		uint32_t* firstPassOutScratchData = nullptr;
		{
			video::IDriverMemoryAllocation::MappedMemoryRange mappedRange = {};
			mappedRange.memory = firstPassOutScratchBuffer->getBoundMemory();
			mappedRange.offset = 0ull;
			mappedRange.length = firstPassOutScratchBuffer->getCachedCreationParams().declaredSize;
			void* mappedPtr = logicalDevice->mapMemory(mappedRange);

			firstPassOutScratchData = reinterpret_cast<uint32_t*>(mappedPtr);
		}

		uint32_t* secondPassOutScratchData = nullptr;
		{
			video::IDriverMemoryAllocation::MappedMemoryRange mappedRange = {};
			mappedRange.memory = secondPassOutScratchBuffer->getBoundMemory();
			mappedRange.offset = 0ull;
			mappedRange.length = secondPassOutScratchBuffer->getCachedCreationParams().declaredSize;
			void* mappedPtr = logicalDevice->mapMemory(mappedRange);

			secondPassOutScratchData = reinterpret_cast<uint32_t*>(mappedPtr);
		}

		const uint32_t intersectionRecordCount = *firstPassOutScratchData++;
		firstPassOutScratchData++; // ignore padding

		const uint32_t intersectionRecordCount_2 = *secondPassOutScratchData++;
		secondPassOutScratchData++; // ignore padding


		printf("After first pass, intersection record count: %d\n", intersectionRecordCount);

		uint32_t intersectionRecordCount_nextpass = 0u;
		for (uint32_t i = 0u; i < intersectionRecordCount; ++i)
		{
			const uint32_t packedX = *firstPassOutScratchData++;
			const uint32_t packedY = *firstPassOutScratchData++;

			intersection_record_t gpuRecord;
			gpuRecord.globalLightIndex = (packedY >> 12) & 0x3FFFFF;
			gpuRecord.localClusterID[0] = packedX & 0x7u;
			gpuRecord.localClusterID[1] = (packedX >> 7) & 0x7u;
			gpuRecord.localClusterID[2] = (packedX >> 14) & 0x7u;

			const float voxelSideLength = genesisVoxelExtent / 4.f;

			const auto parentVoxel = getAABBFromLocalClusterID(gpuRecord.localClusterID[0], gpuRecord.localClusterID[1], gpuRecord.localClusterID[2], genesisVoxelExtent / 2.f);

			nbl_glsl_shapes_AABB_t nodes[2][2][2];
			for (uint32_t z = 0u; z < 2u; ++z)
			{
				for (uint32_t y = 0u; y < 2u; ++y)
				{
					for (uint32_t x = 0u; x < 2u; ++x)
					{
						nodes[z][y][x] = getAABBFromLocalClusterID((gpuRecord.localClusterID[0] << 1) + x, (gpuRecord.localClusterID[1] << 1) + y, (gpuRecord.localClusterID[2] << 1) + z, voxelSideLength);
						const auto& aabb = nodes[z][y][x];

						if (doesConeIntersectAABB(getLightVolume(lights[gpuRecord.globalLightIndex]), nodes[z][y][x]))
							++intersectionRecordCount_nextpass;
					}
				}
			}
		}

		printf("After the next level traversal: %d\n", intersectionRecordCount_nextpass);
		printf("On the GPU: %d\n", intersectionRecordCount_2);
		__debugbreak();	
#endif
	}

	bool keepRunning() override
	{
		return windowCb->isWindowOpen();
	}

private:
	// Only for buffers written by a compute shader
	core::smart_refctd_ptr<video::IGPUBuffer> recordDownloadGPUBufferCommands(core::smart_refctd_ptr<video::IGPUBuffer> bufferToDownload, video::IGPUCommandBuffer* commandBuffer, video::ILogicalDevice* logicalDevice)
	{
		const size_t downloadSize = bufferToDownload->getCachedCreationParams().declaredSize;

		// need a memory dependency here to ensure the compute dispatch has finished
		video::IGPUCommandBuffer::SBufferMemoryBarrier downloadReadyBarrier = {};
		downloadReadyBarrier.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
		downloadReadyBarrier.barrier.dstAccessMask = asset::EAF_TRANSFER_READ_BIT;
		downloadReadyBarrier.srcQueueFamilyIndex = ~0u;
		downloadReadyBarrier.dstQueueFamilyIndex = ~0u;
		downloadReadyBarrier.buffer = bufferToDownload;
		downloadReadyBarrier.offset = 0ull;
		downloadReadyBarrier.size = downloadSize;

		commandBuffer->pipelineBarrier(
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EPSF_TRANSFER_BIT,
			asset::EDF_BY_REGION_BIT,
			0u, nullptr,
			1u, &downloadReadyBarrier,
			0u, nullptr);

		video::IGPUBuffer::SCreationParams creationParams = {};
		creationParams.usage = video::IGPUBuffer::EUF_TRANSFER_DST_BIT;

		core::smart_refctd_ptr<video::IGPUBuffer> downloadBuffer = logicalDevice->createGPUBuffer(creationParams, downloadSize);

		video::IDriverMemoryBacked::SDriverMemoryRequirements memReqs = {};
		memReqs.vulkanReqs.alignment = downloadBuffer->getMemoryReqs().vulkanReqs.alignment;
		memReqs.vulkanReqs.size = downloadBuffer->getMemoryReqs().vulkanReqs.size;
		memReqs.vulkanReqs.memoryTypeBits = downloadBuffer->getMemoryReqs().vulkanReqs.memoryTypeBits;

		memReqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL;
		memReqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ;
		memReqs.prefersDedicatedAllocation = downloadBuffer->getMemoryReqs().prefersDedicatedAllocation;
		memReqs.requiresDedicatedAllocation = downloadBuffer->getMemoryReqs().requiresDedicatedAllocation;
		auto downloadGPUBufferMemory = logicalDevice->allocateGPUMemory(memReqs);

		video::ILogicalDevice::SBindBufferMemoryInfo bindBufferInfo = {};
		bindBufferInfo.buffer = downloadBuffer.get();
		bindBufferInfo.memory = downloadGPUBufferMemory.get();
		bindBufferInfo.offset = 0ull;
		logicalDevice->bindBufferMemory(1u, &bindBufferInfo);

		asset::SBufferCopy copyRegion = {};
		copyRegion.srcOffset = 0ull;
		copyRegion.dstOffset = 0ull;
		copyRegion.size = downloadSize;
		commandBuffer->copyBuffer(bufferToDownload.get(), downloadBuffer.get(), 1u, &copyRegion);

		return downloadBuffer;
	};

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
	}

	// Returns true if the point lies in negative-half-space (space in which the normal to the plane isn't present) of the plane
	// Point on the plane returns true.
	// In other words, if you hold the plane such that its normal points towards your face, then this
	// will return true if the point is "behind" the plane (or farther from your face than the plane's surface)
	bool isPointBehindPlane(const core::vector3df_SIMD& point, const core::plane3dSIMDf& plane)
	{
		// As an optimization we can add an epsilon to 0, to ignore cones which have a
		// very very small intersecting region with the AABB, could help with FP precision
		// too when the point is on the plane
		const float result = (core::dot(point, plane.getNormal()).x + plane.getDistance());
		return  result <= 0.f /* + EPSILON*/;
	}

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

			const bool isTipBehindPlane = isPointBehindPlane(cone.tip, planes[i]);
			const bool isFBPBehindPlane = isPointBehindPlane(farthestBasePoint, planes[i]);

			// There are two edge cases here:
			// 1. When cone's direction and plane's normal are anti-parallel
			//		There is no reason to check farthestBasePoint in this case, because cone's tip is the farthest point!
			//		But there is no harm in doing so.
			// 2. When cone's direction and plane's normal are parallel
			//		This edge case will get handled nicely by the farthestBasePoint coming as center of the base of the cone itself
			// if (isPointBehindPlane(cone.tip, planes[i], printMoreStuff) && isPointBehindPlane(farthestBasePoint, planes[i], printMoreStuff))
			if (isTipBehindPlane && isFBPBehindPlane)
				return false;
		}

		return true;
	};

	bool bakeSecondaryCommandBufferForSubpass(
		const uint32_t subpass,
		video::IGPUCommandBuffer* cmdbuf,
		const core::vector<core::smart_refctd_ptr<video::IGPUGraphicsPipeline>>& graphicsPipelines,
		video::IGPUMesh* gpuMesh)
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
		int16_t secondComp((encoded >> 16) & 0xFFFFu);
		result.x = core::clamp(firstComp / 32727.f, -1.f, 1.f);
		result.y = core::clamp(secondComp / 32727.f, -1.f, 1.f);

		return result;
	};

	inline float getLightImportanceMagnitude(const nbl_glsl_ext_ClusteredLighting_SpotLight& light, const core::vectorSIMDf& cameraPosition)
	{
		core::vectorSIMDf intensity;
		{
			uint64_t packedIntensity = light.intensity.y;
			packedIntensity = (packedIntensity << 32) | light.intensity.x;
			core::rgb32f result = core::rgb19e7_to_rgb32f(packedIntensity);
			intensity.r = result.x;
			intensity.g = result.y;
			intensity.b = result.z;
		}

		const core::vectorSIMDf lightToCamera = cameraPosition - core::vectorSIMDf(light.position.x, light.position.y, light.position.z);
		const float lenSq = core::dot(lightToCamera, lightToCamera).x;
		const float radiusSq = LIGHT_RADIUS * LIGHT_RADIUS;
		const float attenuation = 0.5f * radiusSq * (1.f - core::inversesqrt(1.f + radiusSq / lenSq));
		const core::vectorSIMDf importance = intensity * attenuation;
		return core::sqrt(core::dot(importance, importance).x);
	};

	inline void generateLights(const uint32_t lightCount)
	{
		const float cosOuterHalfAngle = core::cos(core::radians(2.38f));
		const float cosInnerHalfAngle = core::cos(core::radians(0.5f));
		const float cosineRange = (cosInnerHalfAngle - cosOuterHalfAngle);

		uvec2 lightDirection_encoded;
		{
			core::vectorSIMDf lightDirection(0.045677, 0.032760, -0.998440, 0.001499); // normalized!

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

		core::vectorSIMDf bottomRight(-809.f, 32.317501f, -34.f, 1.f);
		core::vectorSIMDf topLeft(964.f, 1266.120117f, -34.f, 1.f);
		const core::vectorSIMDf displacementBetweenLights(25.f, 25.f, 0.f);

		const uint32_t gridDimX = core::floor((topLeft.x - bottomRight.x)/displacementBetweenLights.x);
		const uint32_t gridDimY = core::floor((topLeft.y - bottomRight.y)/displacementBetweenLights.y);

		const uint32_t maxLightCount = 2u * gridDimX * gridDimY;

		uint32_t actualLightCount = lightCount;
		if (actualLightCount > maxLightCount)
		{
			logger->log("Hey I know you asked for %d lights but I can't do that yet while still making sure"
				"the lights have such an arrangment which facilitates easy catching of any artifacts.\n"
				"So I'm generating only %d lights..\n", system::ILogger::ELL_PERFORMANCE, lightCount, maxLightCount);

			__debugbreak();

			actualLightCount = maxLightCount;
		}

		lights.resize(actualLightCount);

		// Grid #1
		for (uint32_t y = 0u; y < gridDimY; ++y)
		{
			for (uint32_t x = 0u; x < gridDimX; ++x)
			{
				const uint32_t absLightIdx = y * gridDimX + x;
				if (absLightIdx < actualLightCount)
				{
					nbl_glsl_ext_ClusteredLighting_SpotLight& light = lights[absLightIdx];

					const core::vectorSIMDf pos = bottomRight + displacementBetweenLights * core::vectorSIMDf(x, y);

					light.position = { pos.x, pos.y, pos.z };
					light.direction = lightDirection_encoded;
					light.outerCosineOverCosineRange = cosOuterHalfAngle / cosineRange;
					light.intensity = lightIntensity_encoded;
				}
			}
		}

		// Move to the next grid with same number of rows, but flipped light direction and
		// a different startPoint
		bottomRight.z = 3.f;
		topLeft.z = 3.f;
		lightDirection_encoded = {};
		{
			const core::vectorSIMDf lightDirection = core::vectorSIMDf(0.045677, 0.032760, 0.998440, 0.001499); // normalized
			lightDirection_encoded.x = packSnorm2x16(lightDirection.x, lightDirection.y);
			lightDirection_encoded.y = packSnorm2x16(lightDirection.z, cosineRange);
		}

		// Grid #2
		for (uint32_t y = 0u; y < gridDimY; ++y)
		{
			for (uint32_t x = 0u; x < gridDimX; ++x)
			{
				const uint32_t absLightIdx = (gridDimX * gridDimY) + (y * gridDimX) + x;
				if (absLightIdx < actualLightCount)
				{
					nbl_glsl_ext_ClusteredLighting_SpotLight& light = lights[absLightIdx];

					const core::vectorSIMDf pos = bottomRight + displacementBetweenLights * core::vectorSIMDf(x, y);
					light.position = { pos.x, pos.y, pos.z };
					light.direction = lightDirection_encoded;
					light.outerCosineOverCosineRange = cosOuterHalfAngle / cosineRange;
					light.intensity = lightIntensity_encoded;
				}
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

	inline void setupCameraUBO(core::smart_refctd_ptr<asset::ICPUMesh> cpuMesh, const asset::COBJMetadata* metadataOBJ)
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
				FATAL_LOG("Failed to convert Camera CPU DS to GPU DS!\n");
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

		video::IDriverMemoryBacked::SDriverMemoryRequirements ubomemreq = logicalDevice->getDeviceLocalGPUMemoryReqs();
		ubomemreq.vulkanReqs.size = uboSize;
		video::IGPUBuffer::SCreationParams gpuuboCreationParams;
		gpuuboCreationParams.usage = static_cast<asset::IBuffer::E_USAGE_FLAGS>(asset::IBuffer::EUF_UNIFORM_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT);
		gpuuboCreationParams.sharingMode = asset::E_SHARING_MODE::ESM_EXCLUSIVE;
		gpuuboCreationParams.queueFamilyIndexCount = 0u;
		gpuuboCreationParams.queueFamilyIndices = nullptr;
		cameraUbo = logicalDevice->createGPUBufferOnDedMem(gpuuboCreationParams, ubomemreq);
		const uint32_t dsCount = 1u;
		auto descriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &gpuDSLayout.get(), &gpuDSLayout.get() + 1ull, &dsCount);
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

	inline void loadModel(const system::path& archiveFilePath, const system::path& modelFilePath, asset::SAssetBundle& outMeshesBundle)
	{
		auto* quantNormalCache = assetManager->getMeshManipulator()->getQuantNormalCache();
		quantNormalCache->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), sharedOutputCWD / "normalCache101010.sse");

		auto fileArchive = system->openFileArchive(archiveFilePath);
		// test no alias loading (TODO: fix loading from absolute paths)
		system->mount(std::move(fileArchive));

		asset::IAssetLoader::SAssetLoadParams loadParams;
		loadParams.workingDirectory = sharedInputCWD;
		loadParams.logger = logger.get();
		outMeshesBundle = assetManager->getAsset(modelFilePath.string(), loadParams);
		if (outMeshesBundle.getContents().empty())
			FATAL_LOG("Failed to load the model!\n");

		quantNormalCache->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), sharedOutputCWD / "normalCache101010.sse");
	}

	inline core::smart_refctd_ptr<video::IGPUMesh> convertCPUMeshToGPU(asset::ICPUMesh* cpuMesh, asset::ICPUDescriptorSetLayout* lightCPUDSLayout)
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
			meshBuffer->getPipeline()->getLayout()->setDescriptorSetLayout(LIGHT_DS_NUMBER, core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(lightCPUDSLayout));
			meshBuffer->getPipeline()->setShaderAtStage(asset::IShader::ESS_VERTEX, vertSpecShader_cpu.get());
			meshBuffer->getPipeline()->setShaderAtStage(asset::IShader::ESS_FRAGMENT, fragSpecShader_cpu.get());

			// Todo(achal): Can get rid of this probably after
			// https://github.com/Devsh-Graphics-Programming/Nabla/pull/160#discussion_r747185441
			for (size_t i = 0ull; i < nbl::asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
				meshBuffer->getPipeline()->getBlendParams().blendParams[i].attachmentEnabled = (i == 0ull);

			meshBuffer->getPipeline()->getRasterizationParams().frontFaceIsCCW = false;
		}

		cpu2gpuParams.beginCommandBuffers();
		asset::ICPUMesh* meshRaw = static_cast<asset::ICPUMesh*>(cpuMesh);
		auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&meshRaw, &meshRaw + 1, cpu2gpuParams);
		if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
			return nullptr;

		cpu2gpuParams.waitForCreationToComplete();
		return (*gpu_array)[0];
	}

	inline void createZPrePassGraphicsPipelines(core::vector<core::smart_refctd_ptr<video::IGPUGraphicsPipeline>>& outPipelines, video::IGPUMesh* gpuMesh)
	{
		core::unordered_map<const void*, core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> pipelineCache;
		for (size_t i = 0ull; i < outPipelines.size(); ++i)
		{
			const video::IGPURenderpassIndependentPipeline* renderpassIndep = gpuMesh->getMeshBuffers()[i]->getPipeline();
			video::IGPURenderpassIndependentPipeline* renderpassIndep_mutable = const_cast<video::IGPURenderpassIndependentPipeline*>(renderpassIndep);

			auto foundPpln = pipelineCache.find(renderpassIndep);
			if (foundPpln == pipelineCache.end())
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
				foundPpln = pipelineCache.emplace_hint(foundPpln, renderpassIndep, logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(params)));
			}
			outPipelines[i] = foundPpln->second;
		}
	}

	inline void createLightingGraphicsPipelines(core::vector<core::smart_refctd_ptr<video::IGPUGraphicsPipeline>>& outPipelines, video::IGPUMesh* gpuMesh)
	{
		core::unordered_map<const void*, core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> pipelineCache;

		for (size_t i = 0ull; i < outPipelines.size(); ++i)
		{
			const video::IGPURenderpassIndependentPipeline* renderpassIndep = gpuMesh->getMeshBuffers()[i]->getPipeline();
			video::IGPURenderpassIndependentPipeline* renderpassIndep_mutable = const_cast<video::IGPURenderpassIndependentPipeline*>(renderpassIndep);

			auto& rasterizationParams_mutable = const_cast<asset::SRasterizationParams&>(renderpassIndep_mutable->getRasterizationParams());
			rasterizationParams_mutable.depthCompareOp = asset::ECO_GREATER_OR_EQUAL;

			auto foundPpln = pipelineCache.find(renderpassIndep);
			if (foundPpln == pipelineCache.end())
			{
				video::IGPUGraphicsPipeline::SCreationParams params;
				params.renderpassIndependent = core::smart_refctd_ptr<const video::IGPURenderpassIndependentPipeline>(renderpassIndep);
				params.renderpass = core::smart_refctd_ptr(renderpass);
				params.subpassIx = LIGHTING_PASS_INDEX;
				foundPpln = pipelineCache.emplace_hint(foundPpln, renderpassIndep, logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(params)));
			}
			outPipelines[i] = foundPpln->second;
		}
	}

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
	std::array<nbl::core::smart_refctd_ptr<video::IGPUFramebuffer>,CommonAPI::InitOutput::MaxSwapChainImageCount> fbos;
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
	core::smart_refctd_ptr<video::IGPUCommandBuffer> lightingCommandBuffer;
	core::smart_refctd_ptr<video::IGPUCommandBuffer> zPrepassCommandBuffer;

	const asset::COBJMetadata::CRenderpassIndependentPipeline* pipelineMetadata = nullptr;

	float genesisVoxelExtent;

	core::smart_refctd_ptr<video::IGPUBuffer> activeLightIndicesGPUBuffer = nullptr; // should be an input to the system

	core::smart_refctd_ptr<video::IGPUComputePipeline> octreeFirstCullPipeline = nullptr;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> octreeFirstCullDS[SC_IMG_COUNT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUBuffer> octreeScratchBuffers[SC_IMG_COUNT][2u];
	core::smart_refctd_ptr<video::IGPUBuffer> importanceHistogramBuffer[SC_IMG_COUNT];

	core::smart_refctd_ptr<video::IGPUComputePipeline> octreeIntermediateCullPipeline = nullptr;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> octreeIntermediateCullDS[SC_IMG_COUNT][2];

	core::smart_refctd_ptr<video::IGPUComputePipeline> octreeLastCullPipeline = nullptr;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> octreeLastCullDS[SC_IMG_COUNT] = { nullptr };

	uint32_t buildHistogramID = 0u;

#ifdef CLIPMAP
	core::smart_refctd_ptr<video::IGPUComputePipeline> cullPipeline = nullptr;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> cullDs = nullptr;
	core::smart_refctd_ptr<video::IGPUBuffer> intersectionRecordsGPUBuffer = nullptr;
	core::smart_refctd_ptr<video::IGPUBuffer> intersectionRecordCountGPUBuffer = nullptr;
	core::smart_refctd_ptr<video::IGPUBuffer> cullScratchGPUBuffer = nullptr;
	core::smart_refctd_ptr<video::IGPUComputePipeline> scatterPipeline = nullptr;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> scatterDS = nullptr;
#endif

	core::smart_refctd_ptr<video::IGPUImage> lightGridTexture = nullptr;
	core::smart_refctd_ptr<video::IGPUImageView> lightGridTextureView = nullptr;
	core::smart_refctd_ptr<video::IGPUBuffer> lightIndexListGPUBuffer = nullptr;
	core::smart_refctd_ptr<video::IGPUBufferView> lightIndexListSSBOView = nullptr;

	core::smart_refctd_ptr<video::IGPUDescriptorSet> scanDS[SC_IMG_COUNT] = { nullptr };
	video::CScanner::DefaultPushConstants scanPushConstants;
	video::CScanner::DispatchInfo scanDispatchInfo;
	core::smart_refctd_ptr<video::IGPUBuffer> scanScratchGPUBuffer = nullptr;
	core::smart_refctd_ptr<video::IGPUComputePipeline> scanPipeline = nullptr;

	core::vector<nbl_glsl_ext_ClusteredLighting_SpotLight> lights;
	// Todo(achal): Not make it global
	core::vector<uint32_t> activeLightIndices;

	// I need descriptor lifetime tracking!
	using PropertyPoolType = video::CPropertyPool<core::allocator, nbl_glsl_ext_ClusteredLighting_SpotLight>;
	core::smart_refctd_ptr<PropertyPoolType> propertyPool = nullptr;

	core::smart_refctd_ptr<video::IGPUBuffer> cameraUbo;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> cameraDS;
	uint32_t cameraUboBindingNumber;

	core::smart_refctd_ptr<video::IGPUDescriptorSet> lightDS = nullptr;

	video::CDumbPresentationOracle oracle;

	CommonAPI::InputSystem::ChannelReader<ui::IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<ui::IKeyboardEventChannel> keyboard;

	Camera camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());

	int32_t debugActiveLightIndex = -1;
#ifdef DEBUG_VIZ
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
#endif

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
		return apiConnection.get();
	}
	video::ILogicalDevice* getLogicalDevice()  override
	{
		return logicalDevice.get();
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
		return SC_IMG_COUNT;
	}
	virtual nbl::asset::E_FORMAT getDepthFormat() override
	{
		return nbl::asset::EF_D32_SFLOAT;
	}

	APP_CONSTRUCTOR(ClusteredRenderingSampleApp);
};

NBL_COMMON_API_MAIN(ClusteredRenderingSampleApp)

extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }