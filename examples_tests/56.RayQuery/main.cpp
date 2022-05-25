// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "../common/Camera.hpp"
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/video/utilities/CDumbPresentationOracle.h"

using namespace nbl;
using namespace core;
using namespace ui;


using namespace nbl;
using namespace core;
using namespace asset;
using namespace video;

smart_refctd_ptr<IGPUImageView> createHDRImageView(nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> device, asset::E_FORMAT colorFormat, uint32_t width, uint32_t height)
{
	smart_refctd_ptr<IGPUImageView> gpuImageViewColorBuffer;
	{
		IGPUImage::SCreationParams imgInfo;
		imgInfo.format = colorFormat;
		imgInfo.type = IGPUImage::ET_2D;
		imgInfo.extent.width = width;
		imgInfo.extent.height = height;
		imgInfo.extent.depth = 1u;
		imgInfo.mipLevels = 1u;
		imgInfo.arrayLayers = 1u;
		imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
		imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
		imgInfo.usage = core::bitflag(asset::IImage::EUF_STORAGE_BIT) | asset::IImage::EUF_TRANSFER_SRC_BIT;

		// (Erfan -> Cyprian)
		// auto image = device->createGPUImageOnDedMem(std::move(imgInfo),device->getDeviceLocalGPUMemoryReqs());
		auto image = device->createImage(std::move(imgInfo));
		auto imageMemoryReqs = image->getMemoryReqs();
		imageMemoryReqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits(); // getDeviceLocalMemoryTypeBits because of previous code getDeviceLocalGPUMemoryReqs
		auto imageMem = device->allocate(imageMemoryReqs, image.get());

		IGPUImageView::SCreationParams imgViewInfo;
		imgViewInfo.image = std::move(image);
		imgViewInfo.format = colorFormat;
		imgViewInfo.viewType = IGPUImageView::ET_2D;
		imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		imgViewInfo.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
		imgViewInfo.subresourceRange.baseArrayLayer = 0u;
		imgViewInfo.subresourceRange.baseMipLevel = 0u;
		imgViewInfo.subresourceRange.layerCount = 1u;
		imgViewInfo.subresourceRange.levelCount = 1u;

		gpuImageViewColorBuffer = device->createImageView(std::move(imgViewInfo));
	}

	return gpuImageViewColorBuffer;
}

struct ShaderParameters
{
	const uint32_t MaxDepthLog2 = 4; //5
	const uint32_t MaxSamplesLog2 = 10; //18
} kShaderParameters;

enum E_LIGHT_GEOMETRY
{
	ELG_SPHERE,
	ELG_TRIANGLE,
	ELG_RECTANGLE
};

struct DispatchInfo_t
{
	uint32_t workGroupCount[3];
};

_NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_WORK_GROUP_SIZE = 16u;

DispatchInfo_t getDispatchInfo(uint32_t imgWidth, uint32_t imgHeight) {
	DispatchInfo_t ret = {};
	ret.workGroupCount[0] = (uint32_t)core::ceil<float>((float)imgWidth / (float)DEFAULT_WORK_GROUP_SIZE);
	ret.workGroupCount[1] = (uint32_t)core::ceil<float>((float)imgHeight / (float)DEFAULT_WORK_GROUP_SIZE);
	ret.workGroupCount[2] = 1;
	return ret;
}

class RayQuerySampleApp : public ApplicationBase
{
	constexpr static uint32_t WIN_W = 1280u;
	constexpr static uint32_t WIN_H = 720u;
	constexpr static uint32_t FBO_COUNT = 2u;
	constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	static_assert(FRAMES_IN_FLIGHT > FBO_COUNT);

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
	core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbos;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
	core::smart_refctd_ptr<nbl::system::ISystem> system;
	core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	core::smart_refctd_ptr<nbl::system::ILogger> logger;
	core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
	video::IGPUObjectFromAssetConverter cpu2gpu;
	
	int32_t m_resourceIx = -1;
	uint32_t m_acquiredNextFBO = {};

	CDumbPresentationOracle oracle;
	
	// polling for events!
	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
	
	core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };

	core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf[FRAMES_IN_FLIGHT]; // from graphics

	Camera cam;
	
	core::smart_refctd_ptr<video::IGPUBuffer> gpuubo = nullptr;
	core::smart_refctd_ptr<video::IGPUImageView> gpuEnvmapImageView = nullptr;
	core::smart_refctd_ptr<IGPUImageView> gpuScrambleImageView;

	core::smart_refctd_ptr<video::IGPUComputePipeline> gpuComputePipeline = nullptr;
	DispatchInfo_t dispatchInfo = {};
	
	core::smart_refctd_ptr<video::IGPUImageView> outHDRImageViews[FBO_COUNT] = {};

	core::smart_refctd_ptr<video::IGPUDescriptorSet> descriptorSets0[FBO_COUNT] = {};
	core::smart_refctd_ptr<video::IGPUDescriptorSet> descriptorSet2 = nullptr;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> uboDescriptorSet1 = nullptr;
	
	core::smart_refctd_ptr<IGPUBuffer> aabbsBuffer = nullptr;
	core::smart_refctd_ptr<IGPUAccelerationStructure> gpuBlas = nullptr;
	core::smart_refctd_ptr<IGPUAccelerationStructure> gpuBlas2 = nullptr; // Built via CPUObject To GPUObject operations and utility
	core::smart_refctd_ptr<IGPUAccelerationStructure> gpuTlas = nullptr;
	core::smart_refctd_ptr<IGPUBuffer> instancesBuffer = nullptr;
	
	core::smart_refctd_ptr<IGPUBufferView> gpuSequenceBufferView = nullptr;
	
	core::smart_refctd_ptr<video::IGPUSampler> sampler0 = nullptr;
	core::smart_refctd_ptr<video::IGPUSampler> sampler1 = nullptr;
	
	core::smart_refctd_ptr<IGPUBuffer> gpuSequenceBuffer = nullptr;
	
	core::smart_refctd_ptr<IGPUBuffer> spheresBuffer = nullptr;

	struct SBasicViewParametersAligned
	{
		SBasicViewParameters uboData;
	};

public:
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

	APP_CONSTRUCTOR(RayQuerySampleApp);
	
	void onAppInitialized_impl() override
	{
		const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_TRANSFER_DST_BIT);
		const video::ISurface::SFormat surfaceFormat(asset::EF_B8G8R8A8_SRGB, asset::ECP_SRGB, asset::EOTF_sRGB);

		CommonAPI::InitOutput initOutput;
		initOutput.window = core::smart_refctd_ptr(window);
		CommonAPI::InitWithRaytracingExt(
			initOutput,
			video::EAT_VULKAN,
			"56.RayQuery",
			WIN_W, WIN_H, FBO_COUNT,
			swapchainImageUsage,
			surfaceFormat, asset::EF_D32_SFLOAT);

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
		renderpass = std::move(initOutput.renderpass);
		fbos = std::move(initOutput.fbo);
		commandPools = std::move(initOutput.commandPools);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);
		auto graphicsQueue = queues[CommonAPI::InitOutput::EQT_GRAPHICS];
		auto computeQueue = queues[CommonAPI::InitOutput::EQT_GRAPHICS];
		auto graphicsCommandPool = commandPools[CommonAPI::InitOutput::EQT_GRAPHICS];
		auto computeCommandPool =  commandPools[CommonAPI::InitOutput::EQT_COMPUTE];

		video::IGPUObjectFromAssetConverter cpu2gpu;	
		logicalDevice->createCommandBuffers(graphicsCommandPool.get(), nbl::video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, cmdbuf);
	
		constexpr uint32_t maxDescriptorCount = 256u;
		constexpr uint32_t PoolSizesCount = 5u;
		nbl::video::IDescriptorPool::SDescriptorPoolSize poolSizes[PoolSizesCount] = {
			{ EDT_STORAGE_BUFFER, 1},
			{ EDT_STORAGE_IMAGE, 8},
			{ EDT_COMBINED_IMAGE_SAMPLER, 2},
			{ EDT_UNIFORM_TEXEL_BUFFER, 1},
			{ EDT_UNIFORM_BUFFER, 1},
		};
		
		auto descriptorPool = logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxDescriptorCount, PoolSizesCount, poolSizes);
	
		auto geometryCreator = assetManager->getGeometryCreator();
		auto cubeGeom = geometryCreator->createCubeMesh(core::vector3df(1.0f, 1.0f, 1.0f));

		auto dummyPplnLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>();
		auto createMeshBufferFromGeomCreatorReturnType = [&dummyPplnLayout](
			asset::IGeometryCreator::return_type& _data,
			asset::IAssetManager* _manager,
			asset::ICPUSpecializedShader** _shadersBegin, asset::ICPUSpecializedShader** _shadersEnd)
		{
			//creating pipeline just to forward vtx and primitive params
			auto pipeline = core::make_smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>(
				core::smart_refctd_ptr(dummyPplnLayout), _shadersBegin, _shadersEnd,
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

		};
		auto cpuMeshCube = createMeshBufferFromGeomCreatorReturnType(cubeGeom, assetManager.get(), nullptr, nullptr);

		// Initialize Spheres
		constexpr uint32_t SphereCount = 9u;
		constexpr uint32_t INVALID_ID_16BIT = 0xffffu;

		struct alignas(16) Sphere
		{
			Sphere()
				: position(0.0f, 0.0f, 0.0f)
				, radius2(0.0f)
			{
				bsdfLightIDs = core::bitfieldInsert<uint32_t>(0u,INVALID_ID_16BIT,16,16);
			}

			Sphere(core::vector3df _position, float _radius, uint32_t _bsdfID, uint32_t _lightID)
			{
				position = _position;
				radius2 = _radius*_radius;
				bsdfLightIDs = core::bitfieldInsert(_bsdfID,_lightID,16,16);
			}

			IGPUAccelerationStructure::AABB_Position getAABB() const
			{
				float radius = core::sqrt(radius2);
				return IGPUAccelerationStructure::AABB_Position(position-core::vector3df(radius, radius, radius), position+core::vector3df(radius, radius, radius));
			}

			core::vector3df position;
			float radius2;
			uint32_t bsdfLightIDs;
		};
	
		Sphere spheres[SphereCount] = {};
		spheres[0] = Sphere(core::vector3df(0.0,-100.5,-1.0), 100.0, 0u, INVALID_ID_16BIT);
		spheres[1] = Sphere(core::vector3df(3.0,0.0,-1.0), 0.5,	 1u, INVALID_ID_16BIT);
		spheres[2] = Sphere(core::vector3df(0.0,0.0,-1.0), 0.5,	 2u, INVALID_ID_16BIT);
		spheres[3] = Sphere(core::vector3df(-3.0,0.0,-1.0), 0.5, 3u, INVALID_ID_16BIT);
		spheres[4] = Sphere(core::vector3df(3.0,0.0,1.0), 0.5,	 4u, INVALID_ID_16BIT);
		spheres[5] = Sphere(core::vector3df(0.0,0.0,1.0), 0.5,	 4u, INVALID_ID_16BIT);
		spheres[6] = Sphere(core::vector3df(-3.0,0.0,1.0), 0.5,	 5u, INVALID_ID_16BIT);
		spheres[7] = Sphere(core::vector3df(0.5,1.0,0.5), 0.5,	 6u, INVALID_ID_16BIT);
		spheres[8] = Sphere(core::vector3df(-1.5,1.5,0.0), 0.3,  INVALID_ID_16BIT, 0u);

		// Create Spheres Buffer
		uint32_t spheresBufferSize = sizeof(Sphere) * SphereCount;

		{
			IGPUBuffer::SCreationParams params = {};
			params.size = spheresBufferSize; // (Erfan->Cyprian) See How I moved "createDeviceLocalGPUBufferOnDedMem" second parameter to params.size? IGPUBuffer::SCreationParams::size is very important to be filled unlike before
			params.usage = core::bitflag(asset::IBuffer::EUF_STORAGE_BUFFER_BIT) | asset::IBuffer::EUF_TRANSFER_DST_BIT; 
			spheresBuffer = logicalDevice->createBuffer(params);
			auto bufferReqs = spheresBuffer->getMemoryReqs();
			bufferReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits(); // (Erfan->Cyprian) I used `getDeviceLocalMemoryTypeBits` because of previous createDeviceLocalGPUBufferOnDedMem (Focus on DeviceLocal Part)
			auto spheresBufferMem = logicalDevice->allocate(bufferReqs, spheresBuffer.get());
			utilities->updateBufferRangeViaStagingBuffer(graphicsQueue, asset::SBufferRange<IGPUBuffer>{0u,spheresBufferSize,spheresBuffer}, spheres);
		}

#define TEST_CPU_2_GPU_BLAS
#ifdef TEST_CPU_2_GPU_BLAS
		// Acceleration Structure Test
		// Create + Build BLAS (CPU2GPU Version)
		{
			struct AABB {
				IGPUAccelerationStructure::AABB_Position aabb;
			};
			const uint32_t aabbsCount = SphereCount / 2u;
			uint32_t aabbsBufferSize = sizeof(AABB) * aabbsCount;
		
			AABB aabbs[aabbsCount] = {};
			for(uint32_t i = 0; i < aabbsCount; ++i)
			{
				aabbs[i].aabb = spheres[i].getAABB();
			}
		
			// auto raytracingFlags = core::bitflag(asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT) | asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
			// | asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT
			core::smart_refctd_ptr<ICPUBuffer> aabbsBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(aabbsBufferSize);
			memcpy(aabbsBuffer->getPointer(), aabbs, aabbsBufferSize);

			ICPUAccelerationStructure::SCreationParams asCreateParams;
			asCreateParams.type = ICPUAccelerationStructure::ET_BOTTOM_LEVEL;
			asCreateParams.flags = ICPUAccelerationStructure::ECF_NONE;
			core::smart_refctd_ptr<ICPUAccelerationStructure> cpuBlas = ICPUAccelerationStructure::create(std::move(asCreateParams));
		
			using HostGeom = ICPUAccelerationStructure::HostBuildGeometryInfo::Geom;
			core::smart_refctd_dynamic_array<HostGeom> geometries = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<HostGeom>>(1u);

			HostGeom & simpleGeom = geometries->operator[](0u);
			simpleGeom.type = IAccelerationStructure::EGT_AABBS;
			simpleGeom.flags = IAccelerationStructure::EGF_OPAQUE_BIT;
			simpleGeom.data.aabbs.data.offset = 0u;
			simpleGeom.data.aabbs.data.buffer = aabbsBuffer;
			simpleGeom.data.aabbs.stride = sizeof(AABB);

			ICPUAccelerationStructure::HostBuildGeometryInfo buildInfo;
			buildInfo.type = asCreateParams.type;
			buildInfo.buildFlags = ICPUAccelerationStructure::EBF_PREFER_FAST_TRACE_BIT;
			buildInfo.buildMode = ICPUAccelerationStructure::EBM_BUILD;
			buildInfo.geometries = geometries;

			core::smart_refctd_dynamic_array<ICPUAccelerationStructure::BuildRangeInfo> buildRangeInfos = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUAccelerationStructure::BuildRangeInfo>>(1u);
			ICPUAccelerationStructure::BuildRangeInfo & firstBuildRangeInfo = buildRangeInfos->operator[](0u);
			firstBuildRangeInfo.primitiveCount = aabbsCount;
			firstBuildRangeInfo.primitiveOffset = 0u;
			firstBuildRangeInfo.firstVertex = 0u;
			firstBuildRangeInfo.transformOffset = 0u;

			cpuBlas->setBuildInfoAndRanges(std::move(buildInfo), buildRangeInfos);

			// Build BLAS 
			{
				cpu2gpuParams.beginCommandBuffers();
				gpuBlas2 = cpu2gpu.getGPUObjectsFromAssets(&cpuBlas, &cpuBlas + 1u, cpu2gpuParams)->front();
				cpu2gpuParams.waitForCreationToComplete();
			}
		}
#endif

		// Create + Build BLAS
		{
			// Build BLAS with AABBS
			const uint32_t aabbsCount = SphereCount;

			struct AABB {
				IGPUAccelerationStructure::AABB_Position aabb;
			};
	
			AABB aabbs[aabbsCount] = {};
			for(uint32_t i = 0; i < aabbsCount; ++i)
			{
				aabbs[i].aabb = spheres[i].getAABB();
			}
			auto raytracingFlags = core::bitflag(asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT) | asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
			uint32_t aabbsBufferSize = sizeof(AABB) * aabbsCount;

			{
				IGPUBuffer::SCreationParams params = {};
				params.size = aabbsBufferSize;
				params.usage = raytracingFlags | asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT; 
				aabbsBuffer = logicalDevice->createBuffer(params);
				auto bufferReqs = aabbsBuffer->getMemoryReqs();
				bufferReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
				auto aabbBufferMem = logicalDevice->allocate(bufferReqs, aabbsBuffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
				// (Erfan->Cyprian) -> I passed `IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT` as a third parameter to the allocate function because the buffer needs the usage `EUF_SHADER_DEVICE_ADDRESS_BIT`
				//		You don't have to worry about it, it's only used in this example
				utilities->updateBufferRangeViaStagingBuffer(graphicsQueue, asset::SBufferRange<IGPUBuffer>{0u,aabbsBufferSize,aabbsBuffer}, aabbs);
			}

			using DeviceGeom = IGPUAccelerationStructure::DeviceBuildGeometryInfo::Geometry;

			DeviceGeom simpleGeom = {};
			simpleGeom.type = IAccelerationStructure::EGT_AABBS;
			simpleGeom.flags = IAccelerationStructure::EGF_OPAQUE_BIT;
			simpleGeom.data.aabbs.data.offset = 0u;
			simpleGeom.data.aabbs.data.buffer = aabbsBuffer;
			simpleGeom.data.aabbs.stride = sizeof(AABB);

			IGPUAccelerationStructure::DeviceBuildGeometryInfo blasBuildInfo = {};
			blasBuildInfo.type = IGPUAccelerationStructure::ET_BOTTOM_LEVEL;
			blasBuildInfo.buildFlags = IGPUAccelerationStructure::EBF_PREFER_FAST_TRACE_BIT;
			blasBuildInfo.buildMode = IGPUAccelerationStructure::EBM_BUILD;
			blasBuildInfo.srcAS = nullptr;
			blasBuildInfo.dstAS = nullptr;
			blasBuildInfo.geometries = core::SRange<DeviceGeom>(&simpleGeom, &simpleGeom + 1u);
			blasBuildInfo.scratchAddr = {};
	
			// Get BuildSizes
			IGPUAccelerationStructure::BuildSizes buildSizes = {};
			{
				std::vector<uint32_t> maxPrimCount(1u);
				maxPrimCount[0] = aabbsCount;
				buildSizes = logicalDevice->getAccelerationStructureBuildSizes(blasBuildInfo, maxPrimCount.data());
			}
	
			{
				core::smart_refctd_ptr<IGPUBuffer> asBuffer;
				IGPUBuffer::SCreationParams params = {};
				params.size = buildSizes.accelerationStructureSize;
				params.usage = core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT;
				asBuffer = logicalDevice->createBuffer(params);
				auto bufferReqs = asBuffer->getMemoryReqs();
				bufferReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
				auto asBufferMem = logicalDevice->allocate(bufferReqs, asBuffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);

				IGPUAccelerationStructure::SCreationParams blasParams = {};
				blasParams.type = IGPUAccelerationStructure::ET_BOTTOM_LEVEL;
				blasParams.flags = IGPUAccelerationStructure::ECF_NONE;
				blasParams.bufferRange.buffer = asBuffer;
				blasParams.bufferRange.offset = 0u;
				blasParams.bufferRange.size = buildSizes.accelerationStructureSize;
				gpuBlas = logicalDevice->createAccelerationStructure(std::move(blasParams));
			}

			// Allocate ScratchBuffer
			core::smart_refctd_ptr<IGPUBuffer> scratchBuffer;
			{
				IGPUBuffer::SCreationParams params = {};
				params.size = buildSizes.buildScratchSize;
				params.usage = core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_STORAGE_BUFFER_BIT; 
				scratchBuffer = logicalDevice->createBuffer(params);
				auto bufferReqs = scratchBuffer->getMemoryReqs();
				bufferReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
				auto scratchBufferMem = logicalDevice->allocate(bufferReqs, scratchBuffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
			}

			// Complete BLAS Build Info
			{
				blasBuildInfo.dstAS = gpuBlas.get();
				blasBuildInfo.scratchAddr.buffer = scratchBuffer;
				blasBuildInfo.scratchAddr.offset = 0u;
			}
	
			IGPUAccelerationStructure::BuildRangeInfo firstBuildRangeInfos[1u];
			firstBuildRangeInfos[0].primitiveCount = aabbsCount;
			firstBuildRangeInfos[0].primitiveOffset = 0u;
			firstBuildRangeInfos[0].firstVertex = 0u;
			firstBuildRangeInfos[0].transformOffset = 0u;
			IGPUAccelerationStructure::BuildRangeInfo* pRangeInfos[1u];
			pRangeInfos[0] = firstBuildRangeInfos;
			// pRangeInfos[1] = &secondBuildRangeInfos;

			// Build BLAS 
			{
				utilities->buildAccelerationStructures(computeQueue, core::SRange<IGPUAccelerationStructure::DeviceBuildGeometryInfo>(&blasBuildInfo, &blasBuildInfo + 1u), pRangeInfos);
			}
		}
	
		// Create + Build TLAS
		{
			struct Instance {
				IGPUAccelerationStructure::Instance instance;
			};

			const uint32_t instancesCount = 1u;
			Instance instances[instancesCount] = {};
			core::matrix3x4SIMD identity;
			instances[0].instance.mat = identity;
			instances[0].instance.instanceCustomIndex = 0u;
			instances[0].instance.mask = 0xFF;
			instances[0].instance.instanceShaderBindingTableRecordOffset = 0u;
			instances[0].instance.flags = IAccelerationStructure::EIF_TRIANGLE_FACING_CULL_DISABLE_BIT;
#ifdef TEST_CPU_2_GPU_BLAS
			instances[0].instance.accelerationStructureReference = gpuBlas2->getReferenceForDeviceOperations();
#else
			instances[0].instance.accelerationStructureReference = gpuBlas->getReferenceForDeviceOperations();
#endif
			auto raytracingFlags = core::bitflag(asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT) | asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
	
			uint32_t instancesBufferSize = sizeof(Instance);
			{
				IGPUBuffer::SCreationParams params = {};
				params.size = instancesBufferSize;
				params.usage = raytracingFlags | asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT; 
				instancesBuffer = logicalDevice->createBuffer(params);
				auto bufferReqs = instancesBuffer->getMemoryReqs();
				bufferReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
				auto instancesBufferMem = logicalDevice->allocate(bufferReqs, instancesBuffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
				utilities->updateBufferRangeViaStagingBuffer(graphicsQueue, asset::SBufferRange<IGPUBuffer>{0u,instancesBufferSize,instancesBuffer}, instances);
			}
		
			using DeviceGeom = IGPUAccelerationStructure::DeviceBuildGeometryInfo::Geometry;

			DeviceGeom blasInstancesGeom = {};
			blasInstancesGeom.type = IAccelerationStructure::EGT_INSTANCES;
			blasInstancesGeom.flags = IAccelerationStructure::EGF_NONE;
			blasInstancesGeom.data.instances.data.offset = 0u;
			blasInstancesGeom.data.instances.data.buffer = instancesBuffer;

			IGPUAccelerationStructure::DeviceBuildGeometryInfo tlasBuildInfo = {};
			tlasBuildInfo.type = IGPUAccelerationStructure::ET_TOP_LEVEL;
			tlasBuildInfo.buildFlags = IGPUAccelerationStructure::EBF_PREFER_FAST_TRACE_BIT;
			tlasBuildInfo.buildMode = IGPUAccelerationStructure::EBM_BUILD;
			tlasBuildInfo.srcAS = nullptr;
			tlasBuildInfo.dstAS = nullptr;
			tlasBuildInfo.geometries = core::SRange<DeviceGeom>(&blasInstancesGeom, &blasInstancesGeom + 1u);
			tlasBuildInfo.scratchAddr = {};
			
			// Get BuildSizes
			IGPUAccelerationStructure::BuildSizes buildSizes = {};
			{
				std::vector<uint32_t> maxPrimCount(1u); 
				maxPrimCount[0] = instancesCount;
				buildSizes = logicalDevice->getAccelerationStructureBuildSizes(tlasBuildInfo, maxPrimCount.data());
			}
	
			{
				core::smart_refctd_ptr<IGPUBuffer> asBuffer;
				IGPUBuffer::SCreationParams params = {};
				params.size = buildSizes.accelerationStructureSize;
				params.usage = core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT; 
				asBuffer = logicalDevice->createBuffer(params);
				auto bufferReqs = asBuffer->getMemoryReqs();
				bufferReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
				auto asBufferMem = logicalDevice->allocate(bufferReqs, asBuffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);

				IGPUAccelerationStructure::SCreationParams tlasParams = {};
				tlasParams.type = IGPUAccelerationStructure::ET_TOP_LEVEL;
				tlasParams.flags = IGPUAccelerationStructure::ECF_NONE;
				tlasParams.bufferRange.buffer = asBuffer;
				tlasParams.bufferRange.offset = 0u;
				tlasParams.bufferRange.size = buildSizes.accelerationStructureSize;
				gpuTlas = logicalDevice->createAccelerationStructure(std::move(tlasParams));
			}

			// Allocate ScratchBuffer
			core::smart_refctd_ptr<IGPUBuffer> scratchBuffer;
			{
				IGPUBuffer::SCreationParams params = {};
				params.size = buildSizes.buildScratchSize;
				params.usage = core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_STORAGE_BUFFER_BIT; 
				scratchBuffer = logicalDevice->createBuffer(params);
				auto bufferReqs = scratchBuffer->getMemoryReqs();
				bufferReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
				auto scratchBufferMem = logicalDevice->allocate(bufferReqs, scratchBuffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
			}

			// Complete BLAS Build Info
			{
				tlasBuildInfo.dstAS = gpuTlas.get();
				tlasBuildInfo.scratchAddr.buffer = scratchBuffer;
				tlasBuildInfo.scratchAddr.offset = 0u;
			}
		
			IGPUAccelerationStructure::BuildRangeInfo firstBuildRangeInfos[1u];
			firstBuildRangeInfos[0].primitiveCount = instancesCount;
			firstBuildRangeInfos[0].primitiveOffset = 0u;
			firstBuildRangeInfos[0].firstVertex = 0u;
			firstBuildRangeInfos[0].transformOffset = 0u;
			IGPUAccelerationStructure::BuildRangeInfo* pRangeInfos[1u];
			pRangeInfos[0] = firstBuildRangeInfos;

			// Build BLAS 
			{
				utilities->buildAccelerationStructures(computeQueue, core::SRange<IGPUAccelerationStructure::DeviceBuildGeometryInfo>(&tlasBuildInfo, &tlasBuildInfo + 1u), pRangeInfos);
			}
		}
	

		// Camera 
		core::vectorSIMDf cameraPosition(0, 5, -10);
		matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(60.0f), float(WIN_W) / WIN_H, 0.01f, 500.0f);
		cam = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), proj);

		IGPUDescriptorSetLayout::SBinding descriptorSet0Bindings[] = {
			{ 0u, EDT_STORAGE_IMAGE, 1u, IShader::ESS_COMPUTE, nullptr },
		};
		IGPUDescriptorSetLayout::SBinding uboBinding {0, EDT_UNIFORM_BUFFER, 1u, IShader::ESS_COMPUTE, nullptr};
		IGPUDescriptorSetLayout::SBinding descriptorSet3Bindings[] = {
			{ 0u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IShader::ESS_COMPUTE, nullptr },
			{ 1u, EDT_UNIFORM_TEXEL_BUFFER, 1u, IShader::ESS_COMPUTE, nullptr },
			{ 2u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IShader::ESS_COMPUTE, nullptr },
			{ 3u, EDT_ACCELERATION_STRUCTURE_KHR, 1u, IShader::ESS_COMPUTE, nullptr },
			{ 4u, EDT_STORAGE_BUFFER, 1u, IShader::ESS_COMPUTE, nullptr }
		};
	
		auto gpuDescriptorSetLayout0 = logicalDevice->createDescriptorSetLayout(descriptorSet0Bindings, descriptorSet0Bindings + 1u);
		auto gpuDescriptorSetLayout1 = logicalDevice->createDescriptorSetLayout(&uboBinding, &uboBinding + 1u);
		auto gpuDescriptorSetLayout2 = logicalDevice->createDescriptorSetLayout(descriptorSet3Bindings, descriptorSet3Bindings+5u);

		auto createGpuResources = [&](std::string pathToShader) -> core::smart_refctd_ptr<video::IGPUComputePipeline>
		{
			asset::IAssetLoader::SAssetLoadParams params{};
			params.logger = logger.get();
			//params.relativeDir = tmp.c_str();
			auto spec = assetManager->getAsset(pathToShader,params).getContents();
		
			if (spec.empty())
				assert(false);

			auto cpuComputeSpecializedShader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*spec.begin());

			ISpecializedShader::SInfo info = cpuComputeSpecializedShader->getSpecializationInfo();
			info.m_backingBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(ShaderParameters));
			memcpy(info.m_backingBuffer->getPointer(),&kShaderParameters,sizeof(ShaderParameters));
			info.m_entries = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ISpecializedShader::SInfo::SMapEntry>>(2u);
			for (uint32_t i=0; i<2; i++)
				info.m_entries->operator[](i) = {i,i*sizeof(uint32_t),sizeof(uint32_t)};


			cpuComputeSpecializedShader->setSpecializationInfo(std::move(info));

			auto gpuComputeSpecializedShader = cpu2gpu.getGPUObjectsFromAssets(&cpuComputeSpecializedShader, &cpuComputeSpecializedShader + 1, cpu2gpuParams)->front();

			auto gpuPipelineLayout = logicalDevice->createPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout0), core::smart_refctd_ptr(gpuDescriptorSetLayout1), core::smart_refctd_ptr(gpuDescriptorSetLayout2), nullptr);

			auto gpuPipeline = logicalDevice->createComputePipeline(nullptr, std::move(gpuPipelineLayout), std::move(gpuComputeSpecializedShader));

			return gpuPipeline;
		};

		E_LIGHT_GEOMETRY lightGeom = ELG_SPHERE;
		constexpr const char* shaderPaths[] = {"../litBySphere.comp","../litByTriangle.comp","../litByRectangle.comp"};
		gpuComputePipeline = createGpuResources(shaderPaths[lightGeom]);

		dispatchInfo = getDispatchInfo(WIN_W, WIN_H);

		auto createImageView = [&](std::string pathToOpenEXRHDRIImage)
		{
			auto pathToTexture = pathToOpenEXRHDRIImage;
			IAssetLoader::SAssetLoadParams lp(0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES);
			auto cpuTexture = assetManager->getAsset(pathToTexture, lp);
			auto cpuTextureContents = cpuTexture.getContents();
			assert(!cpuTextureContents.empty());
			auto cpuImage = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(*cpuTextureContents.begin());
			cpuImage->setImageUsageFlags(IImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT);

			ICPUImageView::SCreationParams viewParams;
			viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
			viewParams.image = cpuImage;
			viewParams.format = viewParams.image->getCreationParameters().format;
			viewParams.viewType = IImageView<ICPUImage>::ET_2D;
			viewParams.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			viewParams.subresourceRange.baseArrayLayer = 0u;
			viewParams.subresourceRange.layerCount = 1u;
			viewParams.subresourceRange.baseMipLevel = 0u;
			viewParams.subresourceRange.levelCount = 1u;

			auto cpuImageView = ICPUImageView::create(std::move(viewParams));
		
			cpu2gpuParams.beginCommandBuffers();
			auto gpuImageView = cpu2gpu.getGPUObjectsFromAssets(&cpuImageView, &cpuImageView + 1u, cpu2gpuParams)->front();
			cpu2gpuParams.waitForCreationToComplete();

			return gpuImageView;
		};
	
		gpuEnvmapImageView = createImageView("../../media/envmap/envmap_0.exr");

		{
			const uint32_t MaxDimensions = 3u<<kShaderParameters.MaxDepthLog2;
			const uint32_t MaxSamples = 1u<<kShaderParameters.MaxSamplesLog2;

			auto sampleSequence = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t)*MaxDimensions*MaxSamples);
		
			core::OwenSampler sampler(MaxDimensions, 0xdeadbeefu);
			//core::SobolSampler sampler(MaxDimensions);

			auto out = reinterpret_cast<uint32_t*>(sampleSequence->getPointer());
			for (auto dim=0u; dim<MaxDimensions; dim++)
			for (uint32_t i=0; i<MaxSamples; i++)
			{
				out[i*MaxDimensions+dim] = sampler.sample(dim,i);
			}
		
			// TODO: Temp Fix because createFilledDeviceLocalBufferOnDedMem doesn't take in params
			// auto gpuSequenceBuffer = utilities->createFilledDeviceLocalBufferOnDedMem(graphicsQueue, sampleSequence->getSize(), sampleSequence->getPointer());
			{
				IGPUBuffer::SCreationParams params = {};
				params.size = sampleSequence->getSize();
				params.usage = core::bitflag(asset::IBuffer::EUF_TRANSFER_DST_BIT) | asset::IBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT; 
				gpuSequenceBuffer = logicalDevice->createBuffer(params);
				auto bufferReqs = gpuSequenceBuffer->getMemoryReqs();
				bufferReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
				auto gpuSequenceBufferMem = logicalDevice->allocate(bufferReqs, gpuSequenceBuffer.get());
				utilities->updateBufferRangeViaStagingBuffer(graphicsQueue, asset::SBufferRange<IGPUBuffer>{0u,params.size,gpuSequenceBuffer},sampleSequence->getPointer());
			}
			gpuSequenceBufferView = logicalDevice->createBufferView(gpuSequenceBuffer.get(), asset::EF_R32G32B32_UINT);
		}

		{
			IGPUImage::SCreationParams imgParams;
			imgParams.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
			imgParams.type = IImage::ET_2D;
			imgParams.format = EF_R32G32_UINT;
			imgParams.extent = {WIN_W, WIN_H,1u};
			imgParams.mipLevels = 1u;
			imgParams.arrayLayers = 1u;
			imgParams.samples = IImage::ESCF_1_BIT;
			imgParams.usage = core::bitflag(IImage::EUF_SAMPLED_BIT) | IImage::EUF_TRANSFER_DST_BIT;
			imgParams.initialLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;

			IGPUImage::SBufferCopy region = {};
			region.bufferOffset = 0u;
			region.bufferRowLength = 0u;
			region.bufferImageHeight = 0u;
			region.imageExtent = imgParams.extent;
			region.imageOffset = {0u,0u,0u};
			region.imageSubresource.layerCount = 1u;
			region.imageSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;

			constexpr auto ScrambleStateChannels = 2u;
			const auto renderPixelCount = imgParams.extent.width*imgParams.extent.height;
			core::vector<uint32_t> random(renderPixelCount*ScrambleStateChannels);
			{
				core::RandomSampler rng(0xbadc0ffeu);
				for (auto& pixel : random)
					pixel = rng.nextSample();
			}

			// TODO: Temp Fix because createFilledDeviceLocalBufferOnDedMem doesn't take in params
			// auto buffer = utilities->createFilledDeviceLocalBufferOnDedMem(graphicsQueue, random.size()*sizeof(uint32_t), random.data());
			core::smart_refctd_ptr<IGPUBuffer> buffer;
			{
				IGPUBuffer::SCreationParams params = {};
				params.size = random.size() * sizeof(uint32_t);
				params.usage = core::bitflag(asset::IBuffer::EUF_TRANSFER_DST_BIT) | asset::IBuffer::EUF_TRANSFER_SRC_BIT; 
				buffer = logicalDevice->createBuffer(params);
				auto bufferReqs = buffer->getMemoryReqs();
				bufferReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
				auto bufferMem = logicalDevice->allocate(bufferReqs, buffer.get());
				utilities->updateBufferRangeViaStagingBuffer(graphicsQueue, asset::SBufferRange<IGPUBuffer>{0u,params.size,buffer},random.data());
			}

			IGPUImageView::SCreationParams viewParams;
			viewParams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
			viewParams.image = utilities->createFilledDeviceLocalImageOnDedMem(graphicsQueue, std::move(imgParams), buffer.get(), 1u, &region);
			viewParams.viewType = IGPUImageView::ET_2D;
			viewParams.format = EF_R32G32_UINT;
			viewParams.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			viewParams.subresourceRange.levelCount = 1u;
			viewParams.subresourceRange.layerCount = 1u;
			gpuScrambleImageView = logicalDevice->createImageView(std::move(viewParams));
		}
	
		// Create Out Image TODO
		for(uint32_t i = 0; i < FBO_COUNT; ++i) {
			outHDRImageViews[i] = createHDRImageView(logicalDevice, asset::EF_R16G16B16A16_SFLOAT, WIN_W, WIN_H);
		}

		for(uint32_t i = 0; i < FBO_COUNT; ++i)
		{
			auto & descSet = descriptorSets0[i];
			descSet = logicalDevice->createDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(gpuDescriptorSetLayout0));
			video::IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSet;
			writeDescriptorSet.dstSet = descSet.get();
			writeDescriptorSet.binding = 0;
			writeDescriptorSet.count = 1u;
			writeDescriptorSet.arrayElement = 0u;
			writeDescriptorSet.descriptorType = asset::EDT_STORAGE_IMAGE;
			video::IGPUDescriptorSet::SDescriptorInfo info;
			{
				info.desc = outHDRImageViews[i];
				info.image.sampler = nullptr;
				info.image.imageLayout = asset::E_IMAGE_LAYOUT::EIL_GENERAL;
			}
			writeDescriptorSet.info = &info;
			logicalDevice->updateDescriptorSets(1u, &writeDescriptorSet, 0u, nullptr);
		}
	
		IGPUBuffer::SCreationParams gpuuboParams = {};
		gpuuboParams.size = sizeof(SBasicViewParametersAligned);
		gpuuboParams.usage = core::bitflag(IGPUBuffer::EUF_UNIFORM_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT;
		gpuubo = logicalDevice->createBuffer(gpuuboParams);
		auto gpuUboMemReqs = gpuubo->getMemoryReqs();
		gpuUboMemReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		auto gpuUboMem = logicalDevice->allocate(gpuUboMemReqs, gpuubo.get());

		uboDescriptorSet1 = logicalDevice->createDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(gpuDescriptorSetLayout1));
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
				info.buffer.size = sizeof(SBasicViewParametersAligned);
			}
			uboWriteDescriptorSet.info = &info;
			logicalDevice->updateDescriptorSets(1u, &uboWriteDescriptorSet, 0u, nullptr);
		}

		ISampler::SParams samplerParams0 = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
		sampler0 = logicalDevice->createSampler(samplerParams0);
		ISampler::SParams samplerParams1 = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_INT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
		sampler1 = logicalDevice->createSampler(samplerParams1);
		
		descriptorSet2 = logicalDevice->createDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(gpuDescriptorSetLayout2));
		{
			constexpr auto kDescriptorCount = 5;
			IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSet2[kDescriptorCount];
			IGPUDescriptorSet::SDescriptorInfo writeDescriptorInfo[kDescriptorCount];
			for (auto i=0; i<kDescriptorCount; i++)
			{
				writeDescriptorSet2[i].dstSet = descriptorSet2.get();
				writeDescriptorSet2[i].binding = i;
				writeDescriptorSet2[i].arrayElement = 0u;
				writeDescriptorSet2[i].count = 1u;
				writeDescriptorSet2[i].info = writeDescriptorInfo+i;
			}
			writeDescriptorSet2[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
			writeDescriptorSet2[1].descriptorType = EDT_UNIFORM_TEXEL_BUFFER;
			writeDescriptorSet2[2].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
			writeDescriptorSet2[3].descriptorType = EDT_ACCELERATION_STRUCTURE_KHR;
			writeDescriptorSet2[4].descriptorType = EDT_STORAGE_BUFFER;

			writeDescriptorInfo[0].desc = gpuEnvmapImageView;
			{
				// ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
				writeDescriptorInfo[0].image.sampler = sampler0;
				writeDescriptorInfo[0].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
			}
			writeDescriptorInfo[1].desc = gpuSequenceBufferView;
			writeDescriptorInfo[2].desc = gpuScrambleImageView;
			{
				// ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_INT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
				writeDescriptorInfo[2].image.sampler = sampler1;
				writeDescriptorInfo[2].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
			}

			writeDescriptorInfo[3].desc = gpuTlas;

			writeDescriptorInfo[4].desc = spheresBuffer;
			writeDescriptorInfo[4].buffer.offset = 0;
			writeDescriptorInfo[4].buffer.size = spheresBufferSize;

			logicalDevice->updateDescriptorSets(kDescriptorCount, writeDescriptorSet2, 0u, nullptr);
		}

		constexpr uint32_t FRAME_COUNT = 500000u;

		for (uint32_t i=0u; i<FRAMES_IN_FLIGHT; i++)
		{
			imageAcquire[i] = logicalDevice->createSemaphore();
			renderFinished[i] = logicalDevice->createSemaphore();
		}
		
		oracle.reportBeginFrameRecord();
	}

	void onAppTerminated_impl() override
	{
		const auto& fboCreationParams = fbos[m_acquiredNextFBO]->getCreationParameters();
		auto gpuSourceImageView = fboCreationParams.attachments[0];
		logicalDevice->waitIdle();

		bool status = ext::ScreenShot::createScreenShot(
			logicalDevice.get(),
			queues[CommonAPI::InitOutput::EQT_TRANSFER_UP],
			renderFinished[m_resourceIx].get(),
			gpuSourceImageView.get(),
			assetManager.get(),
			"ScreenShot.png",
			asset::EIL_PRESENT_SRC,
			static_cast<asset::E_ACCESS_FLAGS>(0u));

		assert(status);
	}

	void workLoopBody() override
	{
		m_resourceIx++;
		if(m_resourceIx >= FRAMES_IN_FLIGHT) {
			m_resourceIx = 0;
		}
		
		oracle.reportEndFrameRecord();
		double dt = oracle.getDeltaTimeInMicroSeconds() / 1000.0;
		auto nextPresentationTimeStamp = oracle.getNextPresentationTimeStamp();
		oracle.reportBeginFrameRecord();

		// Input 
		inputSystem->getDefaultMouse(&mouse);
		inputSystem->getDefaultKeyboard(&keyboard);

		cam.beginInputProcessing(nextPresentationTimeStamp);
		mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { cam.mouseProcess(events); }, logger.get());
		keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { cam.keyboardProcess(events); }, logger.get());
		cam.endInputProcessing(nextPresentationTimeStamp);
		
		auto& cb = cmdbuf[m_resourceIx];
		auto& fence = frameComplete[m_resourceIx];
		if (fence)
		while (logicalDevice->waitForFences(1u,&fence.get(),false,MAX_TIMEOUT)==video::IGPUFence::ES_TIMEOUT)
		{
		}
		else
			fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
		
		const auto viewMatrix = cam.getViewMatrix();
		const auto viewProjectionMatrix = cam.getConcatenatedMatrix();
				
		// safe to proceed
		cb->begin(IGPUCommandBuffer::EU_NONE);

		// renderpass 
		swapchain->acquireNextImage(MAX_TIMEOUT,imageAcquire[m_resourceIx].get(),nullptr,&m_acquiredNextFBO);
		{
			auto mv = viewMatrix;
			auto mvp = viewProjectionMatrix;
			core::matrix3x4SIMD normalMat;
			mv.getSub3x3InverseTranspose(normalMat);

			SBasicViewParametersAligned viewParams;
			memcpy(viewParams.uboData.MV, mv.pointer(), sizeof(mv));
			memcpy(viewParams.uboData.MVP, mvp.pointer(), sizeof(mvp));
			memcpy(viewParams.uboData.NormalMat, normalMat.pointer(), sizeof(normalMat));
			
			asset::SBufferRange<video::IGPUBuffer> range;
			range.buffer = gpuubo;
			range.offset = 0ull;
			range.size = sizeof(viewParams);
			utilities->updateBufferRangeViaStagingBuffer(queues[CommonAPI::InitOutput::EQT_GRAPHICS], range, &viewParams);
		}
				
		auto graphicsCmdQueueFamIdx = queues[CommonAPI::InitOutput::EQT_GRAPHICS]->getFamilyIndex();
		// TRANSITION outHDRImageViews[m_acquiredNextFBO] to EIL_GENERAL (because of descriptorSets0 -> ComputeShader Writes into the image)
		{
			IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[3u] = {};
			imageBarriers[0].barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0u);
			imageBarriers[0].barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_WRITE_BIT);
			imageBarriers[0].oldLayout = asset::EIL_UNDEFINED;
			imageBarriers[0].newLayout = asset::EIL_GENERAL;
			imageBarriers[0].srcQueueFamilyIndex = graphicsCmdQueueFamIdx;
			imageBarriers[0].dstQueueFamilyIndex = graphicsCmdQueueFamIdx;
			imageBarriers[0].image = outHDRImageViews[m_acquiredNextFBO]->getCreationParameters().image;
			imageBarriers[0].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			imageBarriers[0].subresourceRange.baseMipLevel = 0u;
			imageBarriers[0].subresourceRange.levelCount = 1;
			imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[0].subresourceRange.layerCount = 1;

			imageBarriers[1].barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0u);
			imageBarriers[1].barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_READ_BIT);
			imageBarriers[1].oldLayout = asset::EIL_UNDEFINED;
			imageBarriers[1].newLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;
			imageBarriers[1].srcQueueFamilyIndex = graphicsCmdQueueFamIdx;
			imageBarriers[1].dstQueueFamilyIndex = graphicsCmdQueueFamIdx;
			imageBarriers[1].image = gpuScrambleImageView->getCreationParameters().image;
			imageBarriers[1].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			imageBarriers[1].subresourceRange.baseMipLevel = 0u;
			imageBarriers[1].subresourceRange.levelCount = 1;
			imageBarriers[1].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[1].subresourceRange.layerCount = 1;

			 imageBarriers[2].barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0u);
			 imageBarriers[2].barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_READ_BIT);
			 imageBarriers[2].oldLayout = asset::EIL_UNDEFINED;
			 imageBarriers[2].newLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;
			 imageBarriers[2].srcQueueFamilyIndex = graphicsCmdQueueFamIdx;
			 imageBarriers[2].dstQueueFamilyIndex = graphicsCmdQueueFamIdx;
			 imageBarriers[2].image = gpuEnvmapImageView->getCreationParameters().image;
			 imageBarriers[2].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			 imageBarriers[2].subresourceRange.baseMipLevel = 0u;
			 imageBarriers[2].subresourceRange.levelCount = gpuEnvmapImageView->getCreationParameters().subresourceRange.levelCount;
			 imageBarriers[2].subresourceRange.baseArrayLayer = 0u;
			 imageBarriers[2].subresourceRange.layerCount = gpuEnvmapImageView->getCreationParameters().subresourceRange.layerCount;

			cb->pipelineBarrier(asset::EPSF_TOP_OF_PIPE_BIT, asset::EPSF_COMPUTE_SHADER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 3u, imageBarriers);
		}

		// cube envmap handle
		{
			cb->bindComputePipeline(gpuComputePipeline.get());
			cb->bindDescriptorSets(EPBP_COMPUTE, gpuComputePipeline->getLayout(), 0u, 1u, &descriptorSets0[m_acquiredNextFBO].get());
			cb->bindDescriptorSets(EPBP_COMPUTE, gpuComputePipeline->getLayout(), 1u, 1u, &uboDescriptorSet1.get());
			cb->bindDescriptorSets(EPBP_COMPUTE, gpuComputePipeline->getLayout(), 2u, 1u, &descriptorSet2.get());
			cb->dispatch(dispatchInfo.workGroupCount[0], dispatchInfo.workGroupCount[1], dispatchInfo.workGroupCount[2]);
		}
		// TODO: tone mapping and stuff

		// Copy HDR Image to SwapChain
		auto srcImgViewCreationParams = outHDRImageViews[m_acquiredNextFBO]->getCreationParameters();
		auto dstImgViewCreationParams = fbos[m_acquiredNextFBO]->getCreationParameters().attachments[0]->getCreationParameters();
		
		// Getting Ready for Blit
		// TRANSITION outHDRImageViews[m_acquiredNextFBO] to EIL_TRANSFER_SRC_OPTIMAL
		// TRANSITION `fbos[m_acquiredNextFBO]->getCreationParameters().attachments[0]` to EIL_TRANSFER_DST_OPTIMAL
		{
			IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[2u] = {};
			imageBarriers[0].barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0u);
			imageBarriers[0].barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
			imageBarriers[0].oldLayout = asset::EIL_UNDEFINED;
			imageBarriers[0].newLayout = asset::EIL_TRANSFER_SRC_OPTIMAL;
			imageBarriers[0].srcQueueFamilyIndex = graphicsCmdQueueFamIdx;
			imageBarriers[0].dstQueueFamilyIndex = graphicsCmdQueueFamIdx;
			imageBarriers[0].image = srcImgViewCreationParams.image;
			imageBarriers[0].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			imageBarriers[0].subresourceRange.baseMipLevel = 0u;
			imageBarriers[0].subresourceRange.levelCount = 1;
			imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[0].subresourceRange.layerCount = 1;

			imageBarriers[1].barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0u);
			imageBarriers[1].barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
			imageBarriers[1].oldLayout = asset::EIL_UNDEFINED;
			imageBarriers[1].newLayout = asset::EIL_TRANSFER_DST_OPTIMAL;
			imageBarriers[1].srcQueueFamilyIndex = graphicsCmdQueueFamIdx;
			imageBarriers[1].dstQueueFamilyIndex = graphicsCmdQueueFamIdx;
			imageBarriers[1].image = dstImgViewCreationParams.image;
			imageBarriers[1].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			imageBarriers[1].subresourceRange.baseMipLevel = 0u;
			imageBarriers[1].subresourceRange.levelCount = 1;
			imageBarriers[1].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[1].subresourceRange.layerCount = 1;
			cb->pipelineBarrier(asset::EPSF_TRANSFER_BIT, asset::EPSF_TRANSFER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 2u, imageBarriers);
		}

		// Blit Image
		{
			SImageBlit blit = {};
			blit.srcOffsets[0] = {0, 0, 0};
			blit.srcOffsets[1] = {WIN_W, WIN_H, 1};
		
			blit.srcSubresource.aspectMask = srcImgViewCreationParams.subresourceRange.aspectMask;
			blit.srcSubresource.mipLevel = srcImgViewCreationParams.subresourceRange.baseMipLevel;
			blit.srcSubresource.baseArrayLayer = srcImgViewCreationParams.subresourceRange.baseArrayLayer;
			blit.srcSubresource.layerCount = srcImgViewCreationParams.subresourceRange.layerCount;
			blit.dstOffsets[0] = {0, 0, 0};
			blit.dstOffsets[1] = {WIN_W, WIN_H, 1};
			blit.dstSubresource.aspectMask = dstImgViewCreationParams.subresourceRange.aspectMask;
			blit.dstSubresource.mipLevel = dstImgViewCreationParams.subresourceRange.baseMipLevel;
			blit.dstSubresource.baseArrayLayer = dstImgViewCreationParams.subresourceRange.baseArrayLayer;
			blit.dstSubresource.layerCount = dstImgViewCreationParams.subresourceRange.layerCount;

			auto srcImg = srcImgViewCreationParams.image;
			auto dstImg = dstImgViewCreationParams.image;

			cb->blitImage(srcImg.get(), EIL_TRANSFER_SRC_OPTIMAL, dstImg.get(), EIL_TRANSFER_DST_OPTIMAL, 1u, &blit , ISampler::ETF_NEAREST);
		}
		
		// TRANSITION `fbos[m_acquiredNextFBO]->getCreationParameters().attachments[0]` to EIL_PRESENT
		{
			IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[1u] = {};
			imageBarriers[0].barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
			imageBarriers[0].barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0u);
			imageBarriers[0].oldLayout = asset::EIL_TRANSFER_DST_OPTIMAL;
			imageBarriers[0].newLayout = asset::EIL_PRESENT_SRC;
			imageBarriers[0].srcQueueFamilyIndex = graphicsCmdQueueFamIdx;
			imageBarriers[0].dstQueueFamilyIndex = graphicsCmdQueueFamIdx;
			imageBarriers[0].image = dstImgViewCreationParams.image;
			imageBarriers[0].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			imageBarriers[0].subresourceRange.baseMipLevel = 0u;
			imageBarriers[0].subresourceRange.levelCount = 1;
			imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[0].subresourceRange.layerCount = 1;
			cb->pipelineBarrier(asset::EPSF_TRANSFER_BIT, asset::EPSF_TOP_OF_PIPE_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, imageBarriers);
		}

		cb->end();
		logicalDevice->resetFences(1, &fence.get());
		CommonAPI::Submit(logicalDevice.get(), swapchain.get(), cb.get(), queues[CommonAPI::InitOutput::EQT_GRAPHICS], imageAcquire[m_resourceIx].get(), renderFinished[m_resourceIx].get(), fence.get());
		CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[CommonAPI::InitOutput::EQT_GRAPHICS], renderFinished[m_resourceIx].get(), m_acquiredNextFBO);
	}

	bool keepRunning() override
	{
		return windowCb->isWindowOpen();
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
		return FBO_COUNT;
	}
	virtual nbl::asset::E_FORMAT getDepthFormat() override
	{
		return nbl::asset::EF_D32_SFLOAT;
	}
};

NBL_COMMON_API_MAIN(RayQuerySampleApp)