// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "../common/Camera.hpp"

#include <btBulletDynamicsCommon.h>
#include "BulletCollision/NarrowPhaseCollision/btRaycastCallback.h"

#include "nbl/ext/Bullet/BulletUtility.h"
#include "nbl/ext/Bullet/CPhysicsWorld.h"


using namespace nbl;
using namespace core;
using namespace ui;

class CInstancedMotionState : public ext::Bullet3::IMotionStateBase
{
public:
	btTransform m_correctionMatrix;

	inline CInstancedMotionState() {}
	inline CInstancedMotionState(const void* instancePool, uint32_t objectID, uint32_t instanceID, core::matrix3x4SIMD const& start_mat, core::matrix3x4SIMD const& correction_mat)
		: ext::Bullet3::IMotionStateBase(ext::Bullet3::convertMatrixSIMD(start_mat)), m_correctionMatrix(ext::Bullet3::convertMatrixSIMD(correction_mat)), m_instancePool(instancePool), m_objectID(objectID), m_instanceID(instanceID)
	{
		m_cachedMat = m_startWorldTrans * m_correctionMatrix.inverse();
	}

	inline ~CInstancedMotionState()
	{
	}

	inline virtual void getWorldTransform(btTransform& worldTrans) const override
	{
		worldTrans = m_cachedMat;
	}

	inline virtual void setWorldTransform(const btTransform& worldTrans) override
	{
		// TODO: protect agains simulation "substeps" somehow (redundant update sets)
		m_cachedMat = worldTrans;

		s_updateAddresses.push_back(m_objectID);
		s_updateData.push_back(ext::Bullet3::convertbtTransform(m_cachedMat * m_correctionMatrix));
	}

	inline auto getInstancePool() const { return m_instancePool; }
	inline auto getObjectID() const { return m_objectID; }
	inline auto getInstanceID() const { return m_instanceID; }

	static core::vector<uint32_t> s_updateAddresses;
	static core::vector<core::matrix3x4SIMD> s_updateData;
protected:
	btTransform m_cachedMat;
	uint32_t m_objectID, m_instanceID;
	const void* m_instancePool;
};
core::vector<uint32_t> CInstancedMotionState::s_updateAddresses;
core::vector<core::matrix3x4SIMD> CInstancedMotionState::s_updateData;

class MeshLoadersApp : public ApplicationBase
{
	static constexpr uint32_t WIN_W = 1280;
	static constexpr uint32_t WIN_H = 720;
	static constexpr uint32_t FBO_COUNT = 2u;
	static constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	static constexpr size_t MaxFramesToAverage = 100ull;
	static constexpr uint32_t TransformPropertyID = 1u;
	static_assert(FRAMES_IN_FLIGHT > FBO_COUNT);

	using object_property_pool_t = video::CPropertyPool<core::allocator, core::vectorSIMDf, core::matrix3x4SIMD>;
	using instance_redirect_property_pool_t = video::CPropertyPool<core::allocator, uint32_t>;

	// TODO: replace with an actual scenemanager
	struct GPUObject
	{
		const video::IPropertyPool* pool;
		core::smart_refctd_ptr<video::IGPUMeshBuffer> gpuMesh;
		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> graphicsPipeline;
	};

public:
	struct Nabla : IUserData
	{
		nbl::core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
		nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
		nbl::core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
		nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> gl;
		nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
		nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
		nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> device;
		nbl::video::IPhysicalDevice* gpuPhysicalDevice;
		std::array<nbl::video::IGPUQueue*, CommonAPI::InitOutput<FBO_COUNT>::EQT_COUNT> queues = { nullptr, nullptr, nullptr, nullptr };
		nbl::video::IGPUQueue* graphicsQueue = nullptr;
		nbl::video::IGPUQueue* computeQueue = nullptr;
		nbl::video::IGPUQueue* transferUpQueue = nullptr;
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
		std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, FBO_COUNT> fbos;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool> commandPool;
		nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
		nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
		nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
		nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
		nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;

		nbl::core::smart_refctd_ptr<video::IGPUFence> gpuTransferFence;
		nbl::core::smart_refctd_ptr<video::IGPUFence> gpuComputeFence;
		nbl::video::IGPUObjectFromAssetConverter cpu2gpu;

		core::smart_refctd_ptr<ext::Bullet3::CPhysicsWorld> world;
		core::smart_refctd_ptr<instance_redirect_property_pool_t> cubes, cylinders, spheres, cones;
		core::vector<GPUObject> gpuObjects;
		btRigidBody* basePlateBody;
		ext::Bullet3::CPhysicsWorld::RigidBodyData basePlateRigidBodyData;
		core::vector<btRigidBody*> bodies;

		// Shapes RigidBody Data
		ext::Bullet3::CPhysicsWorld::RigidBodyData cubeRigidBodyData;
		ext::Bullet3::CPhysicsWorld::RigidBodyData cylinderRigidBodyData;
		ext::Bullet3::CPhysicsWorld::RigidBodyData sphereRigidBodyData;
		ext::Bullet3::CPhysicsWorld::RigidBodyData coneRigidBodyData;

		// kept state
		core::vector<uint32_t> scratchObjectIDs;
		core::vector<uint32_t> scratchInstanceRedirects;
		std::array<video::CPropertyPoolHandler::TransferRequest, object_property_pool_t::PropertyCount + 1> transfers;

		core::smart_refctd_ptr<object_property_pool_t> objectPool;
		video::CPropertyPoolHandler* propertyPoolHandler;

		core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
		core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
		core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };

		core::smart_refctd_ptr<video::IGPUPipelineLayout> gpuLayout;
		core::smart_refctd_ptr<video::IGPUDescriptorSet> globalDs;

		// TODO: roll into CommonAPI
		core::smart_refctd_ptr<video::IGPUCommandPool> computeCommandPool;

		bool frameDataFilled = false;
		size_t frame_count = 0ull;
		double time_sum = 0;
		double dtList[MaxFramesToAverage] = {};
		double dt = 0;
		std::chrono::system_clock::time_point lastTime;

		Camera cam = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());
		CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
		CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf[FRAMES_IN_FLIGHT];
		
		int resourceIx = -1;

		void deleteShapes(video::IGPUFence* fence, video::IGPUCommandBuffer* cmdbuf, instance_redirect_property_pool_t* pool, const uint32_t* objectsToDelete, const uint32_t* instanceRedirectsToDelete, const uint32_t count)
		{
			objectPool->freeProperties(objectsToDelete, objectsToDelete + count);
			// a bit of reuse
			scratchObjectIDs.resize(count);
			scratchInstanceRedirects.resize(count);
			uint32_t* srcAddrScratch = scratchObjectIDs.data();
			uint32_t* dstAddrScratch = scratchInstanceRedirects.data();
			//
			const bool needTransfer = video::CPropertyPoolHandler::freeProperties(pool, transfers.data() + 2u, instanceRedirectsToDelete, instanceRedirectsToDelete + count, srcAddrScratch, dstAddrScratch);
			if (needTransfer)
				propertyPoolHandler->transferProperties(utilities->getDefaultUpStreamingBuffer(), nullptr, cmdbuf, fence, transfers.data() + 2, transfers.data() + 3, logger.get());
		};
		void deleteBasedOnPhysicsPredicate(video::IGPUFence* fence, video::IGPUCommandBuffer* cmdbuf, instance_redirect_property_pool_t* pool, auto pred)
		{
			core::vector<uint32_t> objects, instances;
			for (auto& body : bodies)
			{
				if (!body)
					continue;
				auto* motionState = static_cast<CInstancedMotionState*>(body->getMotionState());
				if (motionState->getInstancePool() != pool || !pred(motionState))
					continue;

				objects.emplace_back(motionState->getObjectID());
				instances.emplace_back(motionState->getInstanceID());
				world->unbindRigidBody(body);
				world->deleteRigidBody(body);
				body = nullptr;
			}
			deleteShapes(fence, cmdbuf, pool, objects.data(), instances.data(), objects.size());
		};

		void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
		{
			window = std::move(wnd);
		}
	};

	APP_CONSTRUCTOR(MeshLoadersApp)

	void onAppInitialized_impl(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);

		CommonAPI::InitOutput<FBO_COUNT> initOutput;
		initOutput.window = core::smart_refctd_ptr(engine->window);
		CommonAPI::Init<WIN_W, WIN_H, FBO_COUNT>(initOutput, video::EAT_OPENGL, "MeshLoaders", nbl::asset::EF_D32_SFLOAT);
		engine->system = std::move(initOutput.system);
		engine->window = std::move(initOutput.window);
		engine->windowCb = std::move(initOutput.windowCb);
		engine->gl = std::move(initOutput.apiConnection);
		engine->surface = std::move(initOutput.surface);
		engine->gpuPhysicalDevice = std::move(initOutput.physicalDevice);

		engine->device = std::move(initOutput.logicalDevice);
		engine->utilities = std::move(initOutput.utilities);
		engine->queues = std::move(initOutput.queues);
		engine->graphicsQueue = engine->queues[decltype(initOutput)::EQT_GRAPHICS];
		engine->computeQueue = engine->queues[decltype(initOutput)::EQT_COMPUTE];
		engine->transferUpQueue = engine->queues[decltype(initOutput)::EQT_TRANSFER_UP];
		engine->swapchain = std::move(initOutput.swapchain);
		engine->renderpass = std::move(initOutput.renderpass);
		engine->fbos = std::move(initOutput.fbo);
		engine->commandPool = std::move(initOutput.commandPool);
		engine->assetManager = std::move(initOutput.assetManager);
		engine->cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		engine->logger = std::move(initOutput.logger);
		engine->inputSystem = std::move(initOutput.inputSystem);

		engine->computeCommandPool = engine->device->createCommandPool(engine->computeQueue->getFamilyIndex(), video::IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);

		// property transfer cmdbuffers
		core::smart_refctd_ptr<video::IGPUCommandBuffer> propXferCmdbuf[FRAMES_IN_FLIGHT];
		engine->device->createCommandBuffers(engine->computeCommandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, propXferCmdbuf);

		// Physics Setup
		engine->world = ext::Bullet3::CPhysicsWorld::create();
		engine->world->getWorld()->setGravity(btVector3(0, -5, 0));

		// BasePlate
		{
			core::matrix3x4SIMD baseplateMat;
			baseplateMat.setTranslation(core::vectorSIMDf(0.0, -1.0, 0.0));

			engine->basePlateRigidBodyData.mass = 0.0f;
			engine->basePlateRigidBodyData.shape = engine->world->createbtObject<btBoxShape>(btVector3(64, 1, 64));
			engine->basePlateRigidBodyData.trans = baseplateMat;

			engine->basePlateBody = engine->world->createRigidBody(engine->basePlateRigidBodyData);
			engine->world->bindRigidBody(engine->basePlateBody);
		}

		// set up
		engine->propertyPoolHandler = engine->utilities->getDefaultPropertyPoolHandler();
		auto createPropertyPoolWithMemory = [engine](auto& retval, uint32_t capacity, bool contiguous = false) -> void
		{
			using pool_type = std::remove_reference_t<decltype(retval)>::pointee;
			asset::SBufferRange<video::IGPUBuffer> blocks[pool_type::PropertyCount];

			video::IGPUBuffer::SCreationParams creationParams;
			creationParams.usage = asset::IBuffer::E_USAGE_FLAGS::EUF_STORAGE_BUFFER_BIT;
			creationParams.sharingMode = asset::E_SHARING_MODE::ESM_CONCURRENT;
			creationParams.queueFamilyIndices = 0u;
			creationParams.queueFamilyIndices = nullptr;

			for (auto i = 0u; i < pool_type::PropertyCount; i++)
			{
				auto& block = blocks[i];
				block.offset = 0u;
				block.size = pool_type::PropertySizes[i] * capacity;
				block.buffer = engine->device->createDeviceLocalGPUBufferOnDedMem(creationParams, block.size);
			}
			retval = pool_type::create(engine->device.get(), blocks, capacity, contiguous);
		};

		// Instance Redirects
		createPropertyPoolWithMemory(engine->cubes, 20u, true);
		createPropertyPoolWithMemory(engine->cylinders, 20u, true);
		createPropertyPoolWithMemory(engine->spheres, 20u, true);
		createPropertyPoolWithMemory(engine->cones, 10u, true);
		// global object data pool
		const uint32_t MaxSingleType = core::max(core::max(engine->cubes->getCapacity(), engine->cylinders->getCapacity()), core::max(engine->spheres->getCapacity(), engine->cones->getCapacity()));
		const uint32_t MaxNumObjects = engine->cubes->getCapacity() + engine->cylinders->getCapacity() + engine->spheres->getCapacity() + engine->cones->getCapacity();
		createPropertyPoolWithMemory(engine->objectPool, MaxNumObjects);

		// Physics
		engine->bodies = core::vector<btRigidBody*>(MaxNumObjects, nullptr);
		// Shapes RigidBody Data
		engine->cubeRigidBodyData = [engine]()
		{
			ext::Bullet3::CPhysicsWorld::RigidBodyData rigidBodyData;
			rigidBodyData.mass = 2.0f;
			rigidBodyData.shape = engine->world->createbtObject<btBoxShape>(btVector3(0.5, 0.5, 0.5));
			btVector3 inertia;
			rigidBodyData.shape->calculateLocalInertia(rigidBodyData.mass, inertia);
			rigidBodyData.inertia = ext::Bullet3::frombtVec3(inertia);
			return rigidBodyData;
		}();
		engine->cylinderRigidBodyData = [engine]()
		{
			ext::Bullet3::CPhysicsWorld::RigidBodyData rigidBodyData;
			rigidBodyData.mass = 1.0f;
			rigidBodyData.shape = engine->world->createbtObject<btCylinderShape>(btVector3(0.5, 0.5, 0.5));
			btVector3 inertia;
			rigidBodyData.shape->calculateLocalInertia(rigidBodyData.mass, inertia);
			rigidBodyData.inertia = ext::Bullet3::frombtVec3(inertia);
			return rigidBodyData;
		}();
		engine->sphereRigidBodyData = [engine]()
		{
			ext::Bullet3::CPhysicsWorld::RigidBodyData rigidBodyData;
			rigidBodyData.mass = 1.0f;
			rigidBodyData.shape = engine->world->createbtObject<btSphereShape>(0.5);
			btVector3 inertia;
			rigidBodyData.shape->calculateLocalInertia(rigidBodyData.mass, inertia);
			rigidBodyData.inertia = ext::Bullet3::frombtVec3(inertia);
			return rigidBodyData;
		}();
		engine->coneRigidBodyData = [engine]()
		{
			ext::Bullet3::CPhysicsWorld::RigidBodyData rigidBodyData;
			rigidBodyData.mass = 1.0f;
			rigidBodyData.shape = engine->world->createbtObject<btConeShape>(0.5, 1.0);
			btVector3 inertia;
			rigidBodyData.shape->calculateLocalInertia(rigidBodyData.mass, inertia);
			rigidBodyData.inertia = ext::Bullet3::frombtVec3(inertia);
			return rigidBodyData;
		}();

		// kept state
		uint32_t totalSpawned = 0u;
		core::vector<core::vectorSIMDf> initialColor;
		core::vector<core::matrix3x4SIMD> instanceTransforms;
		for (auto i = 0u; i < engine->transfers.size(); i++)
		{
			engine->transfers[i].flags = video::CPropertyPoolHandler::TransferRequest::EF_NONE;
			engine->transfers[i].device2device = false;
			engine->transfers[i].srcAddresses = nullptr;
		}
		// add a shape
		auto addShapes = [&](
			video::IGPUFence* fence,
			video::IGPUCommandBuffer* cmdbuf,
			ext::Bullet3::CPhysicsWorld::RigidBodyData rigidBodyData,
			instance_redirect_property_pool_t* pool, const uint32_t count,
			const core::matrix3x4SIMD& correction_mat = core::matrix3x4SIMD()
			) -> void
		{
			// prepare all the temporary array sizes
			engine->scratchObjectIDs.resize(count);
			engine->scratchInstanceRedirects.resize(count);
			initialColor.resize(count);
			instanceTransforms.resize(count);
			// allocate the object data
			std::fill_n(engine->scratchObjectIDs.data(), count, object_property_pool_t::invalid);
			engine->objectPool->allocateProperties(engine->scratchObjectIDs.data(), engine->scratchObjectIDs.data() + count);
			// now the redirects
			std::fill_n(engine->scratchInstanceRedirects.data(), count, instance_redirect_property_pool_t::invalid);
			pool->allocateProperties(engine->scratchInstanceRedirects.data(), engine->scratchInstanceRedirects.data() + count);
			// fill with data
			for (auto i = 0u; i < count; i++)
			{
				initialColor[i] = core::vectorSIMDf(float(totalSpawned % MaxNumObjects) / float(MaxNumObjects), 0.5f, 1.f);
				rigidBodyData.trans = instanceTransforms[i] = core::matrix3x4SIMD().setTranslation(core::vectorSIMDf(float(totalSpawned % 3) - 1.0f, totalSpawned * 1.5f, 0.f));
				totalSpawned++;
				// TODO: seems like `rigidBodyData.trans` is redundant to some matrices in the MotionStateBase
				const auto objectID = engine->scratchObjectIDs[i];
				auto& body = engine->bodies[objectID] = engine->world->createRigidBody(rigidBodyData);
				engine->world->bindRigidBody<CInstancedMotionState>(body, pool, objectID, engine->scratchInstanceRedirects[i], rigidBodyData.trans, correction_mat);
			}
			for (auto i = 0u; i < object_property_pool_t::PropertyCount; i++)
			{
				engine->transfers[i].setFromPool(engine->objectPool.get(), i);
				engine->transfers[i].elementCount = count;
				engine->transfers[i].dstAddresses = engine->scratchObjectIDs.data();
			}
			engine->transfers[0].source = initialColor.data();
			engine->transfers[1].source = instanceTransforms.data();
			//
			engine->transfers[2].setFromPool(pool, 0u);
			pool->indicesToAddresses(engine->scratchInstanceRedirects.begin(), engine->scratchInstanceRedirects.end(), engine->scratchInstanceRedirects.begin());
			engine->transfers[2].elementCount = count;
			engine->transfers[2].srcAddresses = nullptr;
			engine->transfers[2].dstAddresses = engine->scratchInstanceRedirects.data();
			engine->transfers[2].device2device = false;
			engine->transfers[2].source = engine->scratchObjectIDs.data();
			// set up the transfer/update
			engine->propertyPoolHandler->transferProperties(
				engine->utilities->getDefaultUpStreamingBuffer(), engine->utilities->getDefaultDownStreamingBuffer(),
				cmdbuf, fence, engine->transfers.data(), engine->transfers.data() + engine->transfers.size(), engine->logger.get()
			);
		};
		auto addCubes = [&](video::IGPUFence* fence, video::IGPUCommandBuffer* cmdbuf, const uint32_t count)
		{
			addShapes(std::move(fence), cmdbuf, engine->cubeRigidBodyData, engine->cubes.get(), count);
		};
		auto addCylinders = [&](video::IGPUFence* fence, video::IGPUCommandBuffer* cmdbuf, const uint32_t count)
		{
			addShapes(std::move(fence), cmdbuf, engine->cylinderRigidBodyData, engine->cylinders.get(), count, core::matrix3x4SIMD().setRotation(core::quaternion(core::PI<float>() / 2.f, 0.f, 0.f)));
		};
		auto addSpheres = [&](video::IGPUFence* fence, video::IGPUCommandBuffer* cmdbuf, const uint32_t count)
		{
			addShapes(std::move(fence), cmdbuf, engine->sphereRigidBodyData, engine->spheres.get(), count);
		};
		auto addCones = [&](video::IGPUFence* fence, video::IGPUCommandBuffer* cmdbuf, const uint32_t count)
		{
			addShapes(std::move(fence), cmdbuf, engine->coneRigidBodyData, engine->cones.get(), count, core::matrix3x4SIMD().setTranslation(core::vector3df_SIMD(0.f, -0.5f, 0.f)));
		};

		// setup scene
		{
			auto& fence = engine->frameComplete[FRAMES_IN_FLIGHT - 1] = engine->device->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
			auto cmdbuf = propXferCmdbuf[FRAMES_IN_FLIGHT - 1].get();

			cmdbuf->begin(video::IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
			addCubes(fence.get(), cmdbuf, 20u);
			addCylinders(fence.get(), cmdbuf, 20u);
			addSpheres(fence.get(), cmdbuf, 20u);
			addCones(fence.get(), cmdbuf, 10u);
			cmdbuf->end();

			video::IGPUQueue::SSubmitInfo submit;
			{
				submit.commandBufferCount = 1u;
				submit.commandBuffers = &cmdbuf;
				submit.signalSemaphoreCount = 0u;
				submit.waitSemaphoreCount = 0u;

				engine->computeQueue->submit(1u, &submit, fence.get());
			}
		}


		video::IGPUObjectFromAssetConverter CPU2GPU;
		// set up shader inputs
		core::smart_refctd_ptr<asset::ICPUPipelineLayout> cpuLayout;
		{
			//
			constexpr auto GLOBAL_DS_COUNT = 2u;
			{
				asset::ICPUDescriptorSetLayout::SBinding bindings[GLOBAL_DS_COUNT];
				for (auto i = 0u; i < GLOBAL_DS_COUNT; i++)
				{
					bindings[i].binding = i;
					bindings[i].type = asset::EDT_STORAGE_BUFFER;
					bindings[i].count = 1u;
					bindings[i].stageFlags = asset::ISpecializedShader::ESS_VERTEX;
					bindings[i].samplers = nullptr;
				}
				auto dsLayout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bindings, bindings + GLOBAL_DS_COUNT);

				asset::SPushConstantRange range[1] = { asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD) };
				cpuLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(range, range + 1u, std::move(dsLayout));
			}
			engine->gpuLayout = CPU2GPU.getGPUObjectsFromAssets(&cpuLayout.get(), &cpuLayout.get() + 1, engine->cpu2gpuParams)->front();

			{
				auto globalDsLayout = engine->gpuLayout->getDescriptorSetLayout(0u);
				auto pool = engine->device->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &globalDsLayout, &globalDsLayout + 1u);
				engine->globalDs = engine->device->createGPUDescriptorSet(pool.get(), core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout>(globalDsLayout)); // TODO: change method signature to make it obvious we're taking shared ownership of a pool

				video::IGPUDescriptorSet::SWriteDescriptorSet writes[GLOBAL_DS_COUNT];
				video::IGPUDescriptorSet::SDescriptorInfo infos[GLOBAL_DS_COUNT];
				for (auto i = 0u; i < GLOBAL_DS_COUNT; i++)
				{
					writes[i].dstSet = engine->globalDs.get();
					writes[i].binding = i;
					writes[i].arrayElement = 0u;
					writes[i].count = 1u;
					writes[i].descriptorType = asset::EDT_STORAGE_BUFFER;
					writes[i].info = infos + i;

					const auto& poolBuff = engine->objectPool->getPropertyMemoryBlock(i);
					infos[i].desc = poolBuff.buffer;
					infos[i].buffer.offset = poolBuff.offset;
					infos[i].buffer.size = poolBuff.size;
				}
				engine->device->updateDescriptorSets(GLOBAL_DS_COUNT, writes, 0u, nullptr);
			}
		}


		// Geom Create
		auto geometryCreator = engine->assetManager->getGeometryCreator();
		auto cubeGeom = geometryCreator->createCubeMesh(core::vector3df(1.0f, 1.0f, 1.0f));
		auto cylinderGeom = geometryCreator->createCylinderMesh(0.5f, 0.5f, 20);
		auto sphereGeom = geometryCreator->createSphereMesh(0.5f);
		auto coneGeom = geometryCreator->createConeMesh(0.5f, 1.0f, 32);

		// Creating CPU Shaders 
		auto createCPUSpecializedShaderFromSource = [=](const char* path, asset::ISpecializedShader::E_SHADER_STAGE stage) -> core::smart_refctd_ptr<asset::ICPUSpecializedShader>
		{
			asset::IAssetLoader::SAssetLoadParams params{};
			params.logger = engine->logger.get();
			//params.relativeDir = tmp.c_str();
			auto spec = engine->assetManager->getAsset(path, params).getContents();
			if (spec.empty())
				assert(false);

			return core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*spec.begin());
		};

		auto vs = createCPUSpecializedShaderFromSource("../mesh.vert", asset::ISpecializedShader::ESS_VERTEX);
		auto fs = createCPUSpecializedShaderFromSource("../mesh.frag", asset::ISpecializedShader::ESS_FRAGMENT);
		asset::ICPUSpecializedShader* shaders[2]{ vs.get(), fs.get() };

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
		auto cpuMeshCube = createMeshBufferFromGeomCreatorReturnType(cubeGeom, engine->assetManager.get(), shaders, shaders + 2);
		auto cpuMeshCylinder = createMeshBufferFromGeomCreatorReturnType(cylinderGeom, engine->assetManager.get(), shaders, shaders + 2);
		auto cpuMeshSphere = createMeshBufferFromGeomCreatorReturnType(sphereGeom, engine->assetManager.get(), shaders, shaders + 2);
		auto cpuMeshCone = createMeshBufferFromGeomCreatorReturnType(coneGeom, engine->assetManager.get(), shaders, shaders + 2);
		dummyPplnLayout = nullptr;

		// Create GPU Objects (IGPUMeshBuffer + GraphicsPipeline)
		auto createGPUObject = [&](const video::IPropertyPool* pool, asset::ICPUMeshBuffer* cpuMesh, asset::E_FACE_CULL_MODE faceCullingMode = asset::EFCM_BACK_BIT) -> GPUObject
		{
			auto pipeline = cpuMesh->getPipeline();
			//
			auto vtxinputParams = pipeline->getVertexInputParams();
			vtxinputParams.bindings[15].inputRate = asset::EVIR_PER_INSTANCE;
			vtxinputParams.bindings[15].stride = sizeof(uint32_t);
			vtxinputParams.attributes[15].binding = 15;
			vtxinputParams.attributes[15].relativeOffset = 0;
			vtxinputParams.attributes[15].format = asset::E_FORMAT::EF_R32_UINT;
			vtxinputParams.enabledAttribFlags |= 0x1u << 15;
			vtxinputParams.enabledBindingFlags |= 0x1u << 15;
			// we're working with RH coordinate system(view proj) and in that case the cubeGeom frontFace is NOT CCW.
			auto rasterParams = pipeline->getRasterizationParams();
			rasterParams.frontFaceIsCCW = 0;
			rasterParams.faceCullingMode = faceCullingMode;
			// for wireframe rendering
#if 0
			pipeline->getRasterizationParams().polygonMode = asset::EPM_LINE;
#endif

			asset::ICPUSpecializedShader* cpuShaders[2] = {
				pipeline->getShaderAtStage(asset::ISpecializedShader::ESS_VERTEX),
				pipeline->getShaderAtStage(asset::ISpecializedShader::ESS_FRAGMENT)
			};
			cpuMesh->setPipeline(core::make_smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>(
				core::smart_refctd_ptr(cpuLayout), &cpuShaders[0], &cpuShaders[0] + 2,
				vtxinputParams,
				pipeline->getBlendParams(),
				pipeline->getPrimitiveAssemblyParams(),
				rasterParams
				));

			GPUObject ret = {};
			ret.pool = pool;
			// get the mesh
			ret.gpuMesh = CPU2GPU.getGPUObjectsFromAssets(&cpuMesh, &cpuMesh + 1, engine->cpu2gpuParams)->front();
			asset::SBufferBinding<video::IGPUBuffer> instanceRedirectBufBnd;
			instanceRedirectBufBnd.offset = 0u;
			instanceRedirectBufBnd.buffer = pool->getPropertyMemoryBlock(0u).buffer;
			ret.gpuMesh->setVertexBufferBinding(std::move(instanceRedirectBufBnd), 15u);
			//
			video::IGPUGraphicsPipeline::SCreationParams gp_params;
			gp_params.rasterizationSamplesHint = asset::IImage::ESCF_1_BIT;
			gp_params.renderpass = core::smart_refctd_ptr<video::IGPURenderpass>(engine->renderpass);
			gp_params.renderpassIndependent = core::smart_refctd_ptr<const video::IGPURenderpassIndependentPipeline>(ret.gpuMesh->getPipeline());
			gp_params.subpassIx = 0u;
			ret.graphicsPipeline = engine->device->createGPUGraphicsPipeline(nullptr, std::move(gp_params));

			return ret;
		};

		engine->gpuObjects = {
			createGPUObject(engine->cubes.get(),cpuMeshCube.get()),
			createGPUObject(engine->cylinders.get(),cpuMeshCylinder.get(),asset::EFCM_NONE),
			createGPUObject(engine->spheres.get(),cpuMeshSphere.get()),
			createGPUObject(engine->cones.get(),cpuMeshCone.get(),asset::EFCM_NONE)
		};


		//
		engine->lastTime = std::chrono::system_clock::now();

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			engine->imageAcquire[i] = engine->device->createSemaphore();
			engine->renderFinished[i] = engine->device->createSemaphore();
		}

		for (size_t i = 0ull; i < MaxFramesToAverage; ++i) {
			engine->dtList[i] = 0.0;
		}

		engine->dt = 0;

		// Camera 
		core::vectorSIMDf cameraPosition(0, 5, -10);
		matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(60), float(WIN_W) / WIN_H, 0.01f, 500.0f);
		engine->cam = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), proj);

		// polling for events!
		CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
		CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		engine->device->createCommandBuffers(engine->commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, engine->cmdbuf);
	}

	void onAppTerminated_impl(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);

		engine->world->unbindRigidBody(engine->basePlateBody, false);
		engine->world->deleteRigidBody(engine->basePlateBody);
		engine->world->deletebtObject(engine->basePlateRigidBodyData.shape);

		auto alwaysTrue = [](auto dummy) -> bool {return true; };
		engine->deleteBasedOnPhysicsPredicate(nullptr, nullptr, engine->cylinders.get(), alwaysTrue);
		engine->deleteBasedOnPhysicsPredicate(nullptr, nullptr, engine->spheres.get(), alwaysTrue);
		engine->deleteBasedOnPhysicsPredicate(nullptr, nullptr, engine->cones.get(), alwaysTrue);
		engine->deleteBasedOnPhysicsPredicate(nullptr, nullptr, engine->cubes.get(), alwaysTrue);

		engine->world->deletebtObject(engine->cubeRigidBodyData.shape);
		engine->world->deletebtObject(engine->cylinderRigidBodyData.shape);
		engine->world->deletebtObject(engine->sphereRigidBodyData.shape);
		engine->world->deletebtObject(engine->coneRigidBodyData.shape);
	}

	void workLoopBody(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);

		engine->resourceIx++;
		if (engine->resourceIx >= FRAMES_IN_FLIGHT) {
			engine->resourceIx = 0;
		}

		// Timing
		auto renderStart = std::chrono::system_clock::now();
		engine->dt = std::chrono::duration_cast<std::chrono::milliseconds>(renderStart - engine->lastTime).count();
		engine->lastTime = renderStart;

		// Calculate Simple Moving Average for FrameTime
		{
			engine->time_sum -= engine->dtList[engine->frame_count];
			engine->time_sum += engine->dt;
			engine->dtList[engine->frame_count] = engine->dt;
			engine->frame_count++;
			if (engine->frame_count >= MaxFramesToAverage) {
				engine->frame_count = 0;
				engine->frameDataFilled = true;
			}
		}
		double averageFrameTime = (engine->frameDataFilled) ? (engine->time_sum / (double)MaxFramesToAverage) : (engine->time_sum / engine->frame_count);
		// logger->log("averageFrameTime = %f",system::ILogger::ELL_INFO, averageFrameTime);

		// Calculate Next Presentation Time Stamp
		auto averageFrameTimeDuration = std::chrono::duration<double, std::milli>(averageFrameTime);
		auto nextPresentationTime = renderStart + averageFrameTimeDuration;
		auto nextPresentationTimeStamp = std::chrono::duration_cast<std::chrono::microseconds>(nextPresentationTime.time_since_epoch());

		// Input 
		engine->inputSystem->getDefaultMouse(&engine->mouse);
		engine->inputSystem->getDefaultKeyboard(&engine->keyboard);

		engine->cam.beginInputProcessing(nextPresentationTimeStamp);
		engine->mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { engine->cam.mouseProcess(events); }, engine->logger.get());
		engine->keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void {
			engine->cam.keyboardProcess(events);
			}, engine->logger.get());
		engine->cam.endInputProcessing(nextPresentationTimeStamp);

		auto& cb = engine->cmdbuf[engine->resourceIx];
		auto& fence = engine->frameComplete[engine->resourceIx];
		if (fence)
			while (engine->device->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT)
			{
			}
		else
			fence = engine->device->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

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
		// acquire image 
		uint32_t imgnum = 0u;
		engine->swapchain->acquireNextImage(MAX_TIMEOUT, engine->imageAcquire[engine->resourceIx].get(), nullptr, &imgnum);
		// Update instances buffer 
		{
			// Update Physics (TODO: fixed timestep)
			engine->world->getWorld()->stepSimulation(engine->dt);

			video::CPropertyPoolHandler::TransferRequest request;
			request.setFromPool(engine->objectPool.get(), TransformPropertyID);
			request.flags = video::CPropertyPoolHandler::TransferRequest::EF_NONE;
			request.elementCount = CInstancedMotionState::s_updateAddresses.size();
			request.srcAddresses = nullptr;
			request.dstAddresses = CInstancedMotionState::s_updateAddresses.data();
			request.device2device = false;
			request.source = CInstancedMotionState::s_updateData.data();
			// TODO: why does the very first update set matrices to identity?
			auto result = engine->propertyPoolHandler->transferProperties(engine->utilities->getDefaultUpStreamingBuffer(), engine->utilities->getDefaultDownStreamingBuffer(), cb.get(), fence.get(), &request, &request + 1u, engine->logger.get());
			assert(result.transferSuccess);
			// ensure dependency from transfer to any following transfers
			{
				asset::SMemoryBarrier memBarrier; // cba to list the buffers one-by-one, but probably should
				memBarrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
				memBarrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
				cb->pipelineBarrier(
					asset::EPSF_COMPUTE_SHADER_BIT, asset::EPSF_COMPUTE_SHADER_BIT, asset::EDF_NONE,
					1u, &memBarrier, 0u, nullptr, 0u, nullptr
				);
			}
			CInstancedMotionState::s_updateAddresses.clear();
			CInstancedMotionState::s_updateData.clear();

			auto falledFromMap = [](CInstancedMotionState* motionState) -> bool
			{
				btTransform tform;
				motionState->getWorldTransform(tform);
				return tform.getOrigin().getY() < -128.f;
			};
			engine->deleteBasedOnPhysicsPredicate(fence.get(), cb.get(), engine->cones.get(), falledFromMap);
			engine->deleteBasedOnPhysicsPredicate(fence.get(), cb.get(), engine->cubes.get(), falledFromMap);
			engine->deleteBasedOnPhysicsPredicate(fence.get(), cb.get(), engine->spheres.get(), falledFromMap);
			engine->deleteBasedOnPhysicsPredicate(fence.get(), cb.get(), engine->cylinders.get(), falledFromMap);
			// ensure dependency from transfer to any following transfers and vertex shaders
			{
				asset::SMemoryBarrier memBarrier; // cba to list the buffers one-by-one, but probably should
				memBarrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
				memBarrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_READ_BIT | asset::EAF_VERTEX_ATTRIBUTE_READ_BIT);
				cb->pipelineBarrier(
					asset::EPSF_COMPUTE_SHADER_BIT, core::bitflag(asset::EPSF_COMPUTE_SHADER_BIT) | asset::EPSF_VERTEX_INPUT_BIT | asset::EPSF_VERTEX_SHADER_BIT, asset::EDF_NONE,
					1u, &memBarrier, 0u, nullptr, 0u, nullptr
				);
			}
		}
		// renderpass
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

			info.renderpass = engine->renderpass;
			info.framebuffer = engine->fbos[imgnum];
			info.clearValueCount = 2u;
			info.clearValues = clearValues;
			info.renderArea.offset = { 0, 0 };
			info.renderArea.extent = { WIN_W, WIN_H };
			cb->beginRenderPass(&info, asset::ESC_INLINE);
		}
		// draw
		cb->bindDescriptorSets(asset::EPBP_GRAPHICS, engine->gpuLayout.get(), 0u, 1u, &engine->globalDs.get());
		{
			auto viewProj = engine->cam.getConcatenatedMatrix();

			// Draw Stuff 
			for (uint32_t i = 0; i < engine->gpuObjects.size(); ++i) {
				auto& gpuObject = engine->gpuObjects[i];

				cb->bindGraphicsPipeline(gpuObject.graphicsPipeline.get());
				cb->pushConstants(gpuObject.graphicsPipeline->getRenderpassIndependentPipeline()->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), viewProj.pointer());
				gpuObject.gpuMesh->setInstanceCount(gpuObject.pool->getAllocated());
				cb->drawMeshBuffer(gpuObject.gpuMesh.get());
			}
		}
		cb->endRenderPass();
		cb->end();

		CommonAPI::Submit(engine->device.get(), engine->swapchain.get(), cb.get(), engine->graphicsQueue, engine->imageAcquire[engine->resourceIx].get(), engine->renderFinished[engine->resourceIx].get(), fence.get());
		CommonAPI::Present(engine->device.get(), engine->swapchain.get(), engine->graphicsQueue, engine->renderFinished[engine->resourceIx].get(), imgnum);
	}

	bool keepRunning(void* params) override
	{
		Nabla* engine = static_cast<Nabla*>(params);
		return engine->windowCb->isWindowOpen();
	}
};

NBL_COMMON_API_MAIN(MeshLoadersApp, MeshLoadersApp::Nabla)