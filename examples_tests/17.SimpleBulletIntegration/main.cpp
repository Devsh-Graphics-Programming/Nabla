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

		inline auto getInstancePool() const {return m_instancePool;}
		inline auto getObjectID() const {return m_objectID;}
		inline auto getInstanceID() const {return m_instanceID;}

		static core::vector<uint32_t> s_updateAddresses;
		static core::vector<core::matrix3x4SIMD> s_updateData;
	protected:
		btTransform m_cachedMat;
		uint32_t m_objectID,m_instanceID;
		const void* m_instancePool;
};
core::vector<uint32_t> CInstancedMotionState::s_updateAddresses;
core::vector<core::matrix3x4SIMD> CInstancedMotionState::s_updateData;

class BulletSampleApp : public ApplicationBase
{
	constexpr static uint32_t WIN_W = 1280u;
	constexpr static uint32_t WIN_H = 720u;
	constexpr static uint32_t SC_IMG_COUNT = 3u;
	constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
	constexpr static size_t MaxFramesToAverage = 100ull;
	static_assert(FRAMES_IN_FLIGHT>SC_IMG_COUNT);

public:
	using instance_redirect_property_pool_t = video::CPropertyPool<core::allocator, uint32_t>;
	using object_property_pool_t = video::CPropertyPool<core::allocator, core::vectorSIMDf, core::matrix3x4SIMD>;

	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	static constexpr uint32_t TransformPropertyID = 1u;

	enum E_OBJECT
	{
		E_CUBE = 0,
		E_CYLINDER,
		E_SPHERE,
		E_CONE,
		E_COUNT
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
	core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbo;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> commandPools; // TODO: Multibuffer and reset the commandpools
	core::smart_refctd_ptr<nbl::system::ISystem> system;
	core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	core::smart_refctd_ptr<nbl::system::ILogger> logger;
	core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
	video::IGPUObjectFromAssetConverter cpu2gpu;

	int32_t m_resourceIx = -1;

	double m_dt = 0.0;
	std::chrono::system_clock::time_point m_lastTime;
	bool m_frameDataFilled = false;
	size_t m_frame_count = 0ull;
	double m_time_sum = 0.0;
	double m_dtList[MaxFramesToAverage] = { 0.0 };

	core::smart_refctd_ptr<ext::Bullet3::CPhysicsWorld> m_world = nullptr;
	btRigidBody* m_basePlateBody = nullptr;
	ext::Bullet3::CPhysicsWorld::RigidBodyData m_basePlateRigidBodyData;
	core::smart_refctd_ptr<object_property_pool_t> m_objectPool;
	core::vector<uint32_t> m_scratchObjectIDs;
	core::vector<uint32_t> m_scratchInstanceRedirects;
	std::array<video::CPropertyPoolHandler::TransferRequest, object_property_pool_t::PropertyCount + 1> m_transfers;
	core::vector<btRigidBody*> m_bodies;
	core::smart_refctd_ptr<instance_redirect_property_pool_t> m_cubes, m_cylinders, m_spheres, m_cones;
	ext::Bullet3::CPhysicsWorld::RigidBodyData m_cubeRigidBodyData;
	ext::Bullet3::CPhysicsWorld::RigidBodyData m_cylinderRigidBodyData;
	ext::Bullet3::CPhysicsWorld::RigidBodyData m_sphereRigidBodyData;
	ext::Bullet3::CPhysicsWorld::RigidBodyData m_coneRigidBodyData;

	// polling for events!
	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> m_mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> m_keyboard;

	std::unique_ptr<Camera> m_cam = nullptr;

	core::smart_refctd_ptr<video::IGPUSemaphore> m_imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> m_renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUFence> m_frameComplete[FRAMES_IN_FLIGHT] = { nullptr };

	core::smart_refctd_ptr<video::IGPUCommandBuffer> m_cmdbuf[FRAMES_IN_FLIGHT] = { nullptr };

	core::smart_refctd_ptr<video::IGPUPipelineLayout> m_gpuLayout = nullptr;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> m_globalDs = nullptr;

	// TODO: replace with an actual scenemanager
	struct GPUObject
	{
		const video::IPropertyPool* pool;
		core::smart_refctd_ptr<video::IGPUMeshBuffer> gpuMesh;
		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> graphicsPipeline;
	};

	core::vector<GPUObject> m_gpuObjects;

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

	APP_CONSTRUCTOR(BulletSampleApp);

	void onAppInitialized_impl() override
	{
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

		const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT);
		const video::ISurface::SFormat surfaceFormat(asset::EF_B8G8R8A8_SRGB, asset::ECP_COUNT, asset::EOTF_UNKNOWN);

		CommonAPI::InitOutput initOutput;
		initOutput.window = core::smart_refctd_ptr(window);
		CommonAPI::Init(initOutput,
			video::EAT_VULKAN,
			"Physics Simulation",
			requiredInstanceFeatures,
			optionalInstanceFeatures,
			requiredDeviceFeatures,
			optionalDeviceFeatures,
			WIN_W, WIN_H, SC_IMG_COUNT,
			swapchainImageUsage,
			surfaceFormat,
			asset::EF_D32_SFLOAT);

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
		fbo = std::move(initOutput.fbo);
		commandPools = std::move(initOutput.commandPools);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);

		const auto& computeCommandPool = commandPools[CommonAPI::InitOutput::EQT_COMPUTE];

		// property transfer cmdbuffers
		core::smart_refctd_ptr<video::IGPUCommandBuffer> propXferCmdbuf[FRAMES_IN_FLIGHT];
		logicalDevice->createCommandBuffers(computeCommandPool.get(),video::IGPUCommandBuffer::EL_PRIMARY,FRAMES_IN_FLIGHT,propXferCmdbuf);

		// Physics Setup
		m_world = ext::Bullet3::CPhysicsWorld::create();
		m_world->getWorld()->setGravity(btVector3(0, -5, 0));

		// BasePlate
		{
			core::matrix3x4SIMD baseplateMat;
			baseplateMat.setTranslation(core::vectorSIMDf(0.0, -1.0, 0.0));

			m_basePlateRigidBodyData.mass = 0.0f;
			m_basePlateRigidBodyData.shape = m_world->createbtObject<btBoxShape>(btVector3(64,1,64));
			m_basePlateRigidBodyData.trans = baseplateMat;

			m_basePlateBody = m_world->createRigidBody(m_basePlateRigidBodyData);
			m_world->bindRigidBody(m_basePlateBody);
		}

		// set up
		auto propertyPoolHandler = utilities->getDefaultPropertyPoolHandler();
		auto createPropertyPoolWithMemory = [this](auto& retval, uint32_t capacity, bool contiguous=false) -> void
		{
			using pool_type = std::remove_reference_t<decltype(retval)>::pointee;
			asset::SBufferRange<video::IGPUBuffer> blocks[pool_type::PropertyCount];

			video::IGPUBuffer::SCreationParams creationParams;
			creationParams.usage = static_cast<asset::IBuffer::E_USAGE_FLAGS>(asset::IBuffer::EUF_STORAGE_BUFFER_BIT | asset::IBuffer::EUF_VERTEX_BUFFER_BIT);
			creationParams.sharingMode = asset::E_SHARING_MODE::ESM_EXCLUSIVE;
			creationParams.queueFamilyIndices = 0u;
			creationParams.queueFamilyIndices = nullptr;

			for (auto i=0u; i<pool_type::PropertyCount; i++)
			{
				auto& block = blocks[i];
				block.offset = 0u;
				block.size = pool_type::PropertySizes[i]*capacity;
				block.buffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, block.size);
			}
			retval = pool_type::create(logicalDevice.get(),blocks,capacity,contiguous);
		};

		// Instance Redirects
		createPropertyPoolWithMemory(m_cubes,20u,true);
		createPropertyPoolWithMemory(m_cylinders,20u,true);
		createPropertyPoolWithMemory(m_spheres,20u,true);
		createPropertyPoolWithMemory(m_cones,10u,true);
		// global object data pool
		const uint32_t MaxSingleType = core::max(core::max(m_cubes->getCapacity(),m_cylinders->getCapacity()),core::max(m_spheres->getCapacity(),m_cones->getCapacity()));
		// const uint32_t MaxSingleType = core::max(core::max(m_cubes->getCapacity(), m_cylinders->getCapacity()), m_spheres->getCapacity());
		const uint32_t MaxNumObjects = m_cubes->getCapacity()+m_cylinders->getCapacity()+m_spheres->getCapacity()+m_cones->getCapacity();
		// const uint32_t MaxNumObjects = m_cubes->getCapacity() + m_cylinders->getCapacity() + m_spheres->getCapacity();/* +m_cones->getCapacity();*/
		createPropertyPoolWithMemory(m_objectPool,MaxNumObjects);

		// Physics
		m_bodies.resize(MaxNumObjects, nullptr);
		// Shapes RigidBody Data
		m_cubeRigidBodyData = [this]()
		{
			ext::Bullet3::CPhysicsWorld::RigidBodyData rigidBodyData;
			rigidBodyData.mass = 2.0f;
			rigidBodyData.shape = m_world->createbtObject<btBoxShape>(btVector3(0.5, 0.5, 0.5));
			btVector3 inertia;
			rigidBodyData.shape->calculateLocalInertia(rigidBodyData.mass, inertia);
			rigidBodyData.inertia = ext::Bullet3::frombtVec3(inertia);
			return rigidBodyData;
		}();
		m_cylinderRigidBodyData = [this]()
		{
			ext::Bullet3::CPhysicsWorld::RigidBodyData rigidBodyData;
			rigidBodyData.mass = 1.0f;
			rigidBodyData.shape = m_world->createbtObject<btCylinderShape>(btVector3(0.5, 0.5, 0.5));
			btVector3 inertia;
			rigidBodyData.shape->calculateLocalInertia(rigidBodyData.mass, inertia);
			rigidBodyData.inertia = ext::Bullet3::frombtVec3(inertia);
			return rigidBodyData;
		}();
		m_sphereRigidBodyData = [this]()
		{
			ext::Bullet3::CPhysicsWorld::RigidBodyData rigidBodyData;
			rigidBodyData.mass = 1.0f;
			rigidBodyData.shape = m_world->createbtObject<btSphereShape>(0.5);
			btVector3 inertia;
			rigidBodyData.shape->calculateLocalInertia(rigidBodyData.mass, inertia);
			rigidBodyData.inertia = ext::Bullet3::frombtVec3(inertia);
			return rigidBodyData;
		}();
		m_coneRigidBodyData = [this]()
		{
			ext::Bullet3::CPhysicsWorld::RigidBodyData rigidBodyData;
			rigidBodyData.mass = 1.0f;
			rigidBodyData.shape = m_world->createbtObject<btConeShape>(0.5, 1.0);
			btVector3 inertia;
			rigidBodyData.shape->calculateLocalInertia(rigidBodyData.mass, inertia);
			rigidBodyData.inertia = ext::Bullet3::frombtVec3(inertia);
			return rigidBodyData;
		}();

		// kept state
		uint32_t totalSpawned = 0u;
		
		core::vector<core::vectorSIMDf> initialColor;
		core::vector<core::matrix3x4SIMD> instanceTransforms;
		for (auto i = 0u; i < m_transfers.size(); i++)
		{
			m_transfers[i].flags = video::CPropertyPoolHandler::TransferRequest::EF_NONE;
			m_transfers[i].device2device = false;
			m_transfers[i].srcAddresses = nullptr;
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
			m_scratchObjectIDs.resize(count);
			m_scratchInstanceRedirects.resize(count);
			initialColor.resize(count);
			instanceTransforms.resize(count);
			// allocate the object data
			std::fill_n(m_scratchObjectIDs.data(), count, object_property_pool_t::invalid);
			m_objectPool->allocateProperties(m_scratchObjectIDs.data(), m_scratchObjectIDs.data() + count);
			// now the redirects
			std::fill_n(m_scratchInstanceRedirects.data(), count, instance_redirect_property_pool_t::invalid);
			pool->allocateProperties(m_scratchInstanceRedirects.data(), m_scratchInstanceRedirects.data() + count);
			// fill with data
			for (auto i = 0u; i < count; i++)
			{
				initialColor[i] = core::vectorSIMDf(float(totalSpawned % MaxNumObjects) / float(MaxNumObjects), 0.5f, 1.f);
				rigidBodyData.trans = instanceTransforms[i] = core::matrix3x4SIMD().setTranslation(core::vectorSIMDf(float(totalSpawned % 3) - 1.0f, totalSpawned * 1.5f, 0.f));
				totalSpawned++;
				// TODO: seems like `rigidBodyData.trans` is redundant to some matrices in the MotionStateBase
				const auto objectID = m_scratchObjectIDs[i];
				auto& body = m_bodies[objectID] = m_world->createRigidBody(rigidBodyData);
				m_world->bindRigidBody<CInstancedMotionState>(body, pool, objectID, m_scratchInstanceRedirects[i], rigidBodyData.trans, correction_mat);
			}
			for (auto i = 0u; i < object_property_pool_t::PropertyCount; i++)
			{
				m_transfers[i].setFromPool(m_objectPool.get(), i);
				m_transfers[i].elementCount = count;
				m_transfers[i].dstAddresses = m_scratchObjectIDs.data();
			}
			m_transfers[0].source = initialColor.data();
			m_transfers[1].source = instanceTransforms.data();
			//
			m_transfers[2].setFromPool(pool, 0u);
			pool->indicesToAddresses(m_scratchInstanceRedirects.begin(), m_scratchInstanceRedirects.end(), m_scratchInstanceRedirects.begin());
			m_transfers[2].elementCount = count;
			m_transfers[2].srcAddresses = nullptr;
			m_transfers[2].dstAddresses = m_scratchInstanceRedirects.data();
			m_transfers[2].device2device = false;
			m_transfers[2].source = m_scratchObjectIDs.data();
			// set up the transfer/update
			propertyPoolHandler->transferProperties(
				utilities->getDefaultUpStreamingBuffer(),
				utilities->getDefaultDownStreamingBuffer(),
				cmdbuf, fence, m_transfers.data(), m_transfers.data() + m_transfers.size(),
				logger.get()
			);
		};
		auto addCubes = [&](video::IGPUFence* fence, video::IGPUCommandBuffer* cmdbuf, const uint32_t count)
		{
			addShapes(std::move(fence), cmdbuf, m_cubeRigidBodyData, m_cubes.get(), count);
		};
		auto addCylinders = [&](video::IGPUFence* fence, video::IGPUCommandBuffer* cmdbuf, const uint32_t count)
		{
			addShapes(std::move(fence), cmdbuf, m_cylinderRigidBodyData, m_cylinders.get(), count, core::matrix3x4SIMD().setRotation(core::quaternion(core::PI<float>() / 2.f, 0.f, 0.f)));
		};
		auto addSpheres = [&](video::IGPUFence* fence, video::IGPUCommandBuffer* cmdbuf, const uint32_t count)
		{
			addShapes(std::move(fence), cmdbuf, m_sphereRigidBodyData, m_spheres.get(), count);
		};
		auto addCones = [&](video::IGPUFence* fence, video::IGPUCommandBuffer* cmdbuf, const uint32_t count)
		{
			addShapes(std::move(fence), cmdbuf, m_coneRigidBodyData, m_cones.get(), count, core::matrix3x4SIMD().setTranslation(core::vector3df_SIMD(0.f, -0.5f, 0.f)));
		};

		// setup scene
		{
			auto& fence = m_frameComplete[FRAMES_IN_FLIGHT - 1] = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
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

				queues[CommonAPI::InitOutput::EQT_COMPUTE]->submit(1u, &submit, fence.get());
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
					bindings[i].stageFlags = asset::IShader::ESS_VERTEX;
					bindings[i].samplers = nullptr;
				}
				auto dsLayout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bindings, bindings + GLOBAL_DS_COUNT);

				asset::SPushConstantRange range[1] = { asset::IShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD) };
				cpuLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(range, range + 1u, std::move(dsLayout));
			}
			m_gpuLayout = CPU2GPU.getGPUObjectsFromAssets(&cpuLayout.get(), &cpuLayout.get() + 1, cpu2gpuParams)->front();

			{
				auto globalDsLayout = m_gpuLayout->getDescriptorSetLayout(0u);
				auto pool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &globalDsLayout, &globalDsLayout + 1u);
				m_globalDs = logicalDevice->createGPUDescriptorSet(pool.get(), core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout>(globalDsLayout)); // TODO: change method signature to make it obvious we're taking shared ownership of a pool

				video::IGPUDescriptorSet::SWriteDescriptorSet writes[GLOBAL_DS_COUNT];
				video::IGPUDescriptorSet::SDescriptorInfo infos[GLOBAL_DS_COUNT];
				for (auto i = 0u; i < GLOBAL_DS_COUNT; i++)
				{
					writes[i].dstSet = m_globalDs.get();
					writes[i].binding = i;
					writes[i].arrayElement = 0u;
					writes[i].count = 1u;
					writes[i].descriptorType = asset::EDT_STORAGE_BUFFER;
					writes[i].info = infos + i;

					const auto& poolBuff = m_objectPool->getPropertyMemoryBlock(i);
					infos[i].desc = poolBuff.buffer;
					infos[i].buffer.offset = poolBuff.offset;
					infos[i].buffer.size = poolBuff.size;
				}
				logicalDevice->updateDescriptorSets(GLOBAL_DS_COUNT, writes, 0u, nullptr);
			}
		}

		// Geom Create
		auto geometryCreator = assetManager->getGeometryCreator();
		auto cubeGeom = geometryCreator->createCubeMesh(core::vector3df(1.0f, 1.0f, 1.0f));
		auto cylinderGeom = geometryCreator->createCylinderMesh(0.5f, 0.5f, 20);
		auto sphereGeom = geometryCreator->createSphereMesh(0.5f);
		auto coneGeom = geometryCreator->createConeMesh(0.5f, 1.0f, 32);

		// Creating CPU Shaders 
		auto createCPUSpecializedShaderFromSource = [=](const char* path, asset::IShader::E_SHADER_STAGE stage) -> core::smart_refctd_ptr<asset::ICPUSpecializedShader>
		{
			asset::IAssetLoader::SAssetLoadParams params{};
			params.workingDirectory = CWDOnStartup;
			params.logger = logger.get();
			//params.relativeDir = tmp.c_str();
			auto spec = assetManager->getAsset(path, params).getContents();
			if (spec.empty())
				assert(false);

			return core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*spec.begin());
		};

		auto vs = createCPUSpecializedShaderFromSource("../mesh.vert", asset::IShader::ESS_VERTEX);
		auto fs = createCPUSpecializedShaderFromSource("../mesh.frag", asset::IShader::ESS_FRAGMENT);

		auto dummyPplnLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>();
		auto createMeshBufferFromGeomCreatorReturnType = [&dummyPplnLayout,&vs,&fs,this](
			asset::IGeometryCreator::return_type& _data,
			asset::IAssetManager* _manager,
			const BulletSampleApp::E_OBJECT object)
		{
			uint32_t pos_attrib_location = 0u;
			uint32_t normal_attrib_location = object == BulletSampleApp::E_CONE ? 2u : 3u;
			assert((pos_attrib_location != 15u && normal_attrib_location != 15u) && "This attribute location is used for instance IDs!");

			const uint16_t enabledAttribFlags = (0x1u << pos_attrib_location) | (0x1u << normal_attrib_location);
			_data.inputParams.enabledAttribFlags = enabledAttribFlags;

			auto revamped_vs = asset::IGLSLCompiler::createOverridenCopy(
				vs->getUnspecialized(),
				"#define _NBL_ATTRIB_POS_LOCATION_ %d\n"
				"#define _NBL_ATTRIB_NORMAL_LOCATION_ %d\n", pos_attrib_location, normal_attrib_location);

			asset::ICPUSpecializedShader::SInfo specInfo = { nullptr, nullptr, "main" };
			const auto revamped_vs_spec = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(
				std::move(revamped_vs),
				std::move(specInfo));

			asset::ICPUSpecializedShader* shaders[2] = { revamped_vs_spec.get(), fs.get() };

			//creating pipeline just to forward vtx and primitive params
			auto pipeline = core::make_smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>(
				core::smart_refctd_ptr(dummyPplnLayout),
				shaders, shaders + 2,
				_data.inputParams,
				asset::SBlendParams(),
				_data.assemblyParams,
				asset::SRasterizationParams());

			auto mb = core::make_smart_refctd_ptr<asset::ICPUMeshBuffer>(
				nullptr, nullptr,
				_data.bindings, std::move(_data.indexBuffer));

			mb->setIndexCount(_data.indexCount);
			mb->setIndexType(_data.indexType);
			mb->setBoundingBox(_data.bbox);
			mb->setPipeline(std::move(pipeline));
			mb->setNormalAttributeIx(normal_attrib_location);

			return mb;

		};

		auto cpuMeshCube = createMeshBufferFromGeomCreatorReturnType(cubeGeom, assetManager.get(), BulletSampleApp::E_CUBE);
		auto cpuMeshCylinder = createMeshBufferFromGeomCreatorReturnType(cylinderGeom, assetManager.get(), BulletSampleApp::E_CYLINDER);
		auto cpuMeshSphere = createMeshBufferFromGeomCreatorReturnType(sphereGeom, assetManager.get(), BulletSampleApp::E_SPHERE);
		auto cpuMeshCone = createMeshBufferFromGeomCreatorReturnType(coneGeom, assetManager.get(), BulletSampleApp::E_CONE);
		dummyPplnLayout = nullptr;
		
		// Create GPU Objects (IGPUMeshBuffer + GraphicsPipeline)
		auto createGPUObject = [&](const video::IPropertyPool* pool, asset::ICPUMeshBuffer* cpuMesh, asset::E_FACE_CULL_MODE faceCullingMode = asset::EFCM_BACK_BIT) -> BulletSampleApp::GPUObject
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
			auto& blendParams = pipeline->getBlendParams();
			blendParams.logicOpEnable = false;
			blendParams.logicOp = nbl::asset::ELO_NO_OP;
			for (size_t i = 0ull; i < nbl::asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
				blendParams.blendParams[i].attachmentEnabled = (i == 0ull);

			asset::ICPUSpecializedShader* cpuShaders[2] = {
				pipeline->getShaderAtStage(asset::IShader::ESS_VERTEX),
				pipeline->getShaderAtStage(asset::IShader::ESS_FRAGMENT)
			};
			cpuMesh->setPipeline(core::make_smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>(
				core::smart_refctd_ptr(cpuLayout), &cpuShaders[0], &cpuShaders[0] + 2,
				vtxinputParams,
				pipeline->getBlendParams(),
				pipeline->getPrimitiveAssemblyParams(),
				rasterParams
				));

			BulletSampleApp::GPUObject ret = {};
			ret.pool = pool;
			// get the mesh
			cpu2gpuParams.beginCommandBuffers();
			ret.gpuMesh = CPU2GPU.getGPUObjectsFromAssets(&cpuMesh, &cpuMesh + 1, cpu2gpuParams)->front();
			cpu2gpuParams.waitForCreationToComplete(false);
			asset::SBufferBinding<video::IGPUBuffer> instanceRedirectBufBnd;
			instanceRedirectBufBnd.offset = 0u;
			instanceRedirectBufBnd.buffer = pool->getPropertyMemoryBlock(0u).buffer;
			ret.gpuMesh->setVertexBufferBinding(std::move(instanceRedirectBufBnd), 15u);
			//
			video::IGPUGraphicsPipeline::SCreationParams gp_params;
			gp_params.rasterizationSamplesHint = asset::IImage::ESCF_1_BIT;
			gp_params.renderpass = core::smart_refctd_ptr<video::IGPURenderpass>(renderpass);
			gp_params.renderpassIndependent = core::smart_refctd_ptr<const video::IGPURenderpassIndependentPipeline>(ret.gpuMesh->getPipeline());
			gp_params.subpassIx = 0u;
			ret.graphicsPipeline = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(gp_params));

			return ret;
		};

		m_gpuObjects.emplace_back(createGPUObject(m_cubes.get(), cpuMeshCube.get()));
		m_gpuObjects.emplace_back(createGPUObject(m_cylinders.get(), cpuMeshCylinder.get(), asset::EFCM_NONE));
		m_gpuObjects.emplace_back(createGPUObject(m_spheres.get(), cpuMeshSphere.get()));
		m_gpuObjects.emplace_back(createGPUObject(m_cones.get(), cpuMeshCone.get(), asset::EFCM_NONE));

		//
		m_lastTime = std::chrono::system_clock::now();

		for (uint32_t i=0u; i<FRAMES_IN_FLIGHT; i++)
		{
			m_imageAcquire[i] = logicalDevice->createSemaphore();
			m_renderFinished[i] = logicalDevice->createSemaphore();
		}

		// Camera 
		core::vectorSIMDf cameraPosition(0, 5, -10);
		matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(60), float(WIN_W) / WIN_H, 0.01f, 500.0f);
		m_cam = std::make_unique<Camera>(cameraPosition, core::vectorSIMDf(0, 0, 0), proj);
	
		logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, m_cmdbuf);
	}

	void onAppTerminated_impl() override
	{
		const auto& device = logicalDevice;
		device->waitIdle();

		m_world->unbindRigidBody(m_basePlateBody, false);
		m_world->deleteRigidBody(m_basePlateBody);
		m_world->deletebtObject(m_basePlateRigidBodyData.shape);

		auto alwaysTrue = [](auto dummy) -> bool {return true; };

		deleteBasedOnPhysicsPredicate(nullptr, nullptr, m_cylinders.get(), alwaysTrue);
		deleteBasedOnPhysicsPredicate(nullptr, nullptr, m_spheres.get(), alwaysTrue);
		deleteBasedOnPhysicsPredicate(nullptr, nullptr, m_cones.get(), alwaysTrue);
		deleteBasedOnPhysicsPredicate(nullptr, nullptr, m_cubes.get(), alwaysTrue);

		m_world->deletebtObject(m_cubeRigidBodyData.shape);
		m_world->deletebtObject(m_cylinderRigidBodyData.shape);
		m_world->deletebtObject(m_sphereRigidBodyData.shape);
		m_world->deletebtObject(m_coneRigidBodyData.shape);
	}

	void workLoopBody() override
	{
		m_resourceIx++;
		if(m_resourceIx >= FRAMES_IN_FLIGHT)
			m_resourceIx = 0;

		// Timing
		auto renderStart = std::chrono::system_clock::now();
		m_dt = std::chrono::duration_cast<std::chrono::milliseconds>(renderStart-m_lastTime).count();
		m_lastTime = renderStart;

		// Calculate Simple Moving Average for FrameTime
		{
			m_time_sum -= m_dtList[m_frame_count];
			m_time_sum += m_dt;
			m_dtList[m_frame_count] = m_dt;
			m_frame_count++;
			if(m_frame_count >= MaxFramesToAverage)
			{
				m_frame_count = 0;
				m_frameDataFilled = true;
			}
		}
		double averageFrameTime = (m_frameDataFilled) ? (m_time_sum / (double)MaxFramesToAverage) : (m_time_sum / m_frame_count);
		// logger->log("averageFrameTime = %f",system::ILogger::ELL_INFO, averageFrameTime);
		
		// Calculate Next Presentation Time Stamp
		auto averageFrameTimeDuration = std::chrono::duration<double, std::milli>(averageFrameTime);
		auto nextPresentationTime = renderStart + averageFrameTimeDuration;
		auto nextPresentationTimeStamp = std::chrono::duration_cast<std::chrono::microseconds>(nextPresentationTime.time_since_epoch());

		// Input 
		inputSystem->getDefaultMouse(&m_mouse);
		inputSystem->getDefaultKeyboard(&m_keyboard);

		m_cam->beginInputProcessing(nextPresentationTimeStamp);
		m_mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { m_cam->mouseProcess(events); }, logger.get());
		m_keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void {
			m_cam->keyboardProcess(events);
			}, logger.get());
		m_cam->endInputProcessing(nextPresentationTimeStamp);

		auto& cb = m_cmdbuf[m_resourceIx];
		auto& fence = m_frameComplete[m_resourceIx];
		if (fence)
		{
			while (logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT)
			{
			}
			logicalDevice->resetFences(1u, &fence.get());
		}
		else
			fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

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

			VkRect2D scissor;
			scissor.extent = { WIN_W, WIN_H };
			scissor.offset = { 0, 0 };
			cb->setScissor(0u, 1u, &scissor);
		}
		// acquire image 
		uint32_t imgnum = 0u;
		swapchain->acquireNextImage(MAX_TIMEOUT, m_imageAcquire[m_resourceIx].get(), nullptr, &imgnum);
		// Update instances buffer 
		{
			// Update Physics (TODO: fixed timestep)
			m_world->getWorld()->stepSimulation(m_dt);

			video::CPropertyPoolHandler::TransferRequest request;
			request.setFromPool(m_objectPool.get(), TransformPropertyID);
			request.flags = video::CPropertyPoolHandler::TransferRequest::EF_NONE;
			request.elementCount = CInstancedMotionState::s_updateAddresses.size();
			request.srcAddresses = nullptr;
			request.dstAddresses = CInstancedMotionState::s_updateAddresses.data();
			request.device2device = false;
			request.source = CInstancedMotionState::s_updateData.data();
			// TODO: why does the very first update set matrices to identity?
			auto result = utilities->getDefaultPropertyPoolHandler()->transferProperties(utilities->getDefaultUpStreamingBuffer(), utilities->getDefaultDownStreamingBuffer(), cb.get(), fence.get(), &request, &request + 1u, logger.get());
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
			deleteBasedOnPhysicsPredicate(fence.get(), cb.get(), m_cones.get(), falledFromMap);
			deleteBasedOnPhysicsPredicate(fence.get(), cb.get(), m_cubes.get(), falledFromMap);
			deleteBasedOnPhysicsPredicate(fence.get(), cb.get(), m_spheres.get(), falledFromMap);
			deleteBasedOnPhysicsPredicate(fence.get(), cb.get(), m_cylinders.get(), falledFromMap);
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

			info.renderpass = renderpass;
			info.framebuffer = fbo[imgnum];
			info.clearValueCount = 2u;
			info.clearValues = clearValues;
			info.renderArea.offset = { 0, 0 };
			info.renderArea.extent = { WIN_W, WIN_H };
			cb->beginRenderPass(&info, asset::ESC_INLINE);
		}
		// draw
		cb->bindDescriptorSets(asset::EPBP_GRAPHICS, m_gpuLayout.get(), 0u, 1u, &m_globalDs.get());
		{
			auto viewProj = m_cam->getConcatenatedMatrix();

			// Draw Stuff 
			for (uint32_t i = 0; i < m_gpuObjects.size(); ++i)
			{
				auto& gpuObject = m_gpuObjects[i];

				cb->bindGraphicsPipeline(gpuObject.graphicsPipeline.get());
				cb->pushConstants(gpuObject.graphicsPipeline->getRenderpassIndependentPipeline()->getLayout(), asset::IShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), viewProj.pointer());
				gpuObject.gpuMesh->setInstanceCount(gpuObject.pool->getAllocated());
				cb->drawMeshBuffer(gpuObject.gpuMesh.get());
			}
		}
		cb->endRenderPass();
		cb->end();

		CommonAPI::Submit(
			logicalDevice.get(),
			swapchain.get(),
			cb.get(),
			queues[CommonAPI::InitOutput::EQT_GRAPHICS],
			m_imageAcquire[m_resourceIx].get(),
			m_renderFinished[m_resourceIx].get(),
			fence.get());

		CommonAPI::Present(
			logicalDevice.get(),
			swapchain.get(),
			queues[CommonAPI::InitOutput::EQT_GRAPHICS],
			m_renderFinished[m_resourceIx].get(),
			imgnum);
	}

	bool keepRunning() override
	{
		return windowCb->isWindowOpen();
	}

private:
	void deleteShapes(
		video::IGPUFence* fence,
		video::IGPUCommandBuffer* cmdbuf,
		instance_redirect_property_pool_t* pool,
		const uint32_t* objectsToDelete,
		const uint32_t* instanceRedirectsToDelete,
		const uint32_t count)
	{
		m_objectPool->freeProperties(objectsToDelete, objectsToDelete + count);
		// a bit of reuse
		m_scratchObjectIDs.resize(count);
		m_scratchInstanceRedirects.resize(count);
		uint32_t* srcAddrScratch = m_scratchObjectIDs.data();
		uint32_t* dstAddrScratch = m_scratchInstanceRedirects.data();
		//
		const bool needTransfer = video::CPropertyPoolHandler::freeProperties(pool, m_transfers.data() + 2u, instanceRedirectsToDelete, instanceRedirectsToDelete + count, srcAddrScratch, dstAddrScratch);
		if (needTransfer)
			utilities->getDefaultPropertyPoolHandler()->transferProperties(utilities->getDefaultUpStreamingBuffer(), nullptr, cmdbuf, fence, m_transfers.data() + 2, m_transfers.data() + 3, logger.get());
	};

	void deleteBasedOnPhysicsPredicate(
		video::IGPUFence* fence,
		video::IGPUCommandBuffer* cmdbuf,
		instance_redirect_property_pool_t* pool,
		auto pred)
	{
		core::vector<uint32_t> objects, instances;
		for (auto& body : m_bodies)
		{
			if (!body)
				continue;
			auto* motionState = static_cast<CInstancedMotionState*>(body->getMotionState());
			if (motionState->getInstancePool() != pool || !pred(motionState))
				continue;

			objects.emplace_back(motionState->getObjectID());
			instances.emplace_back(motionState->getInstanceID());
			m_world->unbindRigidBody(body);
			m_world->deleteRigidBody(body);
			body = nullptr;
		}
		deleteShapes(fence, cmdbuf, pool, objects.data(), instances.data(), objects.size());
	};
};

NBL_COMMON_API_MAIN(BulletSampleApp, BulletSampleApp::AppUserData)

extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }