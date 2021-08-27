// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"

#include <btBulletDynamicsCommon.h>
#include "BulletCollision/NarrowPhaseCollision/btRaycastCallback.h"

#include "nbl/ext/Bullet/BulletUtility.h"
#include "nbl/ext/Bullet/CPhysicsWorld.h"
#include "Camera.hpp"

using namespace nbl;
using namespace core;
using namespace ui;


class CInstancedMotionState : public ext::Bullet3::IMotionStateBase
{
	public:
		btTransform m_correctionMatrix;

		inline CInstancedMotionState() {}
		inline CInstancedMotionState(uint32_t objectID, uint32_t instanceID, core::matrix3x4SIMD const & start_mat, core::matrix3x4SIMD const & correction_mat)
			: ext::Bullet3::IMotionStateBase(ext::Bullet3::convertMatrixSIMD(start_mat)), m_correctionMatrix(ext::Bullet3::convertMatrixSIMD(correction_mat)), m_objectID(objectID), m_instanceID(instanceID)
		{
			m_cachedMat = m_startWorldTrans*m_correctionMatrix.inverse();
		}

		inline ~CInstancedMotionState()
		{
		}

		inline virtual void getWorldTransform(btTransform &worldTrans) const override
		{
			worldTrans = m_cachedMat;
		}

		inline virtual void setWorldTransform(const btTransform &worldTrans) override
		{
			// TODO: protect agains simulation "substeps" somehow (redundant update sets)
			m_cachedMat = worldTrans;

			s_updateIndices.push_back(m_objectID);
			s_updateData.push_back(ext::Bullet3::convertbtTransform(m_cachedMat*m_correctionMatrix));
		}

		static core::vector<uint32_t> s_updateIndices;
		static core::vector<core::matrix3x4SIMD> s_updateData;
	protected:
		btTransform m_cachedMat;
		uint32_t m_objectID,m_instanceID;
};
core::vector<uint32_t> CInstancedMotionState::s_updateIndices;
core::vector<core::matrix3x4SIMD> CInstancedMotionState::s_updateData;


int main(int argc, char** argv)
{
	constexpr uint32_t WIN_W = 1280;
	constexpr uint32_t WIN_H = 720;
	constexpr uint32_t FBO_COUNT = 1u;
	constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT>FBO_COUNT);

	auto initOutput = CommonAPI::Init<WIN_W, WIN_H, FBO_COUNT>(video::EAT_OPENGL, "Physics Simulation", asset::EF_D32_SFLOAT);
	auto system = std::move(initOutput.system);
	auto window = std::move(initOutput.window);
	auto gl = std::move(initOutput.apiConnection);
	auto surface = std::move(initOutput.surface);
	auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
	auto device = std::move(initOutput.logicalDevice);
	auto utilities = std::move(initOutput.utilities);
	auto queues = std::move(initOutput.queues);
	auto graphicsQueue = queues[decltype(initOutput)::EQT_GRAPHICS];
	auto computeQueue = queues[decltype(initOutput)::EQT_COMPUTE];
	auto transferUpQueue = queues[decltype(initOutput)::EQT_TRANSFER_UP];
	auto swapchain = std::move(initOutput.swapchain);
	auto renderpass = std::move(initOutput.renderpass);
	auto fbo = std::move(initOutput.fbo[0]);
	auto commandPool = std::move(initOutput.commandPool);
	auto assetManager = std::move(initOutput.assetManager);
	auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
	auto logger = std::move(initOutput.logger);
	auto inputSystem = std::move(initOutput.inputSystem);

	// TODO: roll into CommonAPI
	core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
	auto computeCommandPool = device->createCommandPool(computeQueue->getFamilyIndex(),video::IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);

	// property transfer cmdbuffers
	core::smart_refctd_ptr<video::IGPUCommandBuffer> propXferCmdbuf[FRAMES_IN_FLIGHT];
	device->createCommandBuffers(computeCommandPool.get(),video::IGPUCommandBuffer::EL_PRIMARY,FRAMES_IN_FLIGHT,propXferCmdbuf);
	

	// Physics Setup
	auto world = ext::Bullet3::CPhysicsWorld::create();
	world->getWorld()->setGravity(btVector3(0, -5, 0));

	// BasePlate
	btRigidBody* basePlateBody;
	ext::Bullet3::CPhysicsWorld::RigidBodyData basePlateRigidBodyData;
	{
		core::matrix3x4SIMD baseplateMat;
		baseplateMat.setTranslation(core::vectorSIMDf(0.0, -1.0, 0.0));

		basePlateRigidBodyData.mass = 0.0f;
		basePlateRigidBodyData.shape = world->createbtObject<btBoxShape>(btVector3(300, 1, 300));
		basePlateRigidBodyData.trans = baseplateMat;

		basePlateBody = world->createRigidBody(basePlateRigidBodyData);
		world->bindRigidBody(basePlateBody);
	}
	
	// set up
	auto propertyPoolHandler = utilities->getDefaultPropertyPoolHandler();
	auto createPropertyPoolWithMemory = [device](auto& retval, uint32_t capacity) -> void
	{
		using pool_type = std::remove_reference_t<decltype(retval)>::pointee;
		asset::SBufferRange<video::IGPUBuffer> blocks[pool_type::PropertyCount];
		for (auto i=0u; i<pool_type::PropertyCount; i++)
		{
			auto& block = blocks[i];
			block.offset = 0u;
			block.size = pool_type::PropertySizes[i]*capacity;
			block.buffer = device->createDeviceLocalGPUBufferOnDedMem(block.size);
		}
		retval = pool_type::create(device.get(),blocks,capacity);
	};

	// Instance Redirects
	using instance_redirect_property_pool_t = video::CPropertyPool<core::allocator,uint32_t>;
	core::smart_refctd_ptr<instance_redirect_property_pool_t> cubes,cylinders,spheres,cones;
	createPropertyPoolWithMemory(cubes,20u);
	createPropertyPoolWithMemory(cylinders,20u);
	createPropertyPoolWithMemory(spheres,20u);
	createPropertyPoolWithMemory(cones,10u);
	// global object data pool
	const uint32_t MaxSingleType = core::max(core::max(cubes->getCapacity(),cylinders->getCapacity()),core::max(spheres->getCapacity(),cones->getCapacity()));
	const uint32_t MaxNumObjects = cubes->getCapacity()+cylinders->getCapacity()+spheres->getCapacity()+cones->getCapacity();
	constexpr auto TransformPropertyID = 1u;
	using object_property_pool_t = video::CPropertyPool<core::allocator,core::vectorSIMDf,core::matrix3x4SIMD>;
	core::smart_refctd_ptr<object_property_pool_t> objectPool; createPropertyPoolWithMemory(objectPool,MaxNumObjects);

	// Physics
	core::vector<btRigidBody*> bodies(MaxNumObjects,nullptr);
	// Shapes RigidBody Data
	const ext::Bullet3::CPhysicsWorld::RigidBodyData cubeRigidBodyData = [world]()
	{
		ext::Bullet3::CPhysicsWorld::RigidBodyData rigidBodyData;
		rigidBodyData.mass = 2.0f;
		rigidBodyData.shape = world->createbtObject<btBoxShape>(btVector3(0.5, 0.5, 0.5));
		btVector3 inertia;
		rigidBodyData.shape->calculateLocalInertia(rigidBodyData.mass, inertia);
		rigidBodyData.inertia = ext::Bullet3::frombtVec3(inertia);
		return rigidBodyData;
	}();
	const ext::Bullet3::CPhysicsWorld::RigidBodyData cylinderRigidBodyData = [world]()
	{
		ext::Bullet3::CPhysicsWorld::RigidBodyData rigidBodyData;
		rigidBodyData.mass = 1.0f;
		rigidBodyData.shape = world->createbtObject<btCylinderShape>(btVector3(0.5, 0.5, 0.5));
		btVector3 inertia;
		rigidBodyData.shape->calculateLocalInertia(rigidBodyData.mass, inertia);
		rigidBodyData.inertia = ext::Bullet3::frombtVec3(inertia);
		return rigidBodyData;
	}();
	const ext::Bullet3::CPhysicsWorld::RigidBodyData sphereRigidBodyData = [world]()
	{
		ext::Bullet3::CPhysicsWorld::RigidBodyData rigidBodyData;
		rigidBodyData.mass = 1.0f;
		rigidBodyData.shape = world->createbtObject<btSphereShape>(0.5);
		btVector3 inertia;
		rigidBodyData.shape->calculateLocalInertia(rigidBodyData.mass, inertia);
		rigidBodyData.inertia = ext::Bullet3::frombtVec3(inertia);
		return rigidBodyData;
	}();
	const ext::Bullet3::CPhysicsWorld::RigidBodyData coneRigidBodyData = [world]()
	{
		ext::Bullet3::CPhysicsWorld::RigidBodyData rigidBodyData;
		rigidBodyData.mass = 1.0f;
		rigidBodyData.shape = world->createbtObject<btConeShape>(0.5, 1.0);
		btVector3 inertia;
		rigidBodyData.shape->calculateLocalInertia(rigidBodyData.mass, inertia);
		rigidBodyData.inertia = ext::Bullet3::frombtVec3(inertia);
		return rigidBodyData;
	}();

	// kept state
	uint32_t totalSpawned=0u;
	core::vector<uint32_t> scratchObjectIDs;
	core::vector<uint32_t> scratchInstanceRedirects;
	core::vector<core::vectorSIMDf> initialColor;
	core::vector<core::matrix3x4SIMD> instanceTransforms;
	std::array<video::CPropertyPoolHandler::TransferRequest,object_property_pool_t::PropertyCount+1> transfers;
	transfers[0].propertyID = 0;
	transfers[1].propertyID = 1;
	transfers[2].propertyID = 0;
	// add a shape
	auto addShapes = [&](
		core::smart_refctd_ptr<video::IGPUFence>&& fence,
		video::IGPUCommandBuffer* cmdbuf,
		ext::Bullet3::CPhysicsWorld::RigidBodyData rigidBodyData,
		instance_redirect_property_pool_t* pool, const uint32_t count,
		const core::matrix3x4SIMD& correction_mat=core::matrix3x4SIMD()
	) -> void
	{
		// prepare all the temporary array sizes
		scratchObjectIDs.resize(count);
		scratchInstanceRedirects.resize(count);
		initialColor.resize(count);
		instanceTransforms.resize(count);
		// allocate the object data
		std::fill_n(scratchObjectIDs.data(),count,object_property_pool_t::invalid_index);
		objectPool->allocateProperties(scratchObjectIDs.data(),scratchObjectIDs.data()+count);
		// now the redirects
		std::fill_n(scratchInstanceRedirects.data(),count,instance_redirect_property_pool_t::invalid_index);
		pool->allocateProperties(scratchInstanceRedirects.data(),scratchInstanceRedirects.data()+count);
		// fill with data
		for (auto i=0u; i<count; i++)
		{
			initialColor[i] = core::vectorSIMDf(float(totalSpawned%MaxNumObjects)/float(MaxNumObjects),0.5f,1.f);
			rigidBodyData.trans = instanceTransforms[i] = core::matrix3x4SIMD().setTranslation(core::vectorSIMDf(float(totalSpawned%3)-1.0f,totalSpawned*1.5f,0.f));
			totalSpawned++;
			// TODO: seems like `rigidBodyData.trans` is redundant to some matrices in the MotionStateBase
			const auto objectID = scratchObjectIDs[i];
			auto& body = bodies[objectID] = world->createRigidBody(rigidBodyData);
			world->bindRigidBody<CInstancedMotionState>(body,objectID,scratchInstanceRedirects[i],rigidBodyData.trans,correction_mat);
		}
		for (auto i=0u; i<object_property_pool_t::PropertyCount; i++)
		{
			transfers[i].download = false;
			transfers[i].pool = objectPool.get();
			transfers[i].indices = {scratchObjectIDs.data(),scratchObjectIDs.data()+count};
		}
		transfers[0].data = initialColor.data();
		transfers[1].data = instanceTransforms.data();
		transfers[2].pool = pool;
		transfers[2].indices = {scratchInstanceRedirects.data(),scratchInstanceRedirects.data()+count};
		transfers[2].data = scratchObjectIDs.data();
		// set up the transfer/update
		propertyPoolHandler->transferProperties(
			utilities->getDefaultUpStreamingBuffer(),utilities->getDefaultDownStreamingBuffer(),
			cmdbuf,fence.get(),transfers.data(),transfers.data()+transfers.size(),logger.get()
		);
	};
	auto addCubes = [&](core::smart_refctd_ptr<video::IGPUFence>&& fence, video::IGPUCommandBuffer* cmdbuf, const uint32_t count)
	{
		addShapes(std::move(fence),cmdbuf,cubeRigidBodyData,cubes.get(),count);
	};
	auto addCylinders = [&](core::smart_refctd_ptr<video::IGPUFence>&& fence, video::IGPUCommandBuffer* cmdbuf, const uint32_t count)
	{
		addShapes(std::move(fence),cmdbuf,cylinderRigidBodyData,cylinders.get(),count,core::matrix3x4SIMD().setRotation(core::quaternion(core::PI<float>()/2.f,0.f,0.f)));
	};
	auto addSpheres = [&](core::smart_refctd_ptr<video::IGPUFence>&& fence, video::IGPUCommandBuffer* cmdbuf, const uint32_t count)
	{
		addShapes(std::move(fence),cmdbuf,sphereRigidBodyData,spheres.get(),count);
	};
	auto addCones = [&](core::smart_refctd_ptr<video::IGPUFence>&& fence, video::IGPUCommandBuffer* cmdbuf, const uint32_t count)
	{
		addShapes(std::move(fence),cmdbuf,coneRigidBodyData,cones.get(),count,core::matrix3x4SIMD().setTranslation(core::vector3df_SIMD(0.f,-0.5f,0.f)));
	};

	// setup scene
	{
		auto& fence = frameComplete[FRAMES_IN_FLIGHT-1] = device->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
		auto cmdbuf = propXferCmdbuf[FRAMES_IN_FLIGHT-1].get();

		cmdbuf->begin(video::IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
		addCubes(core::smart_refctd_ptr(fence),cmdbuf,20u);
		addCylinders(core::smart_refctd_ptr(fence),cmdbuf,20u);
		addSpheres(core::smart_refctd_ptr(fence),cmdbuf,20u);
		addCones(core::smart_refctd_ptr(fence),cmdbuf,10u);
		cmdbuf->end();
		
		video::IGPUQueue::SSubmitInfo submit;
		{
			submit.commandBufferCount = 1u;
			submit.commandBuffers = &cmdbuf;
			submit.signalSemaphoreCount = 0u;
			submit.waitSemaphoreCount = 0u;

			computeQueue->submit(1u,&submit,fence.get());
		}
	}


	video::IGPUObjectFromAssetConverter CPU2GPU;
	// set up shader inputs
	core::smart_refctd_ptr<asset::ICPUPipelineLayout> cpuLayout;
	core::smart_refctd_ptr<video::IGPUPipelineLayout> gpuLayout;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> globalDs;
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
			auto dsLayout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bindings,bindings+GLOBAL_DS_COUNT);
		
			asset::SPushConstantRange range[1] = { asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD) };
			cpuLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(range,range+1u,std::move(dsLayout));
		}
		gpuLayout = CPU2GPU.getGPUObjectsFromAssets(&cpuLayout.get(),&cpuLayout.get()+1,cpu2gpuParams)->front();

		{
			auto globalDsLayout = gpuLayout->getDescriptorSetLayout(0u);
			auto pool = device->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE,&globalDsLayout,&globalDsLayout+1u);
			globalDs = device->createGPUDescriptorSet(pool.get(),core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout>(globalDsLayout)); // TODO: change method signature to make it obvious we're taking shared ownership of a pool

			video::IGPUDescriptorSet::SWriteDescriptorSet writes[GLOBAL_DS_COUNT];
			video::IGPUDescriptorSet::SDescriptorInfo infos[GLOBAL_DS_COUNT];
			for (auto i = 0u; i < GLOBAL_DS_COUNT; i++)
			{
				writes[i].dstSet = globalDs.get();
				writes[i].binding = i;
				writes[i].arrayElement = 0u;
				writes[i].count = 1u;
				writes[i].descriptorType = asset::EDT_STORAGE_BUFFER;
				writes[i].info = infos+i;

				const auto& poolBuff = objectPool->getPropertyMemoryBlock(i);
				infos[i].desc = poolBuff.buffer;
				infos[i].buffer.offset = poolBuff.offset;
				infos[i].buffer.size = poolBuff.size;
			}
			device->updateDescriptorSets(GLOBAL_DS_COUNT,writes,0u,nullptr);
		}
	}


	// Geom Create
	auto geometryCreator = assetManager->getGeometryCreator();
	auto cubeGeom = geometryCreator->createCubeMesh(core::vector3df(1.0f, 1.0f, 1.0f));
	auto cylinderGeom = geometryCreator->createCylinderMesh(0.5f, 0.5f, 20);
	auto sphereGeom = geometryCreator->createSphereMesh(0.5f);
	auto coneGeom = geometryCreator->createConeMesh(0.5f, 1.0f, 32);

	// Creating CPU Shaders 
	auto createCPUSpecializedShaderFromSource = [=](const char* path, asset::ISpecializedShader::E_SHADER_STAGE stage) -> core::smart_refctd_ptr<asset::ICPUSpecializedShader>
	{
		// TODO: Change IAssetLoader::SAssetLoadParams::relativeDir to `system::path`
		//auto tmp = system::path(argv[0]).root_directory().string();

		asset::IAssetLoader::SAssetLoadParams params{};
		params.logger = logger.get();
		//params.relativeDir = tmp.c_str();
		auto spec = assetManager->getAsset(path,params).getContents();
		if (spec.empty())
			return nullptr;

		return core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*spec.begin());
	};

	auto vs = createCPUSpecializedShaderFromSource("../mesh.vert",asset::ISpecializedShader::ESS_VERTEX);
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
	auto cpuMeshCube = createMeshBufferFromGeomCreatorReturnType(cubeGeom, assetManager.get(), shaders, shaders+2);
	auto cpuMeshCylinder = createMeshBufferFromGeomCreatorReturnType(cylinderGeom, assetManager.get(), shaders, shaders+2);
	auto cpuMeshSphere = createMeshBufferFromGeomCreatorReturnType(sphereGeom, assetManager.get(), shaders, shaders+2);
	auto cpuMeshCone = createMeshBufferFromGeomCreatorReturnType(coneGeom, assetManager.get(), shaders, shaders+2);
	dummyPplnLayout = nullptr;

	// TODO: replace with an actual scenemanager
	struct GPUObject
	{
		core::smart_refctd_ptr<video::IGPUMeshBuffer> gpuMesh;
		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> graphicsPipeline;
	};
	// Create GPU Objects (IGPUMeshBuffer + GraphicsPipeline)
	auto createGPUObject = [&](
		asset::ICPUMeshBuffer * cpuMesh, uint32_t numInstances, uint64_t rangeStart,
		asset::E_FACE_CULL_MODE faceCullingMode = asset::EFCM_BACK_BIT) -> GPUObject
	{
		auto pipeline = cpuMesh->getPipeline();

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
			core::smart_refctd_ptr(cpuLayout),&cpuShaders[0],&cpuShaders[0]+2,
			pipeline->getVertexInputParams(),
			pipeline->getBlendParams(),
			pipeline->getPrimitiveAssemblyParams(),
			rasterParams
		));

		GPUObject ret = {};
		// get the mesh
		ret.gpuMesh = CPU2GPU.getGPUObjectsFromAssets(&cpuMesh,&cpuMesh+1,cpu2gpuParams)->front();
		ret.gpuMesh->setBaseInstance(rangeStart);
		ret.gpuMesh->setInstanceCount(numInstances);

		video::IGPUGraphicsPipeline::SCreationParams gp_params;
		gp_params.rasterizationSamplesHint = asset::IImage::ESCF_1_BIT;
		gp_params.renderpass = core::smart_refctd_ptr<video::IGPURenderpass>(renderpass);
		gp_params.renderpassIndependent = core::smart_refctd_ptr<const video::IGPURenderpassIndependentPipeline>(ret.gpuMesh->getPipeline());
		gp_params.subpassIx = 0u;

		ret.graphicsPipeline = device->createGPUGraphicsPipeline(nullptr,std::move(gp_params));

		return ret;
	};

	core::vector<GPUObject> gpuObjects = {
		createGPUObject(cpuMeshCube.get(),cubes->getAllocated(), 0u),
		createGPUObject(cpuMeshCylinder.get(),cylinders->getAllocated(), 20u, asset::EFCM_NONE),
		createGPUObject(cpuMeshSphere.get(),spheres->getAllocated(), 40u),
		createGPUObject(cpuMeshCone.get(),cones->getAllocated(), 60u, asset::EFCM_NONE)
	};


	//
	auto lastTime = std::chrono::system_clock::now();
	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;

	core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	for (uint32_t i=0u; i<FRAMES_IN_FLIGHT; i++)
	{
		imageAcquire[i] = device->createSemaphore();
		renderFinished[i] = device->createSemaphore();
	}

	// render loop
	constexpr size_t MaxFramesToAverage = 100ull;
	size_t frame_count = 0ull;
	double time_sum = 0;
	double dtList[MaxFramesToAverage] = {};
	for(size_t i = 0ull; i < MaxFramesToAverage; ++i) {
		dtList[i] = 0.0;
	}

	double dt = 0;

	// Camera 
	core::vectorSIMDf cameraPosition(0, 5, -10);
	matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(60), float(WIN_W) / WIN_H, 0.01f, 500.0f);
	Camera cam = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), proj);
	
	// polling for events!
	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

	core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf[FRAMES_IN_FLIGHT];
	device->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, cmdbuf);
	for (auto i=0u; true; i++)
	{
		// Timing

		auto renderStart = std::chrono::system_clock::now();
		dt = std::chrono::duration_cast<std::chrono::milliseconds>(renderStart-lastTime).count();
		lastTime = renderStart;
		
		// Calculate Simple Moving Average for FrameTime
		{
			time_sum -= dtList[frame_count];
			time_sum += dt;
			dtList[frame_count] = dt;
			frame_count++;
			if(frame_count >= MaxFramesToAverage) {
				frame_count = 0;
			}
		}
		double averageFrameTime = time_sum / (double)MaxFramesToAverage;
		// logger->log("dt = %f ------ averageFrameTime = %f",system::ILogger::ELL_INFO, dt, averageFrameTime);
		
		// Calculate Next Presentation Time Stamp
		auto averageFrameTimeDuration = std::chrono::duration<double, std::milli>(averageFrameTime);
		auto nextPresentationTime = renderStart + averageFrameTimeDuration;
		auto nextPresentationTimeStamp = std::chrono::duration_cast<std::chrono::microseconds>(nextPresentationTime.time_since_epoch());
		
		// Input 
		inputSystem->getDefaultMouse(&mouse);
		inputSystem->getDefaultKeyboard(&keyboard);

		cam.beginInputProcessing(nextPresentationTimeStamp);
		mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { cam.mouseProcess(events); }, logger.get());
		keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { cam.keyboardProcess(events); }, logger.get());
		cam.endInputProcessing(nextPresentationTimeStamp);

		// Render
		const auto resourceIx = i%FRAMES_IN_FLIGHT;
		auto& cb = cmdbuf[resourceIx];
		auto& fence = frameComplete[resourceIx];
		if (fence)
		while (device->waitForFences(1u,&fence.get(),false,MAX_TIMEOUT)==video::IGPUFence::ES_TIMEOUT)
		{
		}
		else
			fence = device->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
		
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
		// acquire image 
		uint32_t imgnum = 0u;
		swapchain->acquireNextImage(MAX_TIMEOUT,imageAcquire[resourceIx].get(),nullptr,&imgnum);
		// Update instances buffer 
		{
			// Update Physics (TODO: fixed timestep)
			world->getWorld()->stepSimulation(dt);
			video::CPropertyPoolHandler::TransferRequest request;
			request.download = false;
			request.pool = objectPool.get();
			request.indices = {CInstancedMotionState::s_updateIndices.data(),CInstancedMotionState::s_updateIndices.data()+CInstancedMotionState::s_updateIndices.size()};
			request.propertyID = TransformPropertyID;
			request.data = CInstancedMotionState::s_updateData.data();
			// TODO: why does the very first update set matrices to identity?
			auto result = propertyPoolHandler->transferProperties(utilities->getDefaultUpStreamingBuffer(),utilities->getDefaultDownStreamingBuffer(),cb.get(),fence.get(),&request,&request+1u,logger.get());
			assert(result.transferSuccess);
			CInstancedMotionState::s_updateIndices.clear();
			CInstancedMotionState::s_updateData.clear();
		}
		// renderpass
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
		// draw
		cb->bindDescriptorSets(asset::EPBP_GRAPHICS,gpuLayout.get(),0u,1u,&globalDs.get());
		{
			auto viewProj = cam.getConcatenatedMatrix();

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
	
	world->unbindRigidBody(basePlateBody,false);
	world->deleteRigidBody(basePlateBody);
	world->deletebtObject(basePlateRigidBodyData.shape);
	
	for (auto body : bodies)
	if (body)
	{
		world->unbindRigidBody(body);
		world->deleteRigidBody(body);
	}

	world->deletebtObject(cubeRigidBodyData.shape);
	world->deletebtObject(cylinderRigidBodyData.shape);
	world->deletebtObject(sphereRigidBodyData.shape);
	world->deletebtObject(coneRigidBodyData.shape);


	return 0;
}