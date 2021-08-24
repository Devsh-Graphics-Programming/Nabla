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
	mat3x4 transposeWorldMat = mat3x4(vWorldMatRow0, vWorldMatRow1, vWorldMatRow2);
	mat4x3 worldMat = transpose(transposeWorldMat);

	vec4 worldPos = vec4(dot(vWorldMatRow0, vPos), dot(vWorldMatRow1, vPos), dot(vWorldMatRow2, vPos), 1);
	vec4 pos = PushConstants.viewProj*worldPos;
	gl_Position = pos;
	Color = vCol.xyz;

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

	float ambient = 0.35;
	float diffuse = 0.8;
	float cos_theta_term = max(dot(normal,vec3(3.0,5.0,-4.0)),0.0);

	float fresnel = 0.0; //not going to implement yet, not important
	float specular = 0.0;///pow(max(dot(halfVector,normal),0.0),shininess);

	const float sunPower = 3.14156*0.3;

	pixelColor = vec4(Color, 1)*sunPower*(ambient+mix(diffuse,specular,fresnel)*cos_theta_term/3.14159);
}
)===";

class CInstancedMotionState : public ext::Bullet3::IMotionStateBase
{
	public:
		btTransform m_correctionMatrix;

		inline CInstancedMotionState() {}
		inline CInstancedMotionState(uint32_t index, core::matrix3x4SIMD const & start_mat, core::matrix3x4SIMD const & correction_mat)
			: m_correctionMatrix(ext::Bullet3::convertMatrixSIMD(correction_mat)), m_index(index), ext::Bullet3::IMotionStateBase(ext::Bullet3::convertMatrixSIMD(start_mat))
		{
			m_cachedMat = m_startWorldTrans*m_correctionMatrix.inverse();
			m_index = index;
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
			m_cachedMat = worldTrans;

			s_updateIndices.push_back(m_index);
			s_updateData.push_back(ext::Bullet3::convertbtTransform(m_cachedMat*m_correctionMatrix));
		}

		static core::vector<uint32_t> s_updateIndices;
		static core::vector<core::matrix3x4SIMD> s_updateData;
	protected:
		btTransform m_cachedMat;
		uint32_t m_index;
};
core::vector<uint32_t> CInstancedMotionState::s_updateIndices;
core::vector<core::matrix3x4SIMD> CInstancedMotionState::s_updateData;


int main()
{
	constexpr uint32_t WIN_W = 1280;
	constexpr uint32_t WIN_H = 720;
	constexpr uint32_t FBO_COUNT = 2u;
	constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT>FBO_COUNT);

	auto initOutput = CommonAPI::Init<WIN_W, WIN_H, FBO_COUNT>(video::EAT_OPENGL, "Physics Simulation", asset::EF_D32_SFLOAT);
	auto system = std::move(initOutput.system);
	auto window = std::move(initOutput.window);
	auto windowCb = std::move(initOutput.windowCb);
	auto gl = std::move(initOutput.apiConnection);
	auto surface = std::move(initOutput.surface);
	auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
	auto device = std::move(initOutput.logicalDevice);
	auto queues = std::move(initOutput.queues);
	auto graphicsQueue = queues[decltype(initOutput)::EQT_GRAPHICS];
	auto computeQueue = queues[decltype(initOutput)::EQT_COMPUTE];
	auto transferUpQueue = queues[decltype(initOutput)::EQT_TRANSFER_UP];
	auto swapchain = std::move(initOutput.swapchain);
	auto renderpass = std::move(initOutput.renderpass);
	auto fbo = std::move(initOutput.fbo);
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


	// Instance Data
	std::array<uint32_t,20u> cubeIDs;
	std::array<uint32_t,20u> cylinderIDs;
	std::array<uint32_t,20u> sphereIDs;
	std::array<uint32_t,10u> coneIDs;
	constexpr uint32_t startIndexCubes = 0;
	constexpr uint32_t startIndexCylinders = startIndexCubes + cubeIDs.size();
	constexpr uint32_t startIndexSpheres = startIndexCylinders + cylinderIDs.size();
	constexpr uint32_t startIndexCones = startIndexSpheres + sphereIDs.size();

	constexpr uint32_t MaxNumInstances = cubeIDs.size() + cylinderIDs.size() + sphereIDs.size() + coneIDs.size();
	// Instances Buffer
	auto colorBuffer = device->createDeviceLocalGPUBufferOnDedMem(sizeof(core::vectorSIMDf)*MaxNumInstances);
	auto transformBuffer = device->createDeviceLocalGPUBufferOnDedMem(sizeof(core::matrix3x4SIMD)*MaxNumInstances);
	// Physics
	core::vector<btRigidBody*> bodies;
	bodies.resize(MaxNumInstances);
	// Shapes RigidBody Data
	ext::Bullet3::CPhysicsWorld::RigidBodyData cubeRigidBodyData;
	ext::Bullet3::CPhysicsWorld::RigidBodyData cylinderRigidBodyData;
	ext::Bullet3::CPhysicsWorld::RigidBodyData sphereRigidBodyData;
	ext::Bullet3::CPhysicsWorld::RigidBodyData coneRigidBodyData;
	{
		cubeRigidBodyData.mass = 2.0f;
		cubeRigidBodyData.shape = world->createbtObject<btBoxShape>(btVector3(0.5, 0.5, 0.5));
		btVector3 inertia;
		cubeRigidBodyData.shape->calculateLocalInertia(cubeRigidBodyData.mass, inertia);
		cubeRigidBodyData.inertia = ext::Bullet3::frombtVec3(inertia);
	}
	{
		cylinderRigidBodyData.mass = 1.0f;
		cylinderRigidBodyData.shape = world->createbtObject<btCylinderShape>(btVector3(0.5, 0.5, 0.5));
		btVector3 inertia;
		cylinderRigidBodyData.shape->calculateLocalInertia(cylinderRigidBodyData.mass, inertia);
		cylinderRigidBodyData.inertia = ext::Bullet3::frombtVec3(inertia);
	}
	{
		sphereRigidBodyData.mass = 1.0f;
		sphereRigidBodyData.shape = world->createbtObject<btSphereShape>(0.5);
		btVector3 inertia;
		sphereRigidBodyData.shape->calculateLocalInertia(sphereRigidBodyData.mass, inertia);
		sphereRigidBodyData.inertia = ext::Bullet3::frombtVec3(inertia);
	}
	{
		coneRigidBodyData.mass = 1.0f;
		coneRigidBodyData.shape = world->createbtObject<btConeShape>(0.5, 1.0);
		btVector3 inertia;
		coneRigidBodyData.shape->calculateLocalInertia(coneRigidBodyData.mass, inertia);
		coneRigidBodyData.inertia = ext::Bullet3::frombtVec3(inertia);
	}

	// GPU data pool
	constexpr auto TransformPropertyID = 1u;
	using instance_property_pool_t = video::CPropertyPool<core::allocator,core::vectorSIMDf,core::matrix3x4SIMD>;
	core::smart_refctd_ptr<instance_property_pool_t> propertyPool;
	auto propertyPoolHandler = device->getDefaultPropertyPoolHandler();
	{
		asset::SBufferRange<video::IGPUBuffer> blocks[2];
		blocks[0] = { 0,colorBuffer->getSize(),colorBuffer };
		blocks[1] = { 0,transformBuffer->getSize(),transformBuffer };
		propertyPool = instance_property_pool_t::create(device.get(),blocks,MaxNumInstances);
		
		core::vector<core::vectorSIMDf> initialColor(MaxNumInstances);
		core::vector<core::matrix3x4SIMD> instanceTransforms(MaxNumInstances);
		for (auto i=0u; i<MaxNumInstances; i++)
		{
			initialColor[i] = core::vector3df_SIMD(float(i)/float(MaxNumInstances),0.5f,1.f);
			instanceTransforms[i].setTranslation(core::vectorSIMDf(float(i % 3) - 1.0f, i * 1.5f, 0.0f));
			// TODO: make this code pretty
			{	
				core::matrix3x4SIMD correction_mat; 

				auto & rigidBodyData = cubeRigidBodyData;
				if(i >= startIndexCones) 
				{
					rigidBodyData = coneRigidBodyData;
					correction_mat.setTranslation(core::vector3df_SIMD(0.0f, -0.5f, 0.0f));
				}
				else if(i >= startIndexSpheres) 
				{
					rigidBodyData = sphereRigidBodyData;
				}
				else if(i >= startIndexCylinders) 
				{
					rigidBodyData = cylinderRigidBodyData;
					correction_mat.setRotation(core::quaternion(core::PI<float>() / 2.0f, 0.0f, 0.0f));
				}
				rigidBodyData.trans = instanceTransforms[i];
		
				auto& body = bodies[i] = world->createRigidBody(rigidBodyData);
				world->bindRigidBody<CInstancedMotionState>(body, i, instanceTransforms[i], correction_mat);
			}
		}

		std::array<video::CPropertyPoolHandler::AllocationRequest,4u> requests;
		const void* data[4u][2u] = {};
		auto setupRequest = [&](const uint32_t i, auto& ix_array, const uint32_t rangeStart) -> void
		{
			data[i][0] = initialColor.data()+rangeStart;
			data[i][1] = instanceTransforms.data()+rangeStart;
			// need to initialize the addresses to invalid for them to get allocated
			std::fill(ix_array.begin(),ix_array.end(),instance_property_pool_t::invalid_index);
			requests[i] = video::CPropertyPoolHandler::AllocationRequest(
				propertyPool.get(),
				core::SRange<uint32_t>{ix_array.data(),ix_array.data()+ix_array.size()},
				reinterpret_cast<const void* const*>(data[i])
			);
		};
		setupRequest(0u,cubeIDs,startIndexCubes);
		setupRequest(1u,cylinderIDs,startIndexCylinders);
		setupRequest(2u,sphereIDs,startIndexSpheres);
		setupRequest(3u,coneIDs,startIndexCones);

		auto& fence = frameComplete[FRAMES_IN_FLIGHT-1] = device->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
		auto cmdbuf = propXferCmdbuf[FRAMES_IN_FLIGHT-1].get();

		cmdbuf->begin(video::IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
		propertyPoolHandler->addProperties(cmdbuf,fence.get(),requests.data(),requests.data()+requests.size(),logger.get());
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


	// weird fix -> do not read the next 6 lines (It doesn't affect the program logically) -> waiting for access_violation_repro branch to fix and merge
	core::smart_refctd_ptr<asset::ICPUShader> computeUnspec;
	{
		system::ISystem::future_t<smart_refctd_ptr<system::IFile>> future;
		system->createFile(future, "../../29.SpecializationConstants/particles.comp", nbl::system::IFile::ECF_READ_WRITE);
		auto file = future.get();
		auto sname = file->getFileName().string();
		char const* shaderName = sname.c_str();
		computeUnspec = assetManager->getGLSLCompiler()->resolveIncludeDirectives(file.get(), asset::ISpecializedShader::ESS_COMPUTE, shaderName);
	}

	// Geom Create
	auto geometryCreator = assetManager->getGeometryCreator();
	auto cubeGeom = geometryCreator->createCubeMesh(core::vector3df(1.0f, 1.0f, 1.0f));
	auto cylinderGeom = geometryCreator->createCylinderMesh(0.5f, 0.5f, 20);
	auto sphereGeom = geometryCreator->createSphereMesh(0.5f);
	auto coneGeom = geometryCreator->createConeMesh(0.5f, 1.0f, 32);

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
	
	auto createMeshBufferFromGeomCreatorReturnType = [](
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

	};
	auto cpuMeshCube = createMeshBufferFromGeomCreatorReturnType(cubeGeom, assetManager.get(), shaders, shaders+2);
	auto cpuMeshCylinder = createMeshBufferFromGeomCreatorReturnType(cylinderGeom, assetManager.get(), shaders, shaders+2);
	auto cpuMeshSphere = createMeshBufferFromGeomCreatorReturnType(sphereGeom, assetManager.get(), shaders, shaders+2);
	auto cpuMeshCone = createMeshBufferFromGeomCreatorReturnType(coneGeom, assetManager.get(), shaders, shaders+2);

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
		GPUObject ret = {};
		
		uint32_t xformBufferBinding = asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT-1u;
		uint32_t colorBufferBinding = asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT-2u;

		auto pipeline = cpuMesh->getPipeline();
		{
			// we're working with RH coordinate system(view proj) and in that case the cubeGeom frontFace is NOT CCW.
			auto & rasterParams = pipeline->getRasterizationParams();
			rasterParams.frontFaceIsCCW = 0;
			rasterParams.faceCullingMode = faceCullingMode;

			auto & vtxinputParams = pipeline->getVertexInputParams();
			vtxinputParams.bindings[colorBufferBinding].inputRate = asset::EVIR_PER_INSTANCE;
			vtxinputParams.bindings[colorBufferBinding].stride = sizeof(core::vectorSIMDf);
			vtxinputParams.bindings[xformBufferBinding].inputRate = asset::EVIR_PER_INSTANCE;
			vtxinputParams.bindings[xformBufferBinding].stride = sizeof(core::matrix3x4SIMD);
			// Color
			vtxinputParams.attributes[4].binding = colorBufferBinding;
			vtxinputParams.attributes[4].relativeOffset = 0;
			vtxinputParams.attributes[4].format = asset::E_FORMAT::EF_R32G32B32A32_SFLOAT;
			// World Row 0
			vtxinputParams.attributes[5].binding = xformBufferBinding;
			vtxinputParams.attributes[5].relativeOffset = 0;
			vtxinputParams.attributes[5].format = asset::E_FORMAT::EF_R32G32B32A32_SFLOAT;
			// World Row 1
			vtxinputParams.attributes[6].binding = xformBufferBinding;
			vtxinputParams.attributes[6].relativeOffset = sizeof(core::vector3df_SIMD);
			vtxinputParams.attributes[6].format = asset::E_FORMAT::EF_R32G32B32A32_SFLOAT;
			// World Row 2
			vtxinputParams.attributes[7].binding = xformBufferBinding;
			vtxinputParams.attributes[7].relativeOffset = sizeof(core::vector3df_SIMD)*2u;
			vtxinputParams.attributes[7].format = asset::E_FORMAT::EF_R32G32B32A32_SFLOAT;


			vtxinputParams.enabledAttribFlags |= 0x1u << 4 | 0x1u << 5 | 0x1u << 6 | 0x1u << 7;
			vtxinputParams.enabledBindingFlags |= (0x1u<<xformBufferBinding)|(0x1u<<colorBufferBinding);

			// for wireframe rendering
			#if 0
			pipeline->getRasterizationParams().polygonMode = asset::EPM_LINE; 
			#endif
		}

		asset::SPushConstantRange range[1] = { asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD) };
		auto gfxLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(range,range+1u);
		pipeline->setLayout(core::smart_refctd_ptr(gfxLayout));

		video::IGPUObjectFromAssetConverter CPU2GPU;
		core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> rpIndependentPipeline = CPU2GPU.getGPUObjectsFromAssets(&pipeline,&pipeline+1,cpu2gpuParams)->front();
		ret.gpuMesh = CPU2GPU.getGPUObjectsFromAssets(&cpuMesh,&cpuMesh+1,cpu2gpuParams)->front();
	
		asset::SBufferBinding<video::IGPUBuffer> vtxInstanceBufBnd;
		vtxInstanceBufBnd.offset = rangeStart*sizeof(core::vectorSIMDf);
		vtxInstanceBufBnd.buffer = colorBuffer;
		ret.gpuMesh->setVertexBufferBinding(std::move(vtxInstanceBufBnd),colorBufferBinding);
		vtxInstanceBufBnd.offset = rangeStart*sizeof(core::matrix3x4SIMD);
		vtxInstanceBufBnd.buffer = transformBuffer;
		ret.gpuMesh->setVertexBufferBinding(std::move(vtxInstanceBufBnd),xformBufferBinding);
		ret.gpuMesh->setInstanceCount(numInstances);

		video::IGPUGraphicsPipeline::SCreationParams gp_params;
		gp_params.rasterizationSamplesHint = asset::IImage::ESCF_1_BIT;
		gp_params.renderpass = core::smart_refctd_ptr<video::IGPURenderpass>(renderpass);
		gp_params.renderpassIndependent = rpIndependentPipeline; // TODO: fix use gpuMesh->getPipeline instead
		gp_params.subpassIx = 0u;

		ret.graphicsPipeline = device->createGPUGraphicsPipeline(nullptr,std::move(gp_params));

		return ret;
	};

	core::vector<GPUObject> gpuObjects;
	if(cubeIDs.size() > 0)
		gpuObjects.push_back(createGPUObject(cpuMeshCube.get(), cubeIDs.size(), startIndexCubes));
	if(cylinderIDs.size() > 0)
		gpuObjects.push_back(createGPUObject(cpuMeshCylinder.get(), cylinderIDs.size(), startIndexCylinders, asset::EFCM_NONE));
	if(sphereIDs.size() > 0)
		gpuObjects.push_back(createGPUObject(cpuMeshSphere.get(), sphereIDs.size(), startIndexSpheres));
	if(coneIDs.size() > 0)
		gpuObjects.push_back(createGPUObject(cpuMeshCone.get(), coneIDs.size(), startIndexCones, asset::EFCM_NONE));


	//
	auto lastTime = std::chrono::system_clock::now();
	constexpr uint32_t FRAME_COUNT = 500000u;
	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;

	core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	for (uint32_t i=0u; i<FRAMES_IN_FLIGHT; i++)
	{
		imageAcquire[i] = device->createSemaphore();
		renderFinished[i] = device->createSemaphore();
	}

	// Render
	constexpr size_t MaxFramesToAverage = 100ull;
	bool frameDataFilled = false;
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
	
	auto resourceIx = -1;

	while(windowCb->isWindowOpen())
	{
		resourceIx++;
		if(resourceIx >= FRAMES_IN_FLIGHT) {
			resourceIx = 0;
		}

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
				frameDataFilled = true;
			}
		}
		double averageFrameTime = (frameDataFilled) ? (time_sum / (double)MaxFramesToAverage) : (time_sum / frame_count);
		// logger->log("averageFrameTime = %f",system::ILogger::ELL_INFO, averageFrameTime);
		
		// Calculate Next Presentation Time Stamp
		auto averageFrameTimeDuration = std::chrono::duration<double, std::milli>(averageFrameTime);
		auto nextPresentationTime = renderStart + averageFrameTimeDuration;
		auto nextPresentationTimeStamp = std::chrono::duration_cast<std::chrono::microseconds>(nextPresentationTime.time_since_epoch());
		
		// Input 
		inputSystem->getDefaultMouse(&mouse);
		inputSystem->getDefaultKeyboard(&keyboard);

		cam.beginInputProcessing(nextPresentationTimeStamp);
		mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { cam.mouseProcess(events); }, logger.get());
		keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void {
			cam.keyboardProcess(events); 
		}, logger.get());
		cam.endInputProcessing(nextPresentationTimeStamp);

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
		swapchain->acquireNextImage(MAX_TIMEOUT,imageAcquire[resourceIx].get(),nullptr,&imgnum);
		// Update instances buffer 
		{
			// Update Physics (TODO: fixed timestep)
			world->getWorld()->stepSimulation(dt);
			video::CPropertyPoolHandler::TransferRequest request;
			request.download = false;
			request.pool = propertyPool.get();
			request.indices = {CInstancedMotionState::s_updateIndices.data(),CInstancedMotionState::s_updateIndices.data()+CInstancedMotionState::s_updateIndices.size()};
			request.propertyID = TransformPropertyID;
			request.data = CInstancedMotionState::s_updateData.data();
			auto result = propertyPoolHandler->transferProperties(cb.get(),fence.get(),&request,&request+1u,logger.get());
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
			info.framebuffer = fbo[imgnum];
			info.clearValueCount = 2u;
			info.clearValues = clearValues;
			info.renderArea.offset = { 0, 0 };
			info.renderArea.extent = { WIN_W, WIN_H };
			cb->beginRenderPass(&info,asset::ESC_INLINE);
		}
		// draw
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
	
	for (uint32_t i=0u; i<MaxNumInstances; ++i)
	{
		world->unbindRigidBody(bodies[i]);
		world->deleteRigidBody(bodies[i]);
	}

	world->deletebtObject(cubeRigidBodyData.shape);
	world->deletebtObject(cylinderRigidBodyData.shape);
	world->deletebtObject(sphereRigidBodyData.shape);
	world->deletebtObject(coneRigidBodyData.shape);


	return 0;
}