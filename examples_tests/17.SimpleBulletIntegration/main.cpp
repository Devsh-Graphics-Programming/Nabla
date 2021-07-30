// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "CFileSystem.h"

#include <btBulletDynamicsCommon.h>
#include "BulletCollision/NarrowPhaseCollision/btRaycastCallback.h"

#include "nbl/ext/Bullet/BulletUtility.h"
#include "nbl/ext/Bullet/CPhysicsWorld.h"

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

#include <nbl/builtin/glsl/utils/common.glsl>
#include <nbl/builtin/glsl/utils/transform.glsl>

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

    float ambient = 0.2;
    float diffuse = 0.8;
    float cos_theta_term = max(dot(normal,vec3(1.0,1.0,1.0)),0.0);

    float fresnel = 0.0; //not going to implement yet, not important
    float specular = 0.0;///pow(max(dot(halfVector,normal),0.0),shininess);

    const float sunPower = 3.14156*0.5;

    pixelColor = vec4(Color, 1)*sunPower*(ambient+mix(diffuse,specular,fresnel)*cos_theta_term/3.14159);
}
)===";

struct InstanceData {
	core::vector3df_SIMD color;
	core::matrix3x4SIMD modelMatrix; // = 3 x vector3df_SIMD
};

class CInstancedMotionState : public ext::Bullet3::IMotionStateBase{
public:
    inline CInstancedMotionState() {}
    inline CInstancedMotionState(core::vector<InstanceData> * instancesData, uint32_t index, core::matrix3x4SIMD const & start_mat)
        : m_instances_data(instancesData), 
          m_index(index),
          ext::Bullet3::IMotionStateBase(ext::Bullet3::convertMatrixSIMD(start_mat))
    {
		m_index = index;
    }

    inline ~CInstancedMotionState() {
    }

	inline virtual void getWorldTransform(btTransform &worldTrans) const override {
		if(m_instances_data != nullptr) {
			auto mat = (*m_instances_data)[m_index].modelMatrix;
			worldTrans = ext::Bullet3::convertMatrixSIMD(mat);
		}
	};

	inline virtual void setWorldTransform(const btTransform &worldTrans) override {
		if(m_instances_data != nullptr) {
			if(m_instances_data->size() > m_index) {
				(*m_instances_data)[m_index].modelMatrix = ext::Bullet3::convertbtTransform(worldTrans);
			}
		}
	};

protected:

    core::vector<InstanceData> * m_instances_data;
    uint32_t m_index;
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
	constexpr uint32_t SC_IMG_COUNT = 3u;
	constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT>SC_IMG_COUNT);

	auto initOutp = CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(video::EAT_OPENGL, "Physics Simulation");
	auto win = std::move(initOutp.window);
	auto gl = std::move(initOutp.apiConnection);
	auto surface = std::move(initOutp.surface);
	auto device = std::move(initOutp.logicalDevice);
	auto gpu = std::move(initOutp.physicalDevice);
	auto queue = std::move(initOutp.queue);
	auto sc = std::move(initOutp.swapchain);
	auto renderpass = std::move(initOutp.renderpass);
	auto fbo = std::move(initOutp.fbo);
	auto cmdpool = std::move(initOutp.commandPool);
	
	core::vector<GPUObject> gpuObjects; 

	// Instance Data
	constexpr uint32_t NumCubes = 20;
	constexpr uint32_t NumCylinders = 0;
	constexpr uint32_t NumSpheres = 20;

	constexpr uint32_t startIndexCubes = 0;
	constexpr uint32_t startIndexCylinders = startIndexCubes + NumCubes;
	constexpr uint32_t startIndexSpheres = startIndexCylinders + NumCylinders;
	
	constexpr uint32_t NumInstances = NumCubes + NumCylinders + NumSpheres;
	
	core::vector<InstanceData> instancesData;
	instancesData.resize(NumInstances);
	
	for(uint32_t i = 0; i < NumInstances; ++i) {
		instancesData[i].color = core::vector3df_SIMD(float(i) / float(NumInstances), 0.5f, 1.0f);
	}

	// Physics Setup
    ext::Bullet3::CPhysicsWorld *world = _NBL_NEW(nbl::ext::Bullet3::CPhysicsWorld);
    world->getWorld()->setGravity(btVector3(0, -5, 0));

	// BasePlate
    core::matrix3x4SIMD baseplateMat;
    baseplateMat.setTranslation(core::vectorSIMDf(0.0, -1.0, 0.0));

    ext::Bullet3::CPhysicsWorld::RigidBodyData basePlateRigidBodyData;
    basePlateRigidBodyData.mass = 0.0f;
    basePlateRigidBodyData.shape = world->createbtObject<btBoxShape>(btVector3(300, 1, 300));
    basePlateRigidBodyData.trans = baseplateMat;

    btRigidBody *body2 = world->createRigidBody(basePlateRigidBodyData);
    world->bindRigidBody(body2);

	core::vector<btRigidBody*> bodies;
	bodies.resize(NumInstances);
		
	// Shapes RigidBody Data
    ext::Bullet3::CPhysicsWorld::RigidBodyData cubeRigidBodyData;
    ext::Bullet3::CPhysicsWorld::RigidBodyData cylinderRigidBodyData;
    ext::Bullet3::CPhysicsWorld::RigidBodyData sphereRigidBodyData;
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
		sphereRigidBodyData.shape = world->createbtObject<btSphereShape>(0.5f);
		btVector3 inertia;
		sphereRigidBodyData.shape->calculateLocalInertia(sphereRigidBodyData.mass, inertia);
		sphereRigidBodyData.inertia = ext::Bullet3::frombtVec3(inertia);
	}

	for(uint32_t i = 0; i < NumInstances; ++i) {
		auto & rigidBodyData = cubeRigidBodyData;
		if(i >= startIndexSpheres) {
			rigidBodyData = sphereRigidBodyData;
		} else if(i >= startIndexCylinders) {
			rigidBodyData = cylinderRigidBodyData;
		}

        core::matrix3x4SIMD mat;
        mat.setTranslation(core::vectorSIMDf(0.0f, i * 5.0f, i * 2.0f));

		instancesData[i].modelMatrix = mat;
		rigidBodyData.trans = mat;
		
		auto & body = bodies[i];

		bodies[i] = world->createRigidBody(rigidBodyData);
		world->bindRigidBody<CInstancedMotionState>(body, &instancesData, i, mat);
	}

	// TODO? Setup Debug Draw

	// Asset Manager
	auto filesystem = core::make_smart_refctd_ptr<io::CFileSystem>("");
	auto am = core::make_smart_refctd_ptr<asset::IAssetManager>(std::move(filesystem));


	// CPU2GPU
	video::IGPUObjectFromAssetConverter CPU2GPU;

	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	cpu2gpuParams.assetManager = am.get();
	cpu2gpuParams.device = device.get();
	cpu2gpuParams.perQueue[video::IGPUObjectFromAssetConverter::EQU_TRANSFER].queue = queue;
	cpu2gpuParams.perQueue[video::IGPUObjectFromAssetConverter::EQU_COMPUTE].queue = queue;
	cpu2gpuParams.finalQueueFamIx = queue->getFamilyIndex();
	cpu2gpuParams.sharingMode = asset::ESM_CONCURRENT;
	cpu2gpuParams.limits = gpu->getLimits();
	cpu2gpuParams.assetManager = am.get();

	// weird fix -> do not read the next 6 lines (It doesn't affect the program logically) -> waiting for access_violation_repro branch to fix and merge
	core::smart_refctd_ptr<asset::ICPUShader> computeUnspec;
	{
		 auto file = am->getFileSystem()->createAndOpenFile("../../29.SpecializationConstants/particles.comp");
		 computeUnspec = am->getGLSLCompiler()->resolveIncludeDirectives(file, asset::ISpecializedShader::ESS_COMPUTE, file->getFileName().c_str());
		 file->drop();
	}

	// Geom Create
	auto geometryCreator = am->getGeometryCreator();
	auto cubeGeom = geometryCreator->createCubeMesh(core::vector3df(1.0f, 1.0f, 1.0f));
	auto cylinderGeom = geometryCreator->createCylinderMesh(0.5f, 0.5f, 20);
	auto sphereGeom = geometryCreator->createSphereMesh(0.5f);

	// Camera Stuff
	core::vectorSIMDf cameraPosition(0, 0, -10);
	matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(60), float(WIN_W) / WIN_H, 0.01, 100);
	matrix3x4SIMD view = matrix3x4SIMD::buildCameraLookAtMatrixRH(cameraPosition, core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 1, 0));
	auto viewProj = matrix4SIMD::concatenateBFollowedByA(proj, matrix4SIMD(view));

	// Creating CPU Shaders 

	auto createCPUSpecializedShaderFromSource = [=](const char* source, asset::ISpecializedShader::E_SHADER_STAGE stage) -> core::smart_refctd_ptr<asset::ICPUSpecializedShader>
	{
		auto unspec = am->getGLSLCompiler()->createSPIRVFromGLSL(source, stage, "main", "runtimeID");
		if (!unspec)
			return nullptr;

		asset::ISpecializedShader::SInfo info(nullptr, nullptr, "main", stage, "");
		return core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspec), std::move(info));
	};

	auto createCPUSpecializedShaderFromSourceWithIncludes = [&](const char* source, asset::ISpecializedShader::E_SHADER_STAGE stage, const char* origFilepath)
	{
		auto resolved_includes = am->getGLSLCompiler()->resolveIncludeDirectives(source, stage, origFilepath);
		return createCPUSpecializedShaderFromSource(reinterpret_cast<const char*>(resolved_includes->getSPVorGLSL()->getPointer()), stage);
	};

	auto vs = createCPUSpecializedShaderFromSourceWithIncludes(vertexSource,asset::ISpecializedShader::ESS_VERTEX, "shader.vert");
	auto fs = createCPUSpecializedShaderFromSource(fragmentSource,asset::ISpecializedShader::ESS_FRAGMENT);
	asset::ICPUSpecializedShader* shaders[2]{ vs.get(), fs.get() };
	
	auto cpuMeshCube = createMeshBufferFromGeomCreatorReturnType(cubeGeom, am.get(), shaders, shaders+2);
	auto cpuMeshCylinder = createMeshBufferFromGeomCreatorReturnType(cylinderGeom, am.get(), shaders, shaders+2);
	auto cpuMeshSphere = createMeshBufferFromGeomCreatorReturnType(sphereGeom, am.get(), shaders, shaders+2);
	
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
		
		auto pipeline = cpuMesh->getPipeline();
		{
			// we're working with RH coordinate system(view proj) and in that case the cubeGeom frontFace is NOT CCW.
			auto & rasterParams = pipeline->getRasterizationParams();
			rasterParams.frontFaceIsCCW = 0;
			rasterParams.faceCullingMode = faceCullingMode;
			rasterParams.depthTestEnable = true;
			rasterParams.depthCompareOp = asset::ECO_GREATER;

			auto & vtxinputParams = pipeline->getVertexInputParams();
			vtxinputParams.bindings[1].inputRate = asset::EVIR_PER_INSTANCE;
			vtxinputParams.bindings[1].stride = sizeof(InstanceData);
			// Color
			vtxinputParams.attributes[4].binding = 1;
			vtxinputParams.attributes[4].relativeOffset = 0;
			vtxinputParams.attributes[4].format = asset::E_FORMAT::EF_R32G32B32A32_SFLOAT;
			// World Row 0
			vtxinputParams.attributes[5].binding = 1;
			vtxinputParams.attributes[5].relativeOffset = sizeof(core::vector3df_SIMD);
			vtxinputParams.attributes[5].format = asset::E_FORMAT::EF_R32G32B32A32_SFLOAT;
			// World Row 1
			vtxinputParams.attributes[6].binding = 1;
			vtxinputParams.attributes[6].relativeOffset = sizeof(core::vector3df_SIMD) * 2;
			vtxinputParams.attributes[6].format = asset::E_FORMAT::EF_R32G32B32A32_SFLOAT;
			// World Row 2
			vtxinputParams.attributes[7].binding = 1;
			vtxinputParams.attributes[7].relativeOffset = sizeof(core::vector3df_SIMD) * 3;
			vtxinputParams.attributes[7].format = asset::E_FORMAT::EF_R32G32B32A32_SFLOAT;


			vtxinputParams.enabledAttribFlags |= 0x1u << 4 | 0x1u << 5 | 0x1u << 6 | 0x1u << 7;
			vtxinputParams.enabledBindingFlags |= 0x1u << 1;
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
		ret.gpuMesh->setVertexBufferBinding(std::move(vtxInstanceBufBnd), 1);
		ret.gpuMesh->setInstanceCount(numInstances);

		video::IGPUGraphicsPipeline::SCreationParams gp_params;
		gp_params.rasterizationSamplesHint = asset::IImage::ESCF_1_BIT;
		gp_params.renderpass = core::smart_refctd_ptr<video::IGPURenderpass>(renderpass);
		gp_params.renderpassIndependent = rpIndependentPipeline; // TODO: fix use gpuMesh->getPipeline instead
		gp_params.subpassIx = 0u;

		ret.graphicsPipeline = device->createGPUGraphicsPipeline(nullptr, std::move(gp_params));

		return ret;
	};

	if(NumCubes > 0) {
		gpuObjects.push_back(createGPUObject(cpuMeshCube.get(), gpuInstancesBuffer, NumCubes, sizeof(InstanceData) * startIndexCubes));
	}
	if(NumCylinders > 0) {
		gpuObjects.push_back(createGPUObject(cpuMeshCylinder.get(), gpuInstancesBuffer, NumCylinders, sizeof(InstanceData) * startIndexCylinders, asset::EFCM_NONE));
	}
	if(NumSpheres > 0) {
		gpuObjects.push_back(createGPUObject(cpuMeshSphere.get(), gpuInstancesBuffer, NumSpheres, sizeof(InstanceData) * startIndexSpheres));
	}

	auto lastTime = std::chrono::high_resolution_clock::now();
	constexpr uint32_t FRAME_COUNT = 500000u;
	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;

	core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf[FRAMES_IN_FLIGHT];
	device->createCommandBuffers(cmdpool.get(),video::IGPUCommandBuffer::EL_PRIMARY,FRAMES_IN_FLIGHT,cmdbuf);
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
			vp.minDepth = 1.f;
			vp.maxDepth = 0.f;
			vp.x = 0u;
			vp.y = 0u;
			vp.width = WIN_W;
			vp.height = WIN_H;
			cb->setViewport(0u, 1u, &vp);
		}
		// renderpass 
		uint32_t imgnum = 0u;
		sc->acquireNextImage(MAX_TIMEOUT,imageAcquire[resourceIx].get(),nullptr,&imgnum);
		{
			video::IGPUCommandBuffer::SRenderpassBeginInfo info;
			asset::SClearValue clear;
			asset::VkRect2D area;
			clear.color.float32[0] = 0.1f;
			clear.color.float32[1] = 0.1f;
			clear.color.float32[2] = 0.1f;
			clear.color.float32[3] = 1.f;
			info.renderpass = renderpass;
			info.framebuffer = fbo[imgnum];
			info.clearValueCount = 1u;
			info.clearValues = &clear;
			info.renderArea.offset = { 0, 0 };
			info.renderArea.extent = { WIN_W, WIN_H };
			cb->beginRenderPass(&info,asset::ESC_INLINE);
		}
		// Update instances buffer 
		{
			// Update Physics
			world->getWorld()->stepSimulation(dt);
			
			asset::SBufferRange<video::IGPUBuffer> range;
			range.buffer = gpuInstancesBuffer;
			range.offset = 0;
			range.size = BUF_SZ;
			device->updateBufferRangeViaStagingBuffer(queue, range, instancesData.data());
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

		CommonAPI::Submit(device.get(), sc.get(), cb.get(), queue, imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
		CommonAPI::Present(device.get(), sc.get(), queue, renderFinished[resourceIx].get(), imgnum);
		
	}
	
    world->unbindRigidBody(body2, false);
    world->deleteRigidBody(body2);
	
    for (uint32_t i = 0; i < NumInstances; ++i) {
        world->unbindRigidBody(bodies[i]);
        world->deleteRigidBody(bodies[i]);
    }

	world->deletebtObject(cubeRigidBodyData.shape);

	world->drop();


	return 0;
}


// If you see this line of code, i forgot to remove it
// It forces the usage of NVIDIA GPU by OpenGL
extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }