#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "irr/ext/ScreenShot/ScreenShot.h"
#include "../common/QToQuitEventReceiver.h"

using namespace irr;
using namespace core;
using namespace asset;


#if 0
const char* uniformNames[] =
{
    "ProjViewWorldMat",
    "WorldMat",
    "ViewWorldMat",
    "eyePos",
    "LoDInvariantMinEdge",
    "LoDInvariantBBoxCenter",
    "LoDInvariantMaxEdge",
    "instanceLoDDistancesSQ"
};

enum E_UNIFORM
{
    EU_PROJ_VIEW_WORLD_MAT = 0,
    EU_WORLD_MAT,
    EU_VIEW_WORLD_MAT,
    EU_EYE_POS,
    EU_LOD_INVARIANT_MIN_EDGE,
    EU_LOD_INVARIANT_BBOX_CENTER,
    EU_LOD_INVARIANT_MAX_EDGE,
    EU_INSTANCE_LOD_DISTANCE_SQ,
    EU_COUNT
};

class SimpleCallBack : public video::IShaderConstantSetCallBack
{
    video::E_MATERIAL_TYPE currentMat;
    int32_t uniformLocation[video::EMT_COUNT+2][EU_COUNT];
    video::E_SHADER_CONSTANT_TYPE uniformType[video::EMT_COUNT+2][EU_COUNT];
    float currentLodPass;
public:
    core::aabbox3df instanceLoDInvariantBBox;

    SimpleCallBack()
    {
        for (size_t i=0; i<EU_COUNT; i++)
        for (size_t j=0; j<video::EMT_COUNT+2; j++)
            uniformLocation[j][i] = -1;
    }

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
    {
        for (size_t i=0; i<constants.size(); i++)
        for (size_t j=0; j<EU_COUNT; j++)
        {
            if (constants[i].name==uniformNames[j])
            {
                uniformLocation[materialType][j] = constants[i].location;
                uniformType[materialType][j] = constants[i].type;
                break;
            }
        }
    }

    virtual void OnSetMaterial(video::IMaterialRendererServices* services, const video::SGPUMaterial& material, const video::SGPUMaterial& lastMaterial)
    {
        currentMat = material.MaterialType;
        currentLodPass = material.MaterialTypeParam;
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        if (uniformLocation[currentMat][EU_PROJ_VIEW_WORLD_MAT]>=0)
            services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(),uniformLocation[currentMat][EU_PROJ_VIEW_WORLD_MAT],uniformType[currentMat][EU_PROJ_VIEW_WORLD_MAT],1);
        if (uniformLocation[currentMat][EU_VIEW_WORLD_MAT]>=0)
            services->setShaderConstant(services->getVideoDriver()->getTransform(video::E4X3TS_WORLD_VIEW).rows[0].pointer,uniformLocation[currentMat][EU_VIEW_WORLD_MAT],uniformType[currentMat][EU_VIEW_WORLD_MAT],1);
        if (uniformLocation[currentMat][EU_WORLD_MAT]>=0)
            services->setShaderConstant(services->getVideoDriver()->getTransform(video::E4X3TS_WORLD).rows[0].pointer,uniformLocation[currentMat][EU_WORLD_MAT],uniformType[currentMat][EU_WORLD_MAT],1);

        if (uniformLocation[currentMat][EU_EYE_POS]>=0)
        {
            core::vectorSIMDf eyePos;
            eyePos.set(services->getVideoDriver()->getTransform(video::E4X3TS_VIEW_INVERSE).getTranslation());
            services->setShaderConstant(eyePos.pointer,uniformLocation[currentMat][EU_EYE_POS],uniformType[currentMat][EU_EYE_POS],1);
        }

        if (uniformLocation[currentMat][EU_LOD_INVARIANT_BBOX_CENTER]>=0)
        {
            services->setShaderConstant(&instanceLoDInvariantBBox.MinEdge,uniformLocation[currentMat][EU_LOD_INVARIANT_MIN_EDGE],uniformType[currentMat][EU_LOD_INVARIANT_MIN_EDGE],1);
            core::vector3df centre = instanceLoDInvariantBBox.getCenter();
            services->setShaderConstant(&centre,uniformLocation[currentMat][EU_LOD_INVARIANT_BBOX_CENTER],uniformType[currentMat][EU_LOD_INVARIANT_BBOX_CENTER],1);
            services->setShaderConstant(&instanceLoDInvariantBBox.MaxEdge,uniformLocation[currentMat][EU_LOD_INVARIANT_MAX_EDGE],uniformType[currentMat][EU_LOD_INVARIANT_MAX_EDGE],1);
        }


        if (uniformLocation[currentMat][EU_INSTANCE_LOD_DISTANCE_SQ]>=0)
        {
            float distancesSQ[2];
            distancesSQ[0] = instanceLoDDistances[0]*instanceLoDDistances[0];
            distancesSQ[1] = instanceLoDDistances[1]*instanceLoDDistances[1];
            services->setShaderConstant(distancesSQ,uniformLocation[currentMat][EU_INSTANCE_LOD_DISTANCE_SQ],uniformType[currentMat][EU_INSTANCE_LOD_DISTANCE_SQ],1);
        }
    }

    virtual void OnUnsetMaterial() {}
};
#endif

#include "common.glsl"

struct CullUniformBuffer
{
	core::matrix4SIMD viewProj;
	core::matrix3x4SIMD view;
	float camPos[3];
	uint32_t pad0;
	float LoDInvariantAABBMinEdge[3];
	uint32_t pad1;
	float LoDInvariantAABBMaxEdge[3];
	uint32_t pad2;
	uint32_t LoDDistancesSq[6969];
};

struct alignas(uint32_t) MaterialProps
{
	uint8_t something[128];
};

/*
EXPAND SHADER

uint culledMeshID = someFunc(gl_GlobalInvocationIndex);
uint meshBufferID = someFunc(gl_GlobalInvocationIndex);

mat4 mvp = inView.objects[culledMeshID].mvp;
vec3 MinEdge = constWorld.meshbuffers[meshBufferID].aabbMinEdge;
vec3 MaxEdge = constWorld.meshbuffers[meshBufferID].aabbMaxEdge;

if (!culled(mvp,MinEdge,MaxEdge))
{
	// TODO: could optimize the reduction
	uint instanceID = atomicAdd(outView.draws[meshBufferID].instanceCount,1u);
	outView.instanceRedirects[atomicAdd(outView.instanceRedirectCount,1u)] = vec2(meshBufferID,objectID);
}



SORT REDIRECTS SHADER

uint selfIx = sortByMeshBufferIDAndThenDistance(outView.instanceRedirects)
if (lowerBoundofMeshBufferID)
	outView.draws[meshBufferID].baseInstance = selfIx;
*/

struct ModelLoD
{
	ModelLoD(IAssetManager* assMgr, video::IVideoDriver* driver, float _distance, const char* modelPath) : mesh(nullptr), distance(_distance)
	{
		// load model
		IAssetLoader::SAssetLoadParams lp;
        auto bundle = assMgr->getAsset(modelPath,lp);
		assert(!bundle.isEmpty());
		auto cpumesh = bundle.getContents().begin()[0];
		auto cpumesh_raw = static_cast<ICPUMesh*>(cpumesh.get());

		const CMTLPipelineMetadata* pipelineMetadata = nullptr;
		core::map<const ICPURenderpassIndependentPipeline*,core::smart_refctd_ptr<ICPURenderpassIndependentPipeline>> modifiedPipelines;
		for (uint32_t i=0u; i<cpumesh_raw->getMeshBufferCount(); i++)
		{
			auto mb = cpumesh_raw->getMeshBuffer(i);
			// assert that the meshbuffers are indexed
			assert(!!mb->getIndexBufferBinding()->buffer);
			// override vertex shader
			auto* pipeline = mb->getPipeline();
			auto found = modifiedPipelines.find(pipeline);
			if (found==modifiedPipelines.end())
			{
				// new pipeline to modify, copy the pipeline
				auto pipeline_cp = core::smart_refctd_ptr_static_cast<ICPURenderpassIndependentPipeline>(pipeline->clone(1u));

				// get metadata
				assert(pipelineMetadata->getLoaderName() == "CGraphicsPipelineLoaderMTL");
				pipelineMetadata = static_cast<const CMTLPipelineMetadata*>(pipeline->getMetadata());

				// add the redirect attribute buffer
				constexpr uint32_t redirectAttributeID = 15u;
				auto& vertexInput = pipeline_cp->getVertexInputParams();
				vertexInput.attributes[redirectAttributeID].binding = 15u;
				vertexInput.attributes[redirectAttributeID].format = EF_R32_UINT;
				vertexInput.attributes[redirectAttributeID].relativeOffset = 0u;
				vertexInput.enabledAttribFlags |= 0x1u<<redirectAttributeID;
				vertexInput.bindings[redirectAttributeID].stride = sizeof(uint32_t);
				vertexInput.bindings[redirectAttributeID].inputRate = EVIR_PER_INSTANCE;
				vertexInput.enabledBindingFlags |= 0x1u<<redirectAttributeID;

				// replace the descriptor set layout with one that ...
				auto* layout = pipeline_cp->getLayout();
//				layout->setDescriptorSetLayout(0, core::smart_refctd_ptr(ds0layout));

				// cache the result
				found = modifiedPipelines.emplace(pipeline,std::move(pipeline_cp)).first;
			}
			mb->setPipeline(core::smart_refctd_ptr(found->second));
		}
		//assert(pipelineMetadata);

		// convert to GPU Mesh
		mesh = driver->getGPUObjectsFromAssets(&cpumesh_raw,&cpumesh_raw+1)->front();
	}

	core::smart_refctd_ptr<video::IGPUMesh> mesh;
	float distance;
};

struct CameraData
{
	CameraData(video::IVideoDriver* driver, scene::ISceneManager* smgr, uint32_t maxUniqueDrawcalls, float moveSpeed=0.01f)
	{
		node = smgr->addCameraSceneNodeFPS(nullptr,100.0f,moveSpeed);

		objectUUIDRedirects = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(uint32_t)*maxUniqueDrawcalls);
	}

	scene::ICameraSceneNode* node;
	core::smart_refctd_ptr<video::IGPUBuffer> objectUUIDRedirects;
};

int main()
{
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	auto device = createDeviceEx(params);

	if (!device)
		return 1; // could not create selected driver.

	device->getCursorControl()->setVisible(false);

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);


	auto assMgr = device->getAssetManager();
	auto driver = device->getVideoDriver();


	constexpr auto kHardMeshbufferLimit = 0x1u<<10u;

	const core::vector<ModelLoD> instanceLoDs = { ModelLoD(assMgr,driver,8.f,"../../media/cow.obj"),ModelLoD(assMgr,driver,50.f,"../../media/yellowflower.obj") };
	const auto kLoDLevels = instanceLoDs.size();
	
	core::aabbox3df lodInvariantAABB;
	{
		lodInvariantAABB = instanceLoDs[0].mesh->getBoundingBox();
		for (auto i=1u; i<kLoDLevels; i++)
			lodInvariantAABB.addInternalBox(instanceLoDs[i].mesh->getBoundingBox());
	}
	


	auto poolHandler = driver->getDefaultPropertyPoolHandler();

	constexpr auto kNumHardwareInstancesX = 10;
	constexpr auto kNumHardwareInstancesY = 20;
	constexpr auto kNumHardwareInstancesZ = 30;

	constexpr auto kHardwareInstancesTOTAL = kNumHardwareInstancesX*kNumHardwareInstancesY*kNumHardwareInstancesZ;

	// create the pool
	auto instanceData = [&]()
	{
		SBufferRange<video::IGPUBuffer> memoryBlock;
		memoryBlock.buffer = driver->createDeviceLocalGPUBufferOnDedMem((sizeof(SceneNode_t)+sizeof(MaterialProps))*kHardwareInstancesTOTAL);
		memoryBlock.size = memoryBlock.buffer->getSize();
		return video::CPropertyPool<core::allocator,SceneNode_t,MaterialProps>::create(std::move(memoryBlock));
	}();

	// use the pool
	core::vector<uint32_t> instanceIDs(kHardwareInstancesTOTAL,video::IPropertyPool::invalid_index);
	{
		// create the instances
		{
			core::vector<SceneNode_t> propsA(kHardwareInstancesTOTAL);
			core::vector<MaterialProps> propsB(kHardwareInstancesTOTAL);
		
			auto propAIt = propsA.begin();
			auto propBIt = propsB.begin();
			for (size_t z=0; z<kNumHardwareInstancesZ; z++)
			for (size_t y=0; y<kNumHardwareInstancesY; y++)
			for (size_t x=0; x<kNumHardwareInstancesX; x++)
			{
				propAIt->worldTransform.setTranslation(core::vectorSIMDf(x,y,z)*2.f);
				/*
				for (auto i=0; i<3; i++)
				{
					propBIt->lodInvariantAABBMinEdge[i] = ;
					propBIt->lodInvariantAABBMaxEdge[i] = ;
				}
				*/
				propAIt++,propBIt++;
			}
			
			const void* data[] = { propsA.data(),propsB.data() };
			video::CPropertyPoolHandler::AllocationRequest req{ instanceData.get(),{instanceIDs.data(),instanceIDs.data()+instanceIDs.size()},data };
			poolHandler->addProperties(&req,&req+1);
		}

		// remove some randomly
		srand(6945);
        for (size_t i=0; i<600; i++)
        {
            uint32_t ix = rand()%kHardwareInstancesTOTAL;
            uint32_t& instanceID = instanceIDs[ix];
            if (instanceID==video::IPropertyPool::invalid_index)
                continue;

			instanceData->freeProperties(&instanceID,&instanceID+1);
            instanceID = video::IPropertyPool::invalid_index;
        }

		// remember which ones we need to get rid of
		auto newEnd = std::remove(instanceIDs.begin(),instanceIDs.end(),video::IPropertyPool::invalid_index);
		instanceIDs.resize(std::distance(instanceIDs.begin(),newEnd));
	}


	auto perViewInstanceData = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(VisibleObject_t)*kHardwareInstancesTOTAL);
	
	uint32_t drawCallCount;
	auto drawCallParameters = [&]()
	{
		core::vector<DrawElementsIndirectCommand_t> indirectCmds;
		for (auto lod : instanceLoDs)
		for (uint32_t i=0u; i<lod.mesh->getMeshBufferCount(); i++)
		{
			const auto meshbuffer = lod.mesh->getMeshBuffer(i);
			const auto indexBufferOffset = meshbuffer->getIndexBufferBinding().offset;

			DrawElementsIndirectCommand_t draw;
			draw.count = meshbuffer->getIndexCount();
			draw.instanceCount = 0u;
			switch (meshbuffer->getIndexType())
			{
				case EIT_16BIT:
					draw.firstIndex = indexBufferOffset/sizeof(uint16_t);
					break;
				case EIT_32BIT:
					draw.firstIndex = indexBufferOffset/sizeof(uint32_t);
					break;
				default:
					assert(false);
					break;
			}
			draw.baseVertex = meshbuffer->getBaseVertex();
			draw.baseInstance = 0u;
			indirectCmds.push_back(draw);
		}
		assert(instanceLoDs.size()<kHardMeshbufferLimit);
		drawCallCount = indirectCmds.size();
		return driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(DrawElementsIndirectCommand_t)*drawCallCount,indirectCmds.data());
	}();



	const auto extensions = driver->getSupportedGLSLExtensions();
	CShaderIntrospector introspector(assMgr->getGLSLCompiler());
	auto createComputePipelineFromFile = [&](const char* shaderPath)
	{
		IAssetLoader::SAssetLoadParams lp;
		auto shaderBundle = assMgr->getAsset(shaderPath,lp);

		auto cpuPipeline = introspector.createApproximateComputePipelineFromIntrospection(
			IAsset::castDown<ICPUSpecializedShader>(shaderBundle.getContents().begin()->get()),
			extensions->begin(),extensions->end()
		);
		return driver->getGPUObjectsFromAssets(&cpuPipeline.get(),&cpuPipeline.get()+1)->operator[](0);
	};

	auto clearDrawsAndCullObjectsPipeline = createComputePipelineFromFile("../clearDrawsAndCullObjects.comp");
	auto expandObjectsIntoDrawcallsPipeline = createComputePipelineFromFile("../expandObjectsIntoDrawcalls.comp");
	auto setupRedirectsPipeline = createComputePipelineFromFile("../setupRedirects.comp");


	auto smgr = device->getSceneManager();

	CameraData camera(driver,smgr,kHardMeshbufferLimit);
	camera.node->setPosition(core::vector3df(-4, 0, 0));
	camera.node->setTarget(core::vector3df(0, 0, 0));
	camera.node->setNearValue(0.01f);
	camera.node->setFarValue(100.0f);

	smgr->setActiveCamera(camera.node);
	uint64_t lastFPSTime = 0;
	// render
	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255, 128, 128, 128));

		//! This animates (moves) the camera and sets the transforms
		camera.node->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera.node->render();

		// TODO: task for a junior, make the instances spin
		{
			//poolHandler->transferProperties();
		}

		// compute shader culling
		{
			auto workgroupCount = [](uint32_t workItems) -> uint32_t
			{
				return (workItems+kOptimalWorkgroupSize-1)/kOptimalWorkgroupSize;
			};

            driver->bindComputePipeline(clearDrawsAndCullObjectsPipeline.get());
            //driver->bindDescriptorSets(video::EPBP_COMPUTE, clearDrawsAndCullObjectsPipeline->getLayout(), 1u, 1u, &cullDescriptorSet.get(), nullptr);
            //driver->pushConstants(clearDrawsAndCullObjectsPipeline->getLayout(), asset::ICPUSpecializedShader::ESS_COMPUTE, 0u, sizeof(CullShaderData_t), &pc);
            driver->dispatch(workgroupCount(core::max<uint32_t>(instanceIDs.size(),drawCallCount)),1u,1u);
            video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
			//
			driver->bindComputePipeline(expandObjectsIntoDrawcallsPipeline.get());
            //driver->dispatch(workgroupCount(instanceIDs.size()),1u,1u);
			video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
			//
			driver->bindComputePipeline(setupRedirectsPipeline.get());
            //driver->dispatch(workgroupCount(instanceIDs.size()),1u,1u);
			video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_COMMAND_BARRIER_BIT);
		}
		
		// draw indirects
	}

	instanceData->freeProperties(instanceIDs.data(),instanceIDs.data()+instanceIDs.size());

#if 0
	//create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		ext::ScreenShot::dirtyCPUStallingScreenshot(driver,device->getAssetManager(), "screenshot.png", sourceRect, EF_R8G8B8_SRGB);
	}
#endif

	return 0;
}
