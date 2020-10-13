#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "irr/ext/ScreenShot/ScreenShot.h"
#include "../common/QToQuitEventReceiver.h"

using namespace irr;
using namespace core;
using namespace scene;


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

struct ViewInvariantProps
{
	core::matrix3x4SIMD tform;
};
struct alignas(uint32_t) MaterialProps
{
	uint8_t something[128];
};

struct ViewDependentObjectInvariantProps
{
	core::matrix4SIMD mvp;
	union
	{
		struct
		{
			float normalMatrixRow0[3];
			uint32_t objectUUID;
			float normalMatrixRow1[3];
			int32_t meshUUID;
			float normalMatrixRow2[3];
		};
		core::matrix3x4SIMD normalMatrix;
	};
};

/*
CULL SHADER (could process numerous cameras)

mat4 viewProj = pc.viewProj;

uint objectUUID = objectIDs[gl_GlobalInvocationIndex];
ViewInvariantProps objProps = props[objectUUID];
mat4x3 world = objProps.tform;
mat4 mvp = viewProj*mat4(vec4(world[0],0.0),vec4(world[1],0.0),vec4(world[2],0.0),vec4(world[3],1.0));

vec3 MinEdge = pc.LoDInvariantAABBMinEdge;
vec3 MaxEdge = pc.LoDInvariantAABBMaxEdge;

if (!culled(mvp,MinEdge,MaxEdge))
{
	vec3 toCamera = pc.camPos-world[3];
	float distanceSq = dot(toCamera,toCamera);

	int lod = specConstantMaxLoDLevels;
	for (int i<0; i<specConstantMaxLoDLevels; i++)
	if (distanceSq<pc.LoDDistancesSq[i])
	{
		lod = i;
		break;
	}

	if (lod<specConstantMaxLoDLevels)
	{
		mat3 normalMatrixT = inverse(mat3(pc.view)*mat3(world));

		ViewDependentObjectInvariantProps objectToDraw;
		objectToDraw.mvp = mvp;
		objectToDraw.normalMatrixRow0 = normalMatrixT[0];
		objectToDraw.objectUUID = objectUUID;
		objectToDraw.normalMatrixRow1 = normalMatrixT[1];
		objectToDraw.meshUUID = pc.LoDMeshes[lod];
		objectToDraw.normalMatrixRow2 = normalMatrixT[2];
		// TODO: could optimize the reduction
		outView.objects[atomicAdd(outView.objCount,1u)] = objectToDraw;
	}
}



SORT AND SETUP VIEW

if (gl_GlobalInvocationIndex<TOTAL_MESHBUFFERS)
	outView.draws[gl_GlobalInvocationIndex] = constWorld.meshbuffers[gl_GlobalInvocationIndex].draw;


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
	ModelLoD(asset::IAssetManager* assMgr, video::IVideoDriver* driver, float _distance, const char* modelPath) : mesh(nullptr), distance(_distance)
	{
		// load model
		asset::IAssetLoader::SAssetLoadParams lp;
        auto bundle = assMgr->getAsset(modelPath,lp);
		assert(!bundle.isEmpty());
		auto cpumesh = bundle.getContents().begin()[0];
		auto cpumesh_raw = static_cast<asset::ICPUMesh*>(cpumesh.get());

		//const asset::CMTLPipelineMetadata* pipelineMetadata = nullptr;
		//core::map<const asset::ICPURenderpassIndependentPipeline*,core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>> modifiedPipelines;
		for (uint32_t i=0u; i<cpumesh_raw->getMeshBufferCount(); i++)
		{
			auto mb = cpumesh_raw->getMeshBuffer(i);
			// assert that the meshbuffers are indexed
			assert(!!mb->getIndexBufferBinding()->buffer);
			// override vertex shader
			auto* pipeline = mb->getPipeline();
			/*
			auto found = modifiedPipelines.find(pipeline);
			if (found==modifiedPipelines.end())
			{
				// new pipeline to modify, copy the pipeline
				auto pipeline_cp = core::smart_refctd_ptr_static_cast<asset::ICPURenderpassIndependentPipeline>(pipeline->clone(1u));

				// insert a geometry shader into the pipeline
				assert(pipelineMetadata->getLoaderName() == "CGraphicsPipelineLoaderMTL");
				pipelineMetadata = static_cast<const asset::CMTLPipelineMetadata*>(pipeline->getMetadata());
				pipeline_cp->setShaderAtIndex(asset::ICPURenderpassIndependentPipeline::ESSI_GEOMETRY_SHADER_IX,(pipelineMetadata->usesShaderWithUVs() ? geomShaderUV:geomShaderNOUV).get());

				// add descriptor set layout with one that has an SSBO and UBO
				auto* layout = pipeline_cp->getLayout();
				layout->setDescriptorSetLayout(0, core::smart_refctd_ptr(ds0layout));

				// cache the result
				found = modifiedPipelines.emplace(pipeline,std::move(pipeline_cp)).first;
			}
			mb->setPipeline(core::smart_refctd_ptr(found->second));
			*/
		}
		//assert(pipelineMetadata);

		// convert to GPU Mesh
		mesh = driver->getGPUObjectsFromAssets(&cpumesh_raw,&cpumesh_raw+1)->front();
	}

	core::smart_refctd_ptr<video::IGPUMesh> mesh;
	float distance;
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



	const ModelLoD instanceLoDs[] = { ModelLoD(assMgr,driver,8.f,"../../media/cow.obj"),ModelLoD(assMgr,driver,50.f,"../../media/yellowflower.obj") };
	constexpr auto kLoDLevels = sizeof(instanceLoDs)/sizeof(ModelLoD);
	
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
		asset::SBufferRange<video::IGPUBuffer> memoryBlock;
		memoryBlock.buffer = driver->createDeviceLocalGPUBufferOnDedMem((sizeof(ViewInvariantProps)+sizeof(MaterialProps))*kHardwareInstancesTOTAL);
		memoryBlock.size = memoryBlock.buffer->getSize();
		return video::CPropertyPool<core::allocator,ViewInvariantProps,MaterialProps>::create(std::move(memoryBlock));
	}();

	// use the pool
	core::vector<uint32_t> instanceIDs(kHardwareInstancesTOTAL,video::IPropertyPool::invalid_index);
	{
		// create the instances
		{
			core::vector<ViewInvariantProps> propsA(kHardwareInstancesTOTAL);
			core::vector<MaterialProps> propsB(kHardwareInstancesTOTAL);
		
			auto propAIt = propsA.begin();
			auto propBIt = propsB.begin();
			for (size_t z=0; z<kNumHardwareInstancesZ; z++)
			for (size_t y=0; y<kNumHardwareInstancesY; y++)
			for (size_t x=0; x<kNumHardwareInstancesX; x++)
			{
				propAIt->tform.setTranslation(core::vectorSIMDf(x,y,z)*2.f);
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

	auto perViewInstanceData = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(ViewDependentObjectInvariantProps)*kHardwareInstancesTOTAL);
	
	auto drawIndirectData = [&]()
	{
		/*
		uint32_t count;
		uint32_t instanceCount;
		uint32_t firstIndex;
		uint32_t baseVertex;
		uint32_t baseInstance;
		*/
		core::vector<asset::DrawElementsIndirectCommand_t> indirectCmds(instanceLoDs[0].mesh->getMeshBufferCount()+instanceLoDs[1].mesh->getMeshBufferCount());
		// TODO: fill
		{
		}
		return driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(asset::DrawElementsIndirectCommand_t)*indirectCmds.size(),indirectCmds.data());
	}();


	scene::ICameraSceneNode* camera = device->getSceneManager()->addCameraSceneNodeFPS(0, 100.0f, 0.01f);
	camera->setPosition(core::vector3df(-4, 0, 0));
	camera->setTarget(core::vector3df(0, 0, 0));
	camera->setNearValue(0.01f);
	camera->setFarValue(100.0f);

	// render
	while (true)
	{
		// TODO: task for junior, make the instances spin

		// compute shader culling

		// barrier
		
		// draw indirects
	}

	instanceData->freeProperties(instanceIDs.data(),instanceIDs.data()+instanceIDs.size());

#if 0
	//create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		ext::ScreenShot::dirtyCPUStallingScreenshot(driver,device->getAssetManager(), "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}
#endif

	return 0;
}
