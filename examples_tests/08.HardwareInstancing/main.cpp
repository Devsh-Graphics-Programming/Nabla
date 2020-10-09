#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../ext/ScreenShot/ScreenShot.h"
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


struct MaterialMetadataProps
{
	uint32_t someData;
};
struct ViewInvariantProps
{
	core::matrix3x4SIMD tform;
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
			uint32_t objectUUID;
			float normalMatrixRow2[3];
			uint32_t padding;
		};
		core::matrix3x4SIMD normalMatrix;
	};
};
// perView.data[gl_DrawID][gl_InstanceIndex]

struct ModelLoD
{
	ModelLoD(asset::IAssetManager* assMgr, video::IVideoDriver* driver, float _distance, const char* modelPath) : mb(nullptr), distance(_distance)
	{
		// load model
		// override vertex shader
		// assert that the mesh is indexed
		// convert to GPU Mesh
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


	constexpr auto kNumHardwareInstancesX = 10;
	constexpr auto kNumHardwareInstancesY = 20;
	constexpr auto kNumHardwareInstancesZ = 30;

	constexpr auto kHardwareInstancesTOTAL = kNumHardwareInstancesX * kNumHardwareInstancesY * kNumHardwareInstancesZ;

	const ModelLoD instanceLoDs[] = { ModelLoD(assMgr,driver,8.f,"../../media/cow.obj"),ModelLoD(assMgr,driver,50.f,"../../media/yellowflower.obj") };
	constexpr auto kLoDLevels = sizeof(instanceLoDs) / sizeof(ModelLoD);

	// create the pool
	auto instanceData = [&]()
	{
		asset::SBufferRange<video::IGPUBuffer> memoryBlock;
		memoryBlock.buffer = driver->createDeviceLocalGPUBufferOnDedMem((sizeof(ViewInvariantProps)+sizeof(MaterialMetadataProps))*kHardwareInstancesTOTAL);
		memoryBlock.size = memoryBlock.buffer->getSize();
		return video::CPropertyPool<core::allocator,ViewInvariantProps,MaterialMetadataProps>::create(std::move(memoryBlock));
	}();

	// use the pool
	core::vector<uint32_t> instanceIDs(kHardwareInstancesTOTAL);
	{
		// create the instances
		{
			core::vector<ViewInvariantProps> propsA(kHardwareInstancesTOTAL);
			core::vector<MaterialMetadataProps> propsB(kHardwareInstancesTOTAL);
		
			auto propAIt = propsA.begin();
			auto propBIt = propsB.begin();
			for (size_t z=0; z<kNumHardwareInstancesZ; z++)
			for (size_t y=0; y<kNumHardwareInstancesY; y++)
			for (size_t x=0; x<kNumHardwareInstancesX; x++)
			{
				propAIt->tform.setTranslation(core::vectorSIMDf(x,y,z)*2.f);
				propAIt++,propBIt++;
			}

			video::IPropertyPool* pool = instanceData.get();
			uint32_t* indicesBegin = instanceIDs.data();
			const void* data[] = {propsA.data(),propsB.data()};
			driver->getDefaultPropertyPoolHandler()->addProperties(&pool,&pool+1u,&indicesBegin,&indicesBegin+1u,reinterpret_cast<const void* const* const*>(&data));
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
		return driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(asset::DrawElementsIndirectCommand_t)*indirectCmds.size(),indirectCmds.data());
	}();

	core::vectorSIMDf lodInvariantAABB[2];
	{
		core::aabbox3df bbox = instanceLoDs[0].mesh->getBoundingBox();
		for (auto i=1u; i<kLoDLevels; i++)
			bbox.addInternalBox(instanceLoDs[i].mesh->getBoundingBox());
	}


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
