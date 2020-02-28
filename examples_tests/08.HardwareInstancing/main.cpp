#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../ext/ScreenShot/ScreenShot.h"
#include "../common/QToQuitEventReceiver.h"

using namespace irr;
using namespace core;
using namespace scene;


#define kNumHardwareInstancesX 10
#define kNumHardwareInstancesY 20
#define kNumHardwareInstancesZ 30

#define kHardwareInstancesTOTAL (kNumHardwareInstancesX*kNumHardwareInstancesY*kNumHardwareInstancesZ)


constexpr float instanceLoDDistances[] = {8.f,50.f};

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


 core::smart_refctd_ptr<asset::IMeshDataFormatDesc<video::IGPUBuffer> > vaoSetupOverride(ISceneManager* smgr, video::IGPUBuffer* instanceDataBuffer, const size_t& dataSizePerInstanceOutput, const asset::IMeshDataFormatDesc<video::IGPUBuffer>* oldVAO, void* userData)
 {
    video::IVideoDriver* driver = smgr->getVideoDriver();
    auto vao = driver->createGPUMeshDataFormatDesc();

    //
    for (size_t k=0; k<asset::EVAI_COUNT; k++)
    {
        asset::E_VERTEX_ATTRIBUTE_ID attrId = (asset::E_VERTEX_ATTRIBUTE_ID)k;
        if (!oldVAO->getMappedBuffer(attrId))
            continue;

        vao->setVertexAttrBuffer(	core::smart_refctd_ptr<video::IGPUBuffer>(const_cast<video::IGPUBuffer*>(oldVAO->getMappedBuffer(attrId))),
									attrId,oldVAO->getAttribFormat(attrId), oldVAO->getMappedBufferStride(attrId),oldVAO->getMappedBufferOffset(attrId),
									oldVAO->getAttribDivisor(attrId));
    }

    // I know what attributes are unused in my mesh and I've set up the shader to use thse as instance data
    vao->setVertexAttrBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(instanceDataBuffer),asset::EVAI_ATTR4,asset::EF_R32G32B32A32_SFLOAT,28*sizeof(float),0,1);
    vao->setVertexAttrBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(instanceDataBuffer),asset::EVAI_ATTR5,asset::EF_R32G32B32A32_SFLOAT,28*sizeof(float),4*sizeof(float),1);
    vao->setVertexAttrBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(instanceDataBuffer),asset::EVAI_ATTR6,asset::EF_R32G32B32A32_SFLOAT,28*sizeof(float),8*sizeof(float),1);
    vao->setVertexAttrBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(instanceDataBuffer),asset::EVAI_ATTR2,asset::EF_R32G32B32A32_SFLOAT,28*sizeof(float),12*sizeof(float),1);

    vao->setVertexAttrBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(instanceDataBuffer),asset::EVAI_ATTR7,asset::EF_R32G32B32_SFLOAT,28*sizeof(float),16*sizeof(float),1);
    vao->setVertexAttrBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(instanceDataBuffer),asset::EVAI_ATTR8,asset::EF_R32G32B32_SFLOAT,28*sizeof(float),19*sizeof(float),1);
    vao->setVertexAttrBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(instanceDataBuffer),asset::EVAI_ATTR9,asset::EF_R32G32B32_SFLOAT,28*sizeof(float),22*sizeof(float),1);
    vao->setVertexAttrBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(instanceDataBuffer),asset::EVAI_ATTR10,asset::EF_R32G32B32_SFLOAT,28*sizeof(float),25*sizeof(float),1);


    if (oldVAO->getIndexBuffer())
        vao->setIndexBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(const_cast<video::IGPUBuffer*>(oldVAO->getIndexBuffer())));

    return vao;
 }


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
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.

	device->getCursorControl()->setVisible(false);

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);



	video::IVideoDriver* driver = device->getVideoDriver();

    SimpleCallBack* cb = new SimpleCallBack();
    video::E_MATERIAL_TYPE newMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
                                                        "","","", //! No Geometry or Tessellation Shaders
                                                        "../mesh.frag",
                                                        3,video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
                                                        cb, //! Our Shader Callback
                                                        0); //! No custom user data
    cb->drop();



	scene::ISceneManager* smgr = device->getSceneManager();
	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
	scene::ICameraSceneNode* camera =
		smgr->addCameraSceneNodeFPS(0,100.0f,0.01f);
	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(100.0f);
    smgr->setActiveCamera(camera);

	IMeshSceneNodeInstanced* node;
	uint32_t instanceIDs[kNumHardwareInstancesZ][kNumHardwareInstancesY][kNumHardwareInstancesX];
	{
		asset::IAssetLoader::SAssetLoadParams lparams;
		video::created_gpu_object_array<asset::ICPUMesh> gpumeshes;
		//! Test Loading of Obj
		{
			core::vector<core::smart_refctd_ptr<asset::ICPUMesh> > cpumeshes;
			cpumeshes.push_back(core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(*device->getAssetManager()->getAsset("../../media/cow.obj", lparams).getContents().first));
			cpumeshes.push_back(core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(*device->getAssetManager()->getAsset("../../media/yellowflower.obj", lparams).getContents().first));

			gpumeshes = std::move(driver->getGPUObjectsFromAssets(reinterpret_cast<asset::ICPUMesh**>(cpumeshes.data()), reinterpret_cast<asset::ICPUMesh**>(cpumeshes.data())+2));
		}
		for (auto j=0u; j<gpumeshes->size(); j++)
		{
			auto gpumesh = gpumeshes->operator[](j);
			for (size_t i=0; i<gpumesh->getMeshBufferCount(); i++)
				gpumesh->getMeshBuffer(i)->getMaterial().MaterialType = (video::E_MATERIAL_TYPE)newMaterialType;
		}


		video::SGPUMaterial cullingXFormFeedbackShader;
		const char* xformFeedbackOutputs[] =
		{
			"outLoD0_worldViewProjMatCol0",
			"outLoD0_worldViewProjMatCol1",
			"outLoD0_worldViewProjMatCol2",
			"outLoD0_worldViewProjMatCol3",
			"outLoD0_worldViewMatCol0",
			"outLoD0_worldViewMatCol1",
			"outLoD0_worldViewMatCol2",
			"outLoD0_worldViewMatCol3",
			"gl_NextBuffer",
			"outLoD1_worldViewProjMatCol0",
			"outLoD1_worldViewProjMatCol1",
			"outLoD1_worldViewProjMatCol2",
			"outLoD1_worldViewProjMatCol3",
			"outLoD1_worldViewMatCol0",
			"outLoD1_worldViewMatCol1",
			"outLoD1_worldViewMatCol2",
			"outLoD1_worldViewMatCol3"
		};
		cullingXFormFeedbackShader.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../culling.vert","","","../culling.geom","",3,video::EMT_SOLID,cb,xformFeedbackOutputs,17);
		cullingXFormFeedbackShader.RasterizerDiscard = true;

		node = smgr->addMeshSceneNodeInstanced(smgr->getRootSceneNode());
        node->setBBoxUpdateEnabled();
        node->setAutomaticCulling(scene::EAC_FRUSTUM_BOX);
        {
            core::vector<scene::IMeshSceneNodeInstanced::MeshLoD> LevelsOfDetail;
            LevelsOfDetail.resize(2);
            LevelsOfDetail[0].mesh = gpumeshes->operator[](0u).get();
            LevelsOfDetail[0].lodDistance = instanceLoDDistances[0];
            LevelsOfDetail[1].mesh = gpumeshes->operator[](1u).get();
            LevelsOfDetail[1].lodDistance = instanceLoDDistances[1];

            bool success = node->setLoDMeshes(LevelsOfDetail,28*sizeof(float),cullingXFormFeedbackShader,vaoSetupOverride,2,NULL,0);
            assert(success);
            cb->instanceLoDInvariantBBox = node->getLoDInvariantBBox();
        }

        //! Special Juice for INSTANCING
        for (size_t z=0; z<kNumHardwareInstancesZ; z++)
        for (size_t y=0; y<kNumHardwareInstancesY; y++)
        for (size_t x=0; x<kNumHardwareInstancesX; x++)
        {
            core::matrix3x4SIMD mat;
            mat.setTranslation(core::vectorSIMDf(x,y,z)*2.f);
            instanceIDs[z][y][x] = node->addInstance(mat);
        }

        srand(6945);

        for (size_t i=0; i<600; i++)
        {
            uint32_t x = rand()%kNumHardwareInstancesX;
            uint32_t y = rand()%kNumHardwareInstancesY;
            uint32_t z = rand()%kNumHardwareInstancesZ;
            uint32_t& instanceID = instanceIDs[z][y][x];
            if (instanceID==0xdeadbeefu)
                continue;

            node->removeInstance(instanceID);
            instanceID = 0xdeadbeefu;
        }
	}
	

    size_t removeCount = 0u;
    uint32_t instancesToRemove[kHardwareInstancesTOTAL];
    for (size_t z=0; z<kNumHardwareInstancesZ; z++)
    for (size_t y=0; y<kNumHardwareInstancesY; y++)
    for (size_t x=0; x<kNumHardwareInstancesX; x++)
    {
        auto instanceID = instanceIDs[z][y][x];
        if (instanceID==0xdeadbeefu)
            continue;

        instancesToRemove[removeCount++] = instanceID;
    }

	uint64_t lastFPSTime = 0;
	while(device->run() && receiver.keepOpen() )
	{
		driver->beginScene(true, true, video::SColor(255,0,0,255) );

        //! This animates (moves) the camera and sets the transforms
        //! Also draws the meshbuffer
        smgr->drawAll();

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Builtin Nodes Demo - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str());
			lastFPSTime = time;
		}
	}

    node->removeInstances(removeCount,instancesToRemove);
    node->remove();


	//create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		ext::ScreenShot::dirtyCPUStallingScreenshot(driver,device->getAssetManager(), "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}


	device->drop();

	return 0;
}
