#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include "../source/Irrlicht/COpenGLExtensionHandler.h"

#include <btBulletDynamicsCommon.h>
#include "../ext/Bullet3/BulletUtility.h"
#include "../ext/Bullet3/CPhysicsWorld.h"
#include "../ext/Bullet3/CDefaultMotionState.h"

using namespace irr;
using namespace core;
using namespace scene;


#define kNumHardwareInstancesX 10
#define kNumHardwareInstancesY 20
#define kNumHardwareInstancesZ 30

#define kHardwareInstancesTOTAL (kNumHardwareInstancesX*kNumHardwareInstancesY*kNumHardwareInstancesZ)


const float instanceLoDDistances[] = {8.f,50.f};

bool quit = false;


//!Same As Last Example
class MyEventReceiver : public IEventReceiver
{
public:

	MyEventReceiver()
	{
	}

	bool OnEvent(const SEvent& event)
	{
        if (event.EventType == irr::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
        {
            switch (event.KeyInput.Key)
            {
            case irr::KEY_KEY_Q: // switch wire frame mode
                quit = true;
                return true;
            default:
                break;
            }
        }

		return false;
	}

private:
};

const char* uniformNames[] =
{
    "ProjViewWorldMat",
    "WorldMat",
    "ViewWorldMat",
    "eyePos",
    "LoDInvariantMinEdge",
    "LoDInvariantBBoxCenter",
    "LoDInvariantMaxEdge"
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

    virtual void OnSetMaterial(video::IMaterialRendererServices* services, const video::SMaterial& material, const video::SMaterial& lastMaterial)
    {
        currentMat = material.MaterialType;
        currentLodPass = material.MaterialTypeParam;
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        if (uniformLocation[currentMat][EU_PROJ_VIEW_WORLD_MAT]>=0)
            services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(),uniformLocation[currentMat][EU_PROJ_VIEW_WORLD_MAT],uniformType[currentMat][EU_PROJ_VIEW_WORLD_MAT],1);
        if (uniformLocation[currentMat][EU_VIEW_WORLD_MAT]>=0)
            services->setShaderConstant(services->getVideoDriver()->getTransform(video::E4X3TS_WORLD_VIEW).pointer(),uniformLocation[currentMat][EU_VIEW_WORLD_MAT],uniformType[currentMat][EU_VIEW_WORLD_MAT],1);
        if (uniformLocation[currentMat][EU_WORLD_MAT]>=0)
            services->setShaderConstant(services->getVideoDriver()->getTransform(video::E4X3TS_WORLD).pointer(),uniformLocation[currentMat][EU_WORLD_MAT],uniformType[currentMat][EU_WORLD_MAT],1);

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
    }

    virtual void OnUnsetMaterial() {}
};


 scene::IMeshDataFormatDesc<video::IGPUBuffer>* vaoSetupOverride(ISceneManager* smgr, video::IGPUBuffer* instanceDataBuffer, const size_t& dataSizePerInstanceOutput, const scene::IMeshDataFormatDesc<video::IGPUBuffer>* oldVAO, void* userData)
 {
    video::IVideoDriver* driver = smgr->getVideoDriver();
    scene::IMeshDataFormatDesc<video::IGPUBuffer>* vao = driver->createGPUMeshDataFormatDesc();

    //
    for (size_t k=0; k<EVAI_COUNT; k++)
    {
        E_VERTEX_ATTRIBUTE_ID attrId = (E_VERTEX_ATTRIBUTE_ID)k;
        if (!oldVAO->getMappedBuffer(attrId))
            continue;

        vao->mapVertexAttrBuffer(const_cast<video::IGPUBuffer*>(oldVAO->getMappedBuffer(attrId)),attrId,oldVAO->getAttribComponentCount(attrId),oldVAO->getAttribType(attrId),
                                 oldVAO->getMappedBufferStride(attrId),oldVAO->getMappedBufferOffset(attrId),oldVAO->getAttribDivisor(attrId));
    }

    vao->mapVertexAttrBuffer(instanceDataBuffer,EVAI_ATTR1,ECPA_FOUR,ECT_NORMALIZED_UNSIGNED_BYTE,29*sizeof(float),28*sizeof(float),1);

    // I know what attributes are unused in my mesh and I've set up the shader to use thse as instance data
    vao->mapVertexAttrBuffer(instanceDataBuffer,EVAI_ATTR4,ECPA_FOUR,ECT_FLOAT,29*sizeof(float),0,1);
    vao->mapVertexAttrBuffer(instanceDataBuffer,EVAI_ATTR5,ECPA_FOUR,ECT_FLOAT,29*sizeof(float),4*sizeof(float),1);
    vao->mapVertexAttrBuffer(instanceDataBuffer,EVAI_ATTR6,ECPA_FOUR,ECT_FLOAT,29*sizeof(float),8*sizeof(float),1);
    vao->mapVertexAttrBuffer(instanceDataBuffer,EVAI_ATTR2,ECPA_FOUR,ECT_FLOAT,29*sizeof(float),12*sizeof(float),1);

    vao->mapVertexAttrBuffer(instanceDataBuffer,EVAI_ATTR7,ECPA_THREE,ECT_FLOAT,29*sizeof(float),16*sizeof(float),1);
    vao->mapVertexAttrBuffer(instanceDataBuffer,EVAI_ATTR8,ECPA_THREE,ECT_FLOAT,29*sizeof(float),19*sizeof(float),1);
    vao->mapVertexAttrBuffer(instanceDataBuffer,EVAI_ATTR9,ECPA_THREE,ECT_FLOAT,29*sizeof(float),22*sizeof(float),1);
    vao->mapVertexAttrBuffer(instanceDataBuffer,EVAI_ATTR10,ECPA_THREE,ECT_FLOAT,29*sizeof(float),25*sizeof(float),1);


    if (oldVAO->getIndexBuffer())
        vao->mapIndexBuffer(const_cast<video::IGPUBuffer*>(oldVAO->getIndexBuffer()));

    return vao;
 }




int main()
{


    btTransform m;
    m.setOrigin(btVector3(0, 100, 0));

    irr::ext::Bullet3::CPhysicsWorld *world = _IRR_NEW(irr::ext::Bullet3::CPhysicsWorld);


    irr::ext::Bullet3::CPhysicsWorld::RigidBodyData data;
    data.mass = 1.0f;
    data.shape = _IRR_NEW(btSphereShape, 5.0f);



    btRigidBody *body = world->createRigidBody(data);
    world->bindRigidBody<irr::ext::Bullet3::CDefaultMotionState>(body);

    irr::ext::Bullet3::CPhysicsWorld::RigidBodyData data2;
    data2.mass = 0.0f;
    data2.shape = _IRR_NEW(btBoxShape, btVector3(100, 1, 100)); 

    btRigidBody *body2 = world->createRigidBody(data2);
    world->bindRigidBody<irr::ext::Bullet3::CDefaultMotionState>(body2);

    world->unbindRigidBody(body2);


    body->setWorldTransform(m);
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
	device->getCursorControl()->setVisible(false);
	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);

	//!
    scene::IGPUMesh* gpumesh = smgr->getGeometryCreator()->createCubeMeshGPU(driver,core::vector3df(5.f,1.f,1.f));
    for (size_t i=0; i<gpumesh->getMeshBufferCount(); i++)
        gpumesh->getMeshBuffer(i)->getMaterial().MaterialType = (video::E_MATERIAL_TYPE)newMaterialType;


    video::SMaterial cullingXFormFeedbackShader;
    const char* xformFeedbackOutputs[] =
    {
        "outLoD0_worldViewProjMatCol0",
        "outLoD0_worldViewProjMatCol1",
        "outLoD0_worldViewProjMatCol2",
        "outLoD0_worldViewProjMatCol3",
        "outLoD0_worldMatCol0",
        "outLoD0_worldMatCol1",
        "outLoD0_worldMatCol2",
        "outLoD0_worldMatCol3",
        "outLoD0_instanceColor"
    };
    cullingXFormFeedbackShader.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../culling.vert","","","../culling.geom","",3,video::EMT_SOLID,cb,xformFeedbackOutputs,9);
    cullingXFormFeedbackShader.RasterizerDiscard = true;



    //!
	uint64_t lastFPSTime = 0;
    
    IMeshSceneNodeInstanced* node = smgr->addMeshSceneNodeInstanced(smgr->getRootSceneNode());
    node->setBBoxUpdateEnabled();
    node->setAutomaticCulling(scene::EAC_FRUSTUM_BOX);
    ///node->setAutomaticCulling(scene::EAC_OFF);
    {
        core::vector<scene::IMeshSceneNodeInstanced::MeshLoD> LevelsOfDetail;
        LevelsOfDetail.resize(1);
        LevelsOfDetail[0].mesh = gpumesh;
        LevelsOfDetail[0].lodDistance = camera->getFarValue();

        bool success = node->setLoDMeshes(LevelsOfDetail,29*sizeof(float),cullingXFormFeedbackShader,vaoSetupOverride,1,NULL,4);
        assert(success);
        cb->instanceLoDInvariantBBox = node->getLoDInvariantBBox();
    }

    srand(6945);

    const uint32_t towerHeight = 20;
    const uint32_t towerWidth = 5;
    //! Special Juice for INSTANCING
    uint32_t instances[towerHeight*towerWidth];
    for (size_t y=0; y<towerHeight; y++)
    for (size_t z=0; z<towerWidth; z++)
    {
        core::matrix4x3 mat;
        if (y&0x1u)
        {
            mat.setTranslation(core::vector3df(1.5f,y,z));
        }
        else
        {
            core::vectorSIMDf eulerXYZ;
            core::quaternion(0.f,1.f,0.f,1.f).toEuler(eulerXYZ);
            mat.setRotationRadians(eulerXYZ.getAsVector3df());
            mat.setTranslation(core::vector3df(z,y,1.5f));
        }
        uint8_t color[4];
        color[0] = rand()%256;
        color[1] = rand()%256;
        color[2] = rand()%256;
        color[3] = 255u;
        instances[y*towerWidth+z] = node->addInstance(mat,color);
        
    }

	while(device->run()&&(!quit))
	{
		driver->beginScene(true, true, video::SColor(255,0,0,255) );

        //! This animates (moves) the camera and sets the transforms
        //! Also draws the meshbuffer
        smgr->drawAll();

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();

        world->getWorld()->stepSimulation(1);

		if (time-lastFPSTime > 1000)
		{
            for (size_t y = 0; y < towerHeight; y++) {
                for (size_t z = 0; z < towerWidth; z++) {
                    uint32_t instanceID = instances[y*towerWidth + z];

                    matrix4x3 currentMat = node->getInstanceTransform(instanceID);
                    currentMat.setTranslation(currentMat.getTranslation() + core::vector3df(0.0, 1.0, 0.0));
             
                    node->setInstanceTransform(instanceID, currentMat);

                }
            }
            

			std::wostringstream str;
			str << L"Builtin Nodes Demo - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str());
			lastFPSTime = time;
		}
	}

    world->unbindRigidBody(body);

    _IRR_DELETE(data.shape);
    _IRR_DELETE(data2.shape);

    node->removeInstances(towerHeight*towerWidth,instances);
    node->remove();

    gpumesh->drop();

    //create a screenshot
	video::IImage* screenshot = driver->createImage(video::ECF_A8R8G8B8,params.WindowSize);
    glReadPixels(0,0, params.WindowSize.Width,params.WindowSize.Height, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, screenshot->getData());
    {
        // images are horizontally flipped, so we have to fix that here.
        uint8_t* pixels = (uint8_t*)screenshot->getData();

        const int32_t pitch=screenshot->getPitch();
        uint8_t* p2 = pixels + (params.WindowSize.Height - 1) * pitch;
        uint8_t* tmpBuffer = new uint8_t[pitch];
        for (uint32_t i=0; i < params.WindowSize.Height; i += 2)
        {
            memcpy(tmpBuffer, pixels, pitch);
            memcpy(pixels, p2, pitch);
            memcpy(p2, tmpBuffer, pitch);
            pixels += pitch;
            p2 -= pitch;
        }
        delete [] tmpBuffer;
    }
	driver->writeImageToFile(screenshot,"./screenshot.png");
	screenshot->drop();


	device->drop();

	return 0;
}
