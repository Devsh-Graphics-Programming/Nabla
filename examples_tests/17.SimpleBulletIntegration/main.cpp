#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include "../source/Irrlicht/COpenGLExtensionHandler.h"
#include <btBulletDynamicsCommon.h>
#include "BulletCollision/NarrowPhaseCollision/btRaycastCallback.h"

#include "../../ext/Bullet/BulletUtility.h"
#include "../../ext/Bullet/CPhysicsWorld.h"

#include "../../ext/Bullet/CInstancedMotionState.h"
#include "../../ext/Bullet/CDebugRender.h"

using namespace irr;
using namespace core;
using namespace scene;





#define kNumHardwareInstancesX 10
#define kNumHardwareInstancesY 20
#define kNumHardwareInstancesZ 30

#define kHardwareInstancesTOTAL (kNumHardwareInstancesX*kNumHardwareInstancesY*kNumHardwareInstancesZ)


const float instanceLoDDistances[] = {8.f,50.f};

bool quit = false;

void handleRaycast(scene::ICameraSceneNode *cam, ext::Bullet3::CPhysicsWorld *physicsWorld) {
    //TODO - find way to extract rigidybody from closeRay (?)

    using namespace ext::Bullet3;

    core::vectorSIMDf to;
    to.set(cam->getAbsolutePosition());

    core::vectorSIMDf from;
    from.set(cam->getTarget());
    
    btCollisionWorld::ClosestRayResultCallback closeRay(tobtVec3(to), tobtVec3(from));
    closeRay.m_flags = btTriangleRaycastCallback::kF_UseSubSimplexConvexCastRaytest;
    
    physicsWorld->getWorld()->rayTest(tobtVec3(from), tobtVec3(to), closeRay);

    if (closeRay.hasHit()) {
        physicsWorld->getWorld()->getDebugDrawer()->drawSphere(
            tobtVec3(from).lerp(tobtVec3(to), closeRay.m_closestHitFraction), 2, btVector3(1, 0, 0)
        );

      
    };

}


//!Disptaches Raycast when R is pressed
class MyEventReceiver : public IEventReceiver
{
public:

	MyEventReceiver(scene::ICameraSceneNode *cam, ext::Bullet3::CPhysicsWorld *physicsWorld):
        physicsWorld(physicsWorld),
        cam(cam)
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
            case irr::KEY_KEY_R: //Bullet raycast
                handleRaycast(cam, physicsWorld);
                return true;
            

            default:
                break;
            }
        }

       


		return false;
	}

private:
    scene::ICameraSceneNode *cam;
    ext::Bullet3::CPhysicsWorld *physicsWorld;

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


 asset::IMeshDataFormatDesc<video::IGPUBuffer>* vaoSetupOverride(ISceneManager* smgr, video::IGPUBuffer* instanceDataBuffer, const size_t& dataSizePerInstanceOutput, const asset::IMeshDataFormatDesc<video::IGPUBuffer>* oldVAO, void* userData)
 {
    video::IVideoDriver* driver = smgr->getVideoDriver();
    asset::IMeshDataFormatDesc<video::IGPUBuffer>* vao = driver->createGPUMeshDataFormatDesc();

    //
    for (size_t k=0; k<asset::EVAI_COUNT; k++)
    {
        asset::E_VERTEX_ATTRIBUTE_ID attrId = (asset::E_VERTEX_ATTRIBUTE_ID)k;
        if (!oldVAO->getMappedBuffer(attrId))
            continue;

        vao->setVertexAttrBuffer(const_cast<video::IGPUBuffer*>(oldVAO->getMappedBuffer(attrId)),attrId,oldVAO->getAttribFormat(attrId),
                                 oldVAO->getMappedBufferStride(attrId),oldVAO->getMappedBufferOffset(attrId),oldVAO->getAttribDivisor(attrId));
    }

    vao->setVertexAttrBuffer(instanceDataBuffer,asset::EVAI_ATTR1,asset::EF_R8G8B8A8_UNORM,29*sizeof(float),28*sizeof(float),1);

    // I know what attributes are unused in my mesh and I've set up the shader to use thse as instance data
    vao->setVertexAttrBuffer(instanceDataBuffer,asset::EVAI_ATTR4,asset::EF_R32G32B32A32_SFLOAT,29*sizeof(float),0,1);
    vao->setVertexAttrBuffer(instanceDataBuffer,asset::EVAI_ATTR5,asset::EF_R32G32B32A32_SFLOAT,29*sizeof(float),4*sizeof(float),1);
    vao->setVertexAttrBuffer(instanceDataBuffer,asset::EVAI_ATTR6,asset::EF_R32G32B32A32_SFLOAT,29*sizeof(float),8*sizeof(float),1);
    vao->setVertexAttrBuffer(instanceDataBuffer,asset::EVAI_ATTR2,asset::EF_R32G32B32A32_SFLOAT,29*sizeof(float),12*sizeof(float),1);

    vao->setVertexAttrBuffer(instanceDataBuffer,asset::EVAI_ATTR7,asset::EF_R32G32B32_SFLOAT,29*sizeof(float),16*sizeof(float),1);
    vao->setVertexAttrBuffer(instanceDataBuffer,asset::EVAI_ATTR8,asset::EF_R32G32B32_SFLOAT,29*sizeof(float),19*sizeof(float),1);
    vao->setVertexAttrBuffer(instanceDataBuffer,asset::EVAI_ATTR9,asset::EF_R32G32B32_SFLOAT,29*sizeof(float),22*sizeof(float),1);
    vao->setVertexAttrBuffer(instanceDataBuffer,asset::EVAI_ATTR10,asset::EF_R32G32B32_SFLOAT,29*sizeof(float),25*sizeof(float),1);


    if (oldVAO->getIndexBuffer())
        vao->setIndexBuffer(const_cast<video::IGPUBuffer*>(oldVAO->getIndexBuffer()));

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
	
    
    // ! - INITIALIZE BULLET WORLD + FLAT PLANE FOR TESTING
//------------------------------------------------------------------

    ext::Bullet3::CPhysicsWorld *world = _IRR_NEW(irr::ext::Bullet3::CPhysicsWorld);
    world->getWorld()->setGravity(btVector3(0, -5, 0));


    core::matrix3x4SIMD baseplateMat;
    baseplateMat.setTranslation(core::vectorSIMDf(0.0, -1.0, 0.0));

    ext::Bullet3::CPhysicsWorld::RigidBodyData data2;
    data2.mass = 0.0f;
    data2.shape = world->createbtObject<btBoxShape>(btVector3(300, 1, 300));
    data2.trans = baseplateMat;

    btRigidBody *body2 = world->createRigidBody(data2);
    world->bindRigidBody(body2);

    //------------------------------------------------------------------


    
    MyEventReceiver receiver(camera, world);
	device->setEventReceiver(&receiver);

    
   

	//!
    asset::ICPUMesh* cpumesh = device->getAssetManager().getGeometryCreator()->createCubeMesh(core::vector3df(1.f, 1.f, 1.f));
    video::IGPUMesh* gpumesh = driver->getGPUObjectsFromAssets(&cpumesh, (&cpumesh)+1).front();
    for (size_t i=0; i<gpumesh->getMeshBufferCount(); i++)
        gpumesh->getMeshBuffer(i)->getMaterial().MaterialType = (video::E_MATERIAL_TYPE)newMaterialType;


    video::SGPUMaterial cullingXFormFeedbackShader;
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


    const uint32_t towerHeight = 100;
    const uint32_t towerWidth = 2;





    ext::Bullet3::CDebugRender *debugDraw = world->createbtObject<ext::Bullet3::CDebugRender>(driver);
    world->getWorld()->setDebugDrawer(debugDraw);

    

    btRigidBody **bodies = _IRR_NEW_ARRAY(btRigidBody*, towerHeight * towerWidth);



    irr::ext::Bullet3::CPhysicsWorld::RigidBodyData data3;
    data3.mass = 2.0f;
    data3.shape = world->createbtObject<btBoxShape>(btVector3(0.5, 0.5, 0.5));

   
    btVector3 inertia;
    data3.shape->calculateLocalInertia(data3.mass, inertia);

    data3.inertia = ext::Bullet3::frombtVec3(inertia);



    //! Special Juice for INSTANCING
    uint32_t instances[towerHeight*towerWidth];
    for (size_t y=0; y<towerHeight; y++)
    for (size_t z=0; z<towerWidth; z++)
    {
        core::matrix4x3 mat;
        mat.setTranslation(core::vector3df(z, y, 1));

        uint8_t color[4];
        color[0] = rand() % 256;
        color[1] = rand() % 256;
        color[2] = rand() % 256;
        color[3] = 255u;
        instances[y*towerWidth + z] = node->addInstance(mat, color);

        core::vectorSIMDf pos;
        pos.set(node->getInstanceTransform(instances[y*towerWidth + z]).getTranslation());
        core::matrix3x4SIMD instancedMat;
        instancedMat.setTranslation(pos);



        data3.trans = instancedMat;

        bodies[y*towerWidth + z] = world->createRigidBody(data3);

        bodies[y*towerWidth + z]->setUserPointer((uint32_t*)(y*towerWidth + z));

        world->bindRigidBody<irr::ext::Bullet3::CInstancedMotionState>(bodies[y*towerWidth + z], node, instances[y*towerWidth + z]);


    }

    uint64_t timeDiff = 0;
	while(device->run()&&(!quit))
	{

        uint64_t now = device->getTimer()->getRealTime();


        

		driver->beginScene(true, true, video::SColor(255,0,0,255) );

        //! This animates (moves) the camera and sets the transforms
        //! Also draws the meshbuffer
 
        
        smgr->drawAll();

        world->getWorld()->debugDrawWorld();
        handleRaycast(camera, world);
        debugDraw->draw();
       
		driver->endScene();

        world->getWorld()->stepSimulation(timeDiff);
       

        

        uint64_t time = device->getTimer()->getRealTime();
        timeDiff = (time - now);

		// display frames per second in window title
        if (time - lastFPSTime > 250)
        {
            std::wostringstream str;
            str << L"Builtin Nodes Demo - Irrlicht Engine [" << driver->getName() << "] FPS:" << 1000 / timeDiff << " PrimitivesDrawn:" << driver->getPrimitiveCountDrawn();

            device->setWindowCaption(str.str());
            lastFPSTime = time;
        }
	}




    world->unbindRigidBody(body2, false);
    world->deleteRigidBody(body2);
    world->deletebtObject(data2.shape);


    for (size_t i = 0; i < towerHeight * towerWidth; ++i) {
        world->unbindRigidBody(bodies[i]);
        world->deleteRigidBody(bodies[i]);
    }

    _IRR_DELETE_ARRAY(bodies, towerHeight * towerWidth);
    world->deletebtObject(data3.shape);

   
    node->removeInstances(towerHeight*towerWidth,instances);
    node->remove();

    gpumesh->drop();

    world->drop();

    //create a screenshot
	video::IImage* screenshot = driver->createImage(asset::EF_B8G8R8A8_UNORM,params.WindowSize);
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
    asset::CImageData* img = new asset::CImageData(screenshot);
    asset::IAssetWriter::SAssetWriteParams wparams(img);
    device->getAssetManager().writeAsset("screenshot.png", wparams);
    img->drop();
    screenshot->drop();


	device->drop();

	return 0;
}
