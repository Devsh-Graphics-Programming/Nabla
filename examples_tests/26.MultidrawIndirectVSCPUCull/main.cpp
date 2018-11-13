#include <random>

#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include "../source/Irrlicht/COpenGLBuffer.h"
#include "../source/Irrlicht/COpenGLExtensionHandler.h"


using namespace irr;
using namespace core;

bool quit = false;

bool doCulling = false;
bool useDrawIndirect = false;

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
            case irr::KEY_KEY_Q: // so we can quit
                quit = true;
                return true;
            case irr::KEY_KEY_C: // so we can quit
                ///doCulling = !doCulling; // Not enabled/necessary yet
                return true;
            case irr::KEY_SPACE: // toggle between gpu and cpu cull
                useDrawIndirect = !useDrawIndirect;
                return true;
            default:
                break;
            }
        }

		return false;
	}

private:
};


class SimpleCallBack : public video::IShaderConstantSetCallBack
{
    int32_t drawIDUniformLocation;
    video::E_SHADER_CONSTANT_TYPE drawIDUniformType;
public:
    SimpleCallBack() : drawIDUniformLocation(-1) {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
    {
        for (size_t i=0; i<constants.size(); i++)
        {
            if (constants[i].name=="drawID")
            {
                drawIDUniformLocation = constants[i].location;
                drawIDUniformType = constants[i].type;
            }
        }
    }

    virtual void OnSetMaterial(video::IMaterialRendererServices* services, const video::SMaterial& material, const video::SMaterial& lastMaterial)
    {
        if (drawIDUniformLocation!=-1)
        {
            services->setShaderConstant(&material.userData,drawIDUniformLocation,drawIDUniformType,1);
        }
    }

    virtual void OnUnsetMaterial() {}

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData) {}
};

//! read up on std430 packing rules to understand the padding
struct ObjectData_t
{
    core::matrix4 modelViewProjMatrix;
    float normalMat[9];
    float padding[3];
};

//
struct DrawElementsIndirectCommand
{
    uint32_t count;
    uint32_t instanceCount;
    uint32_t firstIndex;
    uint32_t baseVertex;
    uint32_t baseInstance;
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
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.


	video::IVideoDriver* driver = device->getVideoDriver();

    SimpleCallBack* cb = new SimpleCallBack();
    video::E_MATERIAL_TYPE cpuCullMaterial = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../meshCPU.vert",
                                                        "","","", //! No Geometry or Tessellation Shaders
                                                        "../mesh.frag",
                                                        3,video::EMT_SOLID,
                                                        cb);
    cb->drop();

    video::E_MATERIAL_TYPE gpuCullMaterial = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../meshGPU.vert",
                                                        "","","", //! No Geometry or Tessellation Shaders
                                                        "../mesh.frag",
                                                        3,video::EMT_SOLID);


	scene::ISceneManager* smgr = device->getSceneManager();

    #define kInstanceCount 4096
    #define kTotalTriangleLimit (64*1024*1024)

	scene::ICameraSceneNode* camera =
		smgr->addCameraSceneNodeFPS(0,100.0f,0.01f);
	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(250.0f);
    smgr->setActiveCamera(camera);
	device->getCursorControl()->setVisible(false);

	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);

	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);

    //! read cache results -- speeds up mesh generation
	{
        io::IReadFile* cacheFile = device->getFileSystem()->createAndOpenFile("./normalCache101010.sse");
        if (cacheFile)
        {
            scene::normalCacheFor2_10_10_10Quant.resize(cacheFile->getSize()/sizeof(scene::QuantizationCacheEntry2_10_10_10));
            cacheFile->read(scene::normalCacheFor2_10_10_10Quant.data(),cacheFile->getSize());
            cacheFile->drop();

            //make sure its still ok
            std::sort(scene::normalCacheFor2_10_10_10Quant.begin(),scene::normalCacheFor2_10_10_10Quant.end());
        }
	}

	core::matrix4x3 instanceXForm[kInstanceCount];
	scene::IGPUMeshBuffer* mbuff[kInstanceCount] = {NULL};
	video::IGPUBuffer* indirectDrawBuffer = NULL;

    scene::IGPUMeshDataFormatDesc* vaospec = driver->createGPUMeshDataFormatDesc();
	{
        scene::ICPUMesh* cpumesh[kInstanceCount];

        size_t vertexSize = 0;
        std::vector<uint8_t> vertexData;
        std::vector<uint32_t> indexData;

        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<uint32_t> dist(16, 4*1024);
        for (size_t i=0; i<kInstanceCount; i++)
        {
            float poly = sqrtf(dist(mt))+0.5f;
            cpumesh[i] = smgr->getGeometryCreator()->createSphereMeshCPU(2.f,poly,poly);

            //some assumptions about generated mesh
            assert(cpumesh[i]->getMeshBuffer(0)->getPrimitiveType()==scene::EPT_TRIANGLES);
            assert(cpumesh[i]->getMeshBuffer(0)->getIndexType()==scene::EIT_32BIT);
            assert(cpumesh[i]->getMeshBuffer(0)->getBaseVertex()==0);
            assert(cpumesh[i]->getMeshBuffer(0)->getIndexBufferOffset()==0);

            scene::ICPUMeshBuffer* mbuf = cpumesh[i]->getMeshBuffer(0);
            scene::IMeshDataFormatDesc<core::ICPUBuffer>* format = mbuf->getMeshDataAndFormat();
            assert(format->getMappedBuffer(scene::EVAI_ATTR0)!=NULL); //helpful assumption

            const core::ICPUBuffer* databuf = NULL;
            for (size_t j=0; j<scene::EVAI_COUNT; j++)
            {
                scene::E_VERTEX_ATTRIBUTE_ID attrID = static_cast<scene::E_VERTEX_ATTRIBUTE_ID>(j);
                if (!format->getMappedBuffer(attrID)) // don't consider un-used attribute slots
                    continue;

                if (databuf) // if first found
                {
                    assert(databuf==format->getMappedBuffer(attrID)); // all sphere vertex data will be packed in the same buffer
                }
                else
                    databuf = format->getMappedBuffer(attrID);

                if (vertexSize) //if set
                {
                    assert(format->getMappedBufferStride(attrID)==vertexSize); //all data in the same buffer == same vertex stride for all attributes
                }
                else
                    vertexSize = format->getMappedBufferStride(attrID);
            }

            assert(databuf->getSize()%vertexSize==0); //otherwise offsets by baseVertex will not work

            vertexData.insert(vertexData.end(),reinterpret_cast<const uint8_t*>(databuf->getPointer()),reinterpret_cast<const uint8_t*>(databuf->getPointer())+databuf->getSize());
            indexData.insert(indexData.end(),reinterpret_cast<const uint32_t*>(mbuf->getIndices()),reinterpret_cast<const uint32_t*>(mbuf->getIndices())+mbuf->getIndexCount());
        }

        //! cache results -- speeds up mesh generation on second run
        {
            io::IWriteFile* cacheFile = device->getFileSystem()->createAndWriteFile("./normalCache101010.sse");
            cacheFile->write(scene::normalCacheFor2_10_10_10Quant.data(),scene::normalCacheFor2_10_10_10Quant.size()*sizeof(scene::QuantizationCacheEntry2_10_10_10));
            cacheFile->drop();
        }

        //
        {
            video::IGPUBuffer* ixbuf = driver->createFilledDeviceLocalGPUBufferOnDedMem(indexData.size()*sizeof(uint32_t),indexData.data());
            indexData.clear();
            vaospec->mapIndexBuffer(ixbuf);
            ixbuf->drop();
        }

        video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
        reqs.vulkanReqs.size = vertexData.size();
        reqs.vulkanReqs.alignment = 4;
        reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
        reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
        reqs.mappingCapability = video::IDriverMemoryAllocation::EMCAF_NO_MAPPING_ACCESS;
        reqs.prefersDedicatedAllocation = true;
        reqs.requiresDedicatedAllocation = true;
        video::IGPUBuffer* vxbuf = driver->createGPUBufferOnDedMem(reqs,true);
        vxbuf->updateSubRange(video::IDriverMemoryAllocation::MemoryRange(0,reqs.vulkanReqs.size),vertexData.data());
        vertexData.clear();


        DrawElementsIndirectCommand indirectDrawData[kInstanceCount];

        uint32_t baseVertex = 0;
        uint32_t indexOffset = 0;
        std::uniform_real_distribution<float> dist3D(0.f,400.f);
        for (size_t i=0; i<kInstanceCount; i++)
        {
            scene::ICPUMeshBuffer* mbuf = cpumesh[i]->getMeshBuffer(0);
            scene::IMeshDataFormatDesc<core::ICPUBuffer>* format = mbuf->getMeshDataAndFormat();
            if (i==0)
            {
                for (size_t j=0; j<scene::EVAI_COUNT; j++)
                {
                    scene::E_VERTEX_ATTRIBUTE_ID attrID = static_cast<scene::E_VERTEX_ATTRIBUTE_ID>(j);
                    if (!format->getMappedBuffer(attrID))
                        continue;

                    vaospec->mapVertexAttrBuffer(vxbuf,attrID,format->getAttribComponentCount(attrID),
                                                 format->getAttribType(attrID),vertexSize,
                                                 format->getMappedBufferOffset(attrID));
                }
            }

            indirectDrawData[i].count = mbuf->getIndexCount();
            indirectDrawData[i].instanceCount = 1;
            indirectDrawData[i].firstIndex = indexOffset/sizeof(uint32_t);
            indirectDrawData[i].baseVertex = baseVertex;
            indirectDrawData[i].baseInstance = 0;


            mbuff[i] = new scene::IGPUMeshBuffer();
            mbuff[i]->setBaseVertex(baseVertex);
            baseVertex += format->getMappedBuffer(scene::EVAI_ATTR0)->getSize()/vertexSize;

            mbuff[i]->setBoundingBox(cpumesh[i]->getBoundingBox());

            mbuff[i]->setIndexBufferOffset(indexOffset);
            indexOffset += mbuf->getIndexCount()*sizeof(uint32_t);

            mbuff[i]->setIndexCount(mbuf->getIndexCount());
            mbuff[i]->setIndexType(scene::EIT_32BIT);
            mbuff[i]->setMeshDataAndFormat(vaospec);
            mbuff[i]->setPrimitiveType(scene::EPT_TRIANGLES);

            cpumesh[i]->drop();


            instanceXForm[i].setScale(dist3D(mt)*0.0025f+1.f);
            instanceXForm[i].setTranslation(core::vector3df(dist3D(mt),dist3D(mt),dist3D(mt)));
        }
        vxbuf->drop();

        reqs.vulkanReqs.size = sizeof(indirectDrawData);
        reqs.vulkanReqs.alignment = 4;
        reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
        reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
        reqs.mappingCapability = video::IDriverMemoryAllocation::EMCAF_NO_MAPPING_ACCESS;
        reqs.prefersDedicatedAllocation = true;
        reqs.requiresDedicatedAllocation = true;
        indirectDrawBuffer = driver->createGPUBufferOnDedMem(reqs,true);
        indirectDrawBuffer->updateSubRange(video::IDriverMemoryAllocation::MemoryRange(0,reqs.vulkanReqs.size),indirectDrawData);
	}

	ObjectData_t perObjectData[kInstanceCount];

    video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
    reqs.vulkanReqs.size = sizeof(perObjectData);
    reqs.vulkanReqs.alignment = 4;
    reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
    reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
    reqs.mappingCapability = video::IDriverMemoryAllocation::EMCAF_NO_MAPPING_ACCESS;
    reqs.prefersDedicatedAllocation = true;
    reqs.requiresDedicatedAllocation = true;
	video::IGPUBuffer* perObjectSSBO = driver->createGPUBufferOnDedMem(reqs,true);
	perObjectSSBO->updateSubRange(video::IDriverMemoryAllocation::MemoryRange(0,reqs.vulkanReqs.size),perObjectData);

	uint64_t lastFPSTime = 0;
	float lastFastestMeshFrameNr = -1.f;

	while(device->run()&&(!quit))
	//if (device->isWindowActive())
	{
		driver->beginScene(true, true, video::SColor(255,255,255,255) );

        //! Draw the view
        smgr->drawAll();

        if (useDrawIndirect)
        {
            if (doCulling)
            {
                //make sure results are visible
                video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            }
            else
            {
                //do compute shader to produce indirect draw buffer (and cull)
                for (size_t i=0; i<kInstanceCount; i++)
                {
                    perObjectData[i].modelViewProjMatrix = core::concatenateBFollowedByA(driver->getTransform(video::EPTS_PROJ_VIEW),instanceXForm[i]);
                    instanceXForm[i].getSub3x3InverseTranspose(perObjectData[i].normalMat);
                }
                perObjectSSBO->updateSubRange(video::IDriverMemoryAllocation::MemoryRange(0,sizeof(perObjectData)),perObjectData);
            }

            //fire it off
            video::COpenGLExtensionHandler::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER,0,1,
                                                                 &static_cast<video::COpenGLBuffer*>(perObjectSSBO)->getOpenGLName());

            video::SMaterial material;
            material.MaterialType = gpuCullMaterial;
            driver->setMaterial(material);
            driver->drawIndexedIndirect(vaospec,scene::EPT_TRIANGLES,scene::EIT_32BIT,indirectDrawBuffer,0,kInstanceCount,sizeof(DrawElementsIndirectCommand));
        }
        else
        {
            scene::IGPUMeshBuffer* mb2draw[kInstanceCount];

            size_t unculledNum = 0;
            for (size_t i=0; i<kInstanceCount; i++)
            {
                if (doCulling)
                    continue;

                mb2draw[unculledNum] = mbuff[i];
                perObjectData[unculledNum].modelViewProjMatrix = core::concatenateBFollowedByA(driver->getTransform(video::EPTS_PROJ_VIEW),instanceXForm[i]);
                instanceXForm[i].getSub3x3InverseTranspose(perObjectData[unculledNum].normalMat);
                unculledNum++;
            }
            perObjectSSBO->updateSubRange(video::IDriverMemoryAllocation::MemoryRange(0,unculledNum*sizeof(ObjectData_t)),perObjectData);
            video::COpenGLExtensionHandler::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER,0,1,
                                                                 &static_cast<video::COpenGLBuffer*>(perObjectSSBO)->getOpenGLName());
            for (size_t i=0; i<unculledNum; i++)
            {
                video::SMaterial material;
                material.MaterialType = cpuCullMaterial;
                reinterpret_cast<uint32_t&>(material.userData) = i;
                driver->setMaterial(material);
                driver->drawMeshBuffer(mb2draw[i]);
            }
        }
        video::COpenGLExtensionHandler::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER,0,1,NULL);

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
	perObjectSSBO->drop();
	indirectDrawBuffer->drop();
    vaospec->drop();

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


    for (size_t i=0; i<kInstanceCount; i++)
    {
        mbuff[i]->drop();
    }

	device->drop();

	return 0;
}
