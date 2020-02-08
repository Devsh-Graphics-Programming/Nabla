#include <random>

#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "irr/asset/normal_quantization.h"

#include "../../ext/ScreenShot/ScreenShot.h"
#include "../common/QToQuitEventReceiver.h"

#include "../source/Irrlicht/COpenGLBuffer.h"
#include "../source/Irrlicht/COpenGLExtensionHandler.h"


using namespace irr;
using namespace core;


bool doCulling = false;
bool useDrawIndirect = false;

class MyEventReceiver : public QToQuitEventReceiver
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
				return QToQuitEventReceiver::OnEvent(event);
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

    virtual void OnSetMaterial(video::IMaterialRendererServices* services, const video::SGPUMaterial& material, const video::SGPUMaterial& lastMaterial)
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
    core::matrix4SIMD modelViewProjMatrix;
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
	params.Vsync = false;
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.

	device->getCursorControl()->setVisible(false);

	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);


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

	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);

    //! read cache results -- speeds up mesh generation
	{
        io::IReadFile* cacheFile = device->getFileSystem()->createAndOpenFile("../../tmp/normalCache101010.sse");
        if (cacheFile)
        {
            asset::normalCacheFor2_10_10_10Quant.resize(cacheFile->getSize()/sizeof(asset::QuantizationCacheEntry2_10_10_10));
            cacheFile->read(asset::normalCacheFor2_10_10_10Quant.data(),cacheFile->getSize());
            cacheFile->drop();

            //make sure its still ok
            std::sort(asset::normalCacheFor2_10_10_10Quant.begin(),asset::normalCacheFor2_10_10_10Quant.end());
        }
	}

	auto* instanceXForm = new core::matrix3x4SIMD[kInstanceCount];
	video::IGPUMeshBuffer* mbuff[kInstanceCount] = {NULL};
	video::IGPUBuffer* indirectDrawBuffer = NULL;

    auto vaospec = driver->createGPUMeshDataFormatDesc();
	{
        core::smart_refctd_ptr<asset::ICPUMesh> cpumesh[kInstanceCount];

        size_t vertexSize = 0;
        std::vector<uint8_t> vertexData;
        std::vector<uint32_t> indexData;

        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<uint32_t> dist(16, 4*1024);
        for (size_t i=0; i<kInstanceCount; i++)
        {
            float poly = sqrtf(dist(mt))+0.5f;
            cpumesh[i] = device->getAssetManager()->getGeometryCreator()->createSphereMesh(2.f,poly,poly);

            //some assumptions about generated mesh
            assert(cpumesh[i]->getMeshBuffer(0)->getPrimitiveType()==asset::EPT_TRIANGLES);
            assert(cpumesh[i]->getMeshBuffer(0)->getIndexType()==asset::EIT_32BIT);
            assert(cpumesh[i]->getMeshBuffer(0)->getBaseVertex()==0);
            assert(cpumesh[i]->getMeshBuffer(0)->getIndexBufferOffset()==0);

            asset::ICPUMeshBuffer* mbuf = cpumesh[i]->getMeshBuffer(0);
            asset::IMeshDataFormatDesc<asset::ICPUBuffer>* format = mbuf->getMeshDataAndFormat();
            assert(format->getMappedBuffer(asset::EVAI_ATTR0)!=NULL); //helpful assumption

            const asset::ICPUBuffer* databuf = NULL;
            for (size_t j=0; j<asset::EVAI_COUNT; j++)
            {
                asset::E_VERTEX_ATTRIBUTE_ID attrID = static_cast<asset::E_VERTEX_ATTRIBUTE_ID>(j);
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
            io::IWriteFile* cacheFile = device->getFileSystem()->createAndWriteFile("../../tmp/normalCache101010.sse");
            cacheFile->write(asset::normalCacheFor2_10_10_10Quant.data(),asset::normalCacheFor2_10_10_10Quant.size()*sizeof(asset::QuantizationCacheEntry2_10_10_10));
            cacheFile->drop();
        }

        //
        {
            auto ixbuf = core::smart_refctd_ptr<video::IGPUBuffer>(driver->createFilledDeviceLocalGPUBufferOnDedMem(indexData.size()*sizeof(uint32_t),indexData.data()),core::dont_grab);
            indexData.clear();
            vaospec->setIndexBuffer(std::move(ixbuf));
        }

        auto vxbuf = core::smart_refctd_ptr<video::IGPUBuffer>(driver->createFilledDeviceLocalGPUBufferOnDedMem(vertexData.size(),vertexData.data()),core::dont_grab);
        vertexData.clear();


        DrawElementsIndirectCommand indirectDrawData[kInstanceCount];

        uint32_t baseVertex = 0;
        uint32_t indexOffset = 0;
        std::uniform_real_distribution<float> dist3D(0.f,400.f);
        for (size_t i=0; i<kInstanceCount; i++)
        {
            asset::ICPUMeshBuffer* mbuf = cpumesh[i]->getMeshBuffer(0);
            asset::IMeshDataFormatDesc<asset::ICPUBuffer>* format = mbuf->getMeshDataAndFormat();
            if (i==0)
            {
                for (size_t j=0; j<asset::EVAI_COUNT; j++)
                {
                    asset::E_VERTEX_ATTRIBUTE_ID attrID = static_cast<asset::E_VERTEX_ATTRIBUTE_ID>(j);
                    if (!format->getMappedBuffer(attrID))
                        continue;

                    vaospec->setVertexAttrBuffer(	core::smart_refctd_ptr(vxbuf),attrID,format->getAttribFormat(attrID),
													vertexSize,format->getMappedBufferOffset(attrID));
                }
            }

            indirectDrawData[i].count = mbuf->getIndexCount();
            indirectDrawData[i].instanceCount = 1;
            indirectDrawData[i].firstIndex = indexOffset/sizeof(uint32_t);
            indirectDrawData[i].baseVertex = baseVertex;
            indirectDrawData[i].baseInstance = 0;


            mbuff[i] = new video::IGPUMeshBuffer();
            mbuff[i]->setBaseVertex(baseVertex);
            baseVertex += format->getMappedBuffer(asset::EVAI_ATTR0)->getSize()/vertexSize;

            mbuff[i]->setBoundingBox(cpumesh[i]->getBoundingBox());

            mbuff[i]->setIndexBufferOffset(indexOffset);
            indexOffset += mbuf->getIndexCount()*sizeof(uint32_t);

            mbuff[i]->setIndexCount(mbuf->getIndexCount());
            mbuff[i]->setIndexType(asset::EIT_32BIT);
            mbuff[i]->setMeshDataAndFormat(core::smart_refctd_ptr(vaospec));
            mbuff[i]->setPrimitiveType(asset::EPT_TRIANGLES);


            float scale = dist3D(mt)*0.0025f+1.f;
            instanceXForm[i].setScale(core::vectorSIMDf(scale,scale,scale));
            instanceXForm[i].setTranslation(core::vectorSIMDf(dist3D(mt),dist3D(mt),dist3D(mt)));
        }

        indirectDrawBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(indirectDrawData),indirectDrawData);
	}

	ObjectData_t perObjectData[kInstanceCount];

	video::IGPUBuffer* perObjectSSBO = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(perObjectData));
	auto updateSSBO  = [&](const void* data, const uint32_t dataSize)
    {
        auto defaultUploadBuffer = driver->getDefaultUpStreamingBuffer();

        const void* dataPtr = reinterpret_cast<const uint8_t*>(data);
        uint32_t offset = video::StreamingTransientDataBufferMT<>::invalid_address;
        while (offset==video::StreamingTransientDataBufferMT<>::invalid_address)
        {
            uint32_t alignment = 8u;
            defaultUploadBuffer->multi_place(std::chrono::microseconds(500u),1u,(const void* const*)&dataPtr,&offset,&dataSize,&alignment);
        }
        // some platforms expose non-coherent host-visible GPU memory, so writes need to be flushed explicitly
        if (defaultUploadBuffer->needsManualFlushOrInvalidate())
            driver->flushMappedMemoryRanges({{defaultUploadBuffer->getBuffer()->getBoundMemory(),offset,dataSize}});
        // after we make sure writes are in GPU memory (visible to GPU) and not still in a cache, we can copy using the GPU to device-only memory
        driver->copyBuffer(defaultUploadBuffer->getBuffer(),perObjectSSBO,offset,0u,dataSize);
        // this doesn't actually free the memory, the memory is queued up to be freed only after the GPU fence/event is signalled
        defaultUploadBuffer->multi_free(1u,&offset,&dataSize,std::move(driver->placeFence()));
    };
    updateSSBO(perObjectData,sizeof(perObjectData));

	uint64_t lastFPSTime = 0;
	float lastFastestMeshFrameNr = -1.f;

	while(device->run() && receiver.keepOpen())
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
                    instanceXForm[i].getSub3x3InverseTransposePacked(perObjectData[i].normalMat);
                }
                updateSSBO(perObjectData,sizeof(perObjectData));
            }

            //fire it off
            video::COpenGLExtensionHandler::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER,0,1,
                                                                 &static_cast<video::COpenGLBuffer*>(perObjectSSBO)->getOpenGLName());

            video::SGPUMaterial material;
            material.MaterialType = gpuCullMaterial;
            driver->setMaterial(material);
            driver->drawIndexedIndirect(vaospec.get(),asset::EPT_TRIANGLES,asset::EIT_32BIT,indirectDrawBuffer,0,kInstanceCount,sizeof(DrawElementsIndirectCommand));
        }
        else
        {
            video::IGPUMeshBuffer* mb2draw[kInstanceCount];

            size_t unculledNum = 0;
            for (size_t i=0; i<kInstanceCount; i++)
            {
                if (doCulling)
                    continue;

                mb2draw[unculledNum] = mbuff[i];
                perObjectData[unculledNum].modelViewProjMatrix = core::concatenateBFollowedByA(driver->getTransform(video::EPTS_PROJ_VIEW),instanceXForm[i]);
                instanceXForm[i].getSub3x3InverseTransposePacked(perObjectData[unculledNum].normalMat);
                unculledNum++;
            }
            updateSSBO(perObjectData,sizeof(perObjectData));
            video::COpenGLExtensionHandler::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER,0,1,
                                                                 &static_cast<video::COpenGLBuffer*>(perObjectSSBO)->getOpenGLName());
            for (size_t i=0; i<unculledNum; i++)
            {
                video::SGPUMaterial material;
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
			str << L"MultiDrawIndirect Benchmark - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str());
			lastFPSTime = time;
		}
	}
	perObjectSSBO->drop();
	indirectDrawBuffer->drop();

    delete [] instanceXForm;

    //create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		ext::ScreenShot::dirtyCPUStallingScreenshot(device, "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}


    for (size_t i=0; i<kInstanceCount; i++)
    {
        mbuff[i]->drop();
    }

	device->drop();

	return 0;
}
