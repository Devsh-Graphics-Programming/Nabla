#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../../ext/ScreenShot/ScreenShot.h"
#include "../common/QToQuitEventReceiver.h"
#include "../source/Irrlicht/COpenGLExtensionHandler.h"

#include <random>

using namespace irr;
using namespace core;
using namespace asset;


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


#include "common.glsl"
#include "commonIndirect.glsl"

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
	auto device = createDeviceEx(params);

	if (!device)
		return 1; // could not create selected driver.

	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);


    auto* am = device->getAssetManager();
    video::IVideoDriver* driver = device->getVideoDriver();

    IAssetLoader::SAssetLoadParams lp;
    auto cullShaderBundle = am->getAsset("../boxFrustCull.comp", lp); // this doesn't want to load with my shader loader, why?
    auto vertexShaderBundleMDI = am->getAsset("../meshGPU.vert", lp);
    auto vertexShaderBundle = am->getAsset("../meshCPU.vert", lp);
    auto fragShaderBundle = am->getAsset("../mesh.frag", lp);

    CShaderIntrospector introspector(am->getGLSLCompiler());
    const auto extensions = driver->getSupportedGLSLExtensions();
    auto cpuCullPipeline = introspector.createApproximateComputePipelineFromIntrospection(IAsset::castDown<ICPUSpecializedShader>(cullShaderBundle.getContents().first->get()), extensions->begin(), extensions->end());
    ICPUSpecializedShader* shaders[2] = { IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundleMDI.getContents().first->get()),IAsset::castDown<ICPUSpecializedShader>(fragShaderBundle.getContents().first->get()) };
    auto cpuDrawIndirectPipeline = introspector.createApproximateRenderpassIndependentPipelineFromIntrospection(shaders, shaders+2, extensions->begin(), extensions->end());
    auto cpuDrawDirectPipeline = introspector.createApproximateRenderpassIndependentPipelineFromIntrospection(shaders, shaders+2, extensions->begin(), extensions->end());



    /*
	auto 
		auto cpushader = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(*asset.getContents().first);
		auto gpumesh = driver->getGPUObjectsFromAssets(&cpumesh.get(), (&cpumesh.get()) + 1)->operator[](0);
    //auto descriptorSetLayout = driver->createGPUDescriptorSetLayout(bindings,bindings+BindingCount);
    */

    auto* fs = am->getFileSystem();

    //
    auto* qnc = am->getMeshManipulator()->getQuantNormalCache();
    //loading cache from file
    qnc->loadNormalQuantCacheFromFile<asset::E_QUANT_NORM_CACHE_TYPE::Q_2_10_10_10>(fs, "../../tmp/normalCache101010.sse", true);

    #define kInstanceCount 4096
    #define kTotalTriangleLimit (64*1024*1024)


    core::matrix3x4SIMD instanceXForm[kInstanceCount];
	core::smart_refctd_ptr<video::IGPUMeshBuffer> mbuff[kInstanceCount] = {};
	core::smart_refctd_ptr<video::IGPUBuffer> indirectDrawBuffer;

    core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> cpuDrawPipeline,gpuDrawPipeline;
#if 0
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
#endif
        //! cache results -- speeds up mesh generation on second run
        qnc->saveCacheToFile(asset::E_QUANT_NORM_CACHE_TYPE::Q_2_10_10_10, fs, "../../tmp/normalCache101010.sse");
#if 0
        //
        {
            auto ixbuf = core::smart_refctd_ptr<video::IGPUBuffer>(driver->createFilledDeviceLocalGPUBufferOnDedMem(indexData.size()*sizeof(uint32_t),indexData.data()),core::dont_grab);
            indexData.clear();
            vaospec->setIndexBuffer(std::move(ixbuf));
        }

        auto vxbuf = core::smart_refctd_ptr<video::IGPUBuffer>(driver->createFilledDeviceLocalGPUBufferOnDedMem(vertexData.size(),vertexData.data()),core::dont_grab);
        vertexData.clear();


        DrawElementsIndirectCommand_t indirectDrawData[kInstanceCount];

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


            mbuff[i] = core::smart_refctd_ptr<video::IGPUMeshBuffer>();
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
#endif
	ModelData_t perObjectData[kInstanceCount];

	auto perObjectSSBO = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(perObjectData));
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
        driver->copyBuffer(defaultUploadBuffer->getBuffer(),perObjectSSBO.get(),offset,0u,dataSize);
        // this doesn't actually free the memory, the memory is queued up to be freed only after the GPU fence/event is signalled
        defaultUploadBuffer->multi_free(1u,&offset,&dataSize,std::move(driver->placeFence()));
    };
    updateSSBO(perObjectData,sizeof(perObjectData));

    //auto descriptorSet = driver->createGPUDescriptorSet(std::move(descriptorSetLayout));
    

    auto smgr = device->getSceneManager();

	scene::ICameraSceneNode* camera =
		smgr->addCameraSceneNodeFPS(0,100.0f,0.01f);
	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(250.0f);
    smgr->setActiveCamera(camera);

    device->getCursorControl()->setVisible(false);


	uint64_t lastFPSTime = 0;
	float lastFastestMeshFrameNr = -1.f;

	while(device->run() && receiver.keepOpen())
	//if (device->isWindowActive())
	{
		driver->beginScene(true, true, video::SColor(255,255,255,255) );

        //! This animates (moves) the camera and sets the transforms
        camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
        camera->render();

        //driver->bindDescriptorSets(video::EPBP_GRAPHICS,cpuDrawPipeline->getLayout(),1u,1u,&descriptorSet.get(),nullptr);
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
                    perObjectData[i].modelViewProjMatrix = core::concatenateBFollowedByA(camera->getConcatenatedMatrix(),instanceXForm[i]);
                    instanceXForm[i].getSub3x3InverseTranspose(perObjectData[i].normalMat);
                }
                updateSSBO(perObjectData,sizeof(perObjectData));
            }

            driver->bindGraphicsPipeline(gpuDrawPipeline.get());
            //driver->drawIndexedIndirect(,asset::EPT_TRIANGLE_LIST,asset::EIT_32BIT,,indirectDrawBuffer,0,kInstanceCount,sizeof(DrawElementsIndirectCommand_t));
        }
        else
        {
            video::IGPUMeshBuffer* mb2draw[kInstanceCount];

            uint32_t unculledNum = 0;
            for (uint32_t i=0; i<kInstanceCount; i++)
            {
                if (doCulling)
                    continue;

                mb2draw[unculledNum] = mbuff[i].get();
                perObjectData[unculledNum].modelViewProjMatrix = core::concatenateBFollowedByA(camera->getConcatenatedMatrix(),instanceXForm[i]);
                instanceXForm[i].getSub3x3InverseTranspose(perObjectData[unculledNum].normalMat);
                unculledNum++;
            }
            updateSSBO(perObjectData,sizeof(perObjectData));

            driver->bindGraphicsPipeline(cpuDrawPipeline.get());
            for (uint32_t i=0; i<unculledNum; i++)
            {
                driver->pushConstants(cpuDrawPipeline->getLayout(),asset::ICPUSpecializedShader::ESS_VERTEX,0u,sizeof(uint32_t),&i);
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

	return 0;
}
