// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

//! I advise to check out this file, its a basic input handler
#include "../common/QToQuitEventReceiver.h"
#include "nbl/asset/utils/CCPUMeshPackerV1.h"
#include "nbl/asset/CCPUMeshPackerV2.h"

using namespace nbl;
using namespace core;
using namespace asset;
using namespace video;

constexpr const char* SHADER_OVERRIDES =
R"(
#define _NBL_VERT_INPUTS_DEFINED_

//#define nbl_glsl_VirtualAttribute_t uint

vec4 nbl_glsl_decodeRGB10A2_UNORM(in uint x)
{
	const uvec3 rgbMask = uvec3(0x3ffu);
	const uvec4 shifted = uvec4(x,uvec3(x)>>uvec3(10,20,30));
	return vec4(vec3(shifted.rgb&rgbMask),shifted.a)/vec4(vec3(rgbMask),3.0);
}

vec4 nbl_glsl_decodeRGB10A2_SNORM(in uint x)
{
    uvec4 shifted = uvec4(x,uvec3(x)>>uvec3(10,20,30));
    const uvec3 rgbMask = uvec3(0x3ffu);
    const uvec3 rgbBias = uvec3(0x200u);
    return max(vec4(vec3(shifted.rgb&rgbMask)-rgbBias,float(shifted.a)-2.0)/vec4(vec3(rgbBias-uvec3(1u)),1.0),vec4(-1.0));
}

//pos
layout(set = 0, binding = 0) uniform samplerBuffer MeshPackedData_R32G32B32_SFLOAT;

//uv
layout(set = 0, binding = 1) uniform samplerBuffer MeshPackedData_R32G32_SFLOAT;

//normal
layout(set = 0, binding = 2) uniform usamplerBuffer MeshPackedData_A2B10G10R10_SNORM_PACK32;

layout(set = 0, binding = 3) readonly buffer VertexDataOffsetTable
{
    int dataOffsetTable[];
} vertexPosition;

layout(set = 0, binding = 4) readonly buffer VertexUVOffsetTable
{
    int dataOffsetTable[];
} vertexUV;

layout(set = 0, binding = 5) readonly buffer VertexNormalOffsetTable
{
    int dataOffsetTable[];
} vertexNormal;

#define _NBL_BASIC_VTX_ATTRIB_FETCH_FUCTIONS_DEFINED_
#define _NBL_POS_FETCH_FUNCTION_DEFINED
#define _NBL_UV_FETCH_FUNCTION_DEFINED
#define _NBL_NORMAL_FETCH_FUNCTION_DEFINED

//vec4 nbl_glsl_readAttrib(uint offset)
//ivec4 nbl_glsl_readAttrib(uint offset)
//uvec4 nbl_glsl_readAttrib(uint offset)
//vec3 nbl_glsl_readAttrib(uint offset) 
//..

vec3 nbl_glsl_fetchVtxPos()
{
    int vtxPosOffset = int(gl_VertexIndex) + vertexPosition.dataOffsetTable[gl_DrawID];
    return texelFetch(MeshPackedData_R32G32B32_SFLOAT, vtxPosOffset).xyz;
}

vec2 nbl_glsl_fetchVtxUV()
{
    int vtxUVOffset = int(gl_VertexIndex) + vertexUV.dataOffsetTable[gl_DrawID];
    return texelFetch(MeshPackedData_R32G32_SFLOAT, vtxUVOffset).xy;
}

vec3 nbl_glsl_fetchVtxNormal()
{
    int vtxNormOffset = int(gl_VertexIndex) + vertexNormal.dataOffsetTable[gl_DrawID];
    return normalize(nbl_glsl_decodeRGB10A2_SNORM(texelFetch(MeshPackedData_A2B10G10R10_SNORM_PACK32, vtxNormOffset).x).xyz);
}

)";

struct DataOffsetTable
{
    uint32_t binding;
    asset::SBufferBinding<IGPUBuffer> offsetBuffer;
};

core::smart_refctd_ptr<asset::ICPUSpecializedShader> createModifiedVertexShader(const asset::ICPUSpecializedShader* _fs)
{
    const asset::ICPUShader* unspec = _fs->getUnspecialized();
    assert(unspec->containsGLSL());

    auto begin = reinterpret_cast<const char*>(unspec->getSPVorGLSL()->getPointer());
    auto end = begin + unspec->getSPVorGLSL()->getSize();
    std::string resultShaderSrc(begin, end);

    size_t firstNewlineAfterVersion = resultShaderSrc.find("\n", resultShaderSrc.find("#version "));

    const std::string customSrcCode = SHADER_OVERRIDES;

    resultShaderSrc.insert(firstNewlineAfterVersion, customSrcCode);
    resultShaderSrc.replace(resultShaderSrc.find("#version 430 core"), sizeof("#version 430 core"), "#version 460 core\n");

    auto unspecNew = core::make_smart_refctd_ptr<asset::ICPUShader>(resultShaderSrc.c_str());
    auto specinfo = _fs->getSpecializationInfo();
    auto vsNew = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecNew), std::move(specinfo));

    return vsNew;
}

struct DrawIndexedIndirectInput
{
    asset::SBufferBinding<video::IGPUBuffer> vtxBuffer;
    static constexpr asset::E_PRIMITIVE_TOPOLOGY mode = asset::EPT_TRIANGLE_LIST;
    static constexpr asset::E_INDEX_TYPE indexType = asset::EIT_16BIT;
    core::smart_refctd_ptr<video::IGPUBuffer> idxBuff = nullptr;
    core::smart_refctd_ptr<video::IGPUBuffer> indirectDrawBuff = nullptr;
    size_t offset = 0u;
    size_t maxCount = 0u;
    size_t stride = 0u;
    core::smart_refctd_ptr<video::IGPUBuffer> countBuffer = nullptr;
    size_t countOffset = 0u;
};


void packMeshBuffers(video::IVideoDriver* driver, core::vector<ICPUMeshBuffer*>& meshBuffers, DrawIndexedIndirectInput& output, std::array<DataOffsetTable, 3>& offsetTable)
{
    using MeshPacker = CCPUMeshPackerV2<DrawElementsIndirectCommand_t>;

    MeshPacker::AllocationParams allocParams;
    allocParams.indexBuffSupportedCnt = 20000000u;
    allocParams.indexBufferMinAllocSize = 5000u;
    allocParams.vertexBuffSupportedSize = 200000000u;
    allocParams.vertexBufferMinAllocSize = 5000u;
    allocParams.MDIDataBuffSupportedCnt = 20000u;
    allocParams.MDIDataBuffMinAllocSize = 1u; //so structs are adjacent in memory

    CCPUMeshPackerV2 mp(allocParams, std::numeric_limits<uint16_t>::max() / 3u, std::numeric_limits<uint16_t>::max() / 3u);

    core::vector<MeshPacker::ReservedAllocationMeshBuffers> allocData(meshBuffers.size());

    bool allocSuccessfull = mp.alloc(allocData.data(), meshBuffers.begin(), meshBuffers.end());
    if (!allocSuccessfull)
        std::cout << "Alloc failed \n";

    mp.instantiateDataStorage();
    MeshPacker::PackerDataStore packerDataStore = mp.getPackerDataStore();

    core::vector<IMeshPackerBase::PackedMeshBufferData> pmbd(meshBuffers.size());
    
    const uint32_t offsetTableSz = mp.calcDataTableNeededSize(meshBuffers.begin(), meshBuffers.end());
    core::vector<MeshPacker::CombinedDataOffsetTable> cdot(offsetTableSz);

    mp.commit(pmbd.data(), cdot.data(), allocData.data(), meshBuffers.begin(), meshBuffers.end());

    output.vtxBuffer = { 0ull, driver->createFilledDeviceLocalGPUBufferOnDedMem(packerDataStore.vertexBuffer->getSize(), packerDataStore.vertexBuffer->getPointer()) };
    output.idxBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(packerDataStore.indexBuffer->getSize(), packerDataStore.indexBuffer->getPointer());
    output.indirectDrawBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(packerDataStore.MDIDataBuffer->getSize(), packerDataStore.MDIDataBuffer->getPointer());

    output.maxCount = offsetTableSz;
    output.stride = sizeof(DrawElementsIndirectCommand_t);

    auto glsl = mp.generateGLSLBufferDefinitions(0u);

    /*DrawElementsIndirectCommand_t* mdiPtr = static_cast<DrawElementsIndirectCommand_t*>(packerDataStore.MDIDataBuffer->getPointer());
    uint16_t* idxBuffPtr = static_cast<uint16_t*>(packerDataStore.indexBuffer->getPointer());
    float* vtxBuffPtr = static_cast<float*>(packerDataStore.vertexBuffer->getPointer());

    for (uint32_t i = 0u; i < 1188; i++)
    {
        float* firstCoord = vtxBuffPtr + (*(idxBuffPtr + i) * 3u);
        std::cout << "vtx: " << i << " idx: " << *(idxBuffPtr + i) << "    ";
        std::cout << *firstCoord << ' ' << *(firstCoord + 1u) << ' ' << *(firstCoord + 2u) << std::endl;
    }*/

    //setOffsetTables

    core::vector<uint32_t> offsetTableLocal(offsetTableSz);

    for (uint32_t i = 0u; i < offsetTableLocal.size(); i++)
        offsetTableLocal[i] = cdot[i].attribOffset[0];

    offsetTable[0].offsetBuffer.offset = 0u;
    offsetTable[0].offsetBuffer.buffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(uint32_t) * offsetTableLocal.size(), static_cast<void*>(offsetTableLocal.data()));

    for (uint32_t i = 0u; i < offsetTableLocal.size(); i++)
        offsetTableLocal[i] = cdot[i].attribOffset[1];

    offsetTable[1].offsetBuffer.offset = 0u;
    offsetTable[1].offsetBuffer.buffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(uint32_t) * offsetTableLocal.size(), static_cast<void*>(offsetTableLocal.data()));

    for (uint32_t i = 0u; i < offsetTableLocal.size(); i++)
        offsetTableLocal[i] = cdot[i].attribOffset[2];

    offsetTable[2].offsetBuffer.offset = 0u;
    offsetTable[2].offsetBuffer.buffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(uint32_t) * offsetTableLocal.size(), static_cast<void*>(offsetTableLocal.data()));
}

void setPipeline(IVideoDriver* driver, ICPUSpecializedShader* vs, ICPUSpecializedShader* fs,
    core::smart_refctd_ptr<IGPUBuffer>& vtxBuffer, core::smart_refctd_ptr<IGPUBuffer>& outputUBO, std::array<DataOffsetTable, 3>& dataOffsetBuffers,
    core::smart_refctd_ptr<IGPUDescriptorSet>& outputGPUDescriptorSet0,
    core::smart_refctd_ptr<IGPUDescriptorSet>& outputGPUDescriptorSet1,
    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>& outputGpuPipeline)
{
    ICPUSpecializedShader* cpuShaders[2] = { vs, fs };
    auto gpuShaders = driver->getGPUObjectsFromAssets(cpuShaders, cpuShaders + 2);

    core::smart_refctd_ptr<IGPUDescriptorSetLayout> ds0Layout;
    core::smart_refctd_ptr<IGPUDescriptorSetLayout> ds1Layout;
    {
        IGPUDescriptorSetLayout::SBinding b[6];
        b[0].binding = 0u; b[1].binding = 1u; b[2].binding = 2u; b[3].binding = 3u; b[4].binding = 4u; b[5].binding = 5u;
        b[0].type = b[1].type = b[2].type = EDT_UNIFORM_TEXEL_BUFFER;
        b[3].type = b[4].type = b[5].type = EDT_STORAGE_BUFFER;
        b[0].stageFlags = b[1].stageFlags = b[2].stageFlags = b[3].stageFlags = b[4].stageFlags = b[5].stageFlags = ISpecializedShader::ESS_VERTEX;
        b[0].count = b[1].count = b[2].count = b[3].count = b[4].count = b[5].count = 1u;
        ds0Layout = driver->createGPUDescriptorSetLayout(b, b + 6u);

        IGPUDescriptorSetLayout::SBinding b2;
        b2.binding = 0;
        b2.type = EDT_UNIFORM_BUFFER;
        b2.stageFlags = ISpecializedShader::ESS_VERTEX;
        b2.count = 1u;
        ds1Layout = driver->createGPUDescriptorSetLayout(&b2, &b2 + 1);
    }

    asset::SPushConstantRange pcRange = { asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD) };
    auto pipelineLayout = driver->createGPUPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr(ds0Layout), core::smart_refctd_ptr(ds1Layout));

    outputGPUDescriptorSet0 = driver->createGPUDescriptorSet(std::move(ds0Layout));
    outputGPUDescriptorSet1 = driver->createGPUDescriptorSet(std::move(ds1Layout));
    {
        IGPUDescriptorSet::SWriteDescriptorSet w[6];
        w[0].arrayElement = w[1].arrayElement = w[2].arrayElement = w[3].arrayElement = w[4].arrayElement = w[5].arrayElement = 0u;
        w[0].count = w[1].count = w[2].count = w[3].count = w[4].count = w[5].count = 1u;
        w[0].binding = 0u; w[1].binding = 1u; w[2].binding = 2u; w[3].binding = 3u; w[4].binding = 4u; w[5].binding = 4u;
        w[0].descriptorType = w[1].descriptorType = w[2].descriptorType = EDT_UNIFORM_TEXEL_BUFFER;
        w[3].descriptorType = w[4].descriptorType = w[5].descriptorType = EDT_STORAGE_BUFFER;
        w[0].dstSet = w[1].dstSet = w[2].dstSet = w[3].dstSet = w[4].dstSet = w[5].dstSet = outputGPUDescriptorSet0.get();

        IGPUDescriptorSet::SDescriptorInfo info[6];

        info[0].buffer.offset = 0u;
        info[0].buffer.size = vtxBuffer->getSize();
        info[0].desc = driver->createGPUBufferView(vtxBuffer.get(), EF_R32G32B32_SFLOAT);
        info[1].buffer.offset = 0u;
        info[1].buffer.size = vtxBuffer->getSize();
        info[1].desc = driver->createGPUBufferView(vtxBuffer.get(), EF_R32G32_SFLOAT);
        info[2].buffer.offset = 0u;
        info[2].buffer.size = vtxBuffer->getSize();
        info[2].desc = driver->createGPUBufferView(vtxBuffer.get(), EF_R32_UINT);

        //sampler buffers
        w[0].info = &info[0];
        w[1].info = &info[1];
        w[2].info = &info[2];

        //offset tables
        info[3].buffer.offset = dataOffsetBuffers[0].offsetBuffer.offset;
        info[3].buffer.size = dataOffsetBuffers[0].offsetBuffer.buffer->getSize();
        info[3].desc = core::smart_refctd_ptr(dataOffsetBuffers[0].offsetBuffer.buffer);
        w[3].info = &info[3];

        info[4].buffer.offset = dataOffsetBuffers[1].offsetBuffer.offset;
        info[4].buffer.size = dataOffsetBuffers[1].offsetBuffer.buffer->getSize();
        info[4].desc = core::smart_refctd_ptr(dataOffsetBuffers[1].offsetBuffer.buffer);
        w[4].info = &info[4];

        info[5].buffer.offset = dataOffsetBuffers[2].offsetBuffer.offset;
        info[5].buffer.size = dataOffsetBuffers[2].offsetBuffer.buffer->getSize();
        info[5].desc = core::smart_refctd_ptr(dataOffsetBuffers[2].offsetBuffer.buffer);
        w[5].info = &info[5];

        driver->updateDescriptorSets(6u, w, 0u, nullptr);

        IGPUDescriptorSet::SWriteDescriptorSet w2;
        w2.arrayElement = 0u;
        w2.count = 1u;
        w2.binding = 0u;
        w2.descriptorType = EDT_UNIFORM_BUFFER;
        w2.dstSet = outputGPUDescriptorSet1.get();

        outputUBO = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(SBasicViewParameters));

        IGPUDescriptorSet::SDescriptorInfo info2;
        info2.buffer.offset = 0u;
        info2.buffer.size = outputUBO->getSize();
        info2.desc = core::smart_refctd_ptr(outputUBO);
        w2.info = &info2;

        driver->updateDescriptorSets(1u, &w2, 0u, nullptr);
    }



    IGPUSpecializedShader* shaders[2] = { gpuShaders->operator[](0).get(), gpuShaders->operator[](1).get() };

    outputGpuPipeline = driver->createGPURenderpassIndependentPipeline(
        nullptr, std::move(pipelineLayout),
        shaders, shaders + 2u,
        SVertexInputParams(),
        asset::SBlendParams(), asset::SPrimitiveAssemblyParams(), SRasterizationParams());
}

int main()
{
    // create device with full flexibility over creation parameters
    // you can add more parameters if desired, check irr::SIrrlichtCreationParameters
    nbl::SIrrlichtCreationParameters params;
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

    //! disable mouse cursor, since camera will force it to the middle
    //! and we don't want a jittery cursor in the middle distracting us
    device->getCursorControl()->setVisible(false);

    //! Since our cursor will be enslaved, there will be no way to close the window
    //! So we listen for the "Q" key being pressed and exit the application
    QToQuitEventReceiver receiver;
    device->setEventReceiver(&receiver);

    auto* driver = device->getVideoDriver();
    auto* smgr = device->getSceneManager();
    auto* am = device->getAssetManager();
    auto* fs = am->getFileSystem();

    //
    auto* qnc = am->getMeshManipulator()->getQuantNormalCache();
    //loading cache from file
    qnc->loadNormalQuantCacheFromFile<asset::CQuantNormalCache::E_CACHE_TYPE::ECT_2_10_10_10>(fs, "../../tmp/normalCache101010.sse", true);

    // register the zip
    device->getFileSystem()->addFileArchive("../../media/sponza.zip");

    asset::IAssetLoader::SAssetLoadParams lp;
    auto meshes_bundle = am->getAsset("sponza.obj", lp);
    //assert(!meshes_bundle.isEmpty());
    auto mesh = meshes_bundle.getContents().begin()[0];
    auto mesh_raw = static_cast<asset::ICPUMesh*>(mesh.get());

    //saving cache to file
    qnc->saveCacheToFile(asset::CQuantNormalCache::E_CACHE_TYPE::ECT_2_10_10_10, fs, "../../tmp/normalCache101010.sse");

    //TODO: change it to vector of smart pointers
    core::vector<ICPUMeshBuffer*> meshBuffers;
    for (uint32_t i = 0u; i < mesh_raw->getMeshBufferVector().size(); i++)
        meshBuffers.push_back(mesh_raw->getMeshBufferVector()[i].get());

    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuPipeline;
    core::smart_refctd_ptr<IGPUDescriptorSet> ds0;
    core::smart_refctd_ptr<IGPUDescriptorSet> ds1;
    core::smart_refctd_ptr<IGPUBuffer> ubo;
    DrawIndexedIndirectInput mdiCallParams;
    {
        auto* pipeline = meshBuffers[0]->getPipeline();

        auto* vtxShader = pipeline->getShaderAtIndex(asset::ICPURenderpassIndependentPipeline::ESSI_VERTEX_SHADER_IX);
        core::smart_refctd_ptr<ICPUSpecializedShader> vs = createModifiedVertexShader(vtxShader);
        ICPUSpecializedShader* fs = IAsset::castDown<ICPUSpecializedShader>(am->getAsset("../shader.frag", lp).getContents().begin()->get());
        std::array<DataOffsetTable, 3> offsetTable;

        packMeshBuffers(driver, meshBuffers, mdiCallParams, offsetTable);

        setPipeline(driver, vs.get(), fs, mdiCallParams.vtxBuffer.buffer, ubo, offsetTable, ds0, ds1, gpuPipeline);
    }

    //! we want to move around the scene and view it from different angles
    scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0, 100.0f, 0.5f);

    camera->setPosition(core::vector3df(-4, 0, 0));
    camera->setTarget(core::vector3df(0, 0, 0));
    camera->setNearValue(1.f);
    camera->setFarValue(5000.0f);

    smgr->setActiveCamera(camera);

    uint64_t lastFPSTime = 0;
    while (device->run() && receiver.keepOpen())
    {
        video::IGPUDescriptorSet* ds[]{ ds0.get(), ds1.get() };
        driver->bindGraphicsPipeline(gpuPipeline.get());
        driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuPipeline->getLayout(), 0u, 2u, ds, nullptr);

        driver->beginScene(true, true, video::SColor(255, 0, 0, 255));

        //! This animates (moves) the camera and sets the transforms
        camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
        camera->render();

        SBasicViewParameters uboData;

        memcpy(uboData.MVP, camera->getConcatenatedMatrix().pointer(), sizeof(core::matrix4SIMD));
        memcpy(uboData.MV, camera->getViewMatrix().pointer(), sizeof(core::matrix3x4SIMD));
        memcpy(uboData.NormalMat, camera->getViewMatrix().pointer(), sizeof(core::matrix3x4SIMD));

        driver->updateBufferRangeViaStagingBuffer(ubo.get(), 0u, sizeof(SBasicViewParameters), &uboData);

        SBufferBinding<IGPUBuffer> vtxBufferBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
        vtxBufferBindings[0] = mdiCallParams.vtxBuffer;
        driver->drawIndexedIndirect(vtxBufferBindings, mdiCallParams.mode, mdiCallParams.indexType, mdiCallParams.idxBuff.get(), mdiCallParams.indirectDrawBuff.get(), mdiCallParams.offset, mdiCallParams.maxCount, mdiCallParams.stride);

        driver->endScene();

        // display frames per second in window title
        uint64_t time = device->getTimer()->getRealTime();
        if (time - lastFPSTime > 1000)
        {
            std::wostringstream str;
            str << L"Meshloaders Demo - IrrlichtBAW Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

            device->setWindowCaption(str.str().c_str());
            lastFPSTime = time;
        }
    }
}