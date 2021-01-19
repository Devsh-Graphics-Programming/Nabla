// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

//! I advise to check out this file, its a basic input handler
#include "../common/QToQuitEventReceiver.h"
#include "nbl/asset/CCPUMeshPackerV1.h"
#include "nbl/asset/CCPUMeshPackerV2.h"

using namespace nbl;
using namespace core;
using namespace asset;
using namespace video;

constexpr const char* SHADER_OVERRIDES[2] =
{
R"(
#define _NBL_VERT_INPUTS_DEFINED_
#define _NBL_VERT_MAIN_DEFINED_

//pos
layout(set = 0, binding = 0) uniform samplerBuffer MeshPackedData_R32G32B32_SFLOAT;

//uv
//layout(set = 0, binding = 1) uniform samplerBuffer MeshPackedData_R32G32_SFLOAT;

//normal
layout(set = 0, binding = 2) uniform samplerBuffer MeshPackedData_A2B10G10R10_SNORM_PACK32;

layout(set = 0, binding = 3) readonly buffer VertexDataOffsetTable
{
    uint dataOffsetTable[];
} vertexPosition;

layout(set = 0, binding = 4) readonly buffer VertexNormalOffsetTable
{
    uint dataOffsetTable[];
} vertexNormal;

)",

R"(
void main()
{
    vec3 pos = texelFetch(MeshPackedData_R32G32B32_SFLOAT,int(gl_VertexIndex)+vertexPosition.dataOffsetTable[gl_DrawID]).xyz;
    LocalPos = pos;
    gl_Position = nbl_glsl_pseudoMul4x4with3x1(CamData.params.MVP, pos);
    ViewPos = nbl_glsl_pseudoMul3x4with3x1(CamData.params.MV, pos);
    mat3 normalMat = nbl_glsl_SBasicViewParameters_GetNormalMat(CamData.params.NormalMatAndEyePos);

    vec3 normal = texelFetch(MeshPackedData_A2B10G10R10_SNORM_PACK32,int(gl_VertexIndex)+vertexNormal.dataOffsetTable[gl_DrawID]).xyz;
    //Normal = normalMat*normalize(vNormal);
    Normal = vec3(1.0, 0.0, 0.0);
}
)"

}
;

core::smart_refctd_ptr<asset::ICPUSpecializedShader> createModifiedVertexShader(asset::ICPUSpecializedShader* _fs)
{
    const asset::ICPUShader* unspec = _fs->getUnspecialized();
    assert(unspec->containsGLSL());

    auto begin = reinterpret_cast<const char*>(unspec->getSPVorGLSL()->getPointer());
    auto end = begin + unspec->getSPVorGLSL()->getSize();
    std::string resultShaderSrc(begin, end);

    size_t firstNewlineAfterVersion = resultShaderSrc.find("\n", resultShaderSrc.find("#version "));

    const std::string buffersDef = SHADER_OVERRIDES[0];
    const std::string mainDef = SHADER_OVERRIDES[1];

    resultShaderSrc.insert(firstNewlineAfterVersion, buffersDef);
    resultShaderSrc += mainDef;

    auto unspecNew = core::make_smart_refctd_ptr<asset::ICPUShader>(resultShaderSrc.c_str());
    auto specinfo = _fs->getSpecializationInfo();
    auto vsNew = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecNew), std::move(specinfo));

    return vsNew;
}

struct DrawIndexedIndirectInput
{
    asset::SBufferBinding<video::IGPUBuffer> vtxBindings[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
    asset::E_PRIMITIVE_TOPOLOGY mode = asset::EPT_TRIANGLE_LIST;
    asset::E_INDEX_TYPE indexType = asset::EIT_16BIT;
    core::smart_refctd_ptr<video::IGPUBuffer> indexBuff = nullptr;
    core::smart_refctd_ptr<video::IGPUBuffer> indirectDrawBuff = nullptr;
    size_t offset = 0u;
    size_t maxCount = 0u;
    size_t stride = 0u;
    core::smart_refctd_ptr<video::IGPUBuffer> countBuffer = nullptr;
    size_t countOffset = 0u;
};

struct DrawIndexedIndirectInputV2
{
    asset::SBufferBinding<video::IGPUBuffer> vtxBuffer;
    asset::E_PRIMITIVE_TOPOLOGY mode = asset::EPT_TRIANGLE_LIST;
    asset::E_INDEX_TYPE indexType = asset::EIT_16BIT;
    core::smart_refctd_ptr<video::IGPUBuffer> idxBuffer = nullptr;
    core::smart_refctd_ptr<video::IGPUBuffer> indirectDrawBuff = nullptr;
    size_t offset = 0u;
    size_t maxCount = 0u;
    size_t stride = 0u;
    core::smart_refctd_ptr<video::IGPUBuffer> countBuffer = nullptr;
    size_t countOffset = 0u;
};

void packMeshBuffers(video::IVideoDriver* driver, core::vector<ICPUMeshBuffer*>& meshBuffers, SVertexInputParams& vipOutput, DrawIndexedIndirectInput& output)
{
    using MeshPacker = CCPUMeshPackerV1<DrawElementsIndirectCommand_t>;

    MeshPacker::PackerDataStore packedMeshBuffer;
    //core::smart_refctd_ptr<IGPUBuffer> gpuIndirectDrawBuffer;

    MeshPacker::AllocationParams allocParams;
    allocParams.indexBuffSupportedCnt = 20000000u;
    allocParams.indexBufferMinAllocSize = 5000u;
    allocParams.vertexBuffSupportedSize = 20000000u;
    allocParams.vertexBufferMinAllocSize = 5000u;
    allocParams.MDIDataBuffSupportedCnt = 20000u;
    allocParams.MDIDataBuffMinAllocSize = 1u; //so structs are adjacent in memory
    allocParams.perInstanceVertexBuffSupportedSize = 0u;
    allocParams.perInstanceVertexBufferMinAllocSize = 0u;

    CCPUMeshPackerV1 mp(meshBuffers[0]->getPipeline()->getVertexInputParams(), allocParams, std::numeric_limits<uint16_t>::max() / 3u, std::numeric_limits<uint16_t>::max() / 3u);

    //TODO: test for multiple alloc
    //TODO: test mp.getPackerCreationParamsFromMeshBufferRange()
    MeshPacker::ReservedAllocationMeshBuffers ramb = mp.alloc(meshBuffers.begin(), meshBuffers.end());
    assert(ramb.isValid());

    mp.instantiateDataStorage();

    IMeshPackerBase::PackedMeshBufferData pmbd =  mp.commit(meshBuffers.begin(), meshBuffers.end(), ramb);
    assert(pmbd.isValid());

    MeshPacker::PackerDataStore pmb = mp.getPackerDataStore();
    assert(pmb.isValid());

    auto& cpuVtxBuff = pmb.vertexBufferBindings[0].buffer;
    auto gpuVtxBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(cpuVtxBuff->getSize(), cpuVtxBuff->getPointer());

    for (uint32_t i = 0u; i < video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; i++)
        output.vtxBindings[i] = { pmb.vertexBufferBindings[i].offset, gpuVtxBuff };
    output.indexBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(pmb.indexBuffer.buffer->getSize(), pmb.indexBuffer.buffer->getPointer());
    output.indirectDrawBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(pmb.MDIDataBuffer->getSize(), pmb.MDIDataBuffer->getPointer());
    output.maxCount = pmbd.mdiParameterCount;

    vipOutput = pmb.vertexInputParams;
}

void packMeshBuffersV2(video::IVideoDriver* driver, core::vector<ICPUMeshBuffer*>& meshBuffers, DrawIndexedIndirectInputV2& output)
{
    using MeshPacker = CCPUMeshPackerV2<DrawElementsIndirectCommand_t>;

    MeshPacker::AllocationParams allocParams;
    allocParams.indexBuffSupportedCnt = 20000000u;
    allocParams.indexBufferMinAllocSize = 5000u;
    allocParams.vertexBuffSupportedSize = 20000000u;
    allocParams.vertexBufferMinAllocSize = 5000u;
    allocParams.MDIDataBuffSupportedCnt = 20000u;
    allocParams.MDIDataBuffMinAllocSize = 1u; //so structs are adjacent in memory

    CCPUMeshPackerV2 mp(allocParams, std::numeric_limits<uint16_t>::max() / 3u, std::numeric_limits<uint16_t>::max() / 3u);

    MeshPacker::ReservedAllocationMeshBuffers allocData[2u];

    bool allocSuccessfull = mp.alloc(allocData, meshBuffers.begin(), meshBuffers.begin() + 2u);
    if (!allocSuccessfull)
        std::cout << "Alloc failed \n";

    mp.instantiateDataStorage();
    MeshPacker::PackerDataStore<IGPUBuffer> packerDataStore = mp.createGPUPackerDataStore(driver);

    IMeshPackerBase::PackedMeshBufferData pmbd[2];

    mp.commit(pmbd, allocData, meshBuffers.begin(), meshBuffers.begin() + 2u);

    output.vtxBuffer = { 0ull, packerDataStore.vertexBuffer };
    output.idxBuffer = packerDataStore.indexBuffer;
    output.indirectDrawBuff = packerDataStore.MDIDataBuffer;

    //create offset tables
}

void setPipeline(IVideoDriver* driver, IAssetManager* am, SVertexInputParams& vtxInputParams,
    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>& gpuPipeline)
{
    IAssetLoader::SAssetLoadParams lp;

    auto vertexShaderBundle = am->getAsset("../shader.vert", lp);
    auto fragShaderBundle = am->getAsset("../shader.frag", lp);
    ICPUSpecializedShader* shaders[2] =
    {
        IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundle.getContents().begin()->get()),
        IAsset::castDown<ICPUSpecializedShader>(fragShaderBundle.getContents().begin()->get())
    };

    {
        auto gpuShaders = driver->getGPUObjectsFromAssets(shaders, shaders + 2);
        IGPUSpecializedShader* shaders[2] = { gpuShaders->operator[](0).get(), gpuShaders->operator[](1).get() };

        asset::SPushConstantRange pcRange = { asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD) };
        auto pipelineLayout = driver->createGPUPipelineLayout(&pcRange, &pcRange + 1);

        gpuPipeline = driver->createGPURenderpassIndependentPipeline(
            nullptr, core::smart_refctd_ptr(pipelineLayout),
            shaders, shaders + 2u,
            vtxInputParams,
            asset::SBlendParams(), asset::SPrimitiveAssemblyParams(), SRasterizationParams());
    }
}

void setPipelineV2(IVideoDriver* driver, ICPUSpecializedShader* vs, ICPUSpecializedShader* fs,
    core::smart_refctd_ptr<IGPUBuffer>& vtxBuffer, std::array<SBufferRange<IGPUBuffer>, 2>& dataOffsetBuffers,
    core::smart_refctd_ptr<IGPUDescriptorSet>& outputGPUDescriptorSet,
    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>& outputGpuPipeline)
{
    ICPUSpecializedShader* cpuShaders[2] = { vs, fs };
    auto gpuShaders = driver->getGPUObjectsFromAssets(cpuShaders, cpuShaders + 2);

    core::smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
    {
        IGPUDescriptorSetLayout::SBinding b[5];
        b[0].binding = 0u; b[1].binding = 1u; b[2].binding = 2u; b[3].binding = 3u; b[4].binding = 4u;
        b[0].type = b[1].type = b[2].type = EDT_UNIFORM_TEXEL_BUFFER;
        b[3].type = b[4].type = EDT_STORAGE_BUFFER;
        b[0].stageFlags = b[1].stageFlags = b[2].stageFlags = b[3].stageFlags = b[4].stageFlags = ISpecializedShader::ESS_VERTEX;
        b[0].count = b[1].count = b[2].count = b[3].count = b[4].count = 1u;
        dsLayout = driver->createGPUDescriptorSetLayout(b, b + 5u);
    }
    
    asset::SPushConstantRange pcRange = { asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(asset::SBasicViewParameters) };
    auto pipelineLayout = driver->createGPUPipelineLayout(&pcRange, &pcRange + 1, core::smart_refctd_ptr(dsLayout));

    outputGPUDescriptorSet = driver->createGPUDescriptorSet(std::move(dsLayout));
    {
        IGPUDescriptorSet::SWriteDescriptorSet w[5];
        w[0].arrayElement = w[1].arrayElement = w[2].arrayElement = w[3].arrayElement = w[4].arrayElement = 0u;
        w[0].count = w[1].count = w[2].count = w[3].count = w[4].count = 1u;
        w[0].binding = 0u; w[1].binding = 1u; w[2].binding = 2u; w[3].binding = 3u; w[4].binding = 4u;
        w[0].descriptorType = w[1].descriptorType = w[2].descriptorType = EDT_UNIFORM_TEXEL_BUFFER;
        w[3].descriptorType = w[4].descriptorType = EDT_STORAGE_BUFFER;
        w[0].dstSet = w[1].dstSet = w[2].dstSet = w[3].dstSet = w[4].dstSet = outputGPUDescriptorSet.get();

        IGPUDescriptorSet::SDescriptorInfo info[3];
        info[0].buffer.offset = 0u;
        info[0].buffer.size = vtxBuffer->getSize();
        info[0].desc = core::smart_refctd_ptr(vtxBuffer);

        w[0].info = w[1].info = w[2].info = &info[0];

        info[1].buffer.offset = dataOffsetBuffers[0].offset;
        info[1].buffer.size = dataOffsetBuffers[0].size;
        w[3].info = &info[1];

        info[2].buffer.offset = dataOffsetBuffers[1].offset;
        info[2].buffer.size = dataOffsetBuffers[1].size;
        w[4].info = &info[2];

        driver->updateDescriptorSets(5u, w, 0u, nullptr);
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
    assert(!meshes_bundle.isEmpty());
    auto mesh = meshes_bundle.getContents().begin()[0];
    auto mesh_raw = static_cast<asset::ICPUMesh*>(mesh.get());

    //saving cache to file
    qnc->saveCacheToFile(asset::CQuantNormalCache::E_CACHE_TYPE::ECT_2_10_10_10, fs, "../../tmp/normalCache101010.sse");

    core::vector<ICPUMeshBuffer*> meshBuffers;
    for (uint32_t i = 0u; i < mesh_raw->getMeshBufferCount(); i++)
        meshBuffers.push_back(mesh_raw->getMeshBuffer(i));

    //pack mesh buffers
    DrawIndexedIndirectInput mdiCallParams;
    SVertexInputParams vtxInputParams;
    packMeshBuffers(driver, meshBuffers, vtxInputParams, mdiCallParams);

    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuPipeline;
    setPipeline(driver, am, vtxInputParams, gpuPipeline);

    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuPipeline2;
    core::smart_refctd_ptr<IGPUDescriptorSet> ds;
    DrawIndexedIndirectInputV2 mdiCallParamsV2;
    {
        auto* pipeline = meshBuffers[0]->getPipeline();

        auto* vtxShader = pipeline->getShaderAtIndex(asset::ICPURenderpassIndependentPipeline::ESSI_VERTEX_SHADER_IX);
        core::smart_refctd_ptr<ICPUSpecializedShader> vs = createModifiedVertexShader(vtxShader);
        ICPUSpecializedShader* fs = pipeline->getShaderAtIndex(asset::ICPURenderpassIndependentPipeline::ESSI_FRAGMENT_SHADER_IX);

        packMeshBuffersV2(driver, meshBuffers, mdiCallParamsV2);

        smart_refctd_ptr<IGPUBuffer> tmpVtxBuff;
        std::array<SBufferRange<IGPUBuffer>, 2> tmpSSBO;

        setPipelineV2(driver, vs.get(), fs, tmpVtxBuff, tmpSSBO, ds ,gpuPipeline2);
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
        driver->bindGraphicsPipeline(gpuPipeline.get());

        driver->beginScene(true, true, video::SColor(255, 0, 0, 255));

        //! This animates (moves) the camera and sets the transforms
        camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
        camera->render();

        driver->pushConstants(gpuPipeline->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), camera->getConcatenatedMatrix().pointer());
        driver->drawIndexedIndirect(mdiCallParams.vtxBindings, mdiCallParams.mode, mdiCallParams.indexType, mdiCallParams.indexBuff.get(), mdiCallParams.indirectDrawBuff.get(), mdiCallParams.offset, mdiCallParams.maxCount, mdiCallParams.stride);

        driver->endScene();
    }
}