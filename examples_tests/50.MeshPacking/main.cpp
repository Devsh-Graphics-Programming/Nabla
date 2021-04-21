// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

//TESTS:

//visualise results of free methods for both v1 and v2 mesh packers
//see if random batch division works for mpv1
//test for every primitive type calcMDIStructCnt (mpv2)

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

//! I advise to check out this file, its a basic input handler
#include "../common/QToQuitEventReceiver.h"
#include "nbl/asset/utils/CCPUMeshPackerV1.h"
#include "nbl/asset/CCPUMeshPackerV2.h"
#include "nbl/video/CGPUMeshPackerV2.h"

using namespace nbl;
using namespace core;
using namespace asset;
using namespace video;

constexpr const char* VERTEX_SHADER_OVERRIDES =
R"(
#define _NBL_VERT_INPUTS_DEFINED_

layout(location = 4) flat out uint drawID;

#define _NBL_VG_DESCRIPTOR_SET 0

#define _NBL_VG_FLOAT_BUFFERS
#define _NBL_VG_FLOAT_BUFFERS_BINDING 0 
//TODO: %u
#define _NBL_VG_FLOAT_BUFFERS_COUNT 2

//#define _NBL_VG_INT_BUFFERS
//#define _NBL_VG_INT_BUFFERS_BINDING 1
//#define _NBL_VG_INT_BUFFERS_COUNT 0

#define _NBL_VG_UINT_BUFFERS
#define _NBL_VG_UINT_BUFFERS_BINDING 2
#define _NBL_VG_UINT_BUFFERS_COUNT 1

#include <nbl/builtin/glsl/virtual_geometry/virtual_attribute_fetch.glsl>

layout(set = 3, binding = 0) readonly buffer VirtualAttributes
{
    nbl_glsl_VG_VirtualAttributePacked_t vAttr[][3];
} virtualAttribTable;

layout (push_constant) uniform Block 
{
    uint dataBufferOffset;
} pc;

vec3 nbl_glsl_fetchVtxPos(in uint vtxID, in uint drawID)
{
    nbl_glsl_VG_VirtualAttributePacked_t va = virtualAttribTable.vAttr[drawID + pc.dataBufferOffset][0];
    return nbl_glsl_VG_vertexFetch_RGB32_SFLOAT(va, vtxID);
}

vec2 nbl_glsl_fetchVtxUV(in uint vtxID, in uint drawID)
{
    nbl_glsl_VG_VirtualAttributePacked_t va = virtualAttribTable.vAttr[drawID + pc.dataBufferOffset][1];
    return nbl_glsl_VG_vertexFetch_RG32_SFLOAT(va, vtxID);
}

vec3 nbl_glsl_fetchVtxNormal(in uint vtxID, in uint drawID)
{
    nbl_glsl_VG_VirtualAttributePacked_t va = virtualAttribTable.vAttr[drawID + pc.dataBufferOffset][2];
    return nbl_glsl_VG_vertexFetch_RGB10A2_SNORM(va, vtxID).xyz;
}

#define _NBL_BASIC_VTX_ATTRIB_FETCH_FUCTIONS_DEFINED_
#define _NBL_POS_FETCH_FUNCTION_DEFINED
#define _NBL_UV_FETCH_FUNCTION_DEFINED
#define _NBL_NORMAL_FETCH_FUNCTION_DEFINED

)";

struct DataOffsetTable
{
    uint32_t binding;
    asset::SBufferBinding<IGPUBuffer> offsetBuffer;
};

core::smart_refctd_ptr<asset::ICPUSpecializedShader> createModifiedVertexShader(const asset::ICPUSpecializedShader* _fs)
{
    /*const asset::ICPUShader* unspec = _fs->getUnspecialized();
    assert(unspec->containsGLSL());

    auto begin = reinterpret_cast<const char*>(unspec->getSPVorGLSL()->getPointer());
    auto end = begin + unspec->getSPVorGLSL()->getSize();
    std::string resultShaderSrc(begin, end);

    size_t firstNewlineAfterVersion = resultShaderSrc.find("\n", resultShaderSrc.find("#version "));

    const std::string buffersDef = SHADER_OVERRIDES[0];
    const std::string mainDef = SHADER_OVERRIDES[1];

    resultShaderSrc.insert(firstNewlineAfterVersion, buffersDef);
    resultShaderSrc += mainDef;
    resultShaderSrc.replace(resultShaderSrc.find("#version 430 core"), sizeof("#version 430 core"), "#version 460 core\n");

    auto unspecNew = core::make_smart_refctd_ptr<asset::ICPUShader>(resultShaderSrc.c_str());
    auto specinfo = _fs->getSpecializationInfo();
    auto vsNew = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecNew), std::move(specinfo));

    return vsNew;*/

    return nullptr;
}

struct DrawIndexedIndirectInput
{
    asset::SBufferBinding<video::IGPUBuffer> vtxBindings[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
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

struct DrawIndexedIndirectInputV2
{
    asset::SBufferBinding<video::IGPUBuffer> vtxBuffer;
    static constexpr asset::E_PRIMITIVE_TOPOLOGY mode = asset::EPT_TRIANGLE_LIST;
    static constexpr asset::E_INDEX_TYPE indexType = asset::EIT_16BIT;
    core::smart_refctd_ptr<video::IGPUBuffer> idxBuffer = nullptr;
    core::smart_refctd_ptr<video::IGPUBuffer> indirectDrawBuff = nullptr;
    size_t offset = 0u;
    size_t maxCount = 0u;
    size_t stride = 0u;
    core::smart_refctd_ptr<video::IGPUBuffer> countBuffer = nullptr;
    size_t countOffset = 0u;

    core::smart_refctd_ptr<video::IGPUBuffer> virtualAttribTable = nullptr;
    std::array<core::smart_refctd_ptr<IGPUDescriptorSet>, 2> ds;
};

void packMeshBuffers(video::IVideoDriver* driver, core::vector<ICPUMeshBuffer*>& meshBuffers, SVertexInputParams& vipOutput, DrawIndexedIndirectInput& output)
{
    using MeshPacker = CCPUMeshPackerV1<DrawElementsIndirectCommand_t>;

    MeshPacker::PackerDataStore packedMeshBuffer;
    //core::smart_refctd_ptr<IGPUBuffer> gpuIndirectDrawBuffer;

    MeshPacker::AllocationParams allocParams;
    allocParams.indexBuffSupportedCnt = 20000000u;
    allocParams.indexBufferMinAllocCnt = 5000u;
    allocParams.vertexBuffSupportedByteSize = 2147483648u; //2GB
    allocParams.vertexBufferMinAllocByteSize = 5000u;
    allocParams.MDIDataBuffSupportedCnt = 20000u;
    allocParams.MDIDataBuffMinAllocCnt = 1u; //so structs are adjacent in memory //TODO: increase this value and delete this comment from all examples
    allocParams.perInstanceVertexBufferMinAllocByteSize = 2147483648u / 4u; // 0.5GB
    allocParams.perInstanceVertexBufferMinAllocByteSize = 2000u; // 0.5GB

    assert(!meshBuffers.empty());

    CCPUMeshPackerV1 mp((*(meshBuffers.end() - 1u))->getPipeline()->getVertexInputParams(), allocParams, 256u, 256u/*std::numeric_limits<uint16_t>::max() / 3u, std::numeric_limits<uint16_t>::max() / 3u*/);

    //TODO: test for multiple alloc
    //TODO: test mp.getPackerCreationParamsFromMeshBufferRange()
    MeshPacker::ReservedAllocationMeshBuffers ramb = mp.alloc(meshBuffers.begin(), meshBuffers.end());
    assert(ramb.isValid());

    //mp.free(ramb);

    mp.shrinkOutputBuffersSize();
    mp.instantiateDataStorage();

    IMeshPackerBase::PackedMeshBufferData pmbd =  mp.commit(meshBuffers.begin(), meshBuffers.end(), ramb);
    assert(pmbd.isValid());

    MeshPacker::PackerDataStore pmb = mp.getPackerDataStore();
    assert(pmb.isValid());
    
    auto& cpuVtxBuff = pmb.vertexBufferBindings[0].buffer;
    auto& cpuPerInsVtxBuffer = pmb.vertexBufferBindings[15].buffer;
    auto gpuVtxBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(cpuVtxBuff->getSize(), cpuVtxBuff->getPointer());
    auto gpuPerInsVtxBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(cpuPerInsVtxBuffer->getSize(), cpuPerInsVtxBuffer->getPointer());

    for (uint32_t i = 0u; i < video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT - 1; i++)
        output.vtxBindings[i] = { pmb.vertexBufferBindings[i].offset, gpuVtxBuff };

    output.vtxBindings[15] = { pmb.vertexBufferBindings[15].offset,  gpuPerInsVtxBuff };

    output.idxBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(pmb.indexBuffer.buffer->getSize(), pmb.indexBuffer.buffer->getPointer());
    output.indirectDrawBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(pmb.MDIDataBuffer->getSize(), pmb.MDIDataBuffer->getPointer());
    output.maxCount = pmbd.mdiParameterCount;

    vipOutput = pmb.vertexInputParams;
}

using MeshPackerV2 = CCPUMeshPackerV2<DrawElementsIndirectCommand_t>;
using GPUMeshPackerV2 = CGPUMeshPackerV2<DrawElementsIndirectCommand_t>;
GPUMeshPackerV2 packMeshBuffersV2(video::IVideoDriver* driver, core::vector<ICPUMeshBuffer*>& meshBuffers, DrawIndexedIndirectInputV2& drawData)
{
    //constexpr uint16_t minTrisBatch = 256u;
    //constexpr uint16_t maxTrisBatch = 256u;

    constexpr uint16_t minTrisBatch = std::numeric_limits<uint16_t>::max() / 3;
    constexpr uint16_t maxTrisBatch = std::numeric_limits<uint16_t>::max() / 3;

    MeshPackerV2::AllocationParams allocParams;
    allocParams.indexBuffSupportedCnt = 32u * 1024u * 1024u;
    allocParams.indexBufferMinAllocCnt = minTrisBatch * 3u;
    allocParams.vertexBuffSupportedByteSize = 128u * 1024u * 1024u;
    allocParams.vertexBufferMinAllocByteSize = minTrisBatch;
    allocParams.MDIDataBuffSupportedCnt = 8192u;
    allocParams.MDIDataBuffMinAllocCnt = 1u; //so structs are adjacent in memory (TODO: WTF NO!)

    CCPUMeshPackerV2 mp(allocParams, minTrisBatch, maxTrisBatch);

    const uint32_t mdiCntTotal = mp.calcMDIStructCount(meshBuffers.begin(), meshBuffers.end());
    const uint32_t meshBufferCnt = meshBuffers.size();
    auto allocData = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<MeshPackerV2::ReservedAllocationMeshBuffers>>(meshBufferCnt);

    const uint32_t offsetTableSz = mdiCntTotal * 4u;
    core::vector<MeshPackerV2::VirtualAttribute> offsetTableLocal;
    offsetTableLocal.reserve(offsetTableSz);

    const uint32_t mdiCnt = mp.calcMDIStructCount(meshBuffers.begin(), meshBuffers.end());
    bool allocSuccessfull = mp.alloc(allocData->data(), meshBuffers.begin(), meshBuffers.end());
    if (!allocSuccessfull)
    {
        std::cout << "Alloc failed \n";
        _NBL_DEBUG_BREAK_IF(true);
    }

    mp.shrinkOutputBuffersSize();
    mp.instantiateDataStorage();
    MeshPackerV2::PackerDataStore packerDataStore = mp.getPackerDataStore();

    core::vector<IMeshPackerBase::PackedMeshBufferData> pmbd(mdiCnt);

    core::vector<MeshPackerV2::CombinedDataOffsetTable> cdot(mdiCnt);

    bool commitSuccessfull = mp.commit(pmbd.data(), cdot.data(), allocData->data(), meshBuffers.begin(), meshBuffers.end());
    if (!commitSuccessfull)
    {
        std::cout << "Commit failed \n";
        _NBL_DEBUG_BREAK_IF(true);
    }

    drawData.maxCount = mdiCnt;
    drawData.offset = pmbd[0].mdiParameterOffset * sizeof(DrawElementsIndirectCommand_t);

    //setOffsetTables
    for (uint32_t j = 0u; j < mdiCnt; j++)
    {
        MeshPackerV2::CombinedDataOffsetTable& virtualAttribTable = cdot[j];

        offsetTableLocal.push_back(virtualAttribTable.attribInfo[0]);
        offsetTableLocal.push_back(virtualAttribTable.attribInfo[2]);
        offsetTableLocal.push_back(virtualAttribTable.attribInfo[3]);
        //offsetTableLocal.push_back(virtualAttribTable.attribInfo[15]);
    }

    /*DrawElementsIndirectCommand_t* mdiPtr = static_cast<DrawElementsIndirectCommand_t*>(packerDataStore.MDIDataBuffer->getPointer()) + 99u;
    uint16_t* idxBuffPtr = static_cast<uint16_t*>(packerDataStore.indexBuffer->getPointer());
    float* vtxBuffPtr = static_cast<float*>(packerDataStore.vertexBuffer->getPointer());

    for (uint32_t i = 0u; i < 264; i++)
    {
        float* firstCoord = vtxBuffPtr + ((*(idxBuffPtr + i) + cdot[0].attribInfo[0].offset) * 3u);
        std::cout << "vtx: " << i << " idx: " << *(idxBuffPtr + i) << "    ";
        std::cout << *firstCoord << ' ' << *(firstCoord + 1u) << ' ' << *(firstCoord + 2u) << std::endl;
    }*/

    drawData.vtxBuffer = { 0ull, driver->createFilledDeviceLocalGPUBufferOnDedMem(packerDataStore.vertexBuffer->getSize(), packerDataStore.vertexBuffer->getPointer()) };
    drawData.idxBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(packerDataStore.indexBuffer->getSize(), packerDataStore.indexBuffer->getPointer());
    drawData.indirectDrawBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(packerDataStore.MDIDataBuffer->getSize(), packerDataStore.MDIDataBuffer->getPointer());

    drawData.virtualAttribTable = driver->createFilledDeviceLocalGPUBufferOnDedMem(offsetTableLocal.size() * sizeof(MeshPackerV2::VirtualAttribute), offsetTableLocal.data());

    return GPUMeshPackerV2(driver, std::move(mp));
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
    DrawIndexedIndirectInputV2& drawData,
    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>& outputGpuPipeline,
    GPUMeshPackerV2& mp)
{
    ICPUSpecializedShader* cpuShaders[2] = { vs, fs };
    auto gpuShaders = driver->getGPUObjectsFromAssets(cpuShaders, cpuShaders + 2);

    //create bindings
    const uint32_t bindingsCnt = mp.getDSlayoutBindings(nullptr);
    auto bindings = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IGPUDescriptorSetLayout::SBinding>>(bindingsCnt);

    mp.getDSlayoutBindings(bindings->data());

    for (auto it = bindings->begin(); it != bindings->end(); it++)
        it->stageFlags = ISpecializedShader::ESS_VERTEX;

    core::smart_refctd_ptr<IGPUDescriptorSetLayout> ds0Layout;
    core::smart_refctd_ptr<IGPUDescriptorSetLayout> ds1Layout;

    ds0Layout = driver->createGPUDescriptorSetLayout(bindings->data(), bindings->data() + bindings->size());

    IGPUDescriptorSetLayout::SBinding b;
    b.binding = 0u;
    b.count = 1u;
    b.samplers = nullptr;
    b.stageFlags = ISpecializedShader::ESS_VERTEX;
    b.type = EDT_STORAGE_BUFFER;

    ds1Layout = driver->createGPUDescriptorSetLayout(&b, &b + 1);

    asset::SPushConstantRange pcRange = { asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD) };
    auto pipelineLayout = driver->createGPUPipelineLayout(&pcRange, &pcRange + 1, core::smart_refctd_ptr(ds0Layout), core::smart_refctd_ptr(ds1Layout));

    //create ds
    drawData.ds[0] = driver->createGPUDescriptorSet(std::move(ds0Layout));
    drawData.ds[1] = driver->createGPUDescriptorSet(std::move(ds1Layout));

    auto sizes = mp.getDescriptorSetWrites(nullptr, nullptr, nullptr);
    auto writes = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IGPUDescriptorSet::SWriteDescriptorSet>>(sizes.first);
    auto info = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IGPUDescriptorSet::SDescriptorInfo>>(sizes.second);

    mp.getDescriptorSetWrites(writes->data(), info->data(), drawData.ds[0].get());
    driver->updateDescriptorSets(writes->size(), writes->data(), 0u, nullptr);

    IGPUDescriptorSet::SWriteDescriptorSet w;
    IGPUDescriptorSet::SDescriptorInfo i;
    w.binding = 0u;
    w.arrayElement = 0u;
    w.count = 1u;
    w.descriptorType = EDT_STORAGE_BUFFER;
    w.dstSet = drawData.ds[1].get();
    w.info = &i;

    i.buffer.size = drawData.virtualAttribTable->getSize();
    i.buffer.offset = 0u;
    i.desc = std::move(drawData.virtualAttribTable);
    drawData.virtualAttribTable = nullptr;

    driver->updateDescriptorSets(1u, &w, 0u, nullptr);

    //create pipeline
    IGPUSpecializedShader* shaders[2] = { gpuShaders->operator[](0).get(), gpuShaders->operator[](1).get() };

    outputGpuPipeline = driver->createGPURenderpassIndependentPipeline(
        nullptr, std::move(pipelineLayout),
        shaders, shaders + 2u,
        SVertexInputParams(),
        asset::SBlendParams(), asset::SPrimitiveAssemblyParams(), SRasterizationParams());
}

core::smart_refctd_ptr<ICPUMeshBuffer> createInstancedMeshBuffer(IGeometryCreator const* geometryCreator)
{
    auto cylinder = geometryCreator->createCylinderMesh(10.0f,10.0f, 32u);

    //create per instance attribute (position)
    constexpr uint32_t insCnt = 10u;
    constexpr size_t perInsAttribSize = insCnt * sizeof(vector3df);

    core::smart_refctd_ptr<ICPUBuffer> perInsAttribBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(perInsAttribSize);
    auto* perInsAttribBufferBegin = static_cast<vector3df*>(perInsAttribBuffer->getPointer());
    auto* perInsAttribBufferEnd = perInsAttribBufferBegin + insCnt;
    std::generate(perInsAttribBufferBegin, perInsAttribBufferEnd, [n = 10u]() mutable { return vector3df(n += 10); });

    //cylinder.inputParams.enabledAttribFlags = 0x800D;
    //cylinder.inputParams.enabledBindingFlags |= 0x8000;
    cylinder.inputParams.enabledAttribFlags = 0x000D;

    cylinder.inputParams.attributes[1] = SVertexInputAttribParams(); //ignore color attribute

    cylinder.bindings[15] = { 0ull, std::move(perInsAttribBuffer) };

    cylinder.inputParams.attributes[15].binding = 15;
    cylinder.inputParams.attributes[15].format = asset::EF_R32G32B32_SFLOAT;
    cylinder.inputParams.attributes[15].relativeOffset = 0u;

    cylinder.inputParams.bindings[15].inputRate = EVIR_PER_INSTANCE;
    cylinder.inputParams.bindings[15].stride = 0u;

    SRasterizationParams rasterizationParams;
    rasterizationParams.faceCullingMode = EFCM_NONE;

    core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> pipeline = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(
        nullptr,
        nullptr,
        nullptr,
        cylinder.inputParams,
        SBlendParams(),
        SPrimitiveAssemblyParams(),
        rasterizationParams
        );

    core::smart_refctd_ptr<ICPUMeshBuffer> output = core::make_smart_refctd_ptr<ICPUMeshBuffer>(
        std::move(pipeline),
        nullptr,
        cylinder.bindings,
        std::move(cylinder.indexBuffer)
        );

    output->setInstanceCount(insCnt);
    output->setIndexCount(cylinder.indexCount);
    output->setIndexType(cylinder.indexType);

    return output;
}

core::smart_refctd_ptr<ICPUMeshBuffer> createTriangleFanMeshBuffer(IGeometryCreator const* geometryCreator)
{
    auto disk = geometryCreator->createDiskMesh(10.0f, 128u);

    disk.inputParams.enabledAttribFlags = 0b1; //only position;

    core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> pipeline = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(
        nullptr,
        nullptr,
        nullptr,
        disk.inputParams,
        SBlendParams(),
        disk.assemblyParams,
        SRasterizationParams()
        );

    core::smart_refctd_ptr<ICPUMeshBuffer> output = core::make_smart_refctd_ptr<ICPUMeshBuffer>(
        std::move(pipeline),
        nullptr,
        disk.bindings,
        std::move(disk.indexBuffer)
        );

    output->setInstanceCount(1);
    output->setIndexCount(disk.indexCount);
    output->setIndexType(disk.indexType);

    return output;
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
    qnc->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fs, "../../tmp/normalCache101010.sse", true);

    // register the zip
    //fs->addFileArchive("../../media/sponza.zip");

    asset::IAssetLoader::SAssetLoadParams lp;
    //auto meshes_bundle = am->getAsset("sponza.obj", lp);
    //assert(!meshes_bundle.isEmpty());
    //auto mesh = meshes_bundle.getContents().begin()[0];
    //auto mesh_raw = static_cast<asset::ICPUMesh*>(mesh.get());

    //saving cache to file
    //qnc->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fs, "../../tmp/normalCache101010.sse");

    //TODO: change it to vector of smart pointers
    core::vector<ICPUMeshBuffer*> meshBuffers;
    //for (uint32_t i = 0u; i < mesh_raw->getMeshBufferVector().size(); i++)
    //    meshBuffers.push_back(mesh_raw->getMeshBufferVector()[i].get());

    auto instancedMeshBuffer = createInstancedMeshBuffer(am->getGeometryCreator());
    meshBuffers.push_back(instancedMeshBuffer.get());

    /*auto diskMeshBuffer = createTriangleFanMeshBuffer(am->getGeometryCreator());
    meshBuffers.push_back(diskMeshBuffer.get());*/

    //pack mesh buffers
    /*DrawIndexedIndirectInput mdiCallParams;
    SVertexInputParams vtxInputParams;
    packMeshBuffers(driver, meshBuffers, vtxInputParams, mdiCallParams);

    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuPipeline;
    setPipeline(driver, am, vtxInputParams, gpuPipeline);*/

    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuPipeline2;
    DrawIndexedIndirectInputV2 mdiCallParamsV2;
    {
        auto* pipeline = meshBuffers[0]->getPipeline();

        //auto* vtxShader = pipeline->getShaderAtIndex(asset::ICPURenderpassIndependentPipeline::ESSI_VERTEX_SHADER_IX);
        //core::smart_refctd_ptr<ICPUSpecializedShader> vs = createModifiedVertexShader(vtxShader);
        ICPUSpecializedShader* vs = IAsset::castDown<ICPUSpecializedShader>(am->getAsset("../shaderV2.vert", lp).getContents().begin()->get());
        ICPUSpecializedShader* fs = IAsset::castDown<ICPUSpecializedShader>(am->getAsset("../shader.frag", lp).getContents().begin()->get());

        GPUMeshPackerV2 mp = packMeshBuffersV2(driver, meshBuffers, mdiCallParamsV2);

        setPipelineV2(driver, vs, fs, mdiCallParamsV2, gpuPipeline2, mp);
    }

    //! we want to move around the scene and view it from different angles
    scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0, 100.0f, 0.5f);

    camera->setPosition(core::vector3df(-4, 0, 0));
    camera->setTarget(core::vector3df(0, 0, 0));
    camera->setNearValue(1.f);
    camera->setFarValue(5000.0f);

    smgr->setActiveCamera(camera);

    uint64_t lastFPSTime = 0;

#define USE_MPV2

#ifdef USE_MPV1
    while (device->run() && receiver.keepOpen())
    {
        driver->bindGraphicsPipeline(gpuPipeline.get());
        driver->bindDescriptorSets(EPBP_GRAPHICS, gpuPipeline->getLayout(), 0u, 0u, nullptr, nullptr);

        driver->beginScene(true, true, video::SColor(255, 0, 0, 255));

        //! This animates (moves) the camera and sets the transforms
        camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
        camera->render();


        driver->pushConstants(gpuPipeline->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), camera->getConcatenatedMatrix().pointer());
        driver->drawIndexedIndirect(mdiCallParams.vtxBindings, mdiCallParams.mode, mdiCallParams.indexType, mdiCallParams.idxBuff.get(), mdiCallParams.indirectDrawBuff.get(), mdiCallParams.offset, mdiCallParams.maxCount, mdiCallParams.stride);

        driver->endScene();
    }
#endif
#ifdef USE_MPV2
    const video::IGPUDescriptorSet* ds[]{ mdiCallParamsV2.ds[0].get(), mdiCallParamsV2.ds[1].get() };

    while (device->run() && receiver.keepOpen())
    {
        driver->bindGraphicsPipeline(gpuPipeline2.get());
        driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuPipeline2->getLayout(), 0u, 2u, ds, nullptr);

        driver->beginScene(true, true, video::SColor(255, 0, 0, 255));

        //! This animates (moves) the camera and sets the transforms
        camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
        camera->render();

        driver->pushConstants(gpuPipeline2->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), camera->getConcatenatedMatrix().pointer());
        SBufferBinding<IGPUBuffer> vtxBufferBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
        vtxBufferBindings[0] = mdiCallParamsV2.vtxBuffer;
        driver->drawIndexedIndirect(vtxBufferBindings, mdiCallParamsV2.mode, mdiCallParamsV2.indexType, mdiCallParamsV2.idxBuffer.get(), mdiCallParamsV2.indirectDrawBuff.get(), mdiCallParamsV2.offset, mdiCallParamsV2.maxCount, mdiCallParamsV2.stride);

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
#endif
}