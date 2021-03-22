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

constexpr const char* SHADER_OVERRIDES[2] =
{
R"(
#define _NBL_VERT_INPUTS_DEFINED_
#define _NBL_VERT_MAIN_DEFINED_
#define _NBL_VERT_OUTPUTS_DEFINED_
#define _NBL_VERT_SET1_BINDINGS_DEFINED_

vec4 nbl_glsl_decodeRGB10A2(in uint x)
{
	const uvec3 rgbMask = uvec3(0x3ffu);
	const uvec4 shifted = uvec4(x,uvec3(x)>>uvec3(10,20,30));
	return vec4(vec3(shifted.rgb&rgbMask),shifted.a)/vec4(vec3(rgbMask),3.0);
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

layout(set = 0, binding = 4) readonly buffer VertexNormalOffsetTable
{
    int dataOffsetTable[];
} vertexNormal;

layout(push_constant, row_major) uniform PushConstants
{
	mat4 vp;
} pc;

//outputs
layout(location = 0) out vec3 normal;

)",

R"(
void main()
{
    int vtxPosOffset = int(gl_VertexIndex) + vertexPosition.dataOffsetTable[gl_DrawID];
    vec3 pos = texelFetch(MeshPackedData_R32G32B32_SFLOAT, vtxPosOffset).xyz;
    gl_Position = nbl_glsl_pseudoMul4x4with3x1(pc.vp, pos);
    
    int vtxNormOffset = int(gl_VertexIndex) + vertexNormal.dataOffsetTable[gl_DrawID];
    normal = normalize(nbl_glsl_decodeRGB10A2(texelFetch(MeshPackedData_A2B10G10R10_SNORM_PACK32, vtxNormOffset).x).xyz);
    
    //vertex outputs
    //LocalPos = pos;
    //ViewPos = nbl_glsl_pseudoMul3x4with3x1(CamData.params.MV, pos);
    
    //mat3 normalMat = nbl_glsl_SBasicViewParameters_GetNormalMat(CamData.params.NormalMatAndEyePos);

    //vec3 normal = texelFetch(MeshPackedData_A2B10G10R10_SNORM_PACK32, int(gl_VertexIndex)+vertexNormal.dataOffsetTable[gl_DrawID]).xyz;
    //Normal = normalMat*normalize(vNormal);
    //Normal = vec3(1.0, 0.0, 0.0);
}
)"

}
;

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

    const std::string buffersDef = SHADER_OVERRIDES[0];
    const std::string mainDef = SHADER_OVERRIDES[1];

    resultShaderSrc.insert(firstNewlineAfterVersion, buffersDef);
    resultShaderSrc += mainDef;
    resultShaderSrc.replace(resultShaderSrc.find("#version 430 core"), sizeof("#version 430 core"), "#version 460 core\n");

    auto unspecNew = core::make_smart_refctd_ptr<asset::ICPUShader>(resultShaderSrc.c_str());
    auto specinfo = _fs->getSpecializationInfo();
    auto vsNew = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecNew), std::move(specinfo));

    return vsNew;
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
    core::smart_refctd_ptr<video::IGPUBuffer> idxBuff = nullptr;
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
    allocParams.indexBufferMinAllocCnt = 5000u;
    allocParams.vertexBuffSupportedSize = 20000000u;
    allocParams.vertexBufferMinAllocSize = 5000u;
    allocParams.MDIDataBuffSupportedCnt = 20000u;
    allocParams.MDIDataBuffMinAllocCnt = 1u; //so structs are adjacent in memory

    assert(!meshBuffers.empty());

    CCPUMeshPackerV1 mp((*(meshBuffers.end() - 1u))->getPipeline()->getVertexInputParams(), allocParams, std::numeric_limits<uint16_t>::max() / 3u, std::numeric_limits<uint16_t>::max() / 3u);

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

void packMeshBuffersV2(video::IVideoDriver* driver, core::vector<ICPUMeshBuffer*>& meshBuffers, DrawIndexedIndirectInputV2& output, std::array<DataOffsetTable, 3>& offsetTable)
{
    using MeshPacker = CCPUMeshPackerV2<DrawElementsIndirectCommand_t>;

    MeshPacker::AllocationParams allocParams;
    allocParams.indexBuffSupportedCnt = 20000000u;
    allocParams.indexBufferMinAllocCnt = 5000u;
    allocParams.vertexBuffSupportedSize = 200000000u;
    allocParams.vertexBufferMinAllocSize = 5000u;
    allocParams.MDIDataBuffSupportedCnt = 20000u;
    allocParams.MDIDataBuffMinAllocCnt = 1u; //so structs are adjacent in memory

    CCPUMeshPackerV2 mp(allocParams, std::numeric_limits<uint16_t>::max() / 3u, std::numeric_limits<uint16_t>::max() / 3u);

    core::vector<MeshPacker::ReservedAllocationMeshBuffers> allocData(meshBuffers.size());

    bool allocSuccessfull = mp.alloc(allocData.data(), meshBuffers.begin(), meshBuffers.end());
    if (!allocSuccessfull)
        std::cout << "Alloc failed \n";

    mp.instantiateDataStorage();
    MeshPacker::PackerDataStore packerDataStore = mp.getPackerDataStore();

    core::vector<IMeshPackerBase::PackedMeshBufferData> pmbd(meshBuffers.size());
    
    const uint32_t offsetTableSz = mp.calcMDIStructCount(meshBuffers.begin(), meshBuffers.end());
    core::vector<MeshPacker::CombinedDataOffsetTable> cdot(offsetTableSz);

    std::cout << offsetTableSz;

    mp.commit(pmbd.data(), cdot.data(), allocData.data(), meshBuffers.begin(), meshBuffers.end());

    output.vtxBuffer = { 0ull, driver->createFilledDeviceLocalGPUBufferOnDedMem(packerDataStore.vertexBuffer->getSize(), packerDataStore.vertexBuffer->getPointer()) };
    output.idxBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(packerDataStore.indexBuffer->getSize(), packerDataStore.indexBuffer->getPointer());
    output.indirectDrawBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(packerDataStore.MDIDataBuffer->getSize(), packerDataStore.MDIDataBuffer->getPointer());

    output.maxCount = offsetTableSz;
    output.stride = sizeof(DrawElementsIndirectCommand_t);

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

    core::vector<MeshPacker::VirtualAttribute> offsetTableLocal(offsetTableSz);

    for (uint32_t i = 0u; i < offsetTableLocal.size(); i++)
        offsetTableLocal[i] = cdot[i].attribInfo[0];

    offsetTable[0].offsetBuffer.offset = 0u;
    offsetTable[0].offsetBuffer.buffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(MeshPacker::VirtualAttribute) * offsetTableLocal.size(), static_cast<void*>(offsetTableLocal.data()));

    for (uint32_t i = 0u; i < offsetTableLocal.size(); i++)
        offsetTableLocal[i] = cdot[i].attribInfo[2];

    offsetTable[1].offsetBuffer.offset = sizeof(uint32_t) * offsetTableLocal.size();
    offsetTable[1].offsetBuffer.buffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(MeshPacker::VirtualAttribute) * offsetTableLocal.size(), static_cast<void*>(offsetTableLocal.data()));

    for (uint32_t i = 0u; i < offsetTableLocal.size(); i++)
        offsetTableLocal[i] = cdot[i].attribInfo[3];

    offsetTable[2].offsetBuffer.offset = 2u * sizeof(uint32_t) * offsetTableLocal.size();
    offsetTable[2].offsetBuffer.buffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(MeshPacker::VirtualAttribute) * offsetTableLocal.size(), static_cast<void*>(offsetTableLocal.data()));
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
    core::smart_refctd_ptr<IGPUBuffer>& vtxBuffer, std::array<DataOffsetTable, 3>& dataOffsetBuffers,
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
        b[0].count = b[1].count = b[2].count = b[3].count = b[3].count = 1u;
        dsLayout = driver->createGPUDescriptorSetLayout(b, b + 5u);
    }

    asset::SPushConstantRange pcRange = { asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD) };
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

        IGPUDescriptorSet::SDescriptorInfo info[5];

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

        info[4].buffer.offset = dataOffsetBuffers[2].offsetBuffer.offset;
        info[4].buffer.size = dataOffsetBuffers[2].offsetBuffer.buffer->getSize();
        info[4].desc = core::smart_refctd_ptr(dataOffsetBuffers[2].offsetBuffer.buffer);
        w[4].info = &info[4];

        driver->updateDescriptorSets(5u, w, 0u, nullptr);
    }

    IGPUSpecializedShader* shaders[2] = { gpuShaders->operator[](0).get(), gpuShaders->operator[](1).get() };

    outputGpuPipeline = driver->createGPURenderpassIndependentPipeline(
        nullptr, std::move(pipelineLayout),
        shaders, shaders + 2u,
        SVertexInputParams(),
        asset::SBlendParams(), asset::SPrimitiveAssemblyParams(), SRasterizationParams());
}

core::smart_refctd_ptr<ICPUMeshBuffer> createInstancedMeshBuffer(IGeometryCreator const* geometryCreator)
{
    auto cylinder = geometryCreator->createCylinderMesh(10.0f,10.0f, 128u);

    //create per instance attribute (position)
    constexpr uint32_t insCnt = 100u;
    constexpr size_t perInsAttribSize = insCnt * sizeof(vector3df);

    core::smart_refctd_ptr<ICPUBuffer> perInsAttribBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(perInsAttribSize);
    auto* perInsAttribBufferBegin = static_cast<vector3df*>(perInsAttribBuffer->getPointer());
    auto* perInsAttribBufferEnd = perInsAttribBufferBegin + insCnt;
    std::generate(perInsAttribBufferBegin, perInsAttribBufferEnd, [n = 10u]() mutable { return vector3df(n += 10); });

    cylinder.inputParams.enabledAttribFlags = 0x800D;
    cylinder.inputParams.enabledBindingFlags |= 0x8000;

    cylinder.inputParams.attributes[1] = SVertexInputAttribParams(); //ignore color attribute

    cylinder.bindings[15] = { 0ull, std::move(perInsAttribBuffer) };

    cylinder.inputParams.attributes[15].binding = 15;
    cylinder.inputParams.attributes[15].format = asset::EF_R32G32B32_SFLOAT;
    cylinder.inputParams.attributes[15].relativeOffset = 0u;

    cylinder.inputParams.bindings[15].inputRate = EVIR_PER_INSTANCE;
    cylinder.inputParams.bindings[15].stride = 0u;

    core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> pipeline = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(
        nullptr,
        nullptr,
        nullptr,
        cylinder.inputParams,
        SBlendParams(),
        SPrimitiveAssemblyParams(),
        SRasterizationParams()
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
    fs->addFileArchive("../../media/sponza.zip");

    asset::IAssetLoader::SAssetLoadParams lp;
    auto meshes_bundle = am->getAsset("sponza.obj", lp);
    //assert(!meshes_bundle.isEmpty());
    auto mesh = meshes_bundle.getContents().begin()[0];
    auto mesh_raw = static_cast<asset::ICPUMesh*>(mesh.get());

    //saving cache to file
    qnc->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fs, "../../tmp/normalCache101010.sse");

    //TODO: change it to vector of smart pointers
    core::vector<ICPUMeshBuffer*> meshBuffers;
    for (uint32_t i = 0u; i < mesh_raw->getMeshBufferVector().size(); i++)
        meshBuffers.push_back(mesh_raw->getMeshBufferVector()[i].get());

    auto instancedMeshBuffer = createInstancedMeshBuffer(am->getGeometryCreator());
    meshBuffers.push_back(instancedMeshBuffer.get());

    //pack mesh buffers
    DrawIndexedIndirectInput mdiCallParams;
    SVertexInputParams vtxInputParams;
    packMeshBuffers(driver, meshBuffers, vtxInputParams, mdiCallParams);

    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuPipeline;
    setPipeline(driver, am, vtxInputParams, gpuPipeline);

    /*core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuPipeline2;
    core::smart_refctd_ptr<IGPUDescriptorSet> ds;
    DrawIndexedIndirectInputV2 mdiCallParamsV2;
    {
        auto* pipeline = meshBuffers[0]->getPipeline();

        auto* vtxShader = pipeline->getShaderAtIndex(asset::ICPURenderpassIndependentPipeline::ESSI_VERTEX_SHADER_IX);
        core::smart_refctd_ptr<ICPUSpecializedShader> vs = createModifiedVertexShader(vtxShader);
        ICPUSpecializedShader* fs = IAsset::castDown<ICPUSpecializedShader>(am->getAsset("../shader.frag", lp).getContents().begin()->get());
        std::array<DataOffsetTable, 3> offsetTable;

        packMeshBuffersV2(driver, meshBuffers, mdiCallParamsV2, offsetTable);

        setPipelineV2(driver, vs.get(), fs, mdiCallParamsV2.vtxBuffer.buffer, offsetTable, ds, gpuPipeline2);
    }*/

    //! we want to move around the scene and view it from different angles
    scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0, 100.0f, 0.5f);

    camera->setPosition(core::vector3df(-4, 0, 0));
    camera->setTarget(core::vector3df(0, 0, 0));
    camera->setNearValue(1.f);
    camera->setFarValue(5000.0f);

    smgr->setActiveCamera(camera);

    uint64_t lastFPSTime = 0;

#define USE_MPV1

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
    while (device->run() && receiver.keepOpen())
    {
        driver->bindGraphicsPipeline(gpuPipeline2.get());
        driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuPipeline2->getLayout(), 0u, 1u, &ds.get(), nullptr);

        driver->beginScene(true, true, video::SColor(255, 0, 0, 255));

        //! This animates (moves) the camera and sets the transforms
        camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
        camera->render();

        driver->pushConstants(gpuPipeline->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), camera->getConcatenatedMatrix().pointer());
        SBufferBinding<IGPUBuffer> vtxBufferBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
        vtxBufferBindings[0] = mdiCallParamsV2.vtxBuffer;
        driver->drawIndexedIndirect(vtxBufferBindings, mdiCallParamsV2.mode, mdiCallParamsV2.indexType, mdiCallParamsV2.idxBuff.get(), mdiCallParamsV2.indirectDrawBuff.get(), mdiCallParamsV2.offset, mdiCallParamsV2.maxCount, mdiCallParamsV2.stride);

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