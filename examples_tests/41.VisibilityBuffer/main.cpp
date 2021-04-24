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
#include "nbl/video/CGPUMeshPackerV2.h"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;

constexpr const char* VERTEX_SHADER_OVERRIDES =
R"(
#define _NBL_VERT_INPUTS_DEFINED_

layout(location = 4) flat out uint drawID;

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
    return nbl_glsl_VG_attribFetch_RGB32_SFLOAT(va, vtxID);
}

vec2 nbl_glsl_fetchVtxUV(in uint vtxID, in uint drawID)
{
    nbl_glsl_VG_VirtualAttributePacked_t va = virtualAttribTable.vAttr[drawID + pc.dataBufferOffset][1];
    return nbl_glsl_VG_attribFetch_RG32_SFLOAT(va, vtxID);
}

vec3 nbl_glsl_fetchVtxNormal(in uint vtxID, in uint drawID)
{
    nbl_glsl_VG_VirtualAttributePacked_t va = virtualAttribTable.vAttr[drawID + pc.dataBufferOffset][2];
    return nbl_glsl_VG_attribFetch_RGB10A2_SNORM(va, vtxID).xyz;
}

#define _NBL_BASIC_VTX_ATTRIB_FETCH_FUCTIONS_DEFINED_
#define _NBL_POS_FETCH_FUNCTION_DEFINED
#define _NBL_UV_FETCH_FUNCTION_DEFINED
#define _NBL_NORMAL_FETCH_FUNCTION_DEFINED

)";

constexpr const char* FRAGMENT_SHADER_OVERRIDES = //also turns off set3 bindings (textures) because they're not needed anymore as we're using VT
R"(
layout(location = 4) flat in uint drawID;

#ifndef _NO_UV
    #include <nbl/builtin/glsl/virtual_texturing/extensions.glsl>

    #define _NBL_VT_DESCRIPTOR_SET 0
    #define _NBL_VT_PAGE_TABLE_BINDING 4

    #define _NBL_VT_FLOAT_VIEWS_BINDING 5 
    #define _NBL_VT_FLOAT_VIEWS_COUNT %u
    #define _NBL_VT_FLOAT_VIEWS

    #define _NBL_VT_INT_VIEWS_BINDING 6
    #define _NBL_VT_INT_VIEWS_COUNT 0
    #define _NBL_VT_INT_VIEWS

    #define _NBL_VT_UINT_VIEWS_BINDING 7
    #define _NBL_VT_UINT_VIEWS_COUNT 0
    #define _NBL_VT_UINT_VIEWS
    #include <nbl/builtin/glsl/virtual_texturing/descriptors.glsl>

    layout (set = 2, binding = 0, std430) restrict readonly buffer PrecomputedStuffSSBO
    {
        uint pgtab_sz_log2;
        float vtex_sz_rcp;
        float phys_pg_tex_sz_rcp[_NBL_VT_MAX_PAGE_TABLE_LAYERS];
        uint layer_to_sampler_ix[_NBL_VT_MAX_PAGE_TABLE_LAYERS];
    } precomputed;
#endif
#define _NBL_FRAG_SET3_BINDINGS_DEFINED_

struct MaterialParams
{
    vec3 Ka;
    vec3 Kd;
    vec3 Ks;
    vec3 Ke;
    uvec2 map_Ka_data;
    uvec2 map_Kd_data;
    uvec2 map_Ks_data;
    uvec2 map_Ns_data;
    uvec2 map_d_data;
    uvec2 map_bump_data;
    float Ns;
    float d;
    float Ni;
    uint extra; //flags copied from MTL metadata
};

layout(set = 2, binding = 1, std430) readonly buffer MaterialBuffer
{
    MaterialParams materialData[];
};

layout (push_constant) uniform Block 
{
    uint dataBufferOffset;
} pc;
#define _NBL_FRAG_PUSH_CONSTANTS_DEFINED_

#include <nbl/builtin/glsl/loader/mtl/common.glsl>
nbl_glsl_MTLMaterialParameters nbl_glsl_getMaterialParameters() // this function is for MTL's shader only
{
    MaterialParams params = materialData[drawID+pc.dataBufferOffset];

    nbl_glsl_MTLMaterialParameters mtl_params;
    mtl_params.Ka = params.Ka;
    mtl_params.Kd = params.Kd;
    mtl_params.Ks = params.Ks;
    mtl_params.Ke = params.Ke;
    mtl_params.Ns = params.Ns;
    mtl_params.d = params.d;
    mtl_params.Ni = params.Ni;
    mtl_params.extra = params.extra;
    return mtl_params;
}
#define _NBL_FRAG_GET_MATERIAL_PARAMETERS_FUNCTION_DEFINED_

#ifndef _NO_UV
    uint nbl_glsl_VT_layer2pid(in uint layer)
    {
        return precomputed.layer_to_sampler_ix[layer];
    }
    uint nbl_glsl_VT_getPgTabSzLog2()
    {
        return precomputed.pgtab_sz_log2;
    }
    float nbl_glsl_VT_getPhysPgTexSzRcp(in uint layer)
    {
        return precomputed.phys_pg_tex_sz_rcp[layer];
    }
    float nbl_glsl_VT_getVTexSzRcp()
    {
        return precomputed.vtex_sz_rcp;
    }
    #define _NBL_USER_PROVIDED_VIRTUAL_TEXTURING_FUNCTIONS_

    //nbl/builtin/glsl/virtual_texturing/functions.glsl/...
    #include <%s>
#endif


#ifndef _NO_UV
    vec4 nbl_sample_Ka(in vec2 uv, in mat2 dUV)   { return nbl_glsl_vTextureGrad(materialData[drawID+pc.dataBufferOffset].map_Ka_data, uv, dUV); }

    vec4 nbl_sample_Kd(in vec2 uv, in mat2 dUV)   { return nbl_glsl_vTextureGrad(materialData[drawID+pc.dataBufferOffset].map_Kd_data, uv, dUV); }

    vec4 nbl_sample_Ks(in vec2 uv, in mat2 dUV)   { return nbl_glsl_vTextureGrad(materialData[drawID+pc.dataBufferOffset].map_Ks_data, uv, dUV); }

    vec4 nbl_sample_Ns(in vec2 uv, in mat2 dUV)   { return nbl_glsl_vTextureGrad(materialData[drawID+pc.dataBufferOffset].map_Ns_data, uv, dUV); }

    vec4 nbl_sample_d(in vec2 uv, in mat2 dUV)    { return nbl_glsl_vTextureGrad(materialData[drawID+pc.dataBufferOffset].map_d_data, uv, dUV); }

    vec4 nbl_sample_bump(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(materialData[drawID+pc.dataBufferOffset].map_bump_data, uv, dUV); }
#endif
#define _NBL_TEXTURE_SAMPLE_FUNCTIONS_DEFINED_
)";

constexpr uint32_t TEX_OF_INTEREST_CNT = 6u; 
#include "nbl/nblpack.h"
struct MaterialParams
{
    //Ka
    vector3df_SIMD ambient;
    //Kd
    vector3df_SIMD diffuse;
    //Ks
    vector3df_SIMD specular;
    //Ke
    vector3df_SIMD emissive;
    uint64_t map_data[TEX_OF_INTEREST_CNT];
    //Ns, specular exponent in phong model
    float shininess = 32.f;
    //d
    float opacity = 1.f;
    //Ni, index of refraction
    float IoR = 1.6f;
    uint32_t extra;
} PACK_STRUCT;
#include "nbl/nblunpack.h"
static_assert(sizeof(MaterialParams) <= asset::ICPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE, "doesnt fit in push constants");

//mesh packing stuff
struct DrawIndexedIndirectInput
{
    size_t offset = 0u;
    size_t maxCount = 0u;

    static constexpr asset::E_PRIMITIVE_TOPOLOGY mode = asset::EPT_TRIANGLE_LIST;
    static constexpr asset::E_INDEX_TYPE indexType = asset::EIT_16BIT;
    static constexpr size_t stride = 0ull;
};

using MbPipelineRange = std::pair<core::smart_refctd_ptr<ICPURenderpassIndependentPipeline>,const core::smart_refctd_ptr<ICPUMeshBuffer>*>;


struct DrawData
{
    core::vector<core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>> gpuPipelines;
    core::vector<DrawIndexedIndirectInput> drawIndirectInput;
    core::smart_refctd_ptr<IGPUBuffer> vtDataSSBO;
    core::vector<uint32_t> pushConstantsData;
    std::array<core::smart_refctd_ptr<IGPUDescriptorSet>, 4> ds;
    core::smart_refctd_ptr<IGPUVirtualTexture> vt;
    core::smart_refctd_ptr<IGPUBuffer> virtualAttribTable;
    core::smart_refctd_ptr<IGPUBuffer> ubo;

    asset::SBufferBinding<IGPUBuffer> vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
    core::smart_refctd_ptr<IGPUBuffer> idxBuffer;
    core::smart_refctd_ptr<IGPUBuffer> mdiBuffer;
};

using MeshPacker = CCPUMeshPackerV2<DrawElementsIndirectCommand_t>;
using GPUMeshPacker = CGPUMeshPackerV2<DrawElementsIndirectCommand_t>;

GPUMeshPacker packMeshBuffers(video::IVideoDriver* driver, core::vector<MbPipelineRange>& ranges, DrawData& drawData)
{
    assert(ranges.size()>=2u);

    constexpr uint16_t minTrisBatch = std::numeric_limits<uint16_t>::max() / 3u; //64u;
    constexpr uint16_t maxTrisBatch = std::numeric_limits<uint16_t>::max() / 3u; //128u

    MeshPacker::AllocationParams allocParams;
    allocParams.indexBuffSupportedCnt = 32u*1024u*1024u;
    allocParams.indexBufferMinAllocCnt = minTrisBatch*3u;
    allocParams.vertexBuffSupportedByteSize = 128u*1024u*1024u;
    allocParams.vertexBufferMinAllocByteSize = minTrisBatch;
    allocParams.MDIDataBuffSupportedCnt = 8192u;
    allocParams.MDIDataBuffMinAllocCnt = 1u; //so structs are adjacent in memory (TODO: WTF NO!)
    
    CCPUMeshPackerV2 mp(allocParams,minTrisBatch,maxTrisBatch);

    auto wholeMbRangeBegin = ranges.front().second;
    auto wholeMbRangeEnd = ranges.back().second;
    const uint32_t meshBufferCnt = std::distance(wholeMbRangeBegin, wholeMbRangeEnd);

    const uint32_t mdiCntTotal = mp.calcMDIStructMaxCount(wholeMbRangeBegin,wholeMbRangeEnd);
    //shouldn't it be `meshBufferCnt` instead of `mdiCntTotal`?
    auto allocData = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<MeshPacker::ReservedAllocationMeshBuffers>>(mdiCntTotal);

    core::vector<uint32_t> allocDataOffsetForDrawCall(ranges.size());
    allocDataOffsetForDrawCall[0] = 0u;
    uint32_t i = 0u;
    for (auto it=ranges.begin(); it!=ranges.end()-1u; )
    {
        auto mbRangeBegin = &it->second->get();
        auto mbRangeEnd = &(++it)->second->get();

        bool allocSuccessfull = mp.alloc(allocData->data() + allocDataOffsetForDrawCall[i], mbRangeBegin, mbRangeEnd);
        if (!allocSuccessfull)
        {
            std::cout << "Alloc failed \n";
            _NBL_DEBUG_BREAK_IF(true);
        }

        const uint32_t mdiMaxCnt = mp.calcMDIStructMaxCount(mbRangeBegin,mbRangeEnd);
        allocDataOffsetForDrawCall[i + 1] = allocDataOffsetForDrawCall[i] + mdiMaxCnt;
        i++;
    }
    
    mp.shrinkOutputBuffersSize();
    mp.instantiateDataStorage();
    MeshPacker::PackerDataStore packerDataStore = mp.getPackerDataStore();
    
    constexpr uint32_t attribCntPerMeshBuffer = 3u;
    const uint32_t offsetTableSz = mdiCntTotal * attribCntPerMeshBuffer;
    core::vector<MeshPacker::VirtualAttribute> offsetTableLocal;
    offsetTableLocal.reserve(offsetTableSz);

    core::vector<uint32_t> mdiCntForMeshBuffer;
    mdiCntForMeshBuffer.reserve(meshBufferCnt);

    uint32_t offsetForDrawCall = 0u;
    i = 0u;
    for (auto it=ranges.begin(); it!=ranges.end()-1u;)
    {
        auto mbRangeBegin = &it->second->get();
        auto mbRangeEnd = &(++it)->second->get();

        const uint32_t mdiMaxCnt = mp.calcMDIStructMaxCount(mbRangeBegin, mbRangeEnd);
        core::vector<IMeshPackerBase::PackedMeshBufferData> pmbd(mdiMaxCnt); //why mdiMaxCnt and not meshBuffersInRangeCnt??????????
        core::vector<MeshPacker::CombinedDataOffsetTable> cdot(mdiMaxCnt);

        uint32_t mdiCnt = mp.commit(pmbd.data(), cdot.data(), allocData->data() + allocDataOffsetForDrawCall[i], mbRangeBegin, mbRangeEnd);
        if (mdiCnt == 0u)
        {
            std::cout << "Commit failed \n";
            _NBL_DEBUG_BREAK_IF(true);
        }

        drawData.pushConstantsData.push_back(offsetForDrawCall);
        offsetForDrawCall += mdiCnt;

        DrawIndexedIndirectInput mdiCallInput;
        mdiCallInput.maxCount = mdiCnt;
        mdiCallInput.offset = pmbd[0].mdiParameterOffset * sizeof(DrawElementsIndirectCommand_t);

        drawData.drawIndirectInput.push_back(mdiCallInput);

        const uint32_t mbInRangeCnt = std::distance(mbRangeBegin, mbRangeEnd);
        for (uint32_t j = 0u; j < mbInRangeCnt; j++)
        {
            mdiCntForMeshBuffer.push_back(pmbd[j].mdiParameterCount);
        }

        //setOffsetTables
        for (uint32_t j = 0u; j < mdiCnt; j++)
        {
            MeshPacker::CombinedDataOffsetTable& virtualAttribTable = cdot[j];

            offsetTableLocal.push_back(virtualAttribTable.attribInfo[0]);
            offsetTableLocal.push_back(virtualAttribTable.attribInfo[2]);
            offsetTableLocal.push_back(virtualAttribTable.attribInfo[3]);
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

        i++;
    }

    //prepare data for (set = 2, binding = 1) frag shader ssbo
    {
        core::vector<MaterialParams> vtData;
        vtData.reserve(mdiCntTotal);

        uint32_t i = 0u;
        for (auto it = wholeMbRangeBegin; it != wholeMbRangeEnd; it++)
        {
            const uint32_t mdiCntForThisMb = mdiCntForMeshBuffer[i];
            for (uint32_t i = 0u; i < mdiCntForThisMb; i++)
                vtData.push_back(*reinterpret_cast<MaterialParams*>((*it)->getPushConstantsDataPtr()));

            i++;
        }

        drawData.vtDataSSBO = driver->createFilledDeviceLocalGPUBufferOnDedMem(vtData.size() * sizeof(MaterialParams), vtData.data());
    }

    drawData.vtxBindings[0] = { 0ull, driver->createFilledDeviceLocalGPUBufferOnDedMem(packerDataStore.vertexBuffer->getSize(), packerDataStore.vertexBuffer->getPointer()) };
    drawData.idxBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(packerDataStore.indexBuffer->getSize(), packerDataStore.indexBuffer->getPointer());
    drawData.mdiBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(packerDataStore.MDIDataBuffer->getSize(), packerDataStore.MDIDataBuffer->getPointer());

    drawData.virtualAttribTable = driver->createFilledDeviceLocalGPUBufferOnDedMem(offsetTableLocal.size() * sizeof(MeshPacker::VirtualAttribute), offsetTableLocal.data());


    return GPUMeshPacker(driver, std::move(mp));
}

//vt stuff
using STextureData = asset::ICPUVirtualTexture::SMasterTextureData;

constexpr uint32_t PAGE_SZ_LOG2 = 7u;
constexpr uint32_t TILES_PER_DIM_LOG2 = 4u;
constexpr uint32_t PAGE_PADDING = 8u;
constexpr uint32_t MAX_ALLOCATABLE_TEX_SZ_LOG2 = 12u; //4096

constexpr uint32_t VT_SET = 0u;
constexpr uint32_t PGTAB_BINDING = 4u;
constexpr uint32_t PHYSICAL_STORAGE_VIEWS_BINDING = 5u;

struct commit_t
{
    STextureData addr;
    core::smart_refctd_ptr<asset::ICPUImage> texture;
    asset::ICPUImage::SSubresourceRange subresource;
    asset::ICPUSampler::E_TEXTURE_CLAMP uwrap;
    asset::ICPUSampler::E_TEXTURE_CLAMP vwrap;
    asset::ICPUSampler::E_TEXTURE_BORDER_COLOR border;
};

constexpr uint32_t texturesOfInterest[TEX_OF_INTEREST_CNT] =
{
    asset::CMTLMetadata::CRenderpassIndependentPipeline::EMP_AMBIENT,
    asset::CMTLMetadata::CRenderpassIndependentPipeline::EMP_DIFFUSE,
    asset::CMTLMetadata::CRenderpassIndependentPipeline::EMP_SPECULAR,
    asset::CMTLMetadata::CRenderpassIndependentPipeline::EMP_SHININESS,
    asset::CMTLMetadata::CRenderpassIndependentPipeline::EMP_OPACITY,
    asset::CMTLMetadata::CRenderpassIndependentPipeline::EMP_BUMP
};

STextureData getTextureData(core::vector<commit_t>& _out_commits, const asset::ICPUImage* _img, asset::ICPUVirtualTexture* _vt, asset::ISampler::E_TEXTURE_CLAMP _uwrap, asset::ISampler::E_TEXTURE_CLAMP _vwrap, asset::ISampler::E_TEXTURE_BORDER_COLOR _borderColor)
{
    const auto& extent = _img->getCreationParameters().extent;

    auto imgAndOrigSz = asset::ICPUVirtualTexture::createPoTPaddedSquareImageWithMipLevels(_img, _uwrap, _vwrap, _borderColor);

    asset::IImage::SSubresourceRange subres;
    subres.baseMipLevel = 0u;
    subres.levelCount = core::findLSB(core::roundDownToPoT<uint32_t>(std::max(extent.width, extent.height))) + 1;
    subres.baseArrayLayer = 0u;
    subres.layerCount = 1u;

    auto addr = _vt->alloc(_img->getCreationParameters().format, imgAndOrigSz.second, subres, _uwrap, _vwrap);
    commit_t cm{ addr, std::move(imgAndOrigSz.first), subres, _uwrap, _vwrap, _borderColor };

    _out_commits.push_back(cm);

    return addr;
}

inline constexpr bool useSSBO() { return false; }

void createPipeline(IVideoDriver* driver, ICPUSpecializedShader* vs, ICPUSpecializedShader* fs, core::vector<MbPipelineRange>& ranges, DrawData& drawData, const GPUMeshPacker& mp)
{
    ICPUSpecializedShader* cpuShaders[2] = { vs, fs };
    auto gpuShaders = driver->getGPUObjectsFromAssets(cpuShaders, cpuShaders + 2);

    core::smart_refctd_ptr<IGPUDescriptorSetLayout> ds0Layout;
    core::smart_refctd_ptr<IGPUDescriptorSetLayout> ds1Layout;
    core::smart_refctd_ptr<IGPUDescriptorSetLayout> ds2Layout;
    core::smart_refctd_ptr<IGPUDescriptorSetLayout> ds3Layout;
    {

        auto getMpBindingsCnt = [&mp]()->uint32_t
        {
            if constexpr (useSSBO())
                return mp.getDSlayoutBindingsForSSBO(nullptr);
            else
                return mp.getDSlayoutBindingsForUTB(nullptr);
        };

        const uint32_t mpBindingsCnt = getMpBindingsCnt();
        const auto vtBindingsCnt = drawData.vt->getDSlayoutBindings(nullptr, nullptr);
        
        auto bindings = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IGPUDescriptorSetLayout::SBinding>>(mpBindingsCnt + vtBindingsCnt.first);
        auto vtSamplers = core::make_refctd_dynamic_array< core::smart_refctd_dynamic_array<core::smart_refctd_ptr<IGPUSampler>>>(vtBindingsCnt.second);

        IGPUDescriptorSetLayout::SBinding* b = bindings->data();
        IGPUDescriptorSetLayout::SBinding* mpBindingsPtr = b;
        IGPUDescriptorSetLayout::SBinding* vtBindingsPtr = b + mpBindingsCnt;
        
        if constexpr (useSSBO())
            mp.getDSlayoutBindingsForSSBO(mpBindingsPtr);
        else
            mp.getDSlayoutBindingsForUTB(mpBindingsPtr);

        drawData.vt->getDSlayoutBindings(vtBindingsPtr, vtSamplers->data(), PGTAB_BINDING, PHYSICAL_STORAGE_VIEWS_BINDING);

        for (uint32_t i = 0u; i < mpBindingsCnt; i++)
        {
            mpBindingsPtr->stageFlags = ISpecializedShader::ESS_VERTEX;
            mpBindingsPtr++;
        }
        
        for (uint32_t i = 0u; i < vtBindingsCnt.first; i++)
        {
            vtBindingsPtr->stageFlags = ISpecializedShader::ESS_FRAGMENT;
            vtBindingsPtr++;
        }

        ds0Layout = driver->createGPUDescriptorSetLayout(b, b + bindings->size());

        IGPUDescriptorSetLayout::SBinding b1;
        b1.binding = 0u;
        b1.count = 1u;
        b1.samplers = nullptr;
        b1.stageFlags = ISpecializedShader::ESS_VERTEX;
        b1.type = EDT_UNIFORM_BUFFER;

        ds1Layout = driver->createGPUDescriptorSetLayout(&b1, &b1 + 1);

        IGPUDescriptorSetLayout::SBinding b2[2];
        b2[0].binding = 0u;
        b2[0].count = 1u;
        b2[0].samplers = nullptr;
        b2[0].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
        b2[0].type = asset::EDT_STORAGE_BUFFER;

        b2[1].binding = 1u;
        b2[1].count = 1u;
        b2[1].samplers = nullptr;
        b2[1].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
        b2[1].type = asset::EDT_STORAGE_BUFFER;

        ds2Layout = driver->createGPUDescriptorSetLayout(b2, b2 + 2);

        IGPUDescriptorSetLayout::SBinding b3;
        b3.binding = 0u;
        b3.count = 1u;
        b3.samplers = nullptr;
        b3.stageFlags = ISpecializedShader::ESS_VERTEX;
        b3.type = EDT_STORAGE_BUFFER;

        ds3Layout = driver->createGPUDescriptorSetLayout(&b3, &b3 + 1);
    }

    SPushConstantRange pcRange;
    pcRange.size = sizeof(uint32_t);
    pcRange.offset = 0u;
    pcRange.stageFlags = ISpecializedShader::ESS_ALL;

    auto pipelineLayout = driver->createGPUPipelineLayout(&pcRange, &pcRange + 1, core::smart_refctd_ptr(ds0Layout), 
        core::smart_refctd_ptr(ds1Layout), core::smart_refctd_ptr(ds2Layout), core::smart_refctd_ptr(ds3Layout));

    drawData.ds[0] = driver->createGPUDescriptorSet(std::move(ds0Layout));
    drawData.ds[1] = driver->createGPUDescriptorSet(std::move(ds1Layout));
    drawData.ds[2] = driver->createGPUDescriptorSet(std::move(ds2Layout));
    drawData.ds[3] = driver->createGPUDescriptorSet(std::move(ds3Layout));
    {
        auto getMpWriteAndInfoSize = [&mp]() -> std::pair<uint32_t, uint32_t>
        {
            if constexpr (useSSBO())
            {
                uint32_t writeAndInfoSize = mp.getDescriptorSetWritesForSSBO(nullptr, nullptr, nullptr);
                return std::make_pair(writeAndInfoSize, writeAndInfoSize);
            }
            else
                return mp.getDescriptorSetWritesForUTB(nullptr, nullptr, nullptr);
        };

        //mesh packing stuff
        auto sizesMP = getMpWriteAndInfoSize();
        auto writesMP = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IGPUDescriptorSet::SWriteDescriptorSet>>(sizesMP.first);
        auto infoMP = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IGPUDescriptorSet::SDescriptorInfo>>(sizesMP.second);

        auto writesPtr = writesMP->data();
        auto infoPtr = infoMP->data();

        if constexpr (useSSBO())
            mp.getDescriptorSetWritesForSSBO(writesMP->data(), infoMP->data(), drawData.ds[0].get());
        else
            mp.getDescriptorSetWritesForUTB(writesMP->data(), infoMP->data(), drawData.ds[0].get());

        driver->updateDescriptorSets(writesMP->size(), writesMP->data(), 0u, nullptr);

        IGPUDescriptorSet::SWriteDescriptorSet w1;
        w1.arrayElement = 0u;
        w1.count = 1u;
        w1.binding = 0u;
        w1.descriptorType = EDT_UNIFORM_BUFFER;
        w1.dstSet = drawData.ds[1].get();

        drawData.ubo = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(SBasicViewParameters));

        IGPUDescriptorSet::SDescriptorInfo info1;
        info1.buffer.offset = 0u;
        info1.buffer.size = drawData.ubo->getSize();
        info1.desc = core::smart_refctd_ptr(drawData.ubo);
        w1.info = &info1;

        driver->updateDescriptorSets(1u, &w1, 0u, nullptr);

        IGPUDescriptorSet::SWriteDescriptorSet w3;
        w3.arrayElement = 0u;
        w3.count = 1u;
        w3.binding = 0u;
        w3.descriptorType = EDT_STORAGE_BUFFER;
        w3.dstSet = drawData.ds[3].get();

        IGPUDescriptorSet::SDescriptorInfo info3;
        info3.buffer.offset = 0u;
        info3.buffer.size = drawData.virtualAttribTable->getSize();
        info3.desc = core::smart_refctd_ptr(drawData.virtualAttribTable);
        w3.info = &info3;

        driver->updateDescriptorSets(1u, &w3, 0u, nullptr);

        //vt stuff
        auto sizesVT = drawData.vt->getDescriptorSetWrites(nullptr, nullptr, nullptr);
        auto writesVT = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<video::IGPUDescriptorSet::SWriteDescriptorSet>>(sizesVT.first);
        auto infoVT = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<video::IGPUDescriptorSet::SDescriptorInfo>>(sizesVT.second);

        drawData.vt->getDescriptorSetWrites(writesVT->data(), infoVT->data(), drawData.ds[0].get(), PGTAB_BINDING, PHYSICAL_STORAGE_VIEWS_BINDING);

        driver->updateDescriptorSets(writesVT->size(), writesVT->data(), 0u, nullptr);

        IGPUDescriptorSet::SWriteDescriptorSet w2[2];
        w2[0].arrayElement = 0u;
        w2[0].count = 1u;
        w2[0].binding = 0u;
        w2[0].descriptorType = EDT_STORAGE_BUFFER;
        w2[0].dstSet = drawData.ds[2].get();

        w2[1].arrayElement = 0u;
        w2[1].count = 1u;
        w2[1].binding = 1u;
        w2[1].descriptorType = EDT_STORAGE_BUFFER;
        w2[1].dstSet = drawData.ds[2].get();

        core::smart_refctd_ptr<video::IGPUBuffer> buffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(video::IGPUVirtualTexture::SPrecomputedData), &drawData.vt->getPrecomputedData());

        IGPUDescriptorSet::SDescriptorInfo info2[2];
        info2[0].buffer.offset = 0u;
        info2[0].buffer.size = sizeof(video::IGPUVirtualTexture::SPrecomputedData);
        info2[0].desc = buffer;

        info2[1].buffer.offset = 0u;
        info2[1].buffer.size = drawData.vtDataSSBO->getSize();
        info2[1].desc = drawData.vtDataSSBO; // TODO: rename vtData to materialData

        w2[0].info = &info2[0];
        w2[1].info = &info2[1];

        driver->updateDescriptorSets(2u,w2,0u,nullptr);
    }

    IGPUSpecializedShader* shaders[2] = { gpuShaders->operator[](0).get(), gpuShaders->operator[](1).get() };
    
    auto it = ranges.begin();
    drawData.gpuPipelines.resize(ranges.size()-1u);
    for (auto& gpuPpln : drawData.gpuPipelines)
    {
        auto cpuPipeline = (it++)->first;

        gpuPpln = driver->createGPURenderpassIndependentPipeline(
            nullptr, core::smart_refctd_ptr(pipelineLayout),
            shaders, shaders+2u,
            cpuPipeline->getVertexInputParams(),
            cpuPipeline->getBlendParams(), 
            cpuPipeline->getPrimitiveAssemblyParams(), 
            cpuPipeline->getRasterizationParams()
        );
    }
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
    DrawData drawData;
    {
        //
        auto* qnc = am->getMeshManipulator()->getQuantNormalCache();
        //loading cache from file
        qnc->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fs, "../../tmp/normalCache101010.sse", true);

        // register the zip
        device->getFileSystem()->addFileArchive("../../media/sponza.zip");

        asset::IAssetLoader::SAssetLoadParams lp;
        auto meshes_bundle = am->getAsset("sponza.obj", lp);
        assert(!meshes_bundle.getContents().empty());
        auto mesh_raw = static_cast<asset::ICPUMesh*>(meshes_bundle.getContents().begin()->get());
        // ensure memory will be freed as soon as CPU assets are dropped
        // am->clearAllAssetCache();


        //saving cache to file
        qnc->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fs, "../../tmp/normalCache101010.sse");

        //
        auto meshBuffers = mesh_raw->getMeshBufferVector();

        auto pipelineMeshBufferRanges = [&meshBuffers]() -> core::vector<MbPipelineRange>
        {
            if (meshBuffers.empty())
                return {};

            // sort meshbuffers by pipeline
            std::sort(meshBuffers.begin(),meshBuffers.end(),[](const auto& lhs, const auto& rhs)
                {
                    auto lPpln = lhs->getPipeline();
                    auto rPpln = rhs->getPipeline();
                    // render non-transparent things first
                    if (lPpln->getBlendParams().blendParams[0].blendEnable < rPpln->getBlendParams().blendParams[0].blendEnable)
                        return true;
                    if (lPpln->getBlendParams().blendParams[0].blendEnable == rPpln->getBlendParams().blendParams[0].blendEnable)
                        return lPpln < rPpln;
                    return false;
                }
            );

            core::vector<MbPipelineRange> output;
            core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> mbPipeline = nullptr;
            for (const auto& mb : meshBuffers)
            if (mb->getPipeline()!=mbPipeline.get())
            {
                mbPipeline = core::smart_refctd_ptr<ICPURenderpassIndependentPipeline>(mb->getPipeline());
                output.emplace_back(core::smart_refctd_ptr(mbPipeline),&mb);
            }
            output.emplace_back(core::smart_refctd_ptr<ICPURenderpassIndependentPipeline>(),meshBuffers.data()+meshBuffers.size());
            return output;
        }();

        core::smart_refctd_ptr<asset::ICPUVirtualTexture> vt = core::make_smart_refctd_ptr<asset::ICPUVirtualTexture>([](asset::E_FORMAT_CLASS) -> uint32_t { return TILES_PER_DIM_LOG2; }, PAGE_SZ_LOG2, PAGE_PADDING, MAX_ALLOCATABLE_TEX_SZ_LOG2);
        // createVirtualTexture
        [driver,&meshBuffers,&meshes_bundle,&vt,&drawData]() -> void
        {
            core::unordered_map<core::smart_refctd_ptr<asset::ICPUImage>, STextureData> VTtexDataMap;

            core::vector<commit_t> vt_commits;
            //modifying push constants and default fragment shader for VT
            for (auto it = meshBuffers.begin(); it != meshBuffers.end(); it++)
            {
                MaterialParams pushConsts;
                memset(pushConsts.map_data, 0xff, TEX_OF_INTEREST_CNT * sizeof(pushConsts.map_data[0]));
                pushConsts.extra = 0u;

                auto* ds = (*it)->getAttachedDescriptorSet();
                if (!ds)
                    continue;
                for (uint32_t k = 0u; k < TEX_OF_INTEREST_CNT; ++k)
                {
                    uint32_t j = texturesOfInterest[k];

                    auto* view = static_cast<asset::ICPUImageView*>(ds->getDescriptors(j).begin()->desc.get());
                    auto* smplr = ds->getLayout()->getBindings().begin()[j].samplers[0].get();
                    const auto uwrap = static_cast<asset::ISampler::E_TEXTURE_CLAMP>(smplr->getParams().TextureWrapU);
                    const auto vwrap = static_cast<asset::ISampler::E_TEXTURE_CLAMP>(smplr->getParams().TextureWrapV);
                    const auto borderColor = static_cast<asset::ISampler::E_TEXTURE_BORDER_COLOR>(smplr->getParams().BorderColor);
                    auto img = view->getCreationParameters().image;
                    auto extent = img->getCreationParameters().extent;
                    if (extent.width <= 2u || extent.height <= 2u)//dummy 2x2
                        continue;
                    STextureData texData = STextureData::invalid();
                    auto found = VTtexDataMap.find(img);
                    if (found != VTtexDataMap.end())
                        texData = found->second;
                    else
                    {
                        const asset::E_FORMAT fmt = img->getCreationParameters().format;
                        texData = getTextureData(vt_commits, img.get(), vt.get(), uwrap, vwrap, borderColor);
                        VTtexDataMap.insert({ img,texData });
                    }

                    static_assert(sizeof(texData) == sizeof(pushConsts.map_data[0]), "wrong reinterpret_cast");
                    pushConsts.map_data[k] = reinterpret_cast<uint64_t*>(&texData)[0];
                }

                // all pipelines will have the same metadata
                auto pipelineMetadata = static_cast<const asset::CMTLMetadata::CRenderpassIndependentPipeline*>(meshes_bundle.getMetadata()->selfCast<const asset::COBJMetadata>()->getAssetSpecificMetadata((*it)->getPipeline()));
                assert(pipelineMetadata);

                //copy texture presence flags
                pushConsts.extra = pipelineMetadata->m_materialParams.extra;
                pushConsts.ambient = pipelineMetadata->m_materialParams.ambient;
                pushConsts.diffuse = pipelineMetadata->m_materialParams.diffuse;
                pushConsts.emissive = pipelineMetadata->m_materialParams.emissive;
                pushConsts.specular = pipelineMetadata->m_materialParams.specular;
                pushConsts.IoR = pipelineMetadata->m_materialParams.IoR;
                pushConsts.opacity = pipelineMetadata->m_materialParams.opacity;
                pushConsts.shininess = pipelineMetadata->m_materialParams.shininess;
                memcpy((*it)->getPushConstantsDataPtr(), &pushConsts, sizeof(pushConsts));

                //we dont want this DS to be converted into GPU DS, so set to nullptr
                //dont worry about deletion of textures (invalidation of pointers), they're grabbed in VTtexDataMap
                (*it)->setAttachedDescriptorSet(nullptr);
            }

            vt->shrink();
            for (const auto& cm : vt_commits)
            {
                vt->commit(cm.addr, cm.texture.get(), cm.subresource, cm.uwrap, cm.vwrap, cm.border);
            }

            drawData.vt = core::make_smart_refctd_ptr<video::IGPUVirtualTexture>(driver, vt.get());
        }();

        {
            // all pipelines refer to the same shader
            const auto& pipeline = pipelineMeshBufferRanges.front().first;

            auto overrideShaderJustAfterVersionDirective = [](const ICPUSpecializedShader* _specShader, const std::string& extraCode)
            {
                const asset::ICPUShader* unspec = _specShader->getUnspecialized();
                assert(unspec->containsGLSL());

                auto begin = reinterpret_cast<const char*>(unspec->getSPVorGLSL()->getPointer());
                const std::string_view origSource(begin,unspec->getSPVorGLSL()->getSize());

                const size_t firstNewlineAfterVersion = origSource.find("\n",origSource.find("#version "));
                assert(firstNewlineAfterVersion!=std::string_view::npos);
                const std::string_view sourceWithoutVersion(begin+firstNewlineAfterVersion,origSource.size()-firstNewlineAfterVersion);

                std::string newSource("#version 460 core\n");
                newSource += extraCode;
                newSource += sourceWithoutVersion;

                auto unspecNew = core::make_smart_refctd_ptr<asset::ICPUShader>(newSource.c_str());
                auto specinfo = _specShader->getSpecializationInfo();

                return core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecNew), std::move(specinfo));
            };

            GPUMeshPacker mp = packMeshBuffers(driver, pipelineMeshBufferRanges, drawData);
            
            auto getGLSL = [&mp]() -> std::string
            {
                if constexpr (useSSBO())
                    return mp.getGLSLForSSBO();
                else
                    return mp.getGLSLForUTB();
            };

            std::string vertPrelude = getGLSL();
            vertPrelude += VERTEX_SHADER_OVERRIDES;
            core::smart_refctd_ptr<ICPUSpecializedShader> vs = overrideShaderJustAfterVersionDirective(pipeline->getShaderAtIndex(asset::ICPURenderpassIndependentPipeline::ESSI_VERTEX_SHADER_IX),vertPrelude);

            std::string fragPrelude(strlen(FRAGMENT_SHADER_OVERRIDES)+500u,'\0');
            sprintf(fragPrelude.data(),FRAGMENT_SHADER_OVERRIDES,drawData.vt->getFloatViews().size(),drawData.vt->getGLSLFunctionsIncludePath().c_str());
            fragPrelude.resize(strlen(fragPrelude.c_str()));
            core::smart_refctd_ptr<ICPUSpecializedShader> fs = overrideShaderJustAfterVersionDirective(pipeline->getShaderAtIndex(asset::ICPURenderpassIndependentPipeline::ESSI_FRAGMENT_SHADER_IX),fragPrelude);

            createPipeline(driver, vs.get(), fs.get(), pipelineMeshBufferRanges, drawData, mp);
        }
    }

    //! we want to move around the scene and view it from different angles
    scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0, 100.0f, 0.5f);

    camera->setPosition(core::vector3df(-4, 0, 0));
    camera->setTarget(core::vector3df(0, 0, 0));
    camera->setNearValue(1.f);
    camera->setFarValue(5000.0f);

    smgr->setActiveCamera(camera);

    //all pipelines share the same layout
    const video::IGPUDescriptorSet* ds[]{ drawData.ds[0].get(), drawData.ds[1].get(), drawData.ds[2].get(), drawData.ds[3].get() };
    

    uint64_t lastFPSTime = 0;
    while (device->run() && receiver.keepOpen())
    {
        driver->beginScene(true, true, video::SColor(255, 0, 0, 255));

        //! This animates (moves) the camera and sets the transforms
        camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
        camera->render();
        
        ICPUBufferView* bufferView;

        SBasicViewParameters uboData;
        memcpy(uboData.MVP, camera->getConcatenatedMatrix().pointer(), sizeof(core::matrix4SIMD));
        memcpy(uboData.MV, camera->getViewMatrix().pointer(), sizeof(core::matrix3x4SIMD));
        memcpy(uboData.NormalMat, camera->getViewMatrix().pointer(), sizeof(core::matrix3x4SIMD));

        driver->updateBufferRangeViaStagingBuffer(drawData.ubo.get(), 0u, sizeof(SBasicViewParameters), &uboData);

        uint32_t i = 0u;
        for (auto pipeline : drawData.gpuPipelines)
        {
            driver->bindGraphicsPipeline(pipeline.get());
            driver->bindDescriptorSets(video::EPBP_GRAPHICS, pipeline->getLayout(), 0u, 4u, ds, nullptr);

            driver->pushConstants(pipeline->getLayout(), IGPUSpecializedShader::ESS_ALL, 0u, sizeof(uint32_t), &drawData.pushConstantsData[i]);

            driver->drawIndexedIndirect(drawData.vtxBindings, DrawIndexedIndirectInput::mode, DrawIndexedIndirectInput::indexType, 
                drawData.idxBuffer.get(), drawData.mdiBuffer.get(), drawData.drawIndirectInput[i].offset, 
                drawData.drawIndirectInput[i].maxCount, DrawIndexedIndirectInput::stride);

            i++;
        }

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