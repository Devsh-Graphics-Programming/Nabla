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

constexpr const char* VERTEX_SHADER_OVERRIDES =
R"(
#define _NBL_VERT_INPUTS_DEFINED_

#define nbl_glsl_VirtualAttribute_t uint

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
layout(set = 0, binding = 0) uniform samplerBuffer MeshPackedDataFloat[2];

//uv
layout(set = 0, binding = 1) uniform isamplerBuffer MeshPackedDataInt[1];

//normal
layout(set = 0, binding = 2) uniform usamplerBuffer MeshPackedDataUint[1];

layout(set = 0, binding = 3) readonly buffer VirtualAttributes
{
    nbl_glsl_VirtualAttribute_t vAttr[][3];
} virtualAttribTable;

#define _NBL_BASIC_VTX_ATTRIB_FETCH_FUCTIONS_DEFINED_
#define _NBL_POS_FETCH_FUNCTION_DEFINED
#define _NBL_UV_FETCH_FUNCTION_DEFINED
#define _NBL_NORMAL_FETCH_FUNCTION_DEFINED

//vec4 nbl_glsl_readAttrib(uint offset)
//ivec4 nbl_glsl_readAttrib(uint offset)
//uvec4 nbl_glsl_readAttrib(uint offset)
//vec3 nbl_glsl_readAttrib(uint offset) 
//..

struct VirtualAttribute
{
    uint binding;
    int offset;
};

VirtualAttribute unpackVirtualAttribute(in nbl_glsl_VirtualAttribute_t vaPacked)
{
    VirtualAttribute result;
    result.binding = bitfieldExtract(vaPacked, 0, 4);
    result.offset = int(bitfieldExtract(vaPacked, 4, 28));
    
    return result;
}

vec3 nbl_glsl_fetchVtxPos(in uint vtxID)
{
    VirtualAttribute va = unpackVirtualAttribute(virtualAttribTable.vAttr[gl_DrawID][0]);
    return texelFetch(MeshPackedDataFloat[va.binding], va.offset + int(vtxID)).xyz;
}

vec2 nbl_glsl_fetchVtxUV(in uint vtxID)
{
    VirtualAttribute va = unpackVirtualAttribute(virtualAttribTable.vAttr[gl_DrawID][1]);
    return texelFetch(MeshPackedDataFloat[va.binding], va.offset + int(vtxID)).xy;
}

vec3 nbl_glsl_fetchVtxNormal(in uint vtxID)
{
    VirtualAttribute va = unpackVirtualAttribute(virtualAttribTable.vAttr[gl_DrawID][2]);
    return nbl_glsl_decodeRGB10A2_SNORM(texelFetch(MeshPackedDataUint[va.binding], va.offset + int(vtxID)).x).xyz;
}

)";

constexpr const char* FRAGMENT_SHADER_OVERRIDES = //also turns off set3 bindings (textures) because they're not needed anymore as we're using VT
R"(
#ifndef _NO_UV
    #include <nbl/builtin/glsl/virtual_texturing/extensions.glsl>

    #define _NBL_VT_DESCRIPTOR_SET 0
    #define _NBL_VT_PAGE_TABLE_BINDING 0

    #define _NBL_VT_FLOAT_VIEWS_BINDING 1 
    #define _NBL_VT_FLOAT_VIEWS_COUNT %u
    #define _NBL_VT_FLOAT_VIEWS

    #define _NBL_VT_INT_VIEWS_BINDING 2
    #define _NBL_VT_INT_VIEWS_COUNT 0
    #define _NBL_VT_INT_VIEWS

    #define _NBL_VT_UINT_VIEWS_BINDING 3
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

struct PCstruct
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
layout (push_constant) uniform Block {
    PCstruct params;
} PC;
#define _NBL_FRAG_PUSH_CONSTANTS_DEFINED_


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
    vec4 nbl_sample_Ka(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(PC.params.map_Ka_data, uv, dUV); }

    vec4 nbl_sample_Kd(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(PC.params.map_Kd_data, uv, dUV); }

    vec4 nbl_sample_Ks(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(PC.params.map_Ks_data, uv, dUV); }

    vec4 nbl_sample_Ns(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(PC.params.map_Ns_data, uv, dUV); }

    vec4 nbl_sample_d(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(PC.params.map_d_data, uv, dUV); }

    vec4 nbl_sample_bump(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(PC.params.map_bump_data, uv, dUV); }
#endif
#define _NBL_TEXTURE_SAMPLE_FUNCTIONS_DEFINED_
)";

core::smart_refctd_ptr<asset::ICPUSpecializedShader> createModifiedVertexShader(const asset::ICPUSpecializedShader* _fs)
{
    const asset::ICPUShader* unspec = _fs->getUnspecialized();
    assert(unspec->containsGLSL());

    auto begin = reinterpret_cast<const char*>(unspec->getSPVorGLSL()->getPointer());
    auto end = begin + unspec->getSPVorGLSL()->getSize();
    std::string resultShaderSrc(begin, end);

    size_t firstNewlineAfterVersion = resultShaderSrc.find("\n", resultShaderSrc.find("#version "));

    const std::string customSrcCode = VERTEX_SHADER_OVERRIDES;

    resultShaderSrc.insert(firstNewlineAfterVersion, customSrcCode);
    resultShaderSrc.replace(resultShaderSrc.find("#version 430 core"), sizeof("#version 430 core"), "#version 460 core\n");

    auto unspecNew = core::make_smart_refctd_ptr<asset::ICPUShader>(resultShaderSrc.c_str());
    auto specinfo = _fs->getSpecializationInfo();
    auto vsNew = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecNew), std::move(specinfo));

    return vsNew;
}

core::smart_refctd_ptr<asset::ICPUSpecializedShader> createModifiedFragShader(const asset::ICPUSpecializedShader* _fs, const asset::ICPUVirtualTexture* _vt)
{
    const asset::ICPUShader* unspec = _fs->getUnspecialized();
    assert(unspec->containsGLSL());

    auto begin = reinterpret_cast<const char*>(unspec->getSPVorGLSL()->getPointer());
    auto end = begin + unspec->getSPVorGLSL()->getSize();
    std::string glsl(begin, end);

    std::string prelude(strlen(FRAGMENT_SHADER_OVERRIDES) + 500u, '\0');
    sprintf(prelude.data(), FRAGMENT_SHADER_OVERRIDES, _vt->getFloatViews().size(), _vt->getGLSLFunctionsIncludePath().c_str());
    prelude.resize(strlen(prelude.c_str()));

    size_t firstNewlineAfterVersion = glsl.find("\n", glsl.find("#version "));
    glsl.insert(firstNewlineAfterVersion, prelude);

    auto* f = fopen("fs.glsl", "w");
    fwrite(glsl.c_str(), 1, glsl.size(), f);
    fclose(f);

    auto unspecNew = core::make_smart_refctd_ptr<asset::ICPUShader>(glsl.c_str());
    auto specinfo = _fs->getSpecializationInfo();//intentional copy
    auto fsNew = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecNew), std::move(specinfo));

    return fsNew;
}

//mesh packing stuff

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

using Range_t = SRange<void, core::vector<ICPUMeshBuffer*>::iterator>;
using MbPipelineRange = std::pair<ICPURenderpassIndependentPipeline*, Range_t>;

core::vector<MbPipelineRange> sortMeshBuffersByPipeline(core::vector<ICPUMeshBuffer*>& meshBuffers)
{
    core::vector<MbPipelineRange> output;

    if (meshBuffers.empty())
        return output;

    auto sortFunc = [](ICPUMeshBuffer* lhs, ICPUMeshBuffer* rhs)
    {
        return lhs->getPipeline() < rhs->getPipeline();
    };
    std::sort(meshBuffers.begin(), meshBuffers.end(), sortFunc);

    auto mbPipeline = (*meshBuffers.begin())->getPipeline();
    auto rangeBegin = meshBuffers.begin();

    for (auto it = meshBuffers.begin() + 1; ; it++)
    {
        if (it == meshBuffers.end())
        {
            output.push_back(std::make_pair(mbPipeline, Range_t(rangeBegin, meshBuffers.end())));
            break;
        }

        if ((*it)->getPipeline() != mbPipeline)
        {
            output.push_back(std::make_pair(mbPipeline, Range_t(rangeBegin, it)));
            rangeBegin = it;
            mbPipeline = (*rangeBegin)->getPipeline();
        }
    }

    return output;
}

void packMeshBuffers(video::IVideoDriver* driver, core::vector<ICPUMeshBuffer*>& meshBuffers, DrawIndexedIndirectInput& output, core::smart_refctd_ptr<IGPUBuffer>& virtualAttribTableOut)
{
    using MeshPacker = CCPUMeshPackerV2<DrawElementsIndirectCommand_t>;

    MeshPacker::AllocationParams allocParams;
    allocParams.indexBuffSupportedCnt = 20000000u;
    allocParams.indexBufferMinAllocSize = 5000u;
    allocParams.vertexBuffSupportedSize = 200000000u;
    allocParams.vertexBufferMinAllocSize = 5000u;
    allocParams.MDIDataBuffSupportedCnt = 20000u;
    allocParams.MDIDataBuffMinAllocSize = 1u; //so structs are adjacent in memory

    CCPUMeshPackerV2 mp(allocParams, 4096/*std::numeric_limits<uint16_t>::max() / 3u*/, 4096/*std::numeric_limits<uint16_t>::max() / 3u*/);

    uint32_t mdiCnt = mp.calcMDIStructCount(meshBuffers.begin(), meshBuffers.end());

    core::vector<MeshPacker::ReservedAllocationMeshBuffers> allocData(mdiCnt);

    bool allocSuccessfull = mp.alloc(allocData.data(), meshBuffers.begin(), meshBuffers.end());
    if (!allocSuccessfull)
    {
        std::cout << "Alloc failed \n";
        _NBL_DEBUG_BREAK_IF(true);
    }

    mp.instantiateDataStorage();
    MeshPacker::PackerDataStore packerDataStore = mp.getPackerDataStore();

    core::vector<IMeshPackerBase::PackedMeshBufferData> pmbd(mdiCnt);

    const uint32_t offsetTableSz = mdiCnt * 3u;
    core::vector<MeshPacker::CombinedDataOffsetTable> cdot(offsetTableSz);

    bool commitSuccessfull = mp.commit(pmbd.data(), cdot.data(), allocData.data(), meshBuffers.begin(), meshBuffers.end());
    if (!commitSuccessfull)
    {
        std::cout << "Commit failed \n";
        _NBL_DEBUG_BREAK_IF(true);
    }

    output.vtxBuffer = { 0ull, driver->createFilledDeviceLocalGPUBufferOnDedMem(packerDataStore.vertexBuffer->getSize(), packerDataStore.vertexBuffer->getPointer()) };
    output.idxBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(packerDataStore.indexBuffer->getSize(), packerDataStore.indexBuffer->getPointer());
    output.indirectDrawBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(packerDataStore.MDIDataBuffer->getSize(), packerDataStore.MDIDataBuffer->getPointer());

    output.maxCount = mdiCnt;
    output.stride = sizeof(DrawElementsIndirectCommand_t);

    //auto glsl = mp.generateGLSLBufferDefinitions(0u);

    //setOffsetTables

    core::vector<MeshPacker::VirtualAttribute> offsetTableLocal;
    offsetTableLocal.reserve(offsetTableSz);
    for (uint32_t i = 0u; i < mdiCnt; i++)
    {
        MeshPacker::CombinedDataOffsetTable& virtualAttribTable = cdot[i];
        
        offsetTableLocal.push_back(virtualAttribTable.attribInfo[0]);
        offsetTableLocal.push_back(virtualAttribTable.attribInfo[2]);
        offsetTableLocal.push_back(virtualAttribTable.attribInfo[3]);
    }

    /*DrawElementsIndirectCommand_t* mdiPtr = static_cast<DrawElementsIndirectCommand_t*>(packerDataStore.MDIDataBuffer->getPointer()) + 99u;
    uint16_t* idxBuffPtr = static_cast<uint16_t*>(packerDataStore.indexBuffer->getPointer());
    float* vtxBuffPtr = static_cast<float*>(packerDataStore.vertexBuffer->getPointer());

    for (uint32_t i = 0u; i < 264; i++)
    {
        float* firstCoord = vtxBuffPtr + ((*(idxBuffPtr + i) + cdot[99].attribInfo[0].offset) * 3u);
        std::cout << "vtx: " << i << " idx: " << *(idxBuffPtr + i) << "    ";
        std::cout << *firstCoord << ' ' << *(firstCoord + 1u) << ' ' << *(firstCoord + 2u) << std::endl;
    }*/

    virtualAttribTableOut = driver->createFilledDeviceLocalGPUBufferOnDedMem(offsetTableLocal.size() * sizeof(MeshPacker::VirtualAttribute), offsetTableLocal.data());
}

//vt stuff

using STextureData = asset::ICPUVirtualTexture::SMasterTextureData;

constexpr uint32_t PAGE_SZ_LOG2 = 7u;
constexpr uint32_t TILES_PER_DIM_LOG2 = 4u;
constexpr uint32_t PAGE_PADDING = 8u;
constexpr uint32_t MAX_ALLOCATABLE_TEX_SZ_LOG2 = 12u; //4096

constexpr uint32_t VT_SET = 0u;
constexpr uint32_t PGTAB_BINDING = 0u;
constexpr uint32_t PHYSICAL_STORAGE_VIEWS_BINDING = 1u;

struct commit_t
{
    STextureData addr;
    core::smart_refctd_ptr<asset::ICPUImage> texture;
    asset::ICPUImage::SSubresourceRange subresource;
    asset::ICPUSampler::E_TEXTURE_CLAMP uwrap;
    asset::ICPUSampler::E_TEXTURE_CLAMP vwrap;
    asset::ICPUSampler::E_TEXTURE_BORDER_COLOR border;
};

constexpr uint32_t TEX_OF_INTEREST_CNT = 6u;
#include "nbl/nblpack.h"
struct SPushConstants
{
    //Ka
    core::vector3df_SIMD ambient;
    //Kd
    core::vector3df_SIMD diffuse;
    //Ks
    core::vector3df_SIMD specular;
    //Ke
    core::vector3df_SIMD emissive;
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
static_assert(sizeof(SPushConstants) <= asset::ICPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE, "doesnt fit in push constants");

constexpr uint32_t texturesOfInterest[TEX_OF_INTEREST_CNT]{
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

void createVirtualTexture(video::IVideoDriver* driver, core::vector<ICPUMeshBuffer*>& meshBuffers, const asset::COBJMetadata const* meta,
    core::smart_refctd_ptr<asset::ICPUVirtualTexture>& vt, core::smart_refctd_ptr<video::IGPUVirtualTexture>& outputGPUvt)
{
    core::unordered_map<core::smart_refctd_ptr<asset::ICPUImage>, STextureData> VTtexDataMap;

    // all pipelines will have the same metadata
    const asset::CMTLMetadata::CRenderpassIndependentPipeline* pipelineMetadata = nullptr;
    core::vector<commit_t> vt_commits;
    //modifying push constants and default fragment shader for VT
    for (auto it = meshBuffers.begin(); it != meshBuffers.end(); it++)
    {
        SPushConstants pushConsts;
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
            else {
                const asset::E_FORMAT fmt = img->getCreationParameters().format;
                texData = getTextureData(vt_commits, img.get(), vt.get(), uwrap, vwrap, borderColor);
                VTtexDataMap.insert({ img,texData });
            }

            static_assert(sizeof(texData) == sizeof(pushConsts.map_data[0]), "wrong reinterpret_cast");
            pushConsts.map_data[k] = reinterpret_cast<uint64_t*>(&texData)[0];
        }

        pipelineMetadata = static_cast<const asset::CMTLMetadata::CRenderpassIndependentPipeline*>(meta->getAssetSpecificMetadata((*it)->getPipeline()));

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
    assert(pipelineMetadata);

    outputGPUvt = core::make_smart_refctd_ptr<video::IGPUVirtualTexture>(driver, vt.get());
}

void setPipeline(IVideoDriver* driver, ICPUSpecializedShader* vs, ICPUSpecializedShader* fs,
    core::smart_refctd_ptr<IGPUBuffer>& vtxBuffer, core::smart_refctd_ptr<IGPUBuffer>& outputUBO, core::smart_refctd_ptr<IGPUBuffer>& virtualAttribBuffer,
    core::smart_refctd_ptr<IGPUVirtualTexture>& vt,
    core::smart_refctd_ptr<IGPUDescriptorSet>& outputGPUDescriptorSet0,
    core::smart_refctd_ptr<IGPUDescriptorSet>& outputGPUDescriptorSet1,
    core::smart_refctd_ptr<IGPUDescriptorSet>& outputGPUDescriptorSet2,
    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>& outputGpuPipeline)
{
    ICPUSpecializedShader* cpuShaders[2] = { vs, fs };
    auto gpuShaders = driver->getGPUObjectsFromAssets(cpuShaders, cpuShaders + 2);

    core::smart_refctd_ptr<IGPUDescriptorSetLayout> ds0Layout;
    core::smart_refctd_ptr<IGPUDescriptorSetLayout> ds1Layout;
    core::smart_refctd_ptr<IGPUDescriptorSetLayout> ds2Layout;
    {
        //should change it so it is not hardcoded?
        constexpr uint32_t vtBindingCnt = 2u;
        constexpr uint32_t vtSamplersCnt = 4u;
        auto samplers = core::make_refctd_dynamic_array< core::smart_refctd_dynamic_array<core::smart_refctd_ptr<video::IGPUSampler>>>(vtSamplersCnt);

        IGPUDescriptorSetLayout::SBinding b[4 + vtBindingCnt];
        b[0].binding = 0u; b[1].binding = 1u; b[2].binding = 2u; b[3].binding = 3u;
        b[0].type = b[1].type = b[2].type = EDT_UNIFORM_TEXEL_BUFFER;
        b[3].type = EDT_STORAGE_BUFFER;
        b[0].stageFlags = b[1].stageFlags = b[2].stageFlags = b[3].stageFlags = ISpecializedShader::ESS_VERTEX;
        b[0].count = 2u;
        b[1].count = 1u;
        b[2].count = 1u;
        b[3].count = 1u;
        
        vt->getDSlayoutBindings(&b[4], samplers->data(), PGTAB_BINDING, PHYSICAL_STORAGE_VIEWS_BINDING);

        b[4].stageFlags = ISpecializedShader::ESS_FRAGMENT;
        b[5].stageFlags = ISpecializedShader::ESS_FRAGMENT;

        /*for (uint32_t i = 0u; i < vtBindingCnt; i++)
        {
            ICPUDescriptorSetLayout::SBinding cpuBinding = *(bindings->data() + i);
            b[4 + i].binding = cpuBinding.binding;
            b[4 + i].type = cpuBinding.type;
            b[4 + i].count = cpuBinding.count;
            b[4 + i].stageFlags = ISpecializedShader::ESS_FRAGMENT;
            b[4 + i].samplers = driver->getGPUObjectsFromAssets(cpuBinding.samplers, cpuBinding.samplers + cpuBinding.count)->data();
        }*/

        ds0Layout = driver->createGPUDescriptorSetLayout(b, b + 6u);

        IGPUDescriptorSetLayout::SBinding b1;
        b1.binding = 0u;
        b1.type = EDT_UNIFORM_BUFFER;
        b1.stageFlags = ISpecializedShader::ESS_VERTEX;
        b1.count = 1u;

        ds1Layout = driver->createGPUDescriptorSetLayout(&b1, &b1 + 1);

        IGPUDescriptorSetLayout::SBinding b2;
        core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> ds2layout;
        {
            std::array<asset::ICPUDescriptorSetLayout::SBinding, 1> bnd;
            bnd[0].binding = 0u;
            bnd[0].count = 1u;
            bnd[0].samplers = nullptr;
            bnd[0].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
            bnd[0].type = asset::EDT_STORAGE_BUFFER;
            ds2layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bnd.data(), bnd.data() + bnd.size());
        }
    }

    asset::SPushConstantRange pcrng;
    pcrng.offset = 0;
    pcrng.size = 128;
    pcrng.stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;

    asset::SPushConstantRange pcRange = { asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD) };
    auto pipelineLayout = driver->createGPUPipelineLayout(&pcrng, &pcrng + 1, core::smart_refctd_ptr(ds0Layout), 
        core::smart_refctd_ptr(ds1Layout), core::smart_refctd_ptr(ds2Layout));

    outputGPUDescriptorSet0 = driver->createGPUDescriptorSet(std::move(ds0Layout));
    outputGPUDescriptorSet1 = driver->createGPUDescriptorSet(std::move(ds1Layout));
    outputGPUDescriptorSet2 = driver->createGPUDescriptorSet(std::move(ds2Layout));
    {
        //mesh packing stuff

        IGPUDescriptorSet::SWriteDescriptorSet w[5];
        w[0].arrayElement = 0u;
        w[1].arrayElement = 1u;
        w[2].arrayElement = 0u;
        w[3].arrayElement = 0u;
        w[4].arrayElement = 0u;
        w[0].count = w[1].count = w[2].count = w[3].count = w[4].count = 1u;
        w[0].binding = 0u; w[1].binding = 0u; w[2].binding = 1u; w[3].binding = 2u; w[4].binding = 3u;
        w[0].descriptorType = w[1].descriptorType = w[2].descriptorType = w[3].descriptorType = EDT_UNIFORM_TEXEL_BUFFER;
        w[4].descriptorType = EDT_STORAGE_BUFFER;
        w[0].dstSet = w[1].dstSet = w[2].dstSet = w[3].dstSet = w[4].dstSet = w[5].dstSet = outputGPUDescriptorSet0.get();

        IGPUDescriptorSet::SDescriptorInfo info[5];

        info[0].buffer.offset = 0u;
        info[0].buffer.size = vtxBuffer->getSize();
        info[0].desc = driver->createGPUBufferView(vtxBuffer.get(), EF_R32G32B32_SFLOAT);
        info[1].buffer.offset = 0u;
        info[1].buffer.size = vtxBuffer->getSize();
        info[1].desc = driver->createGPUBufferView(vtxBuffer.get(), EF_R32G32_SFLOAT);
        info[2].buffer.offset = 0u;
        info[2].buffer.size = vtxBuffer->getSize();
        info[2].desc = driver->createGPUBufferView(vtxBuffer.get(), EF_R32G32_SFLOAT);
        info[3].buffer.offset = 0u;
        info[3].buffer.size = vtxBuffer->getSize();
        info[3].desc = driver->createGPUBufferView(vtxBuffer.get(), EF_R32_UINT);

        //sampler buffers
        w[0].info = &info[0];
        w[1].info = &info[1];
        w[2].info = &info[2];
        w[3].info = &info[3];

        //offset tables
        info[4].buffer.offset = 0u;
        info[4].buffer.size = virtualAttribBuffer->getSize();
        info[4].desc = core::smart_refctd_ptr(virtualAttribBuffer);
        w[4].info = &info[4];

        driver->updateDescriptorSets(5u, w, 0u, nullptr);

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

        //vt stuff
        auto sizes = vt->getDescriptorSetWrites(nullptr, nullptr, nullptr);
        auto writesVT = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<video::IGPUDescriptorSet::SWriteDescriptorSet>>(sizes.first);
        auto infoVT = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<video::IGPUDescriptorSet::SDescriptorInfo>>(sizes.second);

        vt->getDescriptorSetWrites(writesVT->data(), infoVT->data(), outputGPUDescriptorSet0.get(), PGTAB_BINDING, PHYSICAL_STORAGE_VIEWS_BINDING);

        driver->updateDescriptorSets(writesVT->size(), writesVT->data(), 0u, nullptr);

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
    qnc->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fs, "../../tmp/normalCache101010.sse", true);

    // register the zip
    device->getFileSystem()->addFileArchive("../../media/sponza.zip");

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

    core::vector<MbPipelineRange> ranges = sortMeshBuffersByPipeline(meshBuffers);

    // all pipelines will have the same metadata
    const asset::CMTLMetadata::CRenderpassIndependentPipeline* pipelineMetadata = nullptr;
    core::vector<commit_t> vt_commits;
    const auto meta = meshes_bundle.getMetadata()->selfCast<const asset::COBJMetadata>();

    core::smart_refctd_ptr<asset::ICPUVirtualTexture> vt = core::make_smart_refctd_ptr<asset::ICPUVirtualTexture>([](asset::E_FORMAT_CLASS) -> uint32_t { return TILES_PER_DIM_LOG2; }, PAGE_SZ_LOG2, PAGE_PADDING, MAX_ALLOCATABLE_TEX_SZ_LOG2);
    core::smart_refctd_ptr<video::IGPUVirtualTexture> gpuvt = nullptr;
    createVirtualTexture(driver, meshBuffers, meta, vt, gpuvt);

    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuPipeline;
    core::smart_refctd_ptr<IGPUDescriptorSet> ds0;
    core::smart_refctd_ptr<IGPUDescriptorSet> ds1;
    core::smart_refctd_ptr<IGPUDescriptorSet> ds2;
    core::smart_refctd_ptr<IGPUBuffer> ubo;
    DrawIndexedIndirectInput mdiCallParams;
    {
        auto* pipeline = meshBuffers[0]->getPipeline();

        auto* vtxShader = pipeline->getShaderAtIndex(asset::ICPURenderpassIndependentPipeline::ESSI_VERTEX_SHADER_IX);
        core::smart_refctd_ptr<ICPUSpecializedShader> vs = createModifiedVertexShader(vtxShader);
        auto* fragShader = pipeline->getShaderAtIndex(asset::ICPURenderpassIndependentPipeline::ESSI_FRAGMENT_SHADER_IX);
        //ICPUSpecializedShader* fs = IAsset::castDown<ICPUSpecializedShader>(am->getAsset("../shader.frag", lp).getContents().begin()->get());
        core::smart_refctd_ptr<ICPUSpecializedShader> fs = createModifiedFragShader(fragShader, vt.get());
        core::smart_refctd_ptr<IGPUBuffer> virtualAttribTable;

        packMeshBuffers(driver, meshBuffers, mdiCallParams, virtualAttribTable);

        setPipeline(driver, vs.get(), fs.get(), mdiCallParams.vtxBuffer.buffer, ubo, virtualAttribTable, gpuvt, ds0, ds1, ds2, gpuPipeline);
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