// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/ICPUMeshBuffer.h"

#include "nbl/asset/utils/CSPIRVIntrospector.h"
#include "nbl/asset/utils/spvUtils.h"

#include "nbl_spirv_cross/spirv_parser.hpp"
#include "nbl_spirv_cross/spirv_cross.hpp"

namespace nbl::asset
{

namespace
{
E_FORMAT spvImageFormat2E_FORMAT(spv::ImageFormat _imgfmt)
{
    using namespace spv;
    constexpr E_FORMAT convert[]
    {
        EF_UNKNOWN,
        EF_R32G32B32A32_SFLOAT,
        EF_R16G16B16A16_SFLOAT,
        EF_R32_SFLOAT,
        EF_R8G8B8A8_UNORM,
        EF_R8G8B8A8_SNORM,
        EF_R32G32_SFLOAT,
        EF_R16G16_SFLOAT,
        EF_B10G11R11_UFLOAT_PACK32,
        EF_R16_SFLOAT,
        EF_R16G16B16A16_UNORM,
        EF_A2B10G10R10_UNORM_PACK32,
        EF_R16G16_UNORM,
        EF_R8G8_UNORM,
        EF_R16G16_UNORM,
        EF_R8_UNORM,
        EF_R16G16B16A16_SNORM,
        EF_R16G16_SNORM,
        EF_R8G8_SNORM,
        EF_R16_SNORM,
        EF_R8_SNORM,
        EF_R32G32B32A32_SINT,
        EF_R16G16B16A16_SINT,
        EF_R8G8B8A8_SINT,
        EF_R32_SINT,
        EF_R32G32_SINT,
        EF_R16G16_SINT,
        EF_R8G8_SINT,
        EF_R16_SINT,
        EF_R8_SINT,
        EF_R32G32B32A32_UINT,
        EF_R16G16B16A16_UINT,
        EF_R8G8B8A8_UINT,
        EF_R32_UINT,
        EF_A2B10G10R10_UINT_PACK32,
        EF_R32G32_UINT,
        EF_R16G16_UINT,
        EF_R8G8_UINT,
        EF_R16_UINT,
        EF_R8_UINT,
        EF_UNKNOWN
    };
    return convert[_imgfmt];
}
}//anonymous ns

core::smart_refctd_ptr<const CSPIRVIntrospector::CIntrospectionData> CSPIRVIntrospector::introspect(const SIntrospectionParams& params, bool insertToCache)
{
    if (!params.cpuShader)
        return nullptr;
    if (params.cpuShader->getContentType() != IShader::E_CONTENT_TYPE::ECT_SPIRV)
        return nullptr;
    auto introspectionData = m_introspectionCache.find(params);
    if (introspectionData != m_introspectionCache.end())
    {
        return introspectionData->second;
    }

    const ICPUBuffer* spv = params.cpuShader->getContent();
    spirv_cross::Compiler comp(reinterpret_cast<const uint32_t*>(spv->getPointer()), spv->getSize()/4u);
    auto introspection = doIntrospection(comp,params.entryPoint,params.cpuShader->getStage());
    
    if (insertToCache)
        m_introspectionCache[params] = introspection;

    return introspection;
}

bool CSPIRVIntrospector::introspectAllShaders(core::smart_refctd_ptr<const CIntrospectionData>* outIntrospections, const core::SRange<const ICPUSpecializedShader* const>& _shaders)
{
    auto it = outIntrospections;
    for (auto shader : _shaders)
    {
        const auto& specInfo = shader->getSpecializationInfo();
        SIntrospectionParams params = { specInfo.entryPoint, core::smart_refctd_ptr<const ICPUShader>(shader->getUnspecialized()) };
        *it = introspect(params);
        if (!*it++)
            return false;
    }
    return true;
}

static IDescriptor::E_TYPE resType2descType(E_SHADER_RESOURCE_TYPE _t)
{
    switch (_t)
    {
        case ESRT_COMBINED_IMAGE_SAMPLER:
            return IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER;
            break;
        case ESRT_STORAGE_IMAGE:
            return IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
            break;
        case ESRT_UNIFORM_TEXEL_BUFFER:
            return IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER;
            break;
        case ESRT_STORAGE_TEXEL_BUFFER:
            return IDescriptor::E_TYPE::ET_STORAGE_TEXEL_BUFFER;
            break;
        case ESRT_UNIFORM_BUFFER:
            return IDescriptor::E_TYPE::ET_UNIFORM_BUFFER;
            break;
        case ESRT_STORAGE_BUFFER:
            return IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
            break;
        default:
            break;
    }
    return IDescriptor::E_TYPE::ET_COUNT;
}

template<E_SHADER_RESOURCE_TYPE restype>
static std::pair<bool, IImageView<ICPUImage>::E_TYPE> imageInfoFromResource(const SShaderResource<restype>& _res)
{
    return {_res.shadow, _res.viewType};
}

std::pair<bool, IImageView<ICPUImage>::E_TYPE> CSPIRVIntrospector::getImageInfoFromIntrospection(uint32_t _set, uint32_t _binding, const core::SRange<const ICPUSpecializedShader* const>& _shaders)
{
    std::pair<bool, IImageView<ICPUImage>::E_TYPE> fail = { false, IImageView<ICPUImage>::ET_COUNT };

    core::smart_refctd_ptr<const CIntrospectionData> introspections[MAX_STAGE_COUNT] = {nullptr};
    if (!introspectAllShaders(introspections,_shaders))
        return fail;

    for (auto i=0; i<_shaders.size(); i++)
    {
        for (const auto& bnd : introspections[i]->descriptorSetBindings[_set])
        {
            if (bnd.type==ESRT_COMBINED_IMAGE_SAMPLER || bnd.type==ESRT_STORAGE_IMAGE)
                return (bnd.type==ESRT_COMBINED_IMAGE_SAMPLER) ? imageInfoFromResource(bnd.get<ESRT_COMBINED_IMAGE_SAMPLER>()) : imageInfoFromResource(bnd.get<ESRT_STORAGE_IMAGE>());
        }
    }

    return fail;
}

core::smart_refctd_dynamic_array<SPushConstantRange> CSPIRVIntrospector::createPushConstantRangesFromIntrospection_impl(core::smart_refctd_ptr<const CIntrospectionData>* const introspections, const core::SRange<const ICPUSpecializedShader* const>& shaders)
{
    core::vector<SPushConstantRange> ranges[MAX_STAGE_COUNT];
    {
        auto r = ranges;
        auto introspection = introspections;
        for (auto shader : shaders)
        {
            auto& pc = (*introspection)->pushConstant;
            if (pc.present)
            {
                auto shaderStage = shader->getStage();

                r->reserve(100u);
                auto& members = pc.info.members;
                r->push_back({shaderStage, members.array[0].offset, members.array[0].size});
                for (uint32_t i = 1u; i < members.count; ++i)
                {
                    auto& last = r->back();
                    if (members.array[i].offset == (last.offset + last.size))
                        last.size += members.array[i].size;
                    else
                        r->push_back({shaderStage, members.array[i].offset, members.array[i].size});
                }
            }
            r++;
            introspection++;
        }
    }

    core::vector<SPushConstantRange> merged;

    SPushConstantRange rngToPush; rngToPush.offset = 0u; rngToPush.size = 0u;
    uint32_t stageFlags = 0u;
    for (uint32_t i = 0u; i < ICPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE/sizeof(uint32_t); i += sizeof(uint32_t))
    {
        SPushConstantRange curr; curr.offset = i; curr.size = sizeof(uint32_t);

        uint32_t tmpFlags = 0u;
        for (uint32_t stg=0u; stg<shaders.size(); stg++)
            for (const SPushConstantRange& rng : ranges[stg])
                if (curr.overlap(rng))
                    tmpFlags |= rng.stageFlags;

        if (!i)
            stageFlags = tmpFlags;

        if (!tmpFlags)
            continue;
        if (tmpFlags == stageFlags)
            rngToPush.size += sizeof(uint32_t);
        else
        {
            rngToPush.stageFlags = static_cast<ICPUShader::E_SHADER_STAGE>(stageFlags);
            merged.push_back(rngToPush);
            stageFlags = 0u;

            rngToPush.offset = i;
            rngToPush.size = sizeof(uint32_t);
        }
    }
    if (stageFlags)
    {
        rngToPush.stageFlags = static_cast<ICPUShader::E_SHADER_STAGE>(stageFlags);
        merged.push_back(rngToPush);
    }

    core::smart_refctd_dynamic_array<SPushConstantRange> rngsArray;
    if (merged.size())
    {
        rngsArray = core::make_refctd_dynamic_array<decltype(rngsArray)>(merged.size());
        memcpy(rngsArray->data(),merged.data(),rngsArray->size()*sizeof(SPushConstantRange));
    }

    return rngsArray;
}

core::smart_refctd_ptr<ICPUDescriptorSetLayout> CSPIRVIntrospector::createApproximateDescriptorSetLayoutFromIntrospection_impl(uint32_t _set, core::smart_refctd_ptr<const CIntrospectionData>* const introspections, const core::SRange<const ICPUSpecializedShader* const>& shaders)
{
    uint32_t checkedDescsCnt[MAX_STAGE_COUNT]{};

    core::vector<ICPUDescriptorSetLayout::SBinding> bindings;
    bindings.reserve(100u); //preallocating mem for 100 bindings almost ensures no reallocs
    while (1)
    {
        uint32_t stageFlags = 0u;
        ICPUDescriptorSetLayout::SBinding binding;
        binding.binding = ~0u;
        binding.samplers = nullptr;

        bool anyStageNotFinished = false;
        for (auto i=0u; i<shaders.size(); i++)
        {
            auto& introBindings = introspections[i]->descriptorSetBindings[_set];
            if (checkedDescsCnt[i] == introBindings.size())
                continue;
            anyStageNotFinished = true;

            const auto& introBinding = introBindings[checkedDescsCnt[i]].binding;
            if (introBinding < binding.binding)
                binding.binding = introBinding;
        }
        if (!anyStageNotFinished) //all shader stages finished
            break;

        const ICPUSpecializedShader* refShader = nullptr;
        uint32_t refIndex = ~0u;
        core::smart_refctd_ptr<const CIntrospectionData> refIntro = nullptr;
        {
            auto checkedDescsCntIt = checkedDescsCnt;
            auto introspectionIt = introspections;
            for (auto shader : shaders)
            {
                auto& introBindings = (*introspectionIt)->descriptorSetBindings[_set];
                if (*checkedDescsCntIt!=introBindings.size())
                {
                    const auto& introBinding = introBindings[*checkedDescsCntIt].binding;
                    if (introBinding==binding.binding)
                    {
                        stageFlags |= shader->getStage();
                        refIndex = (*checkedDescsCntIt)++;
                        refIntro = *introspectionIt;
                        refShader = shader;
                    }
                }
                introspectionIt++;
                checkedDescsCntIt++;
            }
        }

        auto& introBnd = refIntro->descriptorSetBindings[_set][refIndex];
        binding.type = resType2descType(introBnd.type);
        binding.stageFlags = static_cast<ICPUShader::E_SHADER_STAGE>(stageFlags);
        binding.count = introBnd.descriptorCount;
        if (introBnd.descCountIsSpecConstant)
        {
            auto& specInfo = refShader->getSpecializationInfo();
            auto val = specInfo.getSpecializationByteValue(binding.count);
            assert(val.second == 4ull);
            memcpy(&binding.count, val.first, 4ull);
        }

        bindings.push_back(binding);
    }

    if (bindings.size())
        return core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(bindings.data(), bindings.data()+bindings.size());
    return nullptr; //returns nullptr if no descriptors are bound in set number `_set`
}

core::smart_refctd_ptr<ICPUPipelineLayout> CSPIRVIntrospector::createApproximatePipelineLayoutFromIntrospection_impl(core::smart_refctd_ptr<const CIntrospectionData>* const introspections, const core::SRange<const ICPUSpecializedShader* const>& shaders)
{
    core::smart_refctd_ptr<ICPUDescriptorSetLayout> dsLayout[ICPUPipelineLayout::DESCRIPTOR_SET_COUNT];
    for (uint32_t i = 0u; i < ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
        dsLayout[i] = createApproximateDescriptorSetLayoutFromIntrospection_impl(i,introspections,shaders);

    auto pcRanges = createPushConstantRangesFromIntrospection_impl(introspections,shaders);

    return core::make_smart_refctd_ptr<ICPUPipelineLayout>(
        (pcRanges ? pcRanges->begin() : nullptr), (pcRanges ? pcRanges->end() : nullptr),
        std::move(dsLayout[0]), std::move(dsLayout[1]), std::move(dsLayout[2]), std::move(dsLayout[3])
    );
}

static E_FORMAT glslType2E_FORMAT(E_GLSL_VAR_TYPE _t, uint32_t _e)
{
    static const E_FORMAT retval[6][4]
    {
        {EF_R64_UINT, EF_R64G64_UINT, EF_R64G64B64_UINT, EF_R64G64B64A64_UINT},
        {EF_R64_SINT, EF_R64G64_SINT, EF_R64G64B64_SINT, EF_R64G64B64A64_SINT},
        {EF_R32_UINT, EF_R32G32_UINT, EF_R32G32B32_UINT, EF_R32G32B32A32_UINT},
        {EF_R32_SINT, EF_R32G32_SINT, EF_R32G32B32_SINT, EF_R32G32B32A32_SINT},
        {EF_R64_SFLOAT, EF_R64G64_SFLOAT, EF_R64G64B64_SFLOAT, EF_R64G64B64A64_SFLOAT},
        {EF_R32_SFLOAT, EF_R32G32_SFLOAT, EF_R32G32B32_SFLOAT, EF_R32G32B32A32_SFLOAT}
    };

    return retval[_t][_e];
}

core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> CSPIRVIntrospector::createApproximateRenderpassIndependentPipelineFromIntrospection(const core::SRange<ICPUSpecializedShader* const>& _shaders)
{
    core::smart_refctd_ptr<const CIntrospectionData> introspections[MAX_STAGE_COUNT] = { nullptr };
    if (!introspectAllShaders(introspections,{_shaders.begin(),_shaders.end()}))
        return nullptr;

    auto vs_introspection = introspections;
    for (auto shader : _shaders)
    {
        if (shader->getStage()==ICPUShader::ESS_VERTEX)
            break;
        vs_introspection++;
    }
    if (vs_introspection==introspections+_shaders.size())
        return nullptr;

    //
    SVertexInputParams vtxInput;
    {
        uint32_t reloffset = 0u;
        for (const auto& io : (*vs_introspection)->inputOutput)
        {
            if (io.type == ESIT_STAGE_INPUT)
            {
                auto& attr = vtxInput.attributes[io.location];
                attr.binding = io.location; //assume attrib number = binding number
                attr.format = glslType2E_FORMAT(io.glslType.basetype, io.glslType.elements);
                attr.relativeOffset = reloffset;

                //all formats returned by glslType2E_FORMAT() already are multiple-of-4 bytes so no need to pad
                reloffset += getTexelOrBlockBytesize(static_cast<E_FORMAT>(attr.format));

                vtxInput.enabledAttribFlags |= (1u << io.location);
            }
        }
        vtxInput.enabledBindingFlags = vtxInput.enabledAttribFlags;

        for (uint32_t i = 0u; i < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; ++i)
        {
            if (vtxInput.enabledBindingFlags & (1u<<i))
                vtxInput.bindings[i].stride = reloffset;
        }
    }

    //all except vertex input are defaulted
    SBlendParams blending;
    SPrimitiveAssemblyParams primAssembly;
    SRasterizationParams raster;

    auto layout = createApproximatePipelineLayoutFromIntrospection_impl(introspections,{_shaders.begin(),_shaders.end()});

    return core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(
        std::move(layout),_shaders.begin(),_shaders.end(),
        vtxInput, blending,primAssembly, raster
    );
}

static E_GLSL_VAR_TYPE spvcrossType2E_TYPE(spirv_cross::SPIRType::BaseType _basetype)
{
    switch (_basetype)
    {
    case spirv_cross::SPIRType::Int:
        return EGVT_I32;
    case spirv_cross::SPIRType::UInt:
        return EGVT_U32;
    case spirv_cross::SPIRType::Float:
        return EGVT_F32;
    case spirv_cross::SPIRType::Int64:
        return EGVT_I64;
    case spirv_cross::SPIRType::UInt64:
        return EGVT_U64;
    case spirv_cross::SPIRType::Double:
        return EGVT_F64;
    default:
        return EGVT_UNKNOWN_OR_STRUCT;
    }
}
static IImageView<ICPUImage>::E_TYPE spvcrossImageType2ImageView(const spirv_cross::SPIRType::ImageType& _type)
{
    static constexpr std::array<IImageView<ICPUImage>::E_TYPE, 8> viewType = { {
        IImageView<ICPUImage>::ET_1D,
        IImageView<ICPUImage>::ET_1D_ARRAY,
        IImageView<ICPUImage>::ET_2D,
        IImageView<ICPUImage>::ET_2D_ARRAY,
        IImageView<ICPUImage>::ET_3D,
        IImageView<ICPUImage>::ET_3D,
        IImageView<ICPUImage>::ET_CUBE_MAP,
        IImageView<ICPUImage>::ET_CUBE_MAP_ARRAY
    } };

    return viewType[_type.dim*2u + _type.arrayed];
}

core::smart_refctd_ptr<const CSPIRVIntrospector::CIntrospectionData> CSPIRVIntrospector::doIntrospection(
    spirv_cross::Compiler& _comp, const std::string& entryPoint, const IShader::E_SHADER_STAGE shaderStage) const
{
    spv::ExecutionModel stage = ESS2spvExecModel(shaderStage);
    if (stage == spv::ExecutionModelMax)
        return nullptr;

    core::smart_refctd_ptr<CIntrospectionData> introData = core::make_smart_refctd_ptr<CIntrospectionData>();
    introData->pushConstant.present = false;
    auto addResource_common = [&introData, &_comp] (const spirv_cross::Resource& r, E_SHADER_RESOURCE_TYPE restype, const mapId2SpecConst_t& _mapId2sconst) -> SShaderResourceVariant& {
        const uint32_t descSet = _comp.get_decoration(r.id, spv::DecorationDescriptorSet);
        assert(descSet < 4u);
        introData->descriptorSetBindings[descSet].emplace_back();

        SShaderResourceVariant& res = introData->descriptorSetBindings[descSet].back();
        res.type = restype;
        res.binding = _comp.get_decoration(r.id, spv::DecorationBinding);

        res.descriptorCount = 1u;
        res.descCountIsSpecConstant = false;
        const spirv_cross::SPIRType& type = _comp.get_type(r.type_id);
        // assuming only 1D arrays because i don't know how desc set layout binding is constructed when it's let's say 2D array (e.g. uniform sampler2D smplr[4][5]; is it even legal?)
        if (type.array.size()) // is array
        {
            // the API for this spec constant checking is truly messed up
            res.descriptorCount = type.array[0]; // ID of spec constant if size is spec constant
            res.descCountIsSpecConstant = !type.array_size_literal[0];
            if (res.descCountIsSpecConstant)
            {
                auto sc_itr = _mapId2sconst.find(res.descriptorCount);
                assert(sc_itr!=_mapId2sconst.end());
                auto sc = sc_itr->second;
                res.count_specID = sc->id;
            }
        }

        return res;
    };
    auto addInfo_common = [&introData, &_comp](const spirv_cross::Resource& r, E_SHADER_INFO_TYPE type) ->SShaderInfoVariant& {
        introData->inputOutput.emplace_back();
        SShaderInfoVariant& info = introData->inputOutput.back();
        info.type = type;
        info.location = _comp.get_decoration(r.id, spv::DecorationLocation);
        return info;
    };

    _comp.set_entry_point(entryPoint, stage);

    // spec constants
    spirv_cross::SmallVector<spirv_cross::SpecializationConstant> sconsts = _comp.get_specialization_constants();
    mapId2SpecConst_t mapId2SpecConst;
    introData->specConstants.reserve(sconsts.size());
    for (size_t i = 0u; i < sconsts.size(); ++i)
    {
        CIntrospectionData::SSpecConstant specConst;
        specConst.id = sconsts[i].constant_id;
        specConst.name = _comp.get_name(sconsts[i].id);

        const spirv_cross::SPIRConstant& sconstval = _comp.get_constant(sconsts[i].id);
        const spirv_cross::SPIRType& type = _comp.get_type(sconstval.constant_type);
        specConst.byteSize = calcBytesizeforType(_comp, type);
        specConst.type = spvcrossType2E_TYPE(type.basetype);

        switch (type.basetype)
        {
        case spirv_cross::SPIRType::Int:
            specConst.defaultValue.i32 = sconstval.scalar_i32();
            break;
        case spirv_cross::SPIRType::UInt:
            specConst.defaultValue.u32 = sconstval.scalar_i32();
            break;
        case spirv_cross::SPIRType::Float:
            specConst.defaultValue.f32 = sconstval.scalar_f32();
            break;
        case spirv_cross::SPIRType::Int64:
            specConst.defaultValue.i64 = sconstval.scalar_i64();
            break;
        case spirv_cross::SPIRType::UInt64:
            specConst.defaultValue.u64 = sconstval.scalar_u64();
            break;
        case spirv_cross::SPIRType::Double:
            specConst.defaultValue.f64 = sconstval.scalar_f64();
            break;
        default: break;
        }

        auto where = std::lower_bound(introData->specConstants.begin(), introData->specConstants.end(), specConst, [](const auto& _lhs, const auto& _rhs) { return _lhs.id < _rhs.id; });
        introData->specConstants.insert(where, specConst);
    }
    for (const auto& sc : sconsts)
    {
        CIntrospectionData::SSpecConstant dummy;
        dummy.id = sc.constant_id;
        auto it = std::lower_bound(introData->specConstants.begin(), introData->specConstants.end(), dummy, [](const auto& _lhs, const auto& _rhs) { return _lhs.id < _rhs.id; });
        if (it==introData->specConstants.end() || it->id!=dummy.id)
            continue;
        mapId2SpecConst.insert({sc.id,&*it});
    }

    spirv_cross::ShaderResources resources = _comp.get_shader_resources(_comp.get_active_interface_variables());
    for (const spirv_cross::Resource& r : resources.uniform_buffers)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_UNIFORM_BUFFER, mapId2SpecConst);
        static_cast<impl::SShaderMemoryBlock&>(res.get<ESRT_UNIFORM_BUFFER>()).name = r.name;
        shaderMemBlockIntrospection(_comp, static_cast<impl::SShaderMemoryBlock&>(res.get<ESRT_UNIFORM_BUFFER>()), r.base_type_id, r.id, mapId2SpecConst);
    }
    for (const spirv_cross::Resource& r : resources.storage_buffers)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_STORAGE_BUFFER, mapId2SpecConst);
        static_cast<impl::SShaderMemoryBlock&>(res.get<ESRT_STORAGE_BUFFER>()).name = r.name;
        shaderMemBlockIntrospection(_comp, static_cast<impl::SShaderMemoryBlock&>(res.get<ESRT_STORAGE_BUFFER>()), r.base_type_id, r.id, mapId2SpecConst);
    }
    for (const spirv_cross::Resource& r : resources.subpass_inputs)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_INPUT_ATTACHMENT, mapId2SpecConst);
        res.get<ESRT_INPUT_ATTACHMENT>().inputAttachmentIndex = _comp.get_decoration(r.id, spv::DecorationInputAttachmentIndex);
    }
    for (const spirv_cross::Resource& r : resources.storage_images)
    {
		const spirv_cross::SPIRType& type = _comp.get_type(r.type_id);
        const bool buffer = type.image.dim == spv::DimBuffer;
        SShaderResourceVariant& res = addResource_common(r, buffer ? ESRT_STORAGE_TEXEL_BUFFER : ESRT_STORAGE_IMAGE, mapId2SpecConst);
        if (!buffer)
        {
            res.get<ESRT_STORAGE_IMAGE>().format = spvImageFormat2E_FORMAT(type.image.format);
            res.get<ESRT_STORAGE_IMAGE>().viewType = spvcrossImageType2ImageView(type.image);
            res.get<ESRT_STORAGE_IMAGE>().shadow = type.image.depth;
        }
    }
    for (const spirv_cross::Resource& r : resources.sampled_images)
    {
		const spirv_cross::SPIRType& type = _comp.get_type(r.type_id);
        const bool buffer = type.image.dim == spv::DimBuffer;
        SShaderResourceVariant& res = addResource_common(r, buffer ? ESRT_UNIFORM_TEXEL_BUFFER : ESRT_COMBINED_IMAGE_SAMPLER, mapId2SpecConst);
        if (!buffer)
        {
            res.get<ESRT_COMBINED_IMAGE_SAMPLER>().viewType = spvcrossImageType2ImageView(type.image);
            res.get<ESRT_COMBINED_IMAGE_SAMPLER>().shadow = type.image.depth;
            res.get<ESRT_COMBINED_IMAGE_SAMPLER>().multisample = type.image.ms;
        }
    }
    for (const spirv_cross::Resource& r : resources.separate_images)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_SAMPLED_IMAGE, mapId2SpecConst);
    }
    for (const spirv_cross::Resource& r : resources.separate_samplers)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_SAMPLER, mapId2SpecConst);
    }
    for (auto& descSet : introData->descriptorSetBindings)
        std::sort(descSet.begin(), descSet.end(), [](const SShaderResourceVariant& _lhs, const SShaderResourceVariant& _rhs) { return _lhs.binding < _rhs.binding; });


    auto getStageIOtype = [&_comp](uint32_t _base_type_id)
    {
        const auto& type = _comp.get_type(_base_type_id);
        decltype(SShaderInfoVariant::glslType) glslType;
        glslType.basetype = spvcrossType2E_TYPE(type.basetype);
        glslType.elements = type.vecsize;

        return glslType;
    };

    // in/out
    for (const spirv_cross::Resource& r : resources.stage_inputs)
    {
        SShaderInfoVariant& res = addInfo_common(r, ESIT_STAGE_INPUT);
        res.glslType = getStageIOtype(r.base_type_id);
    }
    for (const spirv_cross::Resource& r : resources.stage_outputs)
    {
        SShaderInfoVariant& res = addInfo_common(r, ESIT_STAGE_OUTPUT);
        res.glslType = getStageIOtype(r.base_type_id);

        res.get<ESIT_STAGE_OUTPUT>().colorIndex = _comp.get_decoration(r.id, spv::DecorationIndex);
    }
    std::sort(introData->inputOutput.begin(), introData->inputOutput.end(), [](const SShaderInfoVariant& _lhs, const SShaderInfoVariant& _rhs) { return _lhs.location < _rhs.location; });

    // push constants
    if (resources.push_constant_buffers.size())
    {
        const spirv_cross::Resource& r = resources.push_constant_buffers.front();
        introData->pushConstant.present = true;
        static_cast<impl::SShaderMemoryBlock&>(introData->pushConstant.info).name = r.name;
        shaderMemBlockIntrospection(_comp, static_cast<impl::SShaderMemoryBlock&>(introData->pushConstant.info), r.base_type_id, r.id, mapId2SpecConst);
    }

    return introData;
}

namespace {
    struct StackElement
    {
        impl::SShaderMemoryBlock::SMember::SMembers& membersDst;
        const spirv_cross::SPIRType& parentType;
        uint32_t baseOffset;
    };
    using mapId2SpecConst_t = core::unordered_map<uint32_t, const CSPIRVIntrospector::CIntrospectionData::SSpecConstant*>;
}
static void introspectStructType(spirv_cross::Compiler& _comp, impl::SShaderMemoryBlock::SMember::SMembers& _dstMembers, const spirv_cross::SPIRType& _parentType, const spirv_cross::SmallVector<spirv_cross::TypeID>& _allMembersTypes, uint32_t _baseOffset, const mapId2SpecConst_t& _mapId2sconst, core::stack<StackElement>& _pushStack) {
    using MembT = impl::SShaderMemoryBlock::SMember;

    auto MemberDefault = [] {
        MembT m;
        m.count = 1u;
        m.countIsSpecConstant = false;
        m.offset = 0u;
        m.size = 0u;
        m.arrayStride = 0u;
        m.mtxStride = 0u;
        m.mtxRowCnt = m.mtxColCnt = 1u;
        m.rowMajor = false;
        m.type = EGVT_UNKNOWN_OR_STRUCT;
        m.members.array = nullptr;
        m.members.count = 0u;
        return m;
    };

    const uint32_t memberCnt = _allMembersTypes.size();
    _dstMembers.array = _NBL_NEW_ARRAY(MembT, memberCnt);
    _dstMembers.count = memberCnt;
    std::fill(_dstMembers.array, _dstMembers.array+memberCnt, MemberDefault());
    for (uint32_t m = 0u; m < memberCnt; ++m)
	{
        MembT& member = _dstMembers.array[m];
        const spirv_cross::SPIRType& mtype = _comp.get_type(_allMembersTypes[m]);

        member.name = _comp.get_member_name(_parentType.self, m);
        member.size = _comp.get_declared_struct_member_size(_parentType, m);
        member.offset = _baseOffset + _comp.type_struct_member_offset(_parentType, m);
        member.rowMajor = _comp.get_member_decoration(_parentType.self, m, spv::DecorationRowMajor);
        member.type = spvcrossType2E_TYPE(mtype.basetype);
        member.arrayStride = 0u;

        // if array, then we can get array stride from decoration (via spirv-cross)
        // otherwise arrayStride is left with value 0
        if (mtype.array.size())
        {
            member.count = mtype.array[0];
            member.arrayStride = _comp.type_struct_member_array_stride(_parentType, m);
            member.countIsSpecConstant = !mtype.array_size_literal[0];
            if (member.countIsSpecConstant)
			{
                auto sc_itr = _mapId2sconst.find(member.count);
                assert(sc_itr!=_mapId2sconst.end());
                auto sc = sc_itr->second;
                member.count_specID = sc->id;
            }
        }

        if (mtype.basetype == spirv_cross::SPIRType::Struct) //recursive introspection done in DFS manner (and without recursive calls)
            _pushStack.push({member.members, mtype, member.offset});
        else
		{
            member.mtxRowCnt = mtype.vecsize;
            member.mtxColCnt = mtype.columns;
            if (member.mtxColCnt > 1u)
                member.mtxStride = _comp.type_struct_member_matrix_stride(_parentType, m);
        }
    }
}

void CSPIRVIntrospector::shaderMemBlockIntrospection(spirv_cross::Compiler& _comp, impl::SShaderMemoryBlock& _res, uint32_t _blockBaseTypeID, uint32_t _varID, const mapId2SpecConst_t& _sortedId2sconst) const
{
    using MembT = impl::SShaderMemoryBlock::SMember;

    core::stack<StackElement> introspectionStack;
    const spirv_cross::SPIRType& type = _comp.get_type(_blockBaseTypeID);
    introspectionStack.push({_res.members, type, 0u});
    while (!introspectionStack.empty()) {
        StackElement e = introspectionStack.top();
        introspectionStack.pop();
        introspectStructType(_comp, e.membersDst, e.parentType, e.parentType.member_types, 0u, _sortedId2sconst, introspectionStack);
    }

    _res.size = _res.rtSizedArrayOneElementSize = _comp.get_declared_struct_size(type);
    const spirv_cross::SPIRType& lastType = _comp.get_type(type.member_types.back());
    if (lastType.array.size() && lastType.array_size_literal[0] && lastType.array[0] == 0u)
        _res.rtSizedArrayOneElementSize += _res.members.array[_res.members.count-1u].arrayStride;

    spirv_cross::Bitset flags = _comp.get_buffer_block_flags(_varID);
    _res.restrict_ = flags.get(spv::DecorationRestrict);
    _res.volatile_ = flags.get(spv::DecorationVolatile);
    _res.coherent = flags.get(spv::DecorationCoherent);
    _res.readonly = flags.get(spv::DecorationNonWritable);
    _res.writeonly = flags.get(spv::DecorationNonReadable);
}

size_t CSPIRVIntrospector::calcBytesizeforType(spirv_cross::Compiler& _comp, const spirv_cross::SPIRType & _type) const
{
    size_t bytesize = 0u;
    switch (_type.basetype)
    {
    case spirv_cross::SPIRType::Char:
    case spirv_cross::SPIRType::SByte:
    case spirv_cross::SPIRType::UByte:
        bytesize = 1u;
        break;
    case spirv_cross::SPIRType::Short:
    case spirv_cross::SPIRType::UShort:
    case spirv_cross::SPIRType::Half:
        bytesize = 2u;
        break;
    //Vulkan spec: "If the specialization constant is of type boolean, size must be the byte size of VkBool32"
    //https://vulkan.lunarg.com/doc/view/1.0.30.0/linux/vkspec.chunked/ch09s07.html
    case spirv_cross::SPIRType::Boolean:
    case spirv_cross::SPIRType::Int:
    case spirv_cross::SPIRType::UInt:
    case spirv_cross::SPIRType::Float:
        bytesize = 4u;
        break;
    case spirv_cross::SPIRType::Int64:
    case spirv_cross::SPIRType::UInt64:
    case spirv_cross::SPIRType::Double:
        bytesize = 8u;
        break;
    case spirv_cross::SPIRType::Struct:
        bytesize = _comp.get_declared_struct_size(_type);
        assert(_type.columns > 1u || _type.vecsize > 1u); // something went wrong (cannot have matrix/vector of struct type)
        break;
    default:
        assert(0);
        break;
    }
    bytesize *= _type.vecsize * _type.columns; //vector or matrix
    if (_type.array.size()) //array
        bytesize *= _type.array[0];

    return bytesize;
}


static void deinitShdrMemBlock(impl::SShaderMemoryBlock& _res)
{
    using MembersT = impl::SShaderMemoryBlock::SMember::SMembers;
    core::stack<MembersT> stack;
    core::queue<MembersT> q;
    if (_res.members.array)
        q.push(_res.members);
    while (!q.empty()) {//build stack
        MembersT curr = q.front();
        stack.push(curr);
        q.pop();
        for (uint32_t i = 0u; i < curr.count; ++i) {
            const auto& m = curr.array[i];
            if (m.members.array)
                q.push(m.members);
        }
    }
    while (!stack.empty()) {
        MembersT m = stack.top();
        stack.pop();
        _NBL_DELETE_ARRAY(m.array, m.count);
    }
}

static void deinitIntrospectionData(CSPIRVIntrospector::CIntrospectionData* _data)
{
    for (auto& descSet : _data->descriptorSetBindings)
        for (auto& res : descSet)
        {
            switch (res.type)
            {
            case ESRT_STORAGE_BUFFER:
                deinitShdrMemBlock(static_cast<impl::SShaderMemoryBlock&>(res.get<ESRT_STORAGE_BUFFER>()));
                break;
            case ESRT_UNIFORM_BUFFER:
                deinitShdrMemBlock(static_cast<impl::SShaderMemoryBlock&>(res.get<ESRT_UNIFORM_BUFFER>()));
                break;
            default: break;
            }
        }
    if (_data->pushConstant.present)
        deinitShdrMemBlock(static_cast<impl::SShaderMemoryBlock&>(_data->pushConstant.info));
}

CSPIRVIntrospector::CIntrospectionData::~CIntrospectionData()
{
    deinitIntrospectionData(this);
}


} // nbl:asset
