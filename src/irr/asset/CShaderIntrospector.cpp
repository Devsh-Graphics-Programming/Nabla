#include "irr/asset/CShaderIntrospector.h"

#include "irr/asset/spvUtils.h"
#include "spirv_cross/spirv_parser.hpp"
#include "spirv_cross/spirv_cross.hpp"

namespace irr
{
namespace asset
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

const CIntrospectionData* CShaderIntrospector::introspect(const ICPUShader* _shader, const SEntryPoint_Stage_Extensions& _params)
{
    if (!_shader)
        return nullptr;

    auto found = findIntrospection(_shader, _params);
    if (found)
        return found.get();

    auto introspectSPV = [this,&_params](const ICPUShader* _spvshader) {
        const ICPUBuffer* spv = _spvshader->getSPVorGLSL();
        spirv_cross::Compiler comp(reinterpret_cast<const uint32_t*>(spv->getPointer()), spv->getSize()/4u);
        return doIntrospection(comp, _params);
    };

    if (_shader->containsGLSL()) {
        std::string glsl = reinterpret_cast<const char*>(_shader->getSPVorGLSL()->getPointer());
        ICPUShader::insertGLSLExtensionsDefines(glsl, _params.GLSLextensions.get());
        auto spvShader = m_glslCompiler->createSPIRVFromGLSL(
            glsl.c_str(),
            _params.stage,
            _params.entryPoint.c_str(),
            "????"
        );
        if (!spvShader)
            return nullptr;

        return cacheIntrospection(introspectSPV(spvShader.get()), _shader, _params);
    }
    else {
        // TODO (?) when we have enabled_extensions_list it may validate whether all extensions in list are also present in spv
        return cacheIntrospection(introspectSPV(_shader), _shader, _params);
    }
}

static E_DESCRIPTOR_TYPE resType2descType(E_SHADER_RESOURCE_TYPE _t)
{
    static const E_DESCRIPTOR_TYPE descType[9]{
        EDT_COMBINED_IMAGE_SAMPLER,
        EDT_STORAGE_IMAGE,
        EDT_UNIFORM_TEXEL_BUFFER,
        EDT_STORAGE_TEXEL_BUFFER,
        EDT_UNIFORM_BUFFER,
        EDT_STORAGE_BUFFER,
        EDT_UNIFORM_BUFFER_DYNAMIC,
        EDT_STORAGE_BUFFER_DYNAMIC,
        EDT_INPUT_ATTACHMENT
    };
    return descType[_t];
}

template<E_SHADER_RESOURCE_TYPE restype>
static std::pair<bool, IImageView<ICPUImage>::E_TYPE> imageInfoFromResource(const SShaderResource<restype>& _res)
{
    return {_res.shadow, _res.viewType};
}

std::pair<bool, IImageView<ICPUImage>::E_TYPE> CShaderIntrospector::getImageInfoFromIntrospection(uint32_t _set, uint32_t _binding, ICPUSpecializedShader** const _begin, ICPUSpecializedShader** const _end, const std::string* _extensionsBegin, const std::string* _extensionsEnd)
{
    std::pair<bool, IImageView<ICPUImage>::E_TYPE> retval;

    core::smart_refctd_dynamic_array<std::string> extensions;
    if (_extensionsBegin)
    {
        core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<std::string>>(_extensionsEnd - _extensionsBegin);
        std::copy(_extensionsBegin, _extensionsEnd, extensions->begin());
    }

    for (auto shdr = _begin; shdr != _end; ++shdr)
    {
        const CIntrospectionData* introspection = introspect((*shdr)->getUnspecialized(), {(*shdr)->getStage(), (*shdr)->getSpecializationInfo().entryPoint, extensions});

        for (const auto& bnd : introspection->descriptorSetBindings[_set])
        {
            if (bnd.type==ESRT_COMBINED_IMAGE_SAMPLER || bnd.type==ESRT_STORAGE_IMAGE)
                return (bnd.type==ESRT_COMBINED_IMAGE_SAMPLER) ? imageInfoFromResource(bnd.get<ESRT_COMBINED_IMAGE_SAMPLER>()) : imageInfoFromResource(bnd.get<ESRT_STORAGE_IMAGE>());
        }
    }

    return {false, IImageView<ICPUImage>::ET_COUNT};
}

core::smart_refctd_dynamic_array<SPushConstantRange> CShaderIntrospector::createPushConstantRangesFromIntrospection(ICPUSpecializedShader** const _begin, ICPUSpecializedShader** const _end, const std::string* _extensionsBegin, const std::string* _extensionsEnd)
{
    constexpr size_t MAX_STAGE_COUNT = 14ull;

    ICPUSpecializedShader* shaders[MAX_STAGE_COUNT]{};
    for (auto shdr = _begin; shdr != _end; ++shdr)
    {
        shaders[core::findLSB<uint32_t>((*shdr)->getStage())] = (*shdr);
    }

    core::smart_refctd_dynamic_array<std::string> extensions;
    if (_extensionsBegin)
    {
        core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<std::string>>(_extensionsEnd - _extensionsBegin);
        std::copy(_extensionsBegin, _extensionsEnd, extensions->begin());
    }

    core::vector<uint32_t> presentStagesIxs;
    presentStagesIxs.reserve(MAX_STAGE_COUNT);
    const CIntrospectionData* introspection[MAX_STAGE_COUNT]{};
    for (uint32_t i = 0u; i < MAX_STAGE_COUNT; ++i)
        if (shaders[i])
        {
            introspection[i] = introspect(shaders[i]->getUnspecialized(), { shaders[i]->getStage(), shaders[i]->getSpecializationInfo().entryPoint, extensions });
            presentStagesIxs.push_back(i);
        }

    core::vector<SPushConstantRange> ranges[MAX_STAGE_COUNT];
    for (auto& r : ranges)
        r.reserve(100u);

    for (uint32_t stg : presentStagesIxs)
    {
        auto& pc = introspection[stg]->pushConstant;
        if (!pc.present)
            continue;

        auto& members = pc.info.members;
        auto& rngs = ranges[stg];
        rngs.push_back({static_cast<ISpecializedShader::E_SHADER_STAGE>(1u<<stg), members.array[0].offset, members.array[0].size});
        for (uint32_t i = 1u; i < members.count; ++i)
        {
            auto& last = rngs.back();
            if (members.array[i].offset == (last.offset + last.size))
                last.size += members.array[i].size;
            else
                rngs.push_back({static_cast<ISpecializedShader::E_SHADER_STAGE>(1u<<stg), members.array[i].offset, members.array[i].size});
        }
    }

    core::vector<SPushConstantRange> merged;

    SPushConstantRange rngToPush; rngToPush.offset = 0u; rngToPush.size = 0u;
    uint32_t stageFlags = 0u;
    for (uint32_t i = 0u; i < ICPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE/sizeof(uint32_t); i += sizeof(uint32_t))
    {
        SPushConstantRange curr; curr.offset = i; curr.size = sizeof(uint32_t);

        uint32_t tmpFlags = 0u;
        for (uint32_t stg : presentStagesIxs)
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
            rngToPush.stageFlags = static_cast<ICPUSpecializedShader::E_SHADER_STAGE>(stageFlags);
            merged.push_back(rngToPush);
            stageFlags = 0u;

            rngToPush.offset = i;
            rngToPush.size = sizeof(uint32_t);
        }
    }
    if (stageFlags)
    {
        rngToPush.stageFlags = static_cast<ICPUSpecializedShader::E_SHADER_STAGE>(stageFlags);
        merged.push_back(rngToPush);
    }

    core::smart_refctd_dynamic_array<SPushConstantRange> rngsArray;
    if (merged.size())
    {
        rngsArray = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<SPushConstantRange>>(merged.size());
        memcpy(rngsArray->data(), merged.data(), rngsArray->size() * sizeof(SPushConstantRange));
    }

    return rngsArray;
}

core::smart_refctd_ptr<ICPUDescriptorSetLayout> CShaderIntrospector::createApproximateDescriptorSetLayoutFromIntrospection(uint32_t _set, ICPUSpecializedShader** const _begin, ICPUSpecializedShader** const _end, const std::string* _extensionsBegin, const std::string* _extensionsEnd)
{
    constexpr size_t MAX_STAGE_COUNT = 14ull;

    ICPUSpecializedShader* shaders[MAX_STAGE_COUNT]{};
    for (auto shdr = _begin; shdr != _end; ++shdr)
    {
        shaders[core::findLSB<uint32_t>((*shdr)->getStage())] = (*shdr);
    }

    core::smart_refctd_dynamic_array<std::string> extensions;
    if (_extensionsBegin)
    {
        core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<std::string>>(_extensionsEnd - _extensionsBegin);
        std::copy(_extensionsBegin, _extensionsEnd, extensions->begin());
    }

    core::vector<uint32_t> presentStagesIxs;
    presentStagesIxs.reserve(MAX_STAGE_COUNT);
    const CIntrospectionData* introspection[MAX_STAGE_COUNT]{};
    for (uint32_t i = 0u; i < MAX_STAGE_COUNT; ++i)
        if (shaders[i])
        {
            introspection[i] = introspect(shaders[i]->getUnspecialized(), {shaders[i]->getStage(), shaders[i]->getSpecializationInfo().entryPoint, extensions});
            presentStagesIxs.push_back(i);
        }
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
        for (uint32_t stg : presentStagesIxs)
        {
            auto& introBindings = introspection[stg]->descriptorSetBindings[_set];
            if (checkedDescsCnt[stg] == introBindings.size())
                continue;
            anyStageNotFinished = true;

            if (introBindings[checkedDescsCnt[stg]].binding < binding.binding)
                binding.binding = introBindings[checkedDescsCnt[stg]].binding;
        }
        if (!anyStageNotFinished) //all shader stages finished
            break;

        uint32_t refStg = ~0u;
        for (uint32_t stg : presentStagesIxs)
        {
            auto& introBnd = introspection[stg]->descriptorSetBindings[_set][checkedDescsCnt[stg]];
            if (introBnd.binding != binding.binding)
                continue;

            stageFlags |= (1u<<stg);

            ++checkedDescsCnt[stg];
            refStg = stg;
        }

        auto& introBnd = introspection[refStg]->descriptorSetBindings[_set][checkedDescsCnt[refStg]-1u];
        binding.type = resType2descType(introBnd.type);
        binding.stageFlags = static_cast<ICPUSpecializedShader::E_SHADER_STAGE>(stageFlags);
        binding.count = introBnd.descriptorCount;
        if (introBnd.descCountIsSpecConstant)
        {
            auto& specInfo = shaders[refStg]->getSpecializationInfo();
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

core::smart_refctd_ptr<ICPUPipelineLayout> CShaderIntrospector::createApproximatePipelineLayoutFromIntrospection(ICPUSpecializedShader** const _begin, ICPUSpecializedShader** const _end, const std::string* _extensionsBegin, const std::string* _extensionsEnd)
{
    core::smart_refctd_ptr<ICPUDescriptorSetLayout> dsLayout[ICPUPipelineLayout::DESCRIPTOR_SET_COUNT];
    for (uint32_t i = 0u; i < ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
        dsLayout[i] = createApproximateDescriptorSetLayoutFromIntrospection(i, _begin, _end, _extensionsBegin, _extensionsEnd);

    auto pcRanges = createPushConstantRangesFromIntrospection(_begin, _end, _extensionsBegin, _extensionsEnd);

    return core::make_smart_refctd_ptr<ICPUPipelineLayout>(
        (pcRanges ? pcRanges->begin() : nullptr), (pcRanges ? pcRanges->end() : nullptr),
        std::move(dsLayout[0]), std::move(dsLayout[1]), std::move(dsLayout[2]), std::move(dsLayout[3])
    );
}

core::smart_refctd_ptr<ICPUComputePipeline> CShaderIntrospector::createApproximateComputePipelineFromIntrospection(ICPUSpecializedShader* _shader, const std::string* _extensionsBegin, const std::string* _extensionsEnd)
{
    if (_shader->getStage() != ICPUSpecializedShader::ESS_COMPUTE)
        return nullptr;

    auto layout = createApproximatePipelineLayoutFromIntrospection(&_shader, &_shader + 1, _extensionsBegin, _extensionsEnd);

    return core::make_smart_refctd_ptr<ICPUComputePipeline>(
        std::move(layout),
        core::smart_refctd_ptr<ICPUSpecializedShader>(_shader)
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

core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> CShaderIntrospector::createApproximateRenderpassIndependentPipelineFromIntrospection(ICPUSpecializedShader** const _begin, ICPUSpecializedShader** const _end, const std::string* _extensionsBegin, const std::string* _extensionsEnd)
{
    ICPUSpecializedShader* vs = nullptr;
    {
        auto vs_it = std::find_if(_begin, _end, [](ICPUSpecializedShader* shdr) { return shdr->getStage()==ICPUSpecializedShader::ESS_VERTEX; });
        if (vs_it == _end)
            return nullptr; //no vertex shader
        vs = vs_it[0];
    }

    core::smart_refctd_dynamic_array<std::string> extensions;
    if (_extensionsBegin)
    {
        core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<std::string>>(_extensionsEnd - _extensionsBegin);
        std::copy(_extensionsBegin, _extensionsEnd, extensions->begin());
    }

    auto vs_introspection = introspect(vs->getUnspecialized(), {ICPUSpecializedShader::ESS_VERTEX, vs->getSpecializationInfo().entryPoint, extensions});

    SVertexInputParams vtxInput;
    uint32_t reloffset = 0u;
    for (const auto& io : vs_introspection->inputOutput)
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

    //all except vertex input are defaulted
    SBlendParams blending;
    SPrimitiveAssemblyParams primAssembly;
    SRasterizationParams raster;

    auto layout = createApproximatePipelineLayoutFromIntrospection(_begin, _end, _extensionsBegin, _extensionsEnd);

    return core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(
        std::move(layout),
        _begin, _end,
        vtxInput, blending, primAssembly, raster
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
    IImageView<ICPUImage>::E_TYPE viewType[8]{
        IImageView<ICPUImage>::ET_1D,
        IImageView<ICPUImage>::ET_1D_ARRAY,
        IImageView<ICPUImage>::ET_2D,
        IImageView<ICPUImage>::ET_2D_ARRAY,
        IImageView<ICPUImage>::ET_3D,
        IImageView<ICPUImage>::ET_3D,
        IImageView<ICPUImage>::ET_CUBE_MAP,
        IImageView<ICPUImage>::ET_CUBE_MAP_ARRAY
    };

    return viewType[_type.dim*2u + _type.arrayed];
}

core::smart_refctd_ptr<CIntrospectionData> CShaderIntrospector::doIntrospection(spirv_cross::Compiler& _comp, const SEntryPoint_Stage_Extensions& _ep) const
{
    spv::ExecutionModel stage = ESS2spvExecModel(_ep.stage);
    if (stage == spv::ExecutionModelMax)
        return nullptr;

    core::smart_refctd_ptr<CIntrospectionData> introData = core::make_smart_refctd_ptr<CIntrospectionData>();
    introData->pushConstant.present = false;
    auto addResource_common = [&introData, &_comp] (const spirv_cross::Resource& r, E_SHADER_RESOURCE_TYPE restype, const core::unordered_map<uint32_t, const CIntrospectionData::SSpecConstant*>& _mapId2sconst) -> SShaderResourceVariant& {
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
            res.descriptorCount = type.array[0]; // ID of spec constant if size is spec constant
            res.descCountIsSpecConstant = !type.array_size_literal[0];
            if (res.descCountIsSpecConstant) {
                const auto sc_itr = _mapId2sconst.find(res.descriptorCount);
                assert(sc_itr != _mapId2sconst.cend());
                auto sc = sc_itr->second;
                res.descriptorCount = sc->id;
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

    _comp.set_entry_point(_ep.entryPoint, stage);

    // spec constants
    spirv_cross::SmallVector<spirv_cross::SpecializationConstant> sconsts = _comp.get_specialization_constants();
    core::unordered_map<uint32_t, const CIntrospectionData::SSpecConstant*> mapId2SpecConst;
    introData->specConstants.resize(sconsts.size());
    for (size_t i = 0u; i < sconsts.size(); ++i)
    {
        CIntrospectionData::SSpecConstant& specConst = introData->specConstants[i];
        specConst.id = sconsts[i].constant_id;
        specConst.name = _comp.get_name(sconsts[i].id);

        mapId2SpecConst[sconsts[i].id] = &specConst;

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
    }
    using SSpecConstant = CIntrospectionData::SSpecConstant;
    std::sort(introData->specConstants.begin(), introData->specConstants.end(), [](const SSpecConstant& _lhs, const SSpecConstant& _rhs) { return _lhs.id < _rhs.id; });

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
        SShaderResourceVariant& res = addResource_common(r, ESRT_COMBINED_IMAGE_SAMPLER, mapId2SpecConst);
		const spirv_cross::SPIRType& type = _comp.get_type(r.type_id);
        res.get<ESRT_COMBINED_IMAGE_SAMPLER>().viewType = spvcrossImageType2ImageView(type.image);
        res.get<ESRT_COMBINED_IMAGE_SAMPLER>().shadow = type.image.depth;
        res.get<ESRT_COMBINED_IMAGE_SAMPLER>().multisample = type.image.ms;
    }
    for (const spirv_cross::Resource& r : resources.separate_images)
    {
		const spirv_cross::SPIRType& type = _comp.get_type(r.type_id);
        const bool buffer = type.image.dim == spv::DimBuffer;
        SShaderResourceVariant& res = addResource_common(r, buffer ? ESRT_UNIFORM_TEXEL_BUFFER : ESRT_SAMPLED_IMAGE, mapId2SpecConst);
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
}
static void introspectStructType(spirv_cross::Compiler& _comp, impl::SShaderMemoryBlock::SMember::SMembers& _dstMembers, const spirv_cross::SPIRType& _parentType, const spirv_cross::SmallVector<uint32_t>& _allMembersTypes, uint32_t _baseOffset, const core::unordered_map<uint32_t, const CIntrospectionData::SSpecConstant*>& _mapId2sconst, core::stack<StackElement>& _pushStack) {
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
    _dstMembers.array = _IRR_NEW_ARRAY(MembT, memberCnt);
    _dstMembers.count = memberCnt;
    std::fill(_dstMembers.array, _dstMembers.array+memberCnt, MemberDefault());
    for (uint32_t m = 0u; m < memberCnt; ++m)
	{
        MembT& member = _dstMembers.array[m];
        const spirv_cross::SPIRType& mtype = _comp.get_type(_allMembersTypes[m]);

        member.name = _comp.get_member_name(_parentType.self, m);
        member.size = _comp.get_declared_struct_member_size(_parentType, m);
        member.offset = _baseOffset + _comp.type_struct_member_offset(_parentType, m);
        member.rowMajor = _comp.get_member_decoration(_parentType.self, m, spv::DecorationRowMajor);//TODO check whether spirv-cross works with this decor
        member.type = spvcrossType2E_TYPE(mtype.basetype);

        if (mtype.array.size())
        {
            member.count = mtype.array[0];
            member.arrayStride = _comp.type_struct_member_array_stride(_parentType, m);
            member.countIsSpecConstant = !mtype.array_size_literal[0];
            if (member.countIsSpecConstant)
			{
                const auto sc_itr = _mapId2sconst.find(member.count);
                assert(sc_itr != _mapId2sconst.cend());
                auto sc = sc_itr->second;
                member.count = sc->id;
            }
        }
		else if (mtype.basetype != spirv_cross::SPIRType::Struct) // might have to ignore a few more types than structs
			member.arrayStride = core::max(0x1u<<core::findMSB(member.size),16u);

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

void CShaderIntrospector::shaderMemBlockIntrospection(spirv_cross::Compiler& _comp, impl::SShaderMemoryBlock& _res, uint32_t _blockBaseTypeID, uint32_t _varID, const core::unordered_map<uint32_t, const CIntrospectionData::SSpecConstant*>& _mapId2sconst) const
{
    using MembT = impl::SShaderMemoryBlock::SMember;

    core::stack<StackElement> introspectionStack;
    const spirv_cross::SPIRType& type = _comp.get_type(_blockBaseTypeID);
    introspectionStack.push({_res.members, type, 0u});
    while (!introspectionStack.empty()) {
        StackElement e = introspectionStack.top();
        introspectionStack.pop();
        introspectStructType(_comp, e.membersDst, e.parentType, e.parentType.member_types, 0u, _mapId2sconst, introspectionStack);
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

size_t CShaderIntrospector::calcBytesizeforType(spirv_cross::Compiler& _comp, const spirv_cross::SPIRType & _type) const
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
        _IRR_DELETE_ARRAY(m.array, m.count);
    }
}

static void deinitIntrospectionData(CIntrospectionData* _data)
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

CIntrospectionData::~CIntrospectionData()
{
    deinitIntrospectionData(this);
}

}//asset
}//irr