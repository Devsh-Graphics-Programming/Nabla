#include "irr/asset/ICPUShader.h"
#include "spirv_cross/spirv_parser.hpp"
#include "spirv_cross/spirv_cross.hpp"
#include "irr/asset/EFormat.h"
#include "irr/asset/spvUtils.h"

namespace irr { namespace asset
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
}

ICPUShader::ICPUShader(IGLSLCompiler* _glslcompiler, io::IReadFile* _glsl, const std::string& _entryPoint, E_SHADER_STAGE _stage) : m_glslCompiler(_glslcompiler)
{
    assert(_glslcompiler);
    if (m_glslCompiler)
        m_glslCompiler->grab();
    m_entryPoints.push_back({_entryPoint,_stage});//in case of creation from GLSL source, (entry point, stage) tuple given in constructor is the only one
    //in case of creation from SPIR-V there can be many of (EP, stage) tuples and they are retrieved directly from SPIR-V opcodes

    //use 0 if no self-inclusions are allowed (behaviour as with traditional include guards)
    //see description of IGLSLCompiler::resolveIncludeDirectives() for more info
    constexpr uint32_t MAX_SELF_INCL_COUNT = 2u;

    if (!_glsl)
        return;
    m_glsl.resize(_glsl->getSize());
    _glsl->read(m_glsl.data(), m_glsl.size());
    m_glslOriginFilename = _glsl->getFileName().c_str();
    m_glsl = m_glslCompiler->resolveIncludeDirectives(m_glsl.c_str(), _stage, m_glslOriginFilename.c_str(), 3u);
}

void ICPUShader::enableIntrospection()
{
    if (m_introspectionCache.size()) // already enabled
        return;

    if (!m_spirvBytecode)
    {
        //TODO insert extension #define-s
        //also introspection key should be (EP,stage,enabled_exts) tuple

        //if ICPUShader doesnt already contain SPIR-V, it means it was constructed from GLSL source and SPIR-V has to be retrieved and parsed
        const SEntryPointStagePair& theOnlyEP = m_entryPoints.front();
        m_spirvBytecode = m_glslCompiler->createSPIRVFromGLSL(m_glsl.c_str(), theOnlyEP.second, theOnlyEP.first.c_str(), m_glslOriginFilename.c_str());
    }
    if (!m_parsed)
        m_parsed = new IParsedShaderSource(m_spirvBytecode);

    spirv_cross::Compiler comp(m_parsed->getUnderlyingRepresentation());
    auto eps = getStageEntryPoints(comp);

    SIntrospectionPerformer introPerformer;
    for (const auto& ep : eps)
        m_introspectionCache.emplace(ep, introPerformer.doIntrospection(comp, ep));
}

auto ICPUShader::getStageEntryPoints() -> const core::vector<SEntryPointStagePair>&
{
    if (m_entryPoints.size() || !m_parsed) // m_parsed==nullptr implies introspection not enabled (therefore m_entryPoints is empty and will be returned as such)
        return m_entryPoints;

    spirv_cross::Compiler comp(m_parsed->getUnderlyingRepresentation());

    return getStageEntryPoints(comp);
}

auto ICPUShader::getStageEntryPoints(spirv_cross::Compiler& _comp) -> const core::vector<SEntryPointStagePair>&
{
    if (m_entryPoints.size())
        return m_entryPoints;

    auto eps = _comp.get_entry_points_and_stages();
    m_entryPoints.reserve(eps.size());

    for (const auto& ep : eps)
        m_entryPoints.emplace_back(ep.name, spvExecModel2ESS(ep.execution_model));
    std::sort(m_entryPoints.begin(), m_entryPoints.end());

    return m_entryPoints;
}

SIntrospectionData ICPUShader::SIntrospectionPerformer::doIntrospection(spirv_cross::Compiler& _comp, const SEntryPointStagePair& _ep) const
{
    spv::ExecutionModel stage = ESS2spvExecModel(_ep.second);
    if (stage == spv::ExecutionModelMax)
        return SIntrospectionData();

    SIntrospectionData introData;
    introData.pushConstant.present = false;
    auto addResource_common = [&introData, &_comp] (const spirv_cross::Resource& r, E_SHADER_RESOURCE_TYPE restype, const core::unordered_map<uint32_t, const SIntrospectionData::SSpecConstant*>& _mapId2sconst) -> SShaderResourceVariant& {
        const uint32_t descSet = _comp.get_decoration(r.id, spv::DecorationDescriptorSet);
        assert(descSet < 4u);
        introData.descriptorSetBindings[descSet].emplace_back();

        SShaderResourceVariant& res = introData.descriptorSetBindings[descSet].back();
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
        introData.inputOutput.emplace_back();
        SShaderInfoVariant& info = introData.inputOutput.back();
        info.type = type;
        info.location = _comp.get_decoration(r.id, spv::DecorationLocation);
        return info;
    };

    _comp.set_entry_point(_ep.first, stage);

    // spec constants
    spirv_cross::SmallVector<spirv_cross::SpecializationConstant> sconsts = _comp.get_specialization_constants();
    core::unordered_map<uint32_t, const SIntrospectionData::SSpecConstant*> mapId2SpecConst;
    introData.specConstants.resize(sconsts.size());
    for (size_t i = 0u; i < sconsts.size(); ++i)
    {
        SIntrospectionData::SSpecConstant& specConst = introData.specConstants[i];
        specConst.id = sconsts[i].constant_id;
        specConst.name = _comp.get_name(sconsts[i].id);

        mapId2SpecConst[sconsts[i].id] = &specConst;

        const spirv_cross::SPIRConstant& sconstval = _comp.get_constant(sconsts[i].id);
        const spirv_cross::SPIRType& type = _comp.get_type(sconstval.constant_type);
        specConst.byteSize = calcBytesizeforType(_comp, type);

        switch (type.basetype)
        {
        case spirv_cross::SPIRType::Int:
            specConst.type = SIntrospectionData::SSpecConstant::ET_I32;
            specConst.defaultValue.i32 = sconstval.scalar_i32();
            break;
        case spirv_cross::SPIRType::UInt:
            specConst.type = SIntrospectionData::SSpecConstant::ET_U32;
            specConst.defaultValue.u32 = sconstval.scalar_i32();
            break;
        case spirv_cross::SPIRType::Float:
            specConst.type = SIntrospectionData::SSpecConstant::ET_F32;
            specConst.defaultValue.f32 = sconstval.scalar_f32();
            break;
        case spirv_cross::SPIRType::Int64:
            specConst.type = SIntrospectionData::SSpecConstant::ET_I64;
            specConst.defaultValue.i64 = sconstval.scalar_i64();
            break;
        case spirv_cross::SPIRType::UInt64:
            specConst.type = SIntrospectionData::SSpecConstant::ET_U64;
            specConst.defaultValue.u64 = sconstval.scalar_u64();
            break;
        case spirv_cross::SPIRType::Double:
            specConst.type = SIntrospectionData::SSpecConstant::ET_F64;
            specConst.defaultValue.f64 = sconstval.scalar_f64();
            break;
        default: break;
        }
    }
    using SSpecConstant = SIntrospectionData::SSpecConstant;
    std::sort(introData.specConstants.begin(), introData.specConstants.end(), [](const SSpecConstant& _lhs, const SSpecConstant& _rhs) { return _lhs.id < _rhs.id; });

    spirv_cross::ShaderResources resources = _comp.get_shader_resources(_comp.get_active_interface_variables());
    for (const spirv_cross::Resource& r : resources.uniform_buffers)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_UNIFORM_BUFFER, mapId2SpecConst);
        shaderMemBlockIntrospection(_comp, static_cast<impl::SShaderMemoryBlock&>(res.get<ESRT_UNIFORM_BUFFER>()), r.base_type_id, r.id, mapId2SpecConst);
    }
    for (const spirv_cross::Resource& r : resources.storage_buffers)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_STORAGE_BUFFER, mapId2SpecConst);
        shaderMemBlockIntrospection(_comp, static_cast<impl::SShaderMemoryBlock&>(res.get<ESRT_STORAGE_BUFFER>()), r.base_type_id, r.id, mapId2SpecConst);
    }
    for (const spirv_cross::Resource& r : resources.subpass_inputs)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_INPUT_ATTACHMENT, mapId2SpecConst);
        res.get<ESRT_INPUT_ATTACHMENT>().inputAttachmentIndex = _comp.get_decoration(r.id, spv::DecorationInputAttachmentIndex);
    }
    for (const spirv_cross::Resource& r : resources.storage_images)
    {
        const spirv_cross::SPIRType& type = _comp.get_type(r.id);
        const bool buffer = type.image.dim == spv::DimBuffer;
        SShaderResourceVariant& res = addResource_common(r, buffer ? ESRT_STORAGE_TEXEL_BUFFER : ESRT_STORAGE_IMAGE, mapId2SpecConst);
        if (!buffer)
        {
            res.get<ESRT_STORAGE_IMAGE>().approxFormat = spvImageFormat2E_FORMAT(type.image.format);
        }
    }
    for (const spirv_cross::Resource& r : resources.sampled_images)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_COMBINED_IMAGE_SAMPLER, mapId2SpecConst);
        const spirv_cross::SPIRType& type = _comp.get_type(r.id);
        res.get<ESRT_COMBINED_IMAGE_SAMPLER>().arrayed = type.image.arrayed;
        res.get<ESRT_COMBINED_IMAGE_SAMPLER>().multisample = type.image.ms;
    }
    for (const spirv_cross::Resource& r : resources.separate_images)
    {
        const spirv_cross::SPIRType& type = _comp.get_type(r.id);
        const bool buffer = type.image.dim == spv::DimBuffer;
        SShaderResourceVariant& res = addResource_common(r, buffer ? ESRT_UNIFORM_TEXEL_BUFFER : ESRT_SAMPLED_IMAGE, mapId2SpecConst);
    }
    for (const spirv_cross::Resource& r : resources.separate_samplers)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_SAMPLER, mapId2SpecConst);
    }
    for (auto& descSet : introData.descriptorSetBindings)
        std::sort(descSet.begin(), descSet.end(), [](const SShaderResourceVariant& _lhs, const SShaderResourceVariant& _rhs) { return _lhs.binding < _rhs.binding; });


    // in/out
    for (const spirv_cross::Resource& r : resources.stage_inputs)
    {
        SShaderInfoVariant& res = addInfo_common(r, ESIT_STAGE_INPUT);
    }
    for (const spirv_cross::Resource& r : resources.stage_outputs)
    {
        SShaderInfoVariant& res = addInfo_common(r, ESIT_STAGE_OUTPUT);
        res.get<ESIT_STAGE_OUTPUT>().colorIndex = _comp.get_decoration(r.id, spv::DecorationIndex);
    }
    std::sort(introData.inputOutput.begin(), introData.inputOutput.end(), [](const SShaderInfoVariant& _lhs, const SShaderInfoVariant& _rhs) { return _lhs.location < _rhs.location; });

    // push constants
    if (resources.push_constant_buffers.size())
    {
        const spirv_cross::Resource& r = resources.push_constant_buffers.front();
        introData.pushConstant.present = true;
        shaderMemBlockIntrospection(_comp, static_cast<impl::SShaderMemoryBlock&>(introData.pushConstant.info), r.base_type_id, r.id, mapId2SpecConst);
    }

    return introData;
}

void ICPUShader::SIntrospectionPerformer::shaderMemBlockIntrospection(spirv_cross::Compiler& _comp, impl::SShaderMemoryBlock& _res, uint32_t _blockBaseTypeID, uint32_t _varID, const core::unordered_map<uint32_t, const SIntrospectionData::SSpecConstant*>& _mapId2sconst) const
{
    // SShaderMemoryBlock (and its members) cannot define custom default ctor nor even default member values, because then it's "non-trivial default ctor" and
    // union containing it (as member) has deleted default ctor... (union default ctor is deleted if any of its members defines non-trivial default ctor)
    auto shdrMemBlockMemberDefault = [] {
        impl::SShaderMemoryBlock::SMember m;
        m.count = 1u;
        m.countIsSpecConstant = false;
        m.offset = 0u;
        m.size = 0u;
        m.arrayStride = 0u;
        m.mtxStride = 0u;
        m.mtxRowCnt = m.mtxColCnt = 1u;
        return m;
    };

    const spirv_cross::SPIRType& type = _comp.get_type(_blockBaseTypeID);
    const uint32_t memberCnt = type.member_types.size();
    _res.members.array = _IRR_NEW_ARRAY(impl::SShaderMemoryBlock::SMember, memberCnt);
    _res.members.count = memberCnt;
    std::fill(_res.members.array, _res.members.array+memberCnt, shdrMemBlockMemberDefault());
    //TODO introspection should be able to work with members of struct type (recursion)
    //tldr SMember has o have `struct_members` array
    for (uint32_t m = 0u; m < memberCnt; ++m)
    {
        const spirv_cross::SPIRType& mtype = _comp.get_type(type.member_types[m]);
        impl::SShaderMemoryBlock::SMember& member = _res.members.array[m];

        member.size = _comp.get_declared_struct_member_size(type, m);
        member.offset = _comp.type_struct_member_offset(type, m);

        if (mtype.array.size())
        {
            member.count = mtype.array[0];
            member.arrayStride = _comp.type_struct_member_array_stride(type, m);
            member.countIsSpecConstant = !mtype.array_size_literal[0];
            if (member.countIsSpecConstant) {
                const auto sc_itr = _mapId2sconst.find(member.count);
                assert(sc_itr != _mapId2sconst.cend());
                auto sc = sc_itr->second;
                member.count = sc->id;
            }
        }

        member.mtxRowCnt = mtype.vecsize;
        member.mtxColCnt = mtype.columns;
        if (member.mtxColCnt > 1u)
            member.mtxStride = _comp.type_struct_member_matrix_stride(type, m);
    }
    using MembT = impl::SShaderMemoryBlock::SMember;
    std::sort(_res.members.array, _res.members.array+memberCnt, [](const MembT& _lhs, const MembT& _rhs) { return _lhs.offset < _rhs.offset; });

    _res.size = _res.rtSizedArrayOneElementSize = _comp.get_declared_struct_size(type);
    const spirv_cross::SPIRType& lastType = _comp.get_type(type.member_types[memberCnt-1u]);
    if (lastType.array.size() && lastType.array_size_literal[0] && lastType.array[0] == 0u)
        _res.rtSizedArrayOneElementSize += _res.members.array[memberCnt-1u].arrayStride;

    spirv_cross::Bitset flags = _comp.get_buffer_block_flags(_varID);
    _res.restrict_ = flags.get(spv::DecorationRestrict);
    _res.volatile_ = flags.get(spv::DecorationVolatile);
    _res.coherent = flags.get(spv::DecorationCoherent);
    _res.readonly = flags.get(spv::DecorationNonWritable);
    _res.writeonly = flags.get(spv::DecorationNonReadable);
}

size_t ICPUShader::SIntrospectionPerformer::calcBytesizeforType(spirv_cross::Compiler& _comp, const spirv_cross::SPIRType & _type) const
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

void ICPUShader::SIntrospectionPerformer::deinitIntrospectionData(SIntrospectionData& _data)
{
    for (auto& descSet : _data.descriptorSetBindings)
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
    if (_data.pushConstant.present)
        deinitShdrMemBlock(static_cast<impl::SShaderMemoryBlock&>(_data.pushConstant.info));
}

void ICPUShader::SIntrospectionPerformer::deinitShdrMemBlock(impl::SShaderMemoryBlock& _res)
{
    if (_res.members.array)
        _IRR_DELETE_ARRAY(_res.members.array, _res.members.count);
}

}}