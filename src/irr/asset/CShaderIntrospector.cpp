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

const CIntrospectionData* CShaderIntrospector::introspect(const ICPUShader* _shader)
{
    if (!_shader)
        return nullptr;

    auto found = m_introspectionCache.find(core::smart_refctd_ptr<const ICPUShader>(_shader));
    if (found != m_introspectionCache.end())
        return found->second.get();

    auto introspectSPV = [this](const ICPUShader* _spvshader) {
        const ICPUBuffer* spv = _spvshader->getSPVorGLSL();
        spirv_cross::Compiler comp(reinterpret_cast<const uint32_t*>(spv->getPointer()), spv->getSize()/4u);
        return doIntrospection(comp, m_params);
    };

    if (_shader->containsGLSL()) {
        std::string glsl = reinterpret_cast<const char*>(_shader->getSPVorGLSL()->getPointer());
        ICPUShader::insertGLSLExtensionsDefines(glsl, m_params.GLSLextensions.get());
        auto spvShader = m_glslCompiler->createSPIRVFromGLSL(
            glsl.c_str(),
            m_params.stage,
            m_params.entryPoint.c_str(),
            "????"
        );
        if (!spvShader)
            return nullptr;

        return m_introspectionCache.insert({core::smart_refctd_ptr<const ICPUShader>(_shader), introspectSPV(spvShader.get())}).first->second.get();
    }
    else {
        // TODO (?) when we have enabled_extensions_list it may validate whether all extensions in list are also present in spv
        return m_introspectionCache.insert({core::smart_refctd_ptr<const ICPUShader>(_shader), introspectSPV(_shader)}).first->second.get();
    }
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
            res.get<ESRT_STORAGE_IMAGE>().approxFormat = spvImageFormat2E_FORMAT(type.image.format);
        }
    }
    for (const spirv_cross::Resource& r : resources.sampled_images)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_COMBINED_IMAGE_SAMPLER, mapId2SpecConst);
		const spirv_cross::SPIRType& type = _comp.get_type(r.type_id);
        res.get<ESRT_COMBINED_IMAGE_SAMPLER>().arrayed = type.image.arrayed;
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