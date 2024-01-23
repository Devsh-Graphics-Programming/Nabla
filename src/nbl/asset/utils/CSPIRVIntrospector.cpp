// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/utils/CSPIRVIntrospector.h"
#include "nbl/asset/utils/spvUtils.h"

#include "nbl_spirv_cross/spirv_parser.hpp"
#include "nbl_spirv_cross/spirv_cross.hpp"

namespace nbl::asset
{

// returns true if successfully added all the info to self, false if incompatible with what's already in our pipeline or incomplete (e.g. missing spec constants)
NBL_API2 bool CSPIRVIntrospector::CPipelineIntrospectionData::merge(const CSPIRVIntrospector::CStageIntrospectionData* stageData, const ICPUShader::SSpecInfoBase::spec_constant_map_t* specConstants)
{
    return false;
}

//
NBL_API2 core::smart_refctd_dynamic_array<SPushConstantRange> CSPIRVIntrospector::CPipelineIntrospectionData::createPushConstantRangesFromIntrospection()
{
    return nullptr;
}
NBL_API2 core::smart_refctd_ptr<ICPUDescriptorSetLayout> CSPIRVIntrospector::CPipelineIntrospectionData::createApproximateDescriptorSetLayoutFromIntrospection(const uint32_t setID)
{
    return nullptr;
}
NBL_API2 core::smart_refctd_ptr<ICPUPipelineLayout> CSPIRVIntrospector::CPipelineIntrospectionData::createApproximatePipelineLayoutFromIntrospection()
{
    return nullptr;
}

CSPIRVIntrospector::CStageIntrospectionData::SDescriptorVarInfo<>& CSPIRVIntrospector::CStageIntrospectionData::addResource_common(
    const spirv_cross::Compiler& comp,
    const spirv_cross::Resource& r,
    IDescriptor::E_TYPE restype
    )
{
    const uint32_t descSet = comp.get_decoration(r.id, spv::DecorationDescriptorSet);
    assert(descSet < 4u); // TODO: fail/error out of the introspection, don't crash

    const auto nameOff = m_stringPool.size();
    {
        m_stringPool.resize(nameOff + r.name.size() + 1);
        memcpy(m_stringPool.data() + nameOff, r.name.c_str(), r.name.size());
        m_stringPool.back() = '\0';
    }
    CStageIntrospectionData::SDescriptorVarInfo<true> res = {
        {
            .binding = comp.get_decoration(r.id, spv::DecorationBinding),
            .count = {}, // TODO: fill out
            .type = restype
        },
        /*.name = */core::based_span<char>(nameOff,r.name.size())
    };
    reinterpret_cast<core::vector<CStageIntrospectionData::SDescriptorVarInfo<true>>*>(m_descriptorSetBindings)[descSet].push_back(std::move(res));

    auto& arrayInfo = m_arraySizePool.emplace_back();
    arrayInfo.value = 1u;

    // TODO [Przemek]: what about this?
    //res.descCountIsSpecConstant = false;

    const spirv_cross::SPIRType& type = comp.get_type(r.type_id);
    // assuming only 1D arrays because i don't know how desc set layout binding is constructed when it's let's say 2D array (e.g. uniform sampler2D smplr[4][5]; is it even legal?)

    // TODO: log error and return invalid value
    //assert(type.array.size() < 2)

    if (type.array.size()) // is array
    {
        // the API for this spec constant checking is truly messed up
        arrayInfo.specID = type.array[0]; // ID of spec constant if size is spec constant
        arrayInfo.isSpecConstant = !type.array_size_literal[0];
    }

    return m_descriptorSetBindings[descSet].back(); // TODO: remove
}

NBL_API2 core::smart_refctd_ptr<const CSPIRVIntrospector::CStageIntrospectionData> CSPIRVIntrospector::doIntrospection(const CSPIRVIntrospector::CStageIntrospectionData::SParams& params)
{
    const ICPUBuffer* spv = params.shader->getContent();
    spirv_cross::Compiler comp(reinterpret_cast<const uint32_t*>(spv->getPointer()), spv->getSize() / 4u);
    const IShader::E_SHADER_STAGE shaderStage = params.shader->getStage();

    spv::ExecutionModel stage = ESS2spvExecModel(shaderStage);
    if (stage == spv::ExecutionModelMax)
        return nullptr;

    core::smart_refctd_ptr<CStageIntrospectionData> stageIntroData = core::make_smart_refctd_ptr<CStageIntrospectionData>();

    // TODO [Przemek]: now `inputOutput` vector is separated into `m_input` and `m_output`, adapt the code accordingly
    /*auto addInfo_common = [&stageIntroData, &comp](const spirv_cross::Resource& r, IDescriptor::E_TYPE type) -> SShaderInfoVariant& 
    {
        introData->inputOutput.emplace_back();
        SShaderInfoVariant& info = introData->inputOutput.back();
        info.type = type;
        info.location = comp.get_decoration(r.id, spv::DecorationLocation);
        return info;
        };*/

    comp.set_entry_point(params.entryPoint, stage);


    // TODO [Przemog]
    // spec constants
    /*
    spirv_cross::SmallVector<spirv_cross::SpecializationConstant> sconsts = comp.get_specialization_constants();
    mapId2SpecConst_t mapId2SpecConst;
    stageIntroData->m_specConstants.reserve(sconsts.size());
    for (size_t i = 0u; i < sconsts.size(); ++i)
    {
        CSPIRVIntrospector::CStageIntrospectionData::SSpecConstant specConst;
        specConst.id = sconsts[i].constant_id;
        specConst.name = comp.get_name(sconsts[i].id);

        const spirv_cross::SPIRConstant& sconstval = comp.get_constant(sconsts[i].id);
        const spirv_cross::SPIRType& type = comp.get_type(sconstval.constant_type);
        specConst.byteSize = calcBytesizeforType(comp, type);
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
        if (it == introData->specConstants.end() || it->id != dummy.id)
            continue;
        mapId2SpecConst.insert({ sc.id,&*it });
    }
    */

    spirv_cross::ShaderResources resources = comp.get_shader_resources(/*TODO: allow choice in Introspection Parameters, comp.get_active_interface_variables()*/);
    for (const spirv_cross::Resource& r : resources.uniform_buffers)
    {
        CSPIRVIntrospector::CStageIntrospectionData::SDescriptorVarInfo<>& res = stageIntroData->addResource_common(comp, r, IDescriptor::E_TYPE::ET_UNIFORM_BUFFER);
        shaderMemBlockIntrospection(comp, res.uniformBuffer, r.base_type_id, r.id);
    }
    for (const spirv_cross::Resource& r : resources.storage_buffers)
    {
        CSPIRVIntrospector::CStageIntrospectionData::SDescriptorVarInfo<>& res = stageIntroData->addResource_common(comp, r, IDescriptor::E_TYPE::ET_STORAGE_BUFFER);
        shaderMemBlockIntrospection(comp, res.storageBuffer, r.base_type_id, r.id);
    }
    for (const spirv_cross::Resource& r : resources.subpass_inputs)
    {
        //CSPIRVIntrospector::CStageIntrospectionData::SDescriptorVarInfo<>&& res = addResource_common(r, IDescriptor::E_TYPE::ET_INPUT_ATTACHMENT);
        //res.inputAttachmentIndex = comp.get_decoration(r.id, spv::DecorationInputAttachmentIndex);
    }
    for (const spirv_cross::Resource& r : resources.storage_images)
    {
        /*const spirv_cross::SPIRType& type = comp.get_type(r.type_id);
        const bool buffer = type.image.dim == spv::DimBuffer;
        CSPIRVIntrospector::CStageIntrospectionData::SDescriptorVarInfo<>&& res = addResource_common(r, buffer ? IDescriptor::E_TYPE::ET_STORAGE_TEXEL_BUFFER : IDescriptor::E_TYPE::ET_STORAGE_IMAGE);
        if (!buffer)
        {
            res.get<IDescriptor::E_TYPE::ET_STORAGE_IMAGE>().format = spvImageFormat2E_FORMAT(type.image.format);
            res.get<IDescriptor::E_TYPE::ET_STORAGE_IMAGE>().viewType = spvcrossImageType2ImageView(type.image);
            res.get<IDescriptor::E_TYPE::ET_STORAGE_IMAGE>().shadow = type.image.depth;
        }*/
    }
    for (const spirv_cross::Resource& r : resources.sampled_images)
    {
        /*const spirv_cross::SPIRType& type = comp.get_type(r.type_id);
        const bool buffer = type.image.dim == spv::DimBuffer;
        CSPIRVIntrospector::CStageIntrospectionData::SDescriptorVarInfo<>& res = addResource_common(r, buffer ? IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER : IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER);
        if (!buffer)
        {
            res.get<IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER>().viewType = spvcrossImageType2ImageView(type.image);
            res.get<IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER>().shadow = type.image.depth;
            res.get<IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER>().multisample = type.image.ms;
        }*/
    }
    for (const spirv_cross::Resource& r : resources.separate_images)
    {
        //CSPIRVIntrospector::CStageIntrospectionData::SDescriptorVarInfo<>& res = addResource_common(r, ESRT_SAMPLED_IMAGE);
    }
    for (const spirv_cross::Resource& r : resources.separate_samplers)
    {
        //CSPIRVIntrospector::CStageIntrospectionData::SDescriptorVarInfo<>& res = addResource_common(r, ESRT_SAMPLER);
    }
    for (auto& descSet : stageIntroData->m_descriptorSetBindings)
        std::sort(descSet.begin(), descSet.end(), [](const CSPIRVIntrospector::CStageIntrospectionData::SDescriptorVarInfo<>& _lhs, const CSPIRVIntrospector::CStageIntrospectionData::SDescriptorVarInfo<>& _rhs) { return _lhs.binding < _rhs.binding; });


    /*auto getStageIOtype = [&comp](uint32_t _base_type_id)
        {
            const auto& type = comp.get_type(_base_type_id);
            decltype(SShaderInfoVariant::glslType) glslType;
            glslType.basetype = spvcrossType2E_TYPE(type.basetype);
            glslType.elements = type.vecsize;

            return glslType;
        };*/

    // in/out
    /*for (const spirv_cross::Resource& r : resources.stage_inputs)
    {
        CSPIRVIntrospector::CStageIntrospectionData::SDescriptorVarInfo<>& res = addInfo_common(r, ESIT_STAGE_INPUT);
        res.glslType = getStageIOtype(r.base_type_id);
    }
    for (const spirv_cross::Resource& r : resources.stage_outputs)
    {
        CSPIRVIntrospector::CStageIntrospectionData::SDescriptorVarInfo<>& res = addInfo_common(r, ESIT_STAGE_OUTPUT);
        res.glslType = getStageIOtype(r.base_type_id);

        res.get<ESIT_STAGE_OUTPUT>().colorIndex = _comp.get_decoration(r.id, spv::DecorationIndex);
    }
    std::sort(introData->inputOutput.begin(), introData->inputOutput.end(), [](const CSPIRVIntrospector::CStageIntrospectionData::SDescriptorVarInfo<>& _lhs, const CSPIRVIntrospector::CStageIntrospectionData::SDescriptorVarInfo<>& _rhs) { return _lhs.location < _rhs.location; });*/

    // push constants
    /*if (resources.push_constant_buffers.size())
    {
        const spirv_cross::Resource& r = resources.push_constant_buffers.front();
        introData->pushConstant.present = true;
        introData->pushConstant.name = r.name;
        shaderMemBlockIntrospection(_comp, introData->pushConstant.info, r.base_type_id, r.id, mapId2SpecConst);
    }*/

    return stageIntroData;
}
#if 0 
static void introspectStructType(spirv_cross::Compiler& comp, decltype(CSPIRVIntrospector::CStageIntrospectionData::SMemoryBlock<true>::SMember::members)& dstMembers, const spirv_cross::SPIRType& parentType, const spirv_cross::SmallVector<spirv_cross::TypeID>& allMembersTypes, uint32_t _baseOffset, /*const mapId2SpecConst_t& _mapId2sconst, */core::stack<StackElement>& _pushStack)
{
    using MembT = CSPIRVIntrospector::CStageIntrospectionData::SMemoryBlock<true>::SMember;

    auto MemberDefault = [] {
        MembT m;
        m.count.value = 1u;
        m.count.isSpecConstant = false;
        m.offset = 0u;
        m.size = 0u;
        m.stride = 0u;
        m.typeInfo.stride = 0u; m.
        m.typeInfo.rowMajor = false;
        m.typeInfo.type = CSPIRVIntrospector::CStageIntrospectionData::VAR_TYPE::UNKNOWN_OR_STRUCT;
        //m.members.array = nullptr;
        //m.members.count = 0u;
        return m;
        };

    const uint32_t memberCnt = allMembersTypes.size();
    //dstMembers.array = _NBL_NEW_ARRAY(MembT, memberCnt);
    //dstMembers.count = memberCnt;
    std::fill(dstMembers.array, dstMembers.array + memberCnt, MemberDefault());
    for (uint32_t m = 0u; m < memberCnt; ++m)
    {
        MembT& member = dstMembers.array[m];
        const spirv_cross::SPIRType& mtype = comp.get_type(allMembersTypes[m]);

        member.name = comp.get_member_name(parentType.self, m);
        member.size = comp.get_declared_struct_member_size(parentType, m);
        member.offset = _baseOffset + comp.type_struct_member_offset(_parentType, m);
        member.rowMajor = _comp.get_member_decoration(_parentType.self, m, spv::DecorationRowMajor);
        member.type = spvcrossType2E_TYPE(mtype.basetype);
        member.stride = 0u;

        // if array, then we can get array stride from decoration (via spirv-cross)
        // otherwise arrayStride is left with value 0
        if (mtype.array.size())
        {
            member.count = mtype.array[0];
            member.stride = _comp.type_struct_member_array_stride(_parentType, m);
            member.countIsSpecConstant = !mtype.array_size_literal[0];
            if (member.countIsSpecConstant)
            {
                auto sc_itr = _mapId2sconst.find(member.count);
                assert(sc_itr != _mapId2sconst.end());
                auto sc = sc_itr->second;
                member.count_specID = sc->id;
            }
        }

        if (mtype.basetype == spirv_cross::SPIRType::Struct) //recursive introspection done in DFS manner (and without recursive calls)
            pushStack.push({ member.members, mtype, member.offset });
        else
        {
            //member.mtxRowCnt = mtype.vecsize;
            //member.mtxColCnt = mtype.columns;
            //if (member.mtxColCnt > 1u)
            //    member.mtxStride = _comp.type_struct_member_matrix_stride(_parentType, m);
        }
    }
}
#endif

void CSPIRVIntrospector::shaderMemBlockIntrospection(spirv_cross::Compiler& comp, CSPIRVIntrospector::CStageIntrospectionData::SMemoryBlock<true>& res, uint32_t blockBaseTypeID, uint32_t varID/*, const mapId2SpecConst_t& _sortedId2sconst*/) const
{
    // TODO:
    /*
    using MembT = impl::SShaderMemoryBlock::SMember;
    
    core::stack<StackElement> introspectionStack;
    const spirv_cross::SPIRType& type = comp.get_type(blockBaseTypeID);
    introspectionStack.push({ res.members, type, 0u });
    while (!introspectionStack.empty()) {
        StackElement e = introspectionStack.top();
        introspectionStack.pop();
        introspectStructType(comp, e.membersDst, e.parentType, e.parentType.member_types, 0u, sortedId2sconst, introspectionStack);
    }

    res.size = res.rtSizedArrayOneElementSize = comp.get_declared_struct_size(type);
    const spirv_cross::SPIRType& lastType = comp.get_type(type.member_types.back());
    if (lastType.array.size() && lastType.array_size_literal[0] && lastType.array[0] == 0u)
        res.rtSizedArrayOneElementSize += res.members.array[res.members.count - 1u].arrayStride;

    spirv_cross::Bitset flags = comp.get_buffer_block_flags(varID);
    res.restrict_ = flags.get(spv::DecorationRestrict);
    res.volatile_ = flags.get(spv::DecorationVolatile);
    res.coherent = flags.get(spv::DecorationCoherent);
    res.readonly = flags.get(spv::DecorationNonWritable);
    res.writeonly = flags.get(spv::DecorationNonReadable);*/
}

size_t CSPIRVIntrospector::calcBytesizeforType(spirv_cross::Compiler& comp, const spirv_cross::SPIRType& type) const
{
    size_t bytesize = 0u;
    switch (type.basetype)
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
        bytesize = comp.get_declared_struct_size(type);
        assert(type.columns > 1u || type.vecsize > 1u); // something went wrong (cannot have matrix/vector of struct type)
        break;
    default:
        assert(0);
        break;
    }
    bytesize *= type.vecsize * type.columns; //vector or matrix
    if (type.array.size()) //array
        bytesize *= type.array[0];

    return bytesize;
                }

}
