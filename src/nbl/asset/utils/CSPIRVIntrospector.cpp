// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/utils/CSPIRVIntrospector.h"
#include "nbl/asset/utils/spvUtils.h"

#include "nbl_spirv_cross/spirv_parser.hpp"
#include "nbl_spirv_cross/spirv_cross.hpp"

namespace nbl::asset
{

static CSPIRVIntrospector::CStageIntrospectionData::VAR_TYPE spvcrossType2E_TYPE(spirv_cross::SPIRType::BaseType basetype)
{
    switch (basetype)
    {
    case spirv_cross::SPIRType::Int:
        return CSPIRVIntrospector::CStageIntrospectionData::VAR_TYPE::I32;
    case spirv_cross::SPIRType::UInt:
        return CSPIRVIntrospector::CStageIntrospectionData::VAR_TYPE::U32;
    case spirv_cross::SPIRType::Float:
        return CSPIRVIntrospector::CStageIntrospectionData::VAR_TYPE::F32;
    case spirv_cross::SPIRType::Int64:
        return CSPIRVIntrospector::CStageIntrospectionData::VAR_TYPE::I64;
    case spirv_cross::SPIRType::UInt64:
        return CSPIRVIntrospector::CStageIntrospectionData::VAR_TYPE::U64;
    case spirv_cross::SPIRType::Double:
        return CSPIRVIntrospector::CStageIntrospectionData::VAR_TYPE::F64;
    default:
        return CSPIRVIntrospector::CStageIntrospectionData::VAR_TYPE::UNKNOWN_OR_STRUCT;
    }
}

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

CSPIRVIntrospector::CStageIntrospectionData::SDescriptorVarInfo<true>* CSPIRVIntrospector::CStageIntrospectionData::addResource(
    const spirv_cross::Compiler& comp,
    const spirv_cross::Resource& r,
    IDescriptor::E_TYPE restype
)
{
    const uint32_t descSet = comp.get_decoration(r.id,spv::DecorationDescriptorSet);
    if (descSet>DescriptorSetCount)
        return nullptr; // TODO: log fail

    const spirv_cross::SPIRType& type = comp.get_type(r.type_id);
    const auto arrDim = type.array.size();
    // assuming only 1D arrays because i don't know how desc set layout binding is constructed when it's let's say 2D array (e.g. uniform sampler2D smplr[4][5]; is it even legal?)
    if (arrDim>1)
        return nullptr; // TODO: log fail

    CStageIntrospectionData::SDescriptorVarInfo<true> res = {
        {
            .binding = comp.get_decoration(r.id,spv::DecorationBinding),
            .type = restype
        },
        /*.name = */addString(r.name),
        /*.count = */addCounts(arrDim,type.array.data(),type.array_size_literal.data())
    };

    auto& ref = reinterpret_cast<core::vector<CStageIntrospectionData::SDescriptorVarInfo<true>>*>(m_descriptorSetBindings)[descSet].emplace_back(std::move(res));
    return &ref;
}

void CSPIRVIntrospector::CStageIntrospectionData::shaderMemBlockIntrospection(const spirv_cross::Compiler& comp, SMemoryBlock<true>* root, const spirv_cross::Resource& r)
{
    core::unordered_map<uint32_t/*spirv_cross::TypeID*/,type_ptr<true>> typeCache;

    struct StackElement
    {
        // the root type pointer backed by slightly different memory, so handle it specially when detected
        bool isRoot() const {return !parentType;}

        spirv_cross::TypeID selfTypeID;
        spirv_cross::TypeID parentTypeID;
        type_ptr<true> parentType = {};
        uint32_t memberIndex = 0;
    };
    core::stack<StackElement> introspectionStack;

    // NOTE: might need to lookup SPIRType based on `base_type_id` and forward to `SPIRType::type_alias` here instead
    introspectionStack.emplace(r.base_type_id);

    bool first = true;
    while (!introspectionStack.empty())
    {
        const auto entry = introspectionStack.top();
        introspectionStack.pop();

        const spirv_cross::SPIRType& type = comp.get_type(entry.selfTypeID);

        type_ptr<true> pType;
        auto found = typeCache.find(entry.selfTypeID);
        if (found!=typeCache.end())
            pType = found->second;
        else
        {
            const auto memberCount = type.member_types.size();
            pType = addType(memberCount);
            if (type.basetype==spirv_cross::SPIRType::Struct)
            for (uint32_t i=0; i<memberCount; i++)
                introspectionStack.emplace(type.member_types[i],entry.selfTypeID,pType,i);
        }
        // pointer might change after allocation of something new, so always access through this lambda
        auto getTypeStore = [&]()->SType<true>*{return pType(m_memPool.data());};
        
        if (entry.isRoot())
            root->type = pType;
        else
        {
            auto getParentTypeStore = [&]()->auto {return entry.parentType(m_memPool.data());};
            getParentTypeStore()->memberTypes()(m_memPool.data())[entry.memberIndex] = pType;
            getParentTypeStore()->memberNames()(m_memPool.data())[entry.memberIndex] = addString(comp.get_member_name(entry.parentTypeID,entry.memberIndex));
            const auto& parentType = comp.get_type(entry.parentTypeID);
            getParentTypeStore()->memberSizes()(m_memPool.data())[entry.memberIndex] = comp.get_declared_struct_member_size(parentType,entry.memberIndex);
            getParentTypeStore()->memberOffsets()(m_memPool.data())[entry.memberIndex] = comp.type_struct_member_offset(parentType,entry.memberIndex);
            getParentTypeStore()->memberStrides()(m_memPool.data())[entry.memberIndex] = getTypeStore()->isArray() ? comp.type_struct_member_array_stride(parentType,entry.memberIndex):0u;
//            comp.get_declared_struct_size();
//            comp.get_declared_struct_size_runtime_array();
#if 0 
    for (uint32_t m = 0u; m < memberCnt; ++m)
    {
        MembT& member = dstMembers.array[m];
        const spirv_cross::SPIRType& mtype = comp.get_type(allMembersTypes[m]);

        member.rowMajor = _comp.get_member_decoration(_parentType.self, m, spv::DecorationRowMajor); // ?

            //if (member.mtxColCnt > 1u)
            //    member.mtxStride = _comp.type_struct_member_matrix_stride(_parentType, m);
#endif
        }

        // found in cache, then don't need to fill out the rest
        if (found!=typeCache.end())
            continue;

        getTypeStore()->count = addCounts(type.array.size(),type.array.data(),type.array_size_literal.data());
        getTypeStore()->typeName = addString("TODO");
        {
            auto typeEnum = VAR_TYPE::UNKNOWN_OR_STRUCT;
            switch (type.basetype)
            {
                case spirv_cross::SPIRType::BaseType::SByte:
                    typeEnum = VAR_TYPE::I8;
                    break;
                case spirv_cross::SPIRType::BaseType::UByte:
                    typeEnum = VAR_TYPE::U8;
                    break;
                case spirv_cross::SPIRType::BaseType::Short:
                    typeEnum = VAR_TYPE::I16;
                    break;
                case spirv_cross::SPIRType::BaseType::UShort:
                    typeEnum = VAR_TYPE::U16;
                    break;
                case spirv_cross::SPIRType::BaseType::Int:
                    typeEnum = VAR_TYPE::I32;
                    break;
                case spirv_cross::SPIRType::BaseType::UInt:
                    typeEnum = VAR_TYPE::U32;
                    break;
                case spirv_cross::SPIRType::BaseType::Int64:
                    typeEnum = VAR_TYPE::I64;
                    break;
                case spirv_cross::SPIRType::BaseType::UInt64:
                    typeEnum = VAR_TYPE::U64;
                    break;
                case spirv_cross::SPIRType::BaseType::Half:
                    typeEnum = VAR_TYPE::F16;
                    break;
                case spirv_cross::SPIRType::BaseType::Float:
                    typeEnum = VAR_TYPE::F32;
                    break;
                case spirv_cross::SPIRType::BaseType::Double:
                    typeEnum = VAR_TYPE::F64;
                    break;
                default:
                    // TODO: get name of the type
                    //        getTypeStore()->typeName = addString(comp.get_name(type));
                    typeEnum = VAR_TYPE::UNKNOWN_OR_STRUCT;
                    break;
            }
            if (typeEnum!=VAR_TYPE::UNKNOWN_OR_STRUCT)
            {
                // TODO: assign names of simple types
                //getTypeStore()->typeName = m_typenames[typeEnum-VAR_TYPE::UNKNOWN_OR_STRUCT][lastRow][lastColumn];
            }

            auto& info = getTypeStore()->info;
            {
                info.lastRow = type.vecsize-1;
                info.lastCol = type.columns-1;
                info.rowMajor = comp.get_decoration(entry.selfTypeID,spv::DecorationRowMajor);
                info.type = typeEnum;
                info.restrict_ = comp.get_decoration(entry.selfTypeID,spv::DecorationRestrict);
                info.aliased = comp.get_decoration(entry.selfTypeID,spv::DecorationAliased);
            }
        }
    }
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

    comp.set_entry_point(params.entryPoint, stage);


    // spec constants
    spirv_cross::SmallVector<spirv_cross::SpecializationConstant> sconsts = comp.get_specialization_constants();
    stageIntroData->m_specConstants.reserve(sconsts.size());
    for (size_t i = 0u; i < sconsts.size(); ++i)
    {
        CSPIRVIntrospector::CStageIntrospectionData::SSpecConstant specConst;
        specConst.id = sconsts[i].constant_id;
        specConst.name = comp.get_name(sconsts[i].id);

        const spirv_cross::SPIRConstant& sconstval = comp.get_constant(sconsts[i].id);
        const spirv_cross::SPIRType& type = comp.get_type(sconstval.constant_type);
        specConst.byteSize = calcBytesizeForType(comp, type);
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

        auto where = std::lower_bound(stageIntroData->m_specConstants.begin(), stageIntroData->m_specConstants.end(), specConst, [](const auto& _lhs, const auto& _rhs) { return _lhs.id < _rhs.id; });
        stageIntroData->m_specConstants.insert(where, specConst);
    }

    spirv_cross::ShaderResources resources = comp.get_shader_resources(/*TODO: allow choice in Introspection Parameters, comp.get_active_interface_variables()*/);
    for (const spirv_cross::Resource& r : resources.uniform_buffers)
    {
        auto* res = stageIntroData->addResource(comp,r,IDescriptor::E_TYPE::ET_UNIFORM_BUFFER);
        if (!res)
            return nullptr;
        spirv_cross::Bitset flags = comp.get_buffer_block_flags(r.id);
        res->storageBuffer.readonly = flags.get(spv::DecorationNonWritable);
        res->storageBuffer.writeonly = flags.get(spv::DecorationNonReadable);
        res->restrict_ = flags.get(spv::DecorationRestrict);
        res->aliased = flags.get(spv::DecorationAliased);
        stageIntroData->shaderMemBlockIntrospection(comp,&res->uniformBuffer,r);
        res->uniformBuffer.size = comp.get_declared_struct_size(comp.get_type(r.type_id));
    }
    for (const spirv_cross::Resource& r : resources.storage_buffers)
    {
        auto* res = stageIntroData->addResource(comp,r,IDescriptor::E_TYPE::ET_STORAGE_BUFFER);
        if (!res)
            return nullptr;
        spirv_cross::Bitset flags = comp.get_buffer_block_flags(r.id);
        res->storageBuffer.readonly = flags.get(spv::DecorationNonWritable);
        res->storageBuffer.writeonly = flags.get(spv::DecorationNonReadable);
        res->restrict_ = flags.get(spv::DecorationRestrict);
        res->aliased = flags.get(spv::DecorationAliased);
        stageIntroData->shaderMemBlockIntrospection(comp,&res->storageBuffer,r);
        res->storageBuffer.sizeWithoutLastMember = comp.get_declared_struct_size_runtime_array(comp.get_type(r.type_id),0);
    }
    for (const spirv_cross::Resource& r : resources.subpass_inputs)
    {
        auto* res = stageIntroData->addResource(comp,r,IDescriptor::E_TYPE::ET_INPUT_ATTACHMENT);
        if (!res)
            return nullptr;
        res->inputAttachment.index = comp.get_decoration(r.id,spv::DecorationInputAttachmentIndex);
    }
    auto spvcrossImageType2ImageView = [](const spirv_cross::SPIRType::ImageType& img) -> IImageView<ICPUImage>::E_TYPE
    {
        switch (img.type)
        {
            case spv::Dim::Dim1D:
                return img.arrayed ? ICPUImageView::ET_1D_ARRAY:ICPUImageView::ET_1D;
            case spv::Dim::DimSubpassData: [[fallthrough]];
            case spv::Dim::Dim2D:
                return img.arrayed ? ICPUImageView::ET_2D_ARRAY:ICPUImageView::ET_2D;
            case spv::Dim::Dim3D:
                assert(!img.arrayed);
                return ICPUImageView::ET_3D;
            case spv::Dim::DimCube:
                return img.arrayed ? ICPUImageView::ET_CUBE_MAP_ARRAY:ICPUImageView::ET_CUBE_MAP;
        }
        return ICPUImageView::ET_COUNT;
    };
    for (const spirv_cross::Resource& r : resources.storage_images)
    {
        const spirv_cross::SPIRType& type = comp.get_type(r.type_id);
        assert(!type.image.depth);
        assert(!type.image.sampled);
        const bool buffer = type.image.dim==spv::DimBuffer;
        auto* res = stageIntroData->addResource(comp,r,buffer ? IDescriptor::E_TYPE::ET_STORAGE_TEXEL_BUFFER:IDescriptor::E_TYPE::ET_STORAGE_IMAGE);
        if (!res)
            return nullptr;
        if (!buffer)
        {
            res->storageImage.readonly = type.image.access==spv::AccessQualifierReadOnly;
            res->storageImage.writeonly = type.image.access==spv::AccessQualifierWriteOnly;
            res->storageImage.viewType = spvcrossImageType2ImageView(type.image);
            res->storageImage.shadow = type.image.depth;
            switch (type.image.format)
            {
                case spv::ImageFormatRgba32f:
                    res->storageImage.format = EF_R32G32B32A32_SFLOAT;
                    break;
                // TODO: Przemog1 do the rest!
                default:
                    res->storageImage.format = EF_UNKNOWN;
                    break;
            }
        }
        else
        {
            res->storageTexelBuffer.readonly = type.image.access==spv::AccessQualifierReadOnly;
            res->storageTexelBuffer.writeonly = type.image.access==spv::AccessQualifierWriteOnly;
        }
    }
    for (const spirv_cross::Resource& r : resources.sampled_images)
    {
        const spirv_cross::SPIRType& type = comp.get_type(r.type_id);
        const bool buffer = type.image.dim==spv::DimBuffer;
        auto* res = stageIntroData->addResource(comp,r,buffer ? IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER:IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER);
        if (!res)
            return nullptr;
        if (!buffer)
        {
            res->combinedImageSampler.viewType = spvcrossImageType2ImageView(type.image);
            res->combinedImageSampler.shadow = type.image.depth;
            res->combinedImageSampler.multisample = type.image.ms;
        }
    }
    // skip cause we don't support
    //for (const spirv_cross::Resource& r : resources.separate_images) {}
    //for (const spirv_cross::Resource& r : resources.separate_samplers) {}
    for (auto& descSet : stageIntroData->m_descriptorSetBindings)
        std::sort(descSet.begin(),descSet.end());

    auto getStageIOtype = [&comp](CSPIRVIntrospector::CStageIntrospectionData::SInterface& glslType, uint32_t _base_type_id)
        {
            const auto& type = comp.get_type(_base_type_id);
            glslType.baseType = spvcrossType2E_TYPE(type.basetype);
            glslType.elements = type.vecsize;

            return glslType;
        };

    // in/out
    for (const spirv_cross::Resource& r : resources.stage_inputs)
    {
        CSPIRVIntrospector::CStageIntrospectionData::SInputInterface& res = stageIntroData->m_input.emplace_back();
        getStageIOtype(res, r.base_type_id);
    }
    for (const spirv_cross::Resource& r : resources.stage_outputs)
    {
        using OutputVecT = core::vector<CSPIRVIntrospector::CStageIntrospectionData::SOutputInterface>;
        using FragmentOutputVecT = core::vector<CSPIRVIntrospector::CStageIntrospectionData::SFragmentOutputInterface>;

        if (params.shader->getStage() == IShader::ESS_FRAGMENT)
        {
            CSPIRVIntrospector::CStageIntrospectionData::SFragmentOutputInterface res =
                std::get<FragmentOutputVecT>(stageIntroData->m_output).emplace_back();
            getStageIOtype(res, r.base_type_id);

            res.colorIndex = comp.get_decoration(r.id, spv::DecorationIndex);
        }
        else
        {
            CSPIRVIntrospector::CStageIntrospectionData::SOutputInterface res =
                std::get<OutputVecT>(stageIntroData->m_output).emplace_back();
            getStageIOtype(res, r.base_type_id);
        }
    }
    // why do we need it sorted?
    //std::sort(introData->inputOutput.begin(), introData->inputOutput.end(), [](const CSPIRVIntrospector::CStageIntrospectionData::SDescriptorVarInfo<>& _lhs, const CSPIRVIntrospector::CStageIntrospectionData::SDescriptorVarInfo<>& _rhs) { return _lhs.location < _rhs.location; });

    // push constants
    auto* pPushConstantsMutable = reinterpret_cast<CStageIntrospectionData::SPushConstantInfo<true>*>(&stageIntroData->m_pushConstants);
    if (resources.push_constant_buffers.size())
    {
        assert(resources.push_constant_buffers.size()==1);
        const spirv_cross::Resource& r = resources.push_constant_buffers.front();
        pPushConstantsMutable->name = stageIntroData->addString(r.name);
        stageIntroData->shaderMemBlockIntrospection(comp,pPushConstantsMutable,r);
    }
    else
        pPushConstantsMutable->type = {};

    // convert all Mutable to non-mutable
    stageIntroData->finalize();

    return stageIntroData;
}

void CSPIRVIntrospector::CStageIntrospectionData::finalize()
{
    auto* const basePtr = m_memPool.data();
    auto addBaseAndConvertStringToImmutable = [basePtr](std::span<const char>& name)->void
    {
        name = reinterpret_cast<const core::based_span<char>&>(name)(basePtr);
    };
    auto addBaseAndConvertExtentToImmutable = [basePtr](std::span<const CIntrospectionData::SArrayInfo>& count)->void
    {
        count = reinterpret_cast<const core::based_span<CIntrospectionData::SArrayInfo>&>(count)(basePtr);
    };
    // TODO: spec constant finalization 

    auto addBaseAndConvertTypeToImmutable = [&](SType<true>* type)->void
    {
        auto* outImmutable = reinterpret_cast<SType<false>*>(type);
        addBaseAndConvertStringToImmutable(outImmutable->typeName);
        addBaseAndConvertExtentToImmutable(outImmutable->count);
        // get these before they change
        const auto memberTypes = type->memberTypes()(basePtr);
        const auto memberNames = type->memberNames()(basePtr);
        for (auto m=0; m<type->memberCount; m++)
        {
            reinterpret_cast<const SType<false>**>(memberTypes)[m] = reinterpret_cast<const SType<false>*>(memberTypes[m](basePtr));
            reinterpret_cast<std::span<const char>*>(memberNames)[m] = memberNames[m](basePtr);
        }
        outImmutable->memberInfoStorage = type->memberInfoStorage(basePtr);
    };
    auto addBaseAndConvertBlockToImmutable = [&](SMemoryBlock<true>& block)->void
    {
        visitMemoryBlockPreOrderDFS(block,addBaseAndConvertTypeToImmutable);
        auto& asImmutable = reinterpret_cast<SMemoryBlock<false>&>(block);
        asImmutable.type = reinterpret_cast<SType<false>*>(block.type(basePtr));
    };

    addBaseAndConvertBlockToImmutable(reinterpret_cast<SPushConstantInfo<true>&>(m_pushConstants));
    addBaseAndConvertStringToImmutable(m_pushConstants.name);
    std::ostringstream debug = {};
    if (m_pushConstants.type)
    {
        debug << "PUSH CONSTANT BLOCK:\n";
        printType(debug,m_pushConstants.type);
        debug << "} " << m_pushConstants.name.data() << ";\n";
    }
    
    for (auto set=0; set<DescriptorSetCount; set++)
    for (auto& descriptor : m_descriptorSetBindings[set])
    {
        addBaseAndConvertStringToImmutable(descriptor.name);
        addBaseAndConvertExtentToImmutable(descriptor.count);
        auto& asMutable = reinterpret_cast<SDescriptorVarInfo<true>&>(descriptor);
        debug << "(set=" << set << ",binding=" << descriptor.binding << ") ";
        switch (descriptor.type)
        {
            case IDescriptor::E_TYPE::ET_UNIFORM_BUFFER:
                addBaseAndConvertBlockToImmutable(asMutable.uniformBuffer);
                printType(debug,descriptor.uniformBuffer.type);
                break;
            case IDescriptor::E_TYPE::ET_STORAGE_BUFFER:
                addBaseAndConvertBlockToImmutable(asMutable.storageBuffer);
                printType(debug,descriptor.storageBuffer.type);
                break;
            default:
                debug << "TODO_type"; // don't implement if not needed
                break;
        }
        debug << " " << descriptor.name.data();
        printExtents(debug,descriptor.count);
        debug << ";\n";
    }
    printf("%s\n",debug.str().c_str());
}

void CSPIRVIntrospector::CStageIntrospectionData::printExtents(std::ostringstream& out, const std::span<const SArrayInfo> counts)
{
    for (auto i=counts.size()-1; i!=(~0ull); i--)
    {
        const auto extent = counts[i];
        out << "[";
        if (extent.isSpecConstant)
            out << "specID=" << extent.specID;
        else if (!extent.isRuntimeSized())
            out << extent.value;
        out << "]";
    }
}
void CSPIRVIntrospector::CStageIntrospectionData::printType(std::ostringstream& out, const SType<false>* type, const uint32_t depth)
{
    assert(type);
    // pre
    auto indent = [&]() -> std::ostringstream&
    {
        for (auto i=0; i<depth; i++)
            out << "\t";
        return out;
    };
    indent() << type->typeName.data();
    if (type->info.type==VAR_TYPE::UNKNOWN_OR_STRUCT)
    {
        out << "\n";
        indent() << "{\n";
    }

    // in-order
    const auto nextDepth = depth+1;
    for (auto m=0; m<type->memberCount; m++)
    {
        printType(out,type->memberTypes()[m],nextDepth); out << " " << type->memberNames()[m].data() << "; // offset=" << type->memberOffsets()[m] << ",size=" << type->memberSizes()[m] << ",stride=" << type->memberStrides()[m] << "\n";
    }

    // post
    if (type->info.type==VAR_TYPE::UNKNOWN_OR_STRUCT)
        indent() << "}";
    printExtents(out,type->count);
}

size_t CSPIRVIntrospector::calcBytesizeForType(spirv_cross::Compiler& comp, const spirv_cross::SPIRType& type) const
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