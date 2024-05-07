// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

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

    // TODO: delete when we are done testing
    std::string_view getBaseTypeNameFromEnum(CSPIRVIntrospector::CStageIntrospectionData::VAR_TYPE varType)
    {
        constexpr std::string_view typeNames[] = {
            "UNKNOWN_OR_STRUCT",
            "U64",
            "I64",
            "U32",
            "I32",
            "U16",
            "I16",
            "U8",
            "I8",
            "F64",
            "F32",
            "F16"
        };

        return typeNames[static_cast<uint32_t>(varType)];
    };
}//anonymous ns

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

core::smart_refctd_ptr<ICPUComputePipeline> CSPIRVIntrospector::createApproximateComputePipelineFromIntrospection(const ICPUShader::SSpecInfo& info, core::smart_refctd_ptr<ICPUPipelineLayout>&& layout/* = nullptr*/)
{
    if (info.shader->getStage() != IShader::ESS_COMPUTE || info.valid() == ICPUShader::SSpecInfo::INVALID_SPEC_INFO)
        return nullptr;

    CStageIntrospectionData::SParams params;
    params.entryPoint = info.entryPoint;
    params.shader = core::smart_refctd_ptr<ICPUShader>(info.shader);

    auto introspection = introspect(params);

    core::smart_refctd_ptr<CPipelineIntrospectionData> pplnIntrospectData = core::make_smart_refctd_ptr<CPipelineIntrospectionData>();
    if (!pplnIntrospectData->merge(introspection.get()))
        return nullptr;

    if (layout)
    {
        // regarding push constants we only need to validate if every range of push constants determined by the introspection is subset any push constants subset of predefined layout
        core::smart_refctd_ptr<ICPUPipelineLayout> pplnIntrospectionLayout = pplnIntrospectData->createApproximatePipelineLayoutFromIntrospection(introspection);
        const auto& introspectionPushConstantRanges = pplnIntrospectionLayout->getPushConstantRanges();
        if (!introspectionPushConstantRanges.empty())
        {
            const auto& layoutPushConstantRanges = layout->getPushConstantRanges();
            if (layoutPushConstantRanges.empty())
                return nullptr;

            for (auto& introPcRange : introspectionPushConstantRanges)
            {
                auto subsetRangeFound = std::find_if(
                    layoutPushConstantRanges.begin(),
                    layoutPushConstantRanges.end(),
                    [&introPcRange](const SPushConstantRange& layoutPcRange)
                    {
                        const bool isIntrospectionPushConstantRangeSubset =
                            introPcRange.offset <= layoutPcRange.offset &&
                            introPcRange.offset + introPcRange.size <= layoutPcRange.offset + layoutPcRange.size;
                        return isIntrospectionPushConstantRangeSubset;
                    });

                auto asdf = layoutPushConstantRanges.end();

                if (subsetRangeFound == layoutPushConstantRanges.end())
                    return nullptr;
            }
        }

        // now validate if bindings of descriptor sets in `introspection` are also present in `layout` descriptor sets and validate their compatability
        for (uint32_t dstSetIdx = 0; dstSetIdx < ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++dstSetIdx)
        {
            const auto& layoutDescriptorSetLayout = layout->getDescriptorSetLayout(dstSetIdx);

            const auto& introspectionDescriptorSetLayout = introspection->getDescriptorSetInfo(dstSetIdx);
            if (!introspectionDescriptorSetLayout.empty())
            {
                auto dscLayout = pplnIntrospectionLayout->getDescriptorSetLayout(dstSetIdx);
                if (!dscLayout)
                    continue;
                if (!dscLayout->isSubsetOf(layout->getDescriptorSetLayout(dstSetIdx)))
                    return nullptr;
            }
        }
    }
    else
    {
        layout = pplnIntrospectData->createApproximatePipelineLayoutFromIntrospection(introspection);
    }

    ICPUComputePipeline::SCreationParams pplnCreationParams = { {.layout = layout.get()} };
    pplnCreationParams.shader = info;
    pplnCreationParams.layout = layout.get();
    return ICPUComputePipeline::create(pplnCreationParams);
}

// returns true if successfully added all the info to self, false if incompatible with what's already in our pipeline or incomplete (e.g. missing spec constants)
NBL_API2 bool CSPIRVIntrospector::CPipelineIntrospectionData::merge(const CSPIRVIntrospector::CStageIntrospectionData* stageData, const ICPUShader::SSpecInfoBase::spec_constant_map_t* specConstants)
{
    if (!stageData)
        return false;

    // validate if descriptors are compatible
    DescriptorSetBindings descriptorsToMerge[ICPUPipelineLayout::DESCRIPTOR_SET_COUNT];
    HighestBindingArray highestBindingsTmp = m_highestBindingNumbers;
    for (uint32_t i = 0u; i < ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
    {
        const auto& introBindingInfos = stageData->getDescriptorSetInfo(i);
        for (const auto& stageIntroBindingInfo : introBindingInfos)
        {
            CPipelineIntrospectionData::SDescriptorInfo descInfo;
            descInfo.binding = stageIntroBindingInfo.binding;
            descInfo.type = stageIntroBindingInfo.type;
            descInfo.stageMask = stageData->m_shaderStage;
            if (stageIntroBindingInfo.isArray())
            {
                if (stageIntroBindingInfo.isRunTimeSized())
                {
                    descInfo.count = stageIntroBindingInfo.count.count;
                    descInfo.isRuntimeSizedFlag = true;
                }
                else
                {
                    const auto count = stageIntroBindingInfo.count;
                    if (count.countMode == SDescriptorArrayInfo::DESCRIPTOR_COUNT::SPEC_CONSTANT)
                    {
                        if (!specConstants)
                            return false;

                        const auto& specConstantFound = specConstants->find(count.specID);
                        if (specConstantFound == specConstants->end())
                            return false;

                        descInfo.count = specConstantFound->second;
                    }
                    else
                    {
                        descInfo.count = stageIntroBindingInfo.count.count;
                    }

                    if (descInfo.count == 0u)
                        return false;

                    descInfo.isRuntimeSizedFlag = false;
                }
            }
            else
            {
                descInfo.count = 1u;
                descInfo.isRuntimeSizedFlag = false;
            }

            const auto& selfIntersectionFound = descriptorsToMerge[i].find(descInfo);
            if (selfIntersectionFound != descriptorsToMerge[i].end() && selfIntersectionFound->type != stageIntroBindingInfo.type)
                return false;

            const auto& pplnIntroDataFoundBinding = m_descriptorSetBindings[i].find(descInfo);
            if (pplnIntroDataFoundBinding != m_descriptorSetBindings[i].end())
            {
                if (pplnIntroDataFoundBinding->type != stageIntroBindingInfo.type)
                    return false;
                descInfo.count = std::max(pplnIntroDataFoundBinding->count, descInfo.count);
            }

            highestBindingsTmp[i].binding = std::max<const int32_t>(highestBindingsTmp[i].binding, stageIntroBindingInfo.binding);
            highestBindingsTmp[i].isRunTimeSized = descInfo.isRuntimeSized();
            descriptorsToMerge[i].insert(descInfo);
        }
    }

    // validate if only descriptors with the highest bindings are run-time sized
    for (uint32_t i = 0u; i < ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
    {
        const auto& introBindingInfos = stageData->getDescriptorSetInfo(i);

        for (const auto& descriptor : descriptorsToMerge[i])
        {
            if (descriptor.binding < highestBindingsTmp[i].binding && descriptor.isRuntimeSized())
                return false;
        }

        if (m_highestBindingNumbers[i].isRunTimeSized && m_highestBindingNumbers[i].binding < highestBindingsTmp[i].binding)
            return false;
    }

    //// validation successfull, update `CPipelineIntrospectionData` contents
    m_highestBindingNumbers = highestBindingsTmp;
    for (uint32_t i = 0u; i < ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
        m_descriptorSetBindings[i].merge(descriptorsToMerge[i]);

    // can only be success now
    const auto& pc = stageData->getPushConstants();
    auto a = pc.size;
    if (pc.present())
    {
        std::span<core::bitflag<ICPUShader::E_SHADER_STAGE>> pcRangesSpan = {
            m_pushConstantBytes.data() + pc.offset,
            pc.size
        };

        // iterate over all bytes used
        const IShader::E_SHADER_STAGE shaderStage = stageData->getParams().shader->getStage();
        for (auto it = pcRangesSpan.begin(); it != pcRangesSpan.end(); ++it)
            *it |= shaderStage;
    }

    return true;
}

//
NBL_API2 core::smart_refctd_dynamic_array<SPushConstantRange> CSPIRVIntrospector::CPipelineIntrospectionData::createPushConstantRangesFromIntrospection(core::smart_refctd_ptr<const CStageIntrospectionData>& introspection)
{
    auto& pc = introspection->getPushConstants();

    core::vector<SPushConstantRange> tmp; 
    tmp.reserve(MaxPushConstantsSize);
    
    SPushConstantRange prev = {
        .stageFlags = m_pushConstantBytes[0].value,
        .offset = 0,
        .size = 0
    };

    SPushConstantRange current = {
        .offset = 0,
        .size = 0
    };

    // run-length encode m_pushConstantBytes
    for (uint32_t currentByteOffset = 1u; currentByteOffset < MaxPushConstantsSize; ++currentByteOffset)
    {
        current.stageFlags = m_pushConstantBytes[currentByteOffset].value;
        current.offset++;
        current.size++;

        if (current.stageFlags != prev.stageFlags)
        {
            if (prev.stageFlags)
            {
                prev.size = current.offset - prev.offset;
                tmp.push_back(prev);
            }
            prev = current;
            current.size = 0u;
        }
    }

    if (prev.stageFlags)
    {
        prev.size = MaxPushConstantsSize - prev.offset;
        tmp.push_back(prev);
    }

    if(tmp.size() == 0)
        return nullptr;

    return core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<SPushConstantRange>>(tmp);
}
NBL_API2 core::smart_refctd_ptr<ICPUDescriptorSetLayout> CSPIRVIntrospector::CPipelineIntrospectionData::createApproximateDescriptorSetLayoutFromIntrospection(const uint32_t setID)
{
    std::vector<ICPUDescriptorSetLayout::SBinding> outBindings;
    outBindings.reserve(m_descriptorSetBindings[setID].size());

    for (const auto& binding : m_descriptorSetBindings[setID])
    {
        auto& outBinding = outBindings.emplace_back();

        outBinding.binding = binding.binding;
        outBinding.count = binding.count;
        outBinding.type = binding.type;
        outBinding.stageFlags = binding.stageMask;
        outBinding.createFlags = binding.isRuntimeSized() ? ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_VARIABLE_DESCRIPTOR_COUNT_BIT : ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE;
        // TODO: outBinding.samplers
    }

    if (outBindings.empty())
        return nullptr;

    core::smart_refctd_ptr<ICPUDescriptorSetLayout> output = core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(outBindings.data(), outBindings.data() + outBindings.size());
    return output;
}

NBL_API2 core::smart_refctd_ptr<ICPUPipelineLayout> CSPIRVIntrospector::CPipelineIntrospectionData::createApproximatePipelineLayoutFromIntrospection(
    core::smart_refctd_ptr<const CStageIntrospectionData>& introspection)
{
    const auto pcRanges = createPushConstantRangesFromIntrospection(introspection);
    const std::span<const asset::SPushConstantRange> pcRangesSpan = {
        pcRanges ? pcRanges->begin() : nullptr,
        pcRanges ? pcRanges->end() : nullptr
    };

    return core::make_smart_refctd_ptr<ICPUPipelineLayout>(
        pcRangesSpan,
        createApproximateDescriptorSetLayoutFromIntrospection(0),
        createApproximateDescriptorSetLayoutFromIntrospection(1),
        createApproximateDescriptorSetLayoutFromIntrospection(2),
        createApproximateDescriptorSetLayoutFromIntrospection(3)
    );
}

CSPIRVIntrospector::CStageIntrospectionData::SDescriptorVarInfo<true>* CSPIRVIntrospector::CStageIntrospectionData::addResource(
    const spirv_cross::Compiler& comp,
    const spirv_cross::Resource& r,
    IDescriptor::E_TYPE restype
)
{
    const uint32_t descSet = comp.get_decoration(r.id,spv::DecorationDescriptorSet);
    if (descSet > DESCRIPTOR_SET_COUNT)
        return nullptr; // TODO: log fail

    const spirv_cross::SPIRType& type = comp.get_type(r.type_id);
    const size_t arrDim = type.array.size();
    // acording to the Vulkan documentation (15.5.3) only 1D arrays are allowed
    if (arrDim>1)
        return nullptr; // TODO: log fail



    const uint32_t descriptorArraySize = type.array.empty() ? 1u : *type.array.data();    
    const bool isArrayTypeLiteral = type.array_size_literal.empty() ? true : type.array_size_literal[0];
    CStageIntrospectionData::SDescriptorVarInfo<true> res = {
        {
            .binding = comp.get_decoration(r.id,spv::DecorationBinding),
            .type = restype
        },
        /*.name = */addString(r.name),
        /*.count = */addDescriptorCount(descriptorArraySize, isArrayTypeLiteral)
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

            _NBL_DEBUG_BREAK_IF(parentType.columns > 1);

            SType<true>::MatrixInfo matrixInfo = {
                .rowMajor = static_cast<uint8_t>(comp.get_member_decoration(parentType.self, entry.memberIndex, spv::DecorationRowMajor)),
                .rows = static_cast<uint8_t>(parentType.vecsize),
                .columns = static_cast<uint8_t>(parentType.columns)
            };
            std::memcpy(&getParentTypeStore()->memberMatrixInfos()(m_memPool.data())[entry.memberIndex], &matrixInfo, sizeof(SType<true>::MatrixInfo));

            /*getParentTypeStore()->memberMatrixInfos()(m_memPool.data())[entry.memberIndex] = {
                .rowMajor = comp.get_member_decoration(parentType.self, entry.memberIndex, spv::DecorationRowMajor),
                .rows = parentType.vecsize,
                .columns = parentType.columns
            };*/
//          comp.get_declared_struct_size();
//          comp.get_declared_struct_size_runtime_array();
        }

        // found in cache, then don't need to fill out the rest
        if (found!=typeCache.end())
            continue;

        getTypeStore()->count = addCounts(type.array.size(),type.array.data(),type.array_size_literal.data());
        if (!entry.isRoot())
        {
            const auto& parentType = comp.get_type(entry.parentTypeID);
            auto memberId = parentType.member_types[entry.memberIndex];
            auto memberBaseType = comp.get_type(memberId).basetype;

            getTypeStore()->typeName = addString(getBaseTypeNameFromEnum(spvcrossType2E_TYPE(memberBaseType)));
        }
        else
        {
            getTypeStore()->typeName = addString("TODO");
        }

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
                    // TODO: get name of the type ( https://github.com/Devsh-Graphics-Programming/Nabla/pull/677#discussion_r1574860622 )
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
    stageIntroData->m_params = params;
    stageIntroData->m_shaderStage = shaderStage;

    comp.set_entry_point(params.entryPoint, stage);

    // spec constants
        // TODO: hash map instead
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

        stageIntroData->m_specConstants.insert(std::move(specConst));
    }

    spirv_cross::ShaderResources resources = comp.get_shader_resources(comp.get_active_interface_variables()/*TODO: allow choice in Introspection Parameters, comp.get_active_interface_variables()*/);
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
        assert(type.image.sampled == 2);
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
            res->storageImage.format = spvImageFormat2E_FORMAT(type.image.format);
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
        std::sort(descSet.begin(), descSet.end());

    auto getStageIOtype = [&comp](CSPIRVIntrospector::CStageIntrospectionData::SInterface& glslType, uint32_t id, uint32_t base_type_id)
        {
            const auto& type = comp.get_type(base_type_id);
            glslType.baseType = spvcrossType2E_TYPE(type.basetype);
            glslType.elements = type.vecsize;
            glslType.location = comp.get_decoration(id, spv::DecorationLocation);

            return glslType;
        };

    // in/out
    for (const spirv_cross::Resource& r : resources.stage_inputs)
    {
        CSPIRVIntrospector::CStageIntrospectionData::SInputInterface res;
        getStageIOtype(res, r.id, r.base_type_id);
        stageIntroData->m_input.insert(res);
    }

    if (shaderStage == IShader::ESS_FRAGMENT)
        stageIntroData->m_output = FragmentOutputVecT();
    else
        stageIntroData->m_output = OutputVecT();

    for (const spirv_cross::Resource& r : resources.stage_outputs)
    {
        if (shaderStage == IShader::ESS_FRAGMENT)
        {
            CSPIRVIntrospector::CStageIntrospectionData::SFragmentOutputInterface res =
                std::get<FragmentOutputVecT>(stageIntroData->m_output).emplace_back();
            getStageIOtype(res, r.id, r.base_type_id);

            res.colorIndex = comp.get_decoration(r.id, spv::DecorationIndex);
        }
        else
        {
            CSPIRVIntrospector::CStageIntrospectionData::SOutputInterface& res =
                std::get<OutputVecT>(stageIntroData->m_output).emplace_back();

            getStageIOtype(res, r.id, r.base_type_id);
        }
    }

    // push constants
    auto* pPushConstantsMutable = reinterpret_cast<CStageIntrospectionData::SPushConstantInfo<true>*>(&stageIntroData->m_pushConstants);
    
    if (resources.push_constant_buffers.size())
    {
        assert(resources.push_constant_buffers.size()==1);
        const spirv_cross::Resource& r = resources.push_constant_buffers.front();
        pPushConstantsMutable->name = stageIntroData->addString(r.name);
        stageIntroData->shaderMemBlockIntrospection(comp,pPushConstantsMutable,r);
        // believe it or not you can declare an empty PC block
        pPushConstantsMutable->size = comp.get_declared_struct_size(comp.get_type(r.type_id));

        if (pPushConstantsMutable->offset + pPushConstantsMutable->size >= MaxPushConstantsSize)
            return nullptr;

        if (pPushConstantsMutable->size != 0)
            pPushConstantsMutable->offset = comp.type_struct_member_offset(comp.get_type(r.type_id/*TODO: verify if this of base*/), 0);
    }
    else
        pPushConstantsMutable->type = {};

    // convert all Mutable to non-mutable
    stageIntroData->finalize(shaderStage);

    return stageIntroData;
}

void CSPIRVIntrospector::CStageIntrospectionData::finalize(const IShader::E_SHADER_STAGE shaderStage)
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

    // const cast doesn't change hasher or operator==
    for (auto& spec : m_specConstants)
        addBaseAndConvertStringToImmutable(const_cast<std::span<const char>&>(spec.name));

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

    for (auto set=0; set < DESCRIPTOR_SET_COUNT; set++)
    for (auto& descriptor : m_descriptorSetBindings[set])
    {
        addBaseAndConvertStringToImmutable(descriptor.name);
        auto& asMutable = reinterpret_cast<SDescriptorVarInfo<true>&>(descriptor);
        switch (descriptor.type)
        {
            // TODO: all descriptor types
            case IDescriptor::E_TYPE::ET_UNIFORM_BUFFER:
                addBaseAndConvertBlockToImmutable(asMutable.uniformBuffer);
                break;
            case IDescriptor::E_TYPE::ET_STORAGE_BUFFER:
                addBaseAndConvertBlockToImmutable(asMutable.storageBuffer);
                break;
            default:
                break;
        }
    }
}

void CSPIRVIntrospector::CStageIntrospectionData::printExtents(std::ostringstream& out, const SArrayInfo& count)
{
    out << "[";
    if (count.isSpecConstant)
        out << "specID=" << count.specID;
    else if (!count.isRuntimeSized())
        out << count.value;
    out << "]";
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
   // printExtents(out,type->count);
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

void CSPIRVIntrospector::CStageIntrospectionData::debugPrint(system::ILogger* logger) const
{
    auto* const basePtr = m_memPool.data();
    std::ostringstream debug = {};

    if (!m_specConstants.empty())
    {
        debug << "SPEC CONSTATS:\n";

        for (auto& specConstant : m_specConstants)
        {
            // TODO: it gives weird errors, debug
            const std::string_view specConstantName = "SPEC_CONSTANT_NAME"; // std::string_view(specConstant.name.begin(), specConstant.name.end());

            debug << " name: " << specConstantName << ' '
                << "TODO: type "
                << "id: " << specConstant.id << " byte size: " << "TODO: specConstant.defaultValue" << '\n';
        }    }

    if (m_pushConstants.type)
    {
        debug << "PUSH CONSTANT BLOCK:\n";
        printType(debug, m_pushConstants.type);
        debug << "} " << m_pushConstants.name.data() << ";\n";
    }

    debug << "DESCRIPTOR SETS:\n";

    for (auto set = 0; set < DESCRIPTOR_SET_COUNT; set++)
        for (auto& descriptor : m_descriptorSetBindings[set])
        {
            debug << "(set=" << set << ",binding=" << descriptor.binding << ") ";
            switch (descriptor.type)
            {
            case IDescriptor::E_TYPE::ET_UNIFORM_BUFFER:
                printType(debug, descriptor.uniformBuffer.type);
                break;
            case IDescriptor::E_TYPE::ET_STORAGE_BUFFER:
                printType(debug, descriptor.storageBuffer.type);
                break;
            default:
                debug << "TODO_type"; // don't implement if not needed
                break;
            }
            debug << " " << descriptor.name.data();
            //printExtents(debug, descriptor.count);
            debug << ";\n";
        }

    if (!m_input.empty())
    {
        debug << "INPUT VARIABLES:\n";
        for (auto& inputEntity : m_input)
            debug << "type: " << getBaseTypeNameFromEnum(inputEntity.baseType) << " name: " << "TODO" << " location: " << inputEntity.location << " elements: " << inputEntity.elements << '\n';
    }

    // duplicated code which can be replaced with template trickery, but this is temporary code for testing purpose so who cares
    if (m_shaderStage == IShader::ESS_FRAGMENT)
    {
        auto outputVec = std::get<FragmentOutputVecT>(m_output);
        if (!outputVec.empty())
        {
            debug << "OUTPUT VARIABLES:\n";
            for (auto& outputEntity : outputVec)
                debug << "type: " << getBaseTypeNameFromEnum(outputEntity.baseType) << " location : " << outputEntity.location << " elements: " << outputEntity.elements << '\n';
        }
    }
    else
    {
        auto outputVec = std::get<OutputVecT>(m_output);
        if (!outputVec.empty())
        {
            debug << "OUTPUT VARIABLES:\n";
            for (auto& outputEntity : outputVec)
                debug << "type: " << getBaseTypeNameFromEnum(outputEntity.baseType) << " location: " << outputEntity.location << " elements: " << outputEntity.elements << '\n';
        }
    }

    logger->log(debug.str() + '\n');
}

}