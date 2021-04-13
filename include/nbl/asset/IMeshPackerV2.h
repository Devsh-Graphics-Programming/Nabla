// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_MESH_PACKER_V2_H_INCLUDED__
#define __NBL_ASSET_I_MESH_PACKER_V2_H_INCLUDED__

#include <nbl/asset/utils/IMeshPacker.h>

namespace nbl
{
namespace asset
{

class IMeshPackerV2Base
{
protected:
    enum class E_UTB_ARRAY_TYPE
    {
        EUAT_FLOAT,
        EUAT_INT,
        EUAT_UINT,
        EUAT_UNKNOWN
    };

    struct VirtualAttribConfig
    {
        core::unordered_map<E_FORMAT, std::pair<E_UTB_ARRAY_TYPE, uint32_t>> map;
        uint16_t floatArrayElementsCnt = 0u;
        uint16_t intArrayElementsCnt = 0u;
        uint16_t uintArrayElementsCnt = 0u;

        VirtualAttribConfig& operator=(const VirtualAttribConfig& other)
        {
            map = other.map;
            floatArrayElementsCnt = other.floatArrayElementsCnt;
            intArrayElementsCnt = other.intArrayElementsCnt;
            uintArrayElementsCnt = other.uintArrayElementsCnt;

            return *this;
        }

        VirtualAttribConfig& operator=(VirtualAttribConfig&& other)
        {
            map = std::move(other.map);
            floatArrayElementsCnt = other.floatArrayElementsCnt;
            intArrayElementsCnt = other.intArrayElementsCnt;
            uintArrayElementsCnt = other.uintArrayElementsCnt;

            other.floatArrayElementsCnt = 0u;
            other.intArrayElementsCnt = 0u;
            other.uintArrayElementsCnt = 0u;

            return *this;
        }

        inline bool insertAttribFormat(E_FORMAT format)
        {
            auto lookupResult = map.find(format);
            if (lookupResult != map.end())
                return true;

            E_UTB_ARRAY_TYPE utbArrayType = getUTBArrayTypeFromFormat(format);

            uint16_t arrayElement = 0u;
            switch (utbArrayType)
            {
            case E_UTB_ARRAY_TYPE::EUAT_FLOAT:
                arrayElement = floatArrayElementsCnt;
                floatArrayElementsCnt++;
                break;

            case E_UTB_ARRAY_TYPE::EUAT_INT:
                arrayElement = intArrayElementsCnt;
                intArrayElementsCnt++;
                break;

            case E_UTB_ARRAY_TYPE::EUAT_UINT:
                arrayElement = uintArrayElementsCnt;
                uintArrayElementsCnt++;
                break;

            case E_UTB_ARRAY_TYPE::EUAT_UNKNOWN:
                assert(false);
                return false;
            }

            map.insert(std::make_pair(format, std::make_pair(utbArrayType, arrayElement)));

            return true;
        }

        inline E_UTB_ARRAY_TYPE getUTBArrayTypeFromFormat(E_FORMAT format)
        {
            switch (format)
            {
             //float formats
            case EF_R8_UNORM:
            case EF_R8_SNORM:
            case EF_R8_USCALED:
            case EF_R8_SSCALED:
            case EF_R8G8_UNORM:
            case EF_R8G8_SNORM:
            case EF_R8G8_USCALED:
            case EF_R8G8_SSCALED:
            case EF_R8G8B8_UNORM:
            case EF_R8G8B8_SNORM:
            case EF_R8G8B8_USCALED:
            case EF_R8G8B8_SSCALED:
            case EF_R8G8B8A8_UNORM:
            case EF_R8G8B8A8_SNORM:
            case EF_R8G8B8A8_USCALED:
            case EF_R8G8B8A8_SSCALED:
            case EF_R16_UNORM:
            case EF_R16_SNORM:
            case EF_R16_USCALED:
            case EF_R16_SSCALED:
            case EF_R16_SFLOAT:
            case EF_R16G16_UNORM:
            case EF_R16G16_SNORM:
            case EF_R16G16_USCALED:
            case EF_R16G16_SSCALED:
            case EF_R16G16_SFLOAT:
            case EF_R16G16B16_UNORM:
            case EF_R16G16B16_SNORM:
            case EF_R16G16B16_USCALED:
            case EF_R16G16B16_SSCALED:
            case EF_R16G16B16_SFLOAT:
            case EF_R16G16B16A16_UNORM:
            case EF_R16G16B16A16_SNORM:
            case EF_R16G16B16A16_USCALED:
            case EF_R16G16B16A16_SSCALED:
            case EF_R16G16B16A16_SFLOAT:
            case EF_R32_SFLOAT:
            case EF_R32G32_SFLOAT:
            case EF_R32G32B32_SFLOAT:
            case EF_R32G32B32A32_SFLOAT:
            case EF_B10G11R11_UFLOAT_PACK32:
            case EF_A2B10G10R10_UNORM_PACK32:
            case EF_A8B8G8R8_UNORM_PACK32:
            case EF_A8B8G8R8_SNORM_PACK32:
            case EF_A8B8G8R8_USCALED_PACK32:
            case EF_A8B8G8R8_SSCALED_PACK32:
                return E_UTB_ARRAY_TYPE::EUAT_FLOAT;

             //int formats
            case EF_R8_SINT:
            case EF_R8G8_SINT:
            case EF_R8G8B8_SINT:
            case EF_R8G8B8A8_SINT:
            case EF_R16_SINT:
            case EF_R16G16_SINT:
            case EF_R16G16B16_SINT:
            case EF_R16G16B16A16_SINT:
            case EF_R32_SINT:
            case EF_R32G32_SINT:
            case EF_R32G32B32_SINT:
            case EF_R32G32B32A32_SINT:
            case EF_A8B8G8R8_SINT_PACK32:
                return E_UTB_ARRAY_TYPE::EUAT_INT;

             //uint formats
            case EF_R8_UINT:
            case EF_R8G8_UINT:
            case EF_R8G8B8_UINT:
            case EF_R8G8B8A8_UINT:
            case EF_R16_UINT:
            case EF_R16G16_UINT:
            case EF_R16G16B16_UINT:
            case EF_R16G16B16A16_UINT:
            case EF_R32_UINT:
            case EF_R32G32_UINT:
            case EF_R32G32B32_UINT:
            case EF_R32G32B32A32_UINT:
            case EF_A2B10G10R10_SNORM_PACK32:
            case EF_A2B10G10R10_UINT_PACK32:
            case EF_A8B8G8R8_UINT_PACK32:
                return E_UTB_ARRAY_TYPE::EUAT_UINT;

            default:
                return E_UTB_ARRAY_TYPE::EUAT_UNKNOWN;
            }
        }
    };

    VirtualAttribConfig m_virtualAttribConfig;
};

template <typename BufferType, typename MeshBufferType, typename MDIStructType = DrawElementsIndirectCommand_t>
class IMeshPackerV2 : public IMeshPacker<MeshBufferType, MDIStructType>, public IMeshPackerV2Base
{
    static_assert(std::is_base_of<IBuffer, BufferType>::value);

	using base_t = IMeshPacker<MeshBufferType, MDIStructType>;
    using AllocationParams = IMeshPackerBase::AllocationParamsCommon;

public:
    struct AttribAllocParams
    {
        size_t offset = INVALID_ADDRESS;
        size_t size = 0ull;
    };

    struct ReservedAllocationMeshBuffers
    {
        uint32_t mdiAllocationOffset;
        uint32_t mdiAllocationReservedSize;
        uint32_t indexAllocationOffset;
        uint32_t indexAllocationReservedSize;
        AttribAllocParams attribAllocParams[SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT];

        inline bool isValid()
        {
            return this->mdiAllocationOffset != core::GeneralpurposeAddressAllocator<uint32_t>::invalid_address;
        }
    };

    struct VirtualAttribute
    {
        uint32_t arrayElement : 4;
        uint32_t offset : 28;
    };

    struct CombinedDataOffsetTable
    {
        VirtualAttribute attribInfo[SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT];
    };

    struct PackerDataStore : base_t::template PackerDataStoreCommon<BufferType>
    {
        core::smart_refctd_ptr<BufferType> vertexBuffer;
        core::smart_refctd_ptr<BufferType> indexBuffer;
    };

    inline uint32_t getFloatBufferBindingsCnt() { return m_virtualAttribConfig.floatArrayElementsCnt;  } //TODO: better names?
    inline uint32_t getIntBufferBindingsCnt() { return m_virtualAttribConfig.intArrayElementsCnt; }
    inline uint32_t getUintBufferBindingsCnt() { return m_virtualAttribConfig.uintArrayElementsCnt; }

protected:
	IMeshPackerV2(const AllocationParams& allocParams, uint16_t minTriangleCountPerMDIData, uint16_t maxTriangleCountPerMDIData)
		:base_t(minTriangleCountPerMDIData, maxTriangleCountPerMDIData),
         m_allocParams(allocParams)
	{
        initializeCommonAllocators(allocParams);
    };

    template <typename DSlayout>
    uint32_t getDSlayoutBindings_internal(typename DSlayout::SBinding* outBindings, uint32_t fsamplersBinding = 0u, uint32_t isamplersBinding = 1u, uint32_t usamplersBinding = 2u) const
    {
        const uint32_t bindingCount = 
            (m_virtualAttribConfig.floatArrayElementsCnt ? 1u : 0u) + 
            (m_virtualAttribConfig.intArrayElementsCnt ? 1u : 0u) +
            (m_virtualAttribConfig.uintArrayElementsCnt ? 1u : 0u);

        if (!outBindings)
            return bindingCount;

        auto* bindings = outBindings;

        auto fillBinding = [](auto& bnd, uint32_t binding, uint32_t count) {
            bnd.binding = binding;
            bnd.count = count;
            bnd.stageFlags = asset::ISpecializedShader::ESS_ALL;
            bnd.type = asset::EDT_UNIFORM_TEXEL_BUFFER;
            bnd.samplers = nullptr;
        };

        uint32_t i = 0u;
        if (m_virtualAttribConfig.floatArrayElementsCnt)
        {
            fillBinding(bindings[i], fsamplersBinding, m_virtualAttribConfig.floatArrayElementsCnt);
            ++i;
        }
        if (m_virtualAttribConfig.intArrayElementsCnt)
        {
            fillBinding(bindings[i], isamplersBinding, m_virtualAttribConfig.intArrayElementsCnt);
            ++i;
        }
        if (m_virtualAttribConfig.uintArrayElementsCnt)
        {
            fillBinding(bindings[i], usamplersBinding, m_virtualAttribConfig.uintArrayElementsCnt);
        }

        return bindingCount;
    }

    template <typename DS, typename BufferView>
    std::pair<uint32_t, uint32_t> getDescriptorSetWrites_internal(typename DS::SWriteDescriptorSet* outWrites, typename DS::SDescriptorInfo* outInfo, DS* dstSet, std::function<core::smart_refctd_ptr<BufferView>(E_FORMAT)> createBufferView, uint32_t fBuffersBinding = 0u, uint32_t iBuffersBinding = 1u, uint32_t uBuffersBinding = 2u) const
    {
        const uint32_t writeCount = 
            (m_virtualAttribConfig.floatArrayElementsCnt ? 1u : 0u) + 
            (m_virtualAttribConfig.intArrayElementsCnt ? 1u : 0u) +
            (m_virtualAttribConfig.uintArrayElementsCnt ? 1u : 0u);
        const uint32_t infoCount = 
            m_virtualAttribConfig.floatArrayElementsCnt +
            m_virtualAttribConfig.intArrayElementsCnt +
            m_virtualAttribConfig.uintArrayElementsCnt;

        if (!outWrites || !outInfo)
            return std::make_pair(writeCount, infoCount);

        auto* writes = outWrites;

        auto* floatInfoPtr = outInfo;
        auto* intInfoPtr = floatInfoPtr + m_virtualAttribConfig.floatArrayElementsCnt;
        auto* uintInfoPtr = intInfoPtr + m_virtualAttribConfig.intArrayElementsCnt;

        uint32_t i = 0u;
        if (m_virtualAttribConfig.floatArrayElementsCnt)
        {
            writes[i].binding = fBuffersBinding;
            writes[i].arrayElement = 0u;
            writes[i].count = m_virtualAttribConfig.floatArrayElementsCnt;
            writes[i].descriptorType = EDT_UNIFORM_TEXEL_BUFFER;
            writes[i].dstSet = dstSet;
            writes[i].info = floatInfoPtr;

            i++;
        }
        if (m_virtualAttribConfig.intArrayElementsCnt)
        {
            writes[i].binding = iBuffersBinding;
            writes[i].arrayElement = 0u;
            writes[i].count = m_virtualAttribConfig.intArrayElementsCnt;
            writes[i].descriptorType = EDT_UNIFORM_TEXEL_BUFFER;
            writes[i].dstSet = dstSet;
            writes[i].info = intInfoPtr;

            i++;
        }
        if (m_virtualAttribConfig.uintArrayElementsCnt)
        {
            writes[i].binding = uBuffersBinding;
            writes[i].arrayElement = 0u;
            writes[i].count = m_virtualAttribConfig.uintArrayElementsCnt;
            writes[i].descriptorType = EDT_UNIFORM_TEXEL_BUFFER;
            writes[i].dstSet = dstSet;
            writes[i].info = uintInfoPtr;
        }

        auto fillInfoStruct = [&](auto* ptr, E_FORMAT format)
        {
            ptr->desc = createBufferView(format);
            ptr->buffer.offset = 0u;
            ptr->buffer.size = m_packerDataStore.vertexBuffer->getSize();
        };

        for (auto virtualAttribData : m_virtualAttribConfig.map)
        {
            const E_UTB_ARRAY_TYPE utbArrayType = virtualAttribData.second.first;
            E_FORMAT format = virtualAttribData.first;
            const uint32_t arrayElement = virtualAttribData.second.second;

            switch (utbArrayType)
            {
            case E_UTB_ARRAY_TYPE::EUAT_FLOAT:
                fillInfoStruct(floatInfoPtr + arrayElement, format);
                break;

            case E_UTB_ARRAY_TYPE::EUAT_INT:
                fillInfoStruct(intInfoPtr + arrayElement, format);
                break;

            case E_UTB_ARRAY_TYPE::EUAT_UINT:
                if (format == EF_A2B10G10R10_SNORM_PACK32)
                    format = EF_R32_UINT;
                fillInfoStruct(uintInfoPtr + arrayElement, format);
                break;

            case E_UTB_ARRAY_TYPE::EUAT_UNKNOWN:
                assert(false);
                return std::make_pair(0u, 0u);
            }
        }

        return std::make_pair(writeCount, infoCount);
    }

public:
	template <typename MeshBufferIterator>
	bool alloc(ReservedAllocationMeshBuffers* rambOut, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd);

    //TODO: test (free only part of the scene)
    void free(const ReservedAllocationMeshBuffers* rambIn, uint32_t meshBuffersToFreeCnt)
    {
        for (uint32_t i = 0u; i < meshBuffersToFreeCnt; i++)
        {
            const ReservedAllocationMeshBuffers* const ramb = rambIn + i;

            if (ramb->indexAllocationOffset != INVALID_ADDRESS)
                m_idxBuffAlctr.free_addr(ramb->indexAllocationOffset, ramb->indexAllocationReservedSize);
            
            if (ramb->mdiAllocationOffset != INVALID_ADDRESS)
                m_MDIDataAlctr.free_addr(ramb->mdiAllocationOffset, ramb->mdiAllocationReservedSize);
            
            for (uint32_t j = 0; j < SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT; j++)
            {
                const AttribAllocParams& attrAllocParams = ramb->attribAllocParams[j];
                if (attrAllocParams.offset != INVALID_ADDRESS)
                    m_vtxBuffAlctr.free_addr(attrAllocParams.offset, attrAllocParams.size);
            }
        }
    }

    template <typename MeshBufferIterator>
    uint32_t calcMDIStructCount(const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd);

    inline PackerDataStore getPackerDataStore() { return m_packerDataStore; };

protected:
    core::vector<VirtualAttribute> virtualAttribTable;
    uint16_t enabledAttribFlagsCombined = 0u;

    PackerDataStore m_packerDataStore;
    AllocationParams m_allocParams;

};

//TODO: check if offset < 2^28-1

template <typename MeshBufferType, typename BufferType, typename MDIStructType>
template <typename MeshBufferIterator>
bool IMeshPackerV2<MeshBufferType, BufferType, MDIStructType>::alloc(ReservedAllocationMeshBuffers* rambOut, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd)
{
    size_t i = 0ull;
    for (auto it = mbBegin; it != mbEnd; it++)
    {
        ReservedAllocationMeshBuffers& ramb = *(rambOut + i);
        const size_t idxCnt = calcIdxCntAfterConversionToTriangleList(*it);
        const size_t maxVtxCnt = calcVertexCountBoundWithBatchDuplication(*it);
        const uint32_t insCnt = (*it)->getInstanceCount();

        //TODO: in this mesh packer there is only one buffer for both per instance and per vertex attribs
        //modify alloc and commit so these functions act accrodingly to attribute they are wokring on

        //allocate indices
        ramb.indexAllocationOffset = m_idxBuffAlctr.alloc_addr(idxCnt, 1u);
        if (ramb.indexAllocationOffset == INVALID_ADDRESS)
            return false;
        ramb.indexAllocationReservedSize = idxCnt * 2;

        //allocate vertices
        const auto& mbVtxInputParams = (*it)->getPipeline()->getVertexInputParams();
        for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
        {
            if (!(attrBit & mbVtxInputParams.enabledAttribFlags))
                continue;

            const E_FORMAT attribFormat = static_cast<E_FORMAT>(mbVtxInputParams.attributes[location].format);
            const uint32_t attribSize = asset::getTexelOrBlockBytesize(attribFormat);
            const uint32_t binding = mbVtxInputParams.attributes[location].binding;
            const E_VERTEX_INPUT_RATE inputRate = mbVtxInputParams.bindings[binding].inputRate;

            if (inputRate == EVIR_PER_VERTEX)
            {
                const uint32_t allocByteSize = maxVtxCnt * attribSize;
                ramb.attribAllocParams[location].offset = m_vtxBuffAlctr.alloc_addr(allocByteSize, attribSize);
                ramb.attribAllocParams[location].size = allocByteSize;
            }
            else if (inputRate == EVIR_PER_INSTANCE)
            {
                const uint32_t allocByteSize = insCnt * attribSize;
                ramb.attribAllocParams[location].offset = m_vtxBuffAlctr.alloc_addr(allocByteSize, attribSize);
                ramb.attribAllocParams[location].size = allocByteSize;
            }

            if (ramb.attribAllocParams[location].offset == INVALID_ADDRESS)
                return false;

            m_virtualAttribConfig.insertAttribFormat(attribFormat);

            //TODO: reset state when allocation fails
        }

        //allocate MDI structs
        const uint32_t minIdxCntPerPatch = m_minTriangleCountPerMDIData * 3;
        size_t possibleMDIStructsNeededCnt = (idxCnt + minIdxCntPerPatch - 1) / minIdxCntPerPatch;

        ramb.mdiAllocationOffset = m_MDIDataAlctr.alloc_addr(possibleMDIStructsNeededCnt, 1u);
        if (ramb.mdiAllocationOffset == INVALID_ADDRESS)
            return false;
        ramb.mdiAllocationReservedSize = possibleMDIStructsNeededCnt;

        i++;
    }

    return true;
}

template <typename MeshBufferType, typename BufferType, typename MDIStructType>
template <typename MeshBufferIterator>
uint32_t IMeshPackerV2<MeshBufferType, BufferType, MDIStructType>::calcMDIStructCount(const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd)
{
    uint32_t acc = 0u;
    for (auto mbIt = mbBegin; mbIt != mbEnd; mbIt++)
    {
        auto mb = *mbIt;
        assert(mb->getPipeline()->getPrimitiveAssemblyParams().primitiveType==EPT_TRIANGLE_LIST);
        const size_t idxCnt = mb->getIndexCount();
        const uint32_t triCnt = idxCnt / 3;
        assert(idxCnt % 3 == 0);

        acc += calcBatchCountBound(triCnt);
    }
    
    return acc;
}

}
}

#endif