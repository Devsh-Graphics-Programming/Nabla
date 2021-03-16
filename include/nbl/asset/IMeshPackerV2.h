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

template <typename BufferType, typename MeshBufferType, typename MDIStructType = DrawElementsIndirectCommand_t>
class IMeshPackerV2 : public IMeshPacker<MeshBufferType, MDIStructType>
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

protected:
	IMeshPackerV2(const AllocationParams& allocParams, uint16_t minTriangleCountPerMDIData, uint16_t maxTriangleCountPerMDIData)
		:base_t(minTriangleCountPerMDIData, maxTriangleCountPerMDIData),
         m_allocParams(allocParams)
	{
        initializeCommonAllocators(allocParams);
    };

public:
	template <typename MeshBufferIterator>
	bool alloc(ReservedAllocationMeshBuffers* rambOut, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd);

    template <typename MeshBufferIterator>
    uint32_t calcMDIStructCount(const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd);

    inline PackerDataStore getPackerDataStore() { return m_packerDataStore; };

    std::string generateGLSLBufferDefinitions(uint32_t set)
    {
        const std::string setStr = std::to_string(set);
        const std::string UTBTemplate = std::string("layout(set = ") + setStr + std::string(", binding = ) uniform ");
        const std::string SSBOTemplateDec = std::string("layout(set = ") + setStr + std::string(", binding = ) readonly buffer ");
        const std::string SSBOTemplateDef = "{\n    int dataOffsetTable[];\n} ";
        const uint32_t bindingOffset = 26u;

        enum TBOType
        {
            ETT_INT,
            ETT_UINT,
            ETT_FLOAT,
            ETT_UNKNOWN
        };

        std::string result;

        uint32_t binding = 0u;
        for (uint32_t i = 0u; i < virtualAttribTable.size(); i++)
        {
            std::string tmp = UTBTemplate;
            std::string samplerType = "samplerBuffer ";

            //TODO
            TBOType type = ETT_FLOAT;

            switch (type)
            {
            case ETT_INT:
                samplerType = "isamplerBuffer ";
                break;
            case ETT_UINT:
                samplerType = "usamplerBuffer ";
                break;
            case ETT_FLOAT:
                samplerType = "samplerBuffer ";
                break;
            default:
                assert(false);
            }

            tmp.insert(bindingOffset, std::to_string(binding));
            tmp.append(samplerType);
            tmp.append(std::string("MeshPackedData_") + std::to_string(i) + std::string(";\n\n"));

            result += tmp;

            binding++;
        }

        uint32_t activeBindingCnt = 0u;

        for (uint32_t i = 0u; i < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; i++)
        {
            if (enabledAttribFlagsCombined & (1u << i))
                activeBindingCnt++;
        }

        for (uint32_t i = 0u; i < activeBindingCnt; i++)
        {
            std::string tmp = SSBOTemplateDec;

            tmp.insert(bindingOffset, std::to_string(binding));
            tmp.append(std::string("OffsetTable_") + std::to_string(i) + std::string("\n"));
            tmp.append(SSBOTemplateDef);
            tmp.append("offsetTable_" + std::to_string(i) + std::string(";\n\n"));

            result += tmp;

            binding++;
        }

        std::cout << result;

        return result;
    }

protected:
    core::vector<VirtualAttribute> virtualAttribTable;
    uint16_t enabledAttribFlagsCombined = 0u;

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
        uint16_t intArrayElementsCnt   = 0u;
        uint16_t uintArrayElementsCnt  = 0u;

        inline bool insertAttribFormat(E_FORMAT format)
        {
            auto lookupResult = map.find(format);
            if (lookupResult != map.end())
                return true;

            E_UTB_ARRAY_TYPE utbArrayType = getUTBArrayTypeFromFormat(format);

            //TODO:
            //if it falils then state of mesh packer is invalid.. maybe return iterator to the inserted element, store array of these iterators in `alloc`
            //and remove all inserted elements when this function returns false
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

    private:

        inline E_UTB_ARRAY_TYPE getUTBArrayTypeFromFormat(E_FORMAT format)
        {
            //TODO: moar formats!
            switch (format)
            {
            case EF_R32G32B32_SFLOAT:
            case EF_R32G32_SFLOAT:
                return E_UTB_ARRAY_TYPE::EUAT_FLOAT;
            case EF_A2B10G10R10_SNORM_PACK32:
                return E_UTB_ARRAY_TYPE::EUAT_INT;

            default:
                return E_UTB_ARRAY_TYPE::EUAT_UNKNOWN;
            }
        }

    };

    VirtualAttribConfig virtualAttribConfig;

    PackerDataStore m_packerDataStore;
    const AllocationParams m_allocParams;

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
        const size_t idxCnt = (*it)->getIndexCount();
        const size_t maxVtxCnt = IMeshManipulator::upperBoundVertexID(*it); //ahsdfjkasdfgasdklfhasdf TODO: deal with vertex duplication, same for v1

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

            ramb.attribAllocParams[location].offset = m_vtxBuffAlctr.alloc_addr(maxVtxCnt * attribSize, attribSize);

            if (ramb.attribAllocParams[location].offset == INVALID_ADDRESS)
                return false;

            ramb.attribAllocParams[location].size = maxVtxCnt * attribSize;

            virtualAttribConfig.insertAttribFormat(attribFormat);

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
        const size_t idxCnt = (*mbIt)->getIndexCount();
        const uint32_t triCnt = idxCnt / 3;
        assert(idxCnt % 3 == 0);

        acc += calcBatchCount(triCnt);
    }
    
    return acc;
}

}
}

#endif