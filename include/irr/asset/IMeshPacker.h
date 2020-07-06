#ifndef __IRR_I_MESH_PACKER_H_INCLUDED__
#define __IRR_I_MESH_PACKER_H_INCLUDED__

namespace irr
{
namespace asset
{

namespace meshPackerUtil
{
    struct AllocationParams
    {
        size_t indexBuffSupportedSizeInBytes                 = 2147483648ull;      /*  2GB*/
        size_t vertexBuffSupportedSizeInBytes                = 2147483648ull;      /*  2GB*/
        size_t perInstanceVertexBuffSupportedSizeInBytes     = 33554432ull;        /* 32MB*/
        size_t MDIDataBuffSupportedSizeInBytes               = 2147483648ull;      /*  2GB*/
        size_t vertexBufferMinAllocSize                      = 512ull;             /* 512B*/
        size_t indexBufferMinAllocSize                       = 256ull;             /* 256B*/
        size_t perInstanceVertexBufferMinAllocSize           = 32ull;              /*  32B*/
        size_t MDIDataBuffMinAllocSize                       = 32ull;              /*  32B*/
    };

    template <typename MeshBufferType>
    struct PackedMeshBuffer
    {
        core::smart_refctd_ptr<MeshBufferType> packedMeshBuffer = nullptr;
    };

    struct ReservedAllocationMeshBuffers // old PackedMeshBufferData 
    {
        uint32_t mdiAllocationOffset;
        uint32_t mdiAllocationReservedSize;
        uint32_t instanceAllocationOffset;
        uint32_t instanceAllocationReservedSize;
        uint32_t indexAllocationOffset;
        uint32_t indexAllocationReservedSize;
        uint32_t vertexAllocationOffset;
        uint32_t vertexAllocationReservedSize;

        //or private `bool isValidFlag` ?
        inline bool isValid()
        {
            return this->mdiAllocationOffset != core::GeneralpurposeAddressAllocator<uint32_t>::invalid_address;
        }
    };

    struct PackedMeshBufferData
    {
        uint32_t mdiParameterOffset; // add to `CCPUMeshPacker::getMultiDrawIndirectBuffer()->getPointer() to get `DrawElementsIndirectCommand_t` address
        uint32_t mdiParameterCount;
    };

    
}

using namespace meshPackerUtil;

template <typename MeshBufferType, typename MDIStructType>
class IMeshPacker
{
    static_assert(std::is_base_of<DrawElementsIndirectCommand_t, MDIStructType>::value);

public:

    IMeshPacker(const SVertexInputParams& preDefinedLayout, const AllocationParams& allocParams, uint16_t minTriangleCountPerMDIData, uint16_t maxTriangleCountPerMDIData)
        :m_maxTriangleCountPerMDIData(maxTriangleCountPerMDIData),
         m_minTriangleCountPerMDIData(minTriangleCountPerMDIData),
        m_MDIDataAlctrResSpc(nullptr), m_idxBuffAlctrResSpc(nullptr),
        m_vtxBuffAlctrResSpc(nullptr), m_perInsVtxBuffAlctrResSpc(nullptr)
    {
        m_outVtxInputParams.enabledAttribFlags  = preDefinedLayout.enabledAttribFlags;
        m_outVtxInputParams.enabledBindingFlags = preDefinedLayout.enabledBindingFlags;

        memcpy(m_outVtxInputParams.attributes, preDefinedLayout.attributes, sizeof(m_outVtxInputParams.attributes));


        m_vtxSize = calcVertexSize(preDefinedLayout, E_VERTEX_INPUT_RATE::EVIR_PER_VERTEX);
        if (m_vtxSize)
        {
            m_vtxBuffAlctrResSpc = _IRR_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(std::max_align_t), allocParams.vertexBuffSupportedSizeInBytes / m_vtxSize, allocParams.vertexBufferMinAllocSize /*divided by vtxSize?*/), _IRR_SIMD_ALIGNMENT);
            //TODO: check if m_alctrReservedSpace[n] != nullptr
            m_vtxBuffAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(m_vtxBuffAlctrResSpc, 0u, 0u, alignof(std::max_align_t), allocParams.vertexBuffSupportedSizeInBytes / m_vtxSize, allocParams.vertexBufferMinAllocSize);
        }
        
        m_perInstVtxSize = calcVertexSize(preDefinedLayout, E_VERTEX_INPUT_RATE::EVIR_PER_INSTANCE);
        if (m_perInstVtxSize)
        {
            m_perInsVtxBuffAlctrResSpc = _IRR_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(std::max_align_t), allocParams.perInstanceVertexBuffSupportedSizeInBytes / m_perInstVtxSize, allocParams.perInstanceVertexBufferMinAllocSize), _IRR_SIMD_ALIGNMENT);
            m_perInsVtxBuffAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(m_perInsVtxBuffAlctrResSpc, 0u, 0u, alignof(std::max_align_t), allocParams.perInstanceVertexBuffSupportedSizeInBytes / m_perInstVtxSize, allocParams.perInstanceVertexBufferMinAllocSize);
        }

        m_idxBuffAlctrResSpc = _IRR_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(uint16_t), allocParams.indexBuffSupportedSizeInBytes / sizeof(uint16_t), allocParams.indexBufferMinAllocSize), _IRR_SIMD_ALIGNMENT);
        m_idxBuffAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(m_idxBuffAlctrResSpc, 0u, 0u, alignof(uint16_t), allocParams.indexBuffSupportedSizeInBytes / sizeof(uint16_t), allocParams.indexBufferMinAllocSize);


        m_MDIDataAlctrResSpc = _IRR_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(uint32_t), allocParams.MDIDataBuffSupportedSizeInBytes / sizeof(MDIStructType), allocParams.MDIDataBuffMinAllocSize), _IRR_SIMD_ALIGNMENT);
        m_MDIDataAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(m_MDIDataAlctrResSpc, 0u, 0u, alignof(uint32_t), allocParams.MDIDataBuffSupportedSizeInBytes / sizeof(MDIStructType), allocParams.MDIDataBuffMinAllocSize);

        //1 attrib enabled == 1 binding
        for (uint16_t attrBit = 0x0001, location = 0; location < 16; attrBit <<= 1, location++)
        {
            if (m_outVtxInputParams.enabledAttribFlags & attrBit)
            {
                m_outVtxInputParams.bindings[location].stride = getTexelOrBlockBytesize(static_cast<E_FORMAT>(m_outVtxInputParams.attributes[location].format));
                m_outVtxInputParams.bindings[location].inputRate = preDefinedLayout.bindings[preDefinedLayout.attributes[location].binding].inputRate;
            }
        }
    }


    virtual void commit(/*TODO*/) = 0;

protected:
    virtual ~IMeshPacker() 
    {
        _IRR_ALIGNED_FREE(m_MDIDataAlctrResSpc);
        _IRR_ALIGNED_FREE(m_idxBuffAlctrResSpc);
        _IRR_ALIGNED_FREE(m_vtxBuffAlctrResSpc);
        _IRR_ALIGNED_FREE(m_perInsVtxBuffAlctrResSpc);
    }

    inline size_t calcVertexSize(const SVertexInputParams& vtxInputParams, const E_VERTEX_INPUT_RATE inputRate) const
    {
        size_t size = 0ull;
        for (size_t i = 0; i < SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT; ++i)
        {
            if (vtxInputParams.enabledAttribFlags & (1u << i))
                if(vtxInputParams.bindings[i].inputRate == inputRate)
                    size += asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(vtxInputParams.attributes[i].format));
        }

        return size;
    }

protected:
    //output mesh buffers data
    SVertexInputParams m_outVtxInputParams;

    void* m_MDIDataAlctrResSpc;
    void* m_idxBuffAlctrResSpc;
    void* m_vtxBuffAlctrResSpc;
    void* m_perInsVtxBuffAlctrResSpc;
    core::GeneralpurposeAddressAllocator<uint32_t> m_vtxBuffAlctr;
    core::GeneralpurposeAddressAllocator<uint32_t> m_idxBuffAlctr;
    core::GeneralpurposeAddressAllocator<uint32_t> m_perInsVtxBuffAlctr;
    core::GeneralpurposeAddressAllocator<uint32_t> m_MDIDataAlctr;

    const uint16_t m_minTriangleCountPerMDIData;
    const uint16_t m_maxTriangleCountPerMDIData;

    size_t m_vtxSize;
    size_t m_perInstVtxSize;
    
    //TODO?:
    //bool wasCommitCalled;

    _IRR_STATIC_INLINE_CONSTEXPR uint32_t INVALID_ADDRESS = core::GeneralpurposeAddressAllocator<uint32_t>::invalid_address;
    _IRR_STATIC_INLINE_CONSTEXPR ReservedAllocationMeshBuffers invalidReservedAllocationMeshBuffers{ INVALID_ADDRESS, 0, 0, 0, 0, 0, 0, 0 };
};

}
}

#endif