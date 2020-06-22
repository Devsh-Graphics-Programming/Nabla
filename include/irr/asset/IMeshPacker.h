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

    struct PackedMeshBufferData
    {
        uint32_t mdiParameterOffset; // add to `CCPUMeshPacker::getMultiDrawIndirectBuffer()->getPointer() to get `DrawElementsIndirectCommand_t` address
        uint32_t mdiParameterCount;

        inline bool isValid() 
        {
            return this->mdiParameterOffset != core::GeneralpurposeAddressAllocator<uint32_t>::invalid_address &&
                   this->mdiParameterCount != core::GeneralpurposeAddressAllocator<uint32_t>::invalid_address;
        }
    };

    
}

struct DrawElementsIndirectCommand_t
{
    uint32_t count;
    uint32_t instanceCount;
    uint32_t firstIndex;
    uint32_t baseVertex;
    uint32_t baseInstance;
};

using namespace meshPackerUtil;

template <typename MeshBufferType, typename MDIStructType>
class IMeshPacker
{
    static_assert(std::is_base_of<DrawElementsIndirectCommand_t, MDIStructType>::value);

public:

    IMeshPacker(const SVertexInputParams& preDefinedLayout, const AllocationParams& allocParams, uint16_t maxTriangleCountPerMDIData = 1024u)
        :m_maxTriangleCountPerMDIData(maxTriangleCountPerMDIData), /*TODO: delete when all alocators are done: */m_alctrReservedSpace{nullptr, nullptr, nullptr, nullptr}
    {
        m_outVtxInputParams.enabledAttribFlags  = preDefinedLayout.enabledAttribFlags;
        m_outVtxInputParams.enabledBindingFlags = preDefinedLayout.enabledBindingFlags;

        memcpy(m_outVtxInputParams.attributes, preDefinedLayout.attributes, sizeof(m_outVtxInputParams.attributes));

        //TODO: init allocators 
        //m_alctrReservedSpace[0] = _IRR_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator::reserved_size(/**/), _IRR_SIMD_ALIGNMENT);
        //m_vtxBuffAlctr = core::GeneralpurposeAddressAllocator(m_alctrReservedSpace[0], 0u, 0u, 0u, allocParams.vertexBuffSupportedSizeInBytes / allocParams.vertexbufferMinAllocSize, allocParams.vertexbufferMinAllocSize);
        
        m_alctrReservedSpace[1] = static_cast<uint8_t*>(_IRR_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(uint16_t), allocParams.indexBuffSupportedSizeInBytes / sizeof(uint16_t), allocParams.indexBufferMinAllocSize), _IRR_SIMD_ALIGNMENT));
        m_idxBuffAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(m_alctrReservedSpace[1], 0u, 0u, alignof(uint16_t), allocParams.indexBuffSupportedSizeInBytes / sizeof(uint16_t), allocParams.indexBufferMinAllocSize);

        //m_alctrReservedSpace[2] = _IRR_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator::reserved_size(/**/), _IRR_SIMD_ALIGNMENT);
        //m_perInsVtxBuffAlctr = core::GeneralpurposeAddressAllocator(m_alctrReservedSpace[2], 0u, 0u, 0u, allocParams.perInstanceVertexBuffSupportedSizeInBytes / allocParams.perInstanceVertexbufferMinAllocSize, allocParams.perInstanceVertexbufferMinAllocSize);

        //init MDI data allocator
        m_alctrReservedSpace[3] = static_cast<uint8_t*>(_IRR_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(uint32_t), allocParams.MDIDataBuffSupportedSizeInBytes / sizeof(MDIStructType), allocParams.MDIDataBuffMinAllocSize), _IRR_SIMD_ALIGNMENT));
        m_MDIDataAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(m_alctrReservedSpace[3], 0u, 0u, alignof(uint32_t), allocParams.MDIDataBuffSupportedSizeInBytes / sizeof(MDIStructType), allocParams.MDIDataBuffMinAllocSize);

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

    virtual PackedMeshBufferData alloc(const MeshBufferType const* meshBuffer) = 0;
    virtual void commit(/*TODO*/) = 0;

protected:
    virtual ~IMeshPacker() 
    {
        for (int i = 0; i < 4; i++)
            _IRR_ALIGNED_FREE(m_alctrReservedSpace[i]);
    }

protected:
    //output mesh buffers data
    SVertexInputParams m_outVtxInputParams;

    uint8_t* m_alctrReservedSpace[4];
    core::GeneralpurposeAddressAllocator<uint32_t> m_vtxBuffAlctr;
    core::GeneralpurposeAddressAllocator<uint32_t> m_idxBuffAlctr;
    core::GeneralpurposeAddressAllocator<uint32_t> m_perInsVtxBuffAlctr;
    core::GeneralpurposeAddressAllocator<uint32_t> m_MDIDataAlctr;

    const uint16_t m_maxTriangleCountPerMDIData;
    
    _IRR_STATIC_INLINE_CONSTEXPR PackedMeshBufferData invalidPackedMeshBufferData{ ~uint32_t(0), ~uint32_t(0) };
};

}
}

#endif