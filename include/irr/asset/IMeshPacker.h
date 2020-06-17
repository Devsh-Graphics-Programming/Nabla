#ifndef __IRR_I_MESH_PACKER_H_INCLUDED__
#define __IRR_I_MESH_PACKER_H_INCLUDED__

namespace irr
{
namespace asset
{
struct DrawElementsIndirectCommand_t
{
    uint32_t count;
    uint32_t instanceCount;
    uint32_t firstIndex;
    uint32_t baseVertex;
    uint32_t baseInstance;
};

template <typename MeshBufferType>
class IMeshPacker
{
public:
    struct AllocationParams
    {
        size_t indexBuffSupportedSizeInBytes                    = 2147483648ull; /*  2GB*/
        size_t vertexBuffSupportedSizeInBytes                   = 4294967296ull; /*  4GB*/
        size_t perInstanceVertexBuffSupportedSizeInBytes        = 33554432ull;   /* 32MB*/
        size_t MDIDataSizeInBytes                               = 2147483648ull; /*  2GB*/
        size_t vertexBufferMinAllocSize                         = 256ull;        /* 256B*/
        size_t indexBufferMinAllocSize                          = 512ull;        /* 512B*/
        size_t perInstanceVertexBufferMinAllocSize              = 32ull;         /*  32B*/
    };

    template <typename MeshBufferType>
    struct PackedMeshBuffer
    {
        core::smart_refctd_ptr<MeshBufferType> packedMeshBuffer = nullptr;
    };

    struct PackedMeshBufferData
    {
        const size_t mdiParameterByteOffset; // add to `CCPUMeshPacker::getMultiDrawIndirectBuffer()->getPointer() to get `DrawElementsIndirectCommand_t` address
        const uint32_t mdiParameterCount;
    };

public:

    IMeshPacker(const SVertexInputParams& preDefinedLayout, const AllocationParams& allocParams,uint16_t maxIndexCountPerMDIData = std::numeric_limits<uint16_t>::max())
        :m_maxIndexCountPerMDIData(maxIndexCountPerMDIData), m_drawCount(0)
    {
        m_outVtxInputParams.enabledAttribFlags  = preDefinedLayout.enabledAttribFlags;
        m_outVtxInputParams.enabledBindingFlags = preDefinedLayout.enabledBindingFlags;

        memcpy(m_outVtxInputParams.attributes, preDefinedLayout.attributes, sizeof(m_outVtxInputParams.attributes));

        m_outMDIData = core::make_smart_refctd_ptr<ICPUBuffer>(allocParams.MDIDataSizeInBytes);

        //TODO: init allocators 
        //m_alctrReservedSpace[0] = _IRR_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator::reserved_size(/**/), _IRR_SIMD_ALIGNMENT);
        //m_vtxBuffAlctr = core::GeneralpurposeAddressAllocator(m_alctrReservedSpace[0], 0u, 0u, 0u, allocParams.vertexBuffSupportedSizeInBytes / allocParams.vertexbufferMinAllocSize, allocParams.vertexbufferMinAllocSize);

        //m_alctrReservedSpace[1] = _IRR_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator::reserved_size(/**/), _IRR_SIMD_ALIGNMENT);
        //m_idxBuffAlctr = core::GeneralpurposeAddressAllocator(m_alctrReservedSpace[1], 0u, 0u, 0u, allocParams.indexBuffSupportedSizeInBytes / allocParams.indexBufferMinAllocSize, allocParams.indexBufferMinAllocSize);

        //m_alctrReservedSpace[2] = _IRR_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator::reserved_size(/**/), _IRR_SIMD_ALIGNMENT);
        //m_perInsVtxBuffAlctr = core::GeneralpurposeAddressAllocator(m_alctrReservedSpace[2], 0u, 0u, 0u, allocParams.perInstanceVertexBuffSupportedSizeInBytes / allocParams.perInstanceVertexbufferMinAllocSize, allocParams.perInstanceVertexbufferMinAllocSize);
        


        //init MDI data allocator
        m_alctrReservedSpace[3] = static_cast<uint8_t*>(_IRR_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<size_t>::reserved_size(alignof(uint32_t), allocParams.MDIDataSizeInBytes / sizeof(DrawElementsIndirectCommand_t), 1u), _IRR_SIMD_ALIGNMENT));
        m_MDIDataAlctr = core::GeneralpurposeAddressAllocator<size_t>(m_alctrReservedSpace[3], 0u, 0u, alignof(uint32_t), allocParams.MDIDataSizeInBytes / sizeof(DrawElementsIndirectCommand_t), /*i git*/1u);

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

    virtual std::optional<PackedMeshBufferData> alloc(const MeshBufferType* meshBuffer) = 0;
    virtual PackedMeshBuffer<MeshBufferType> commit(const core::vector<std::pair<const ICPUMeshBuffer*, PackedMeshBufferData>>& meshBuffers) = 0;

    asset::ICPUBuffer* getMultiDrawIndirectBuffer() { return m_outMDIData.get(); }

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
    core::GeneralpurposeAddressAllocator<size_t> m_vtxBuffAlctr;
    core::GeneralpurposeAddressAllocator<size_t> m_idxBuffAlctr;
    core::GeneralpurposeAddressAllocator<size_t> m_perInsVtxBuffAlctr;
    core::GeneralpurposeAddressAllocator<size_t> m_MDIDataAlctr;

    core::smart_refctd_ptr<asset::ICPUBuffer> m_outMDIData;

    size_t m_drawCount;


    const uint16_t m_maxIndexCountPerMDIData;
};

}
}

#endif