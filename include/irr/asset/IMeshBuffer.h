#ifndef __IRR_I_MESH_BUFFER_H_INCLUDED__
#define __IRR_I_MESH_BUFFER_H_INCLUDED__

#include "irr/asset/IRenderpassIndependentPipeline.h"
#include <algorithm>

namespace irr
{
namespace asset
{

//!
enum E_INDEX_TYPE
{
    EIT_16BIT = 0,
    EIT_32BIT,
    EIT_UNKNOWN
};

enum E_MESH_BUFFER_TYPE
{
    EMBT_UNKNOWN = 0,
    EMBT_NOT_ANIMATED,
    EMBT_ANIMATED_FRAME_BASED,
    EMBT_ANIMATED_SKINNED
};

template <class BufferType, class DescSetType, class PipelineType>
class IMeshBuffer : public virtual core::IReferenceCounted
{
public:
    struct SBufferBinding {
        uint64_t offset = 0ull;
        core::smart_refctd_ptr<BufferType> buffer = nullptr;
    };
    _IRR_STATIC_INLINE_CONSTEXPR size_t MAX_PUSH_CONSTANT_BYTESIZE = 128u;

    _IRR_STATIC_INLINE_CONSTEXPR size_t MAX_VERTEX_ATTRIB_COUNT = SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT;
    _IRR_STATIC_INLINE_CONSTEXPR size_t MAX_ATTR_BUF_BINDING_COUNT = SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT;

protected:
	virtual ~IMeshBuffer()
	{
        if (leakDebugger)
            leakDebugger->deregisterObj(this);
	}

    core::aabbox3df boundingBox;

    SBufferBinding m_vertexBufferBindings[MAX_ATTR_BUF_BINDING_COUNT];
    SBufferBinding m_indexBufferBinding;

    //! Descriptor set which goes to set=3
    core::smart_refctd_ptr<DescSetType> m_descriptorSet;
    core::smart_refctd_ptr<PipelineType> m_pipeline;

	//indices
	E_INDEX_TYPE indexType;
	int32_t baseVertex;
    alignas(64) uint8_t m_pushConstantsData[MAX_PUSH_CONSTANT_BYTESIZE]{};//by putting m_pushConstantsData here, alignas(64) takes no extra place
    uint64_t indexCount;
    //
    size_t instanceCount;
    uint32_t baseInstance;

public:
	//! Constructor.
    /**
    Note that ALL parameters are move-assigned to meshbuffer's members!
    */
	IMeshBuffer(core::smart_refctd_ptr<PipelineType>&& _pipeline, core::smart_refctd_ptr<DescSetType>&& _ds,
        SBufferBinding _vtxBindings[MAX_ATTR_BUF_BINDING_COUNT],
        SBufferBinding&& _indexBinding
        ) : m_indexBufferBinding(std::move(_indexBinding)), m_descriptorSet(std::move(_ds)), m_pipeline(std::move(_pipeline)),
            boundingBox(), indexType(EIT_UNKNOWN), baseVertex(0), indexCount(0u),
            instanceCount(1ull), baseInstance(0u)
	{
        std::move(_vtxBindings, _vtxBindings+MAX_ATTR_BUF_BINDING_COUNT, m_vertexBufferBindings);
	}

    inline bool isAttributeEnabled(uint32_t attrId) const
    {
        if (attrId >= MAX_VERTEX_ATTRIB_COUNT)
            return false;
        const auto& vtxInputParams = m_pipeline->getVertexInputParams();
        if (!(vtxInputParams.enabledAttribFlags & (1u<<attrId)))
            return false;
        return true;
    }
    inline bool isVertexAttribBufferBindingEnabled(uint32_t bndId) const
    {
        if (bndId >= MAX_ATTR_BUF_BINDING_COUNT)
            return false;
        const auto& vtxInputParams = m_pipeline->getVertexInputParams();
        if (!(vtxInputParams.enabledBindingFlags & (1u<<bndId)))
            return false;
        return true;
    }
    //! WARNING: does not check whether attribute and binding are enabled!
    inline uint32_t getBindingNumForAttribute(uint32_t attrId) const
    {
        const auto& vtxInputParams = m_pipeline->getVertexInputParams();
        return vtxInputParams.attributes[attrId].binding;
    }
    inline E_FORMAT getAttribFormat(uint32_t attrId) const
    {
        const auto& vtxInputParams = m_pipeline->getVertexInputParams();
        return vtxInputParams.attributes[attrId].format;
    }
    inline uint32_t getAttribStride(uint32_t attrId) const
    {
        const auto& vtxInputParams = m_pipeline->getVertexInputParams();
        const uint32_t bnd = getBindingNumForAttribute(attrId);
        const auto& vtxInputParams = m_pipeline->getVertexInputParams();
        return vtxInputParams.bindings[bnd].stride;
    }
    inline uint32_t getAttribOffset(uint32_t attrId) const
    {
        const auto& vtxInputParams = m_pipeline->getVertexInputParams();
        return vtxInputParams.attributes[attrId].relativeOffset;
    }
    inline const SBufferBinding* getAttribBoundBuffer(uint32_t attrId) const
    {
        const uint32_t bnd = getBindingNumForAttribute(attrId);
        return &m_vertexBufferBindings[bnd];
    }
    inline const SBufferBinding* getVertexBufferBindings() const
    {
        return m_vertexBufferBindings;
    }
    inline const SBufferBinding* getIndexBufferBinding() const
    {
        return &m_indexBufferBinding;
    }
    inline const PipelineType* getPipeline() const
    {
        return m_pipeline.get();
    }
    inline const DescSetType* getAttachedDescriptorSet() const
    {
        return m_descriptorSet.get();
    }

	//! Get type of index data which is stored in this meshbuffer.
	/** \return Index type of this buffer. */
	inline const E_INDEX_TYPE& getIndexType() const {return indexType;}
	inline void setIndexType(const E_INDEX_TYPE& type) {indexType = type;}

	//! Get amount of indices in this meshbuffer.
	/** \return Number of indices in this buffer. */
	inline const uint64_t& getIndexCount() const {return indexCount;}
	//! Sets amount of indices.
	/** @returns Whether set amount exceeds mapped buffer's size. Regardless of result the amount is set. */
	inline bool setIndexCount(const uint64_t &newIndexCount)
	{
		/*
		#ifdef _IRR_DEBUG
			if (size<0x7fffffffffffffffuLL&&ixbuf&&(ixbuf->getSize()>size+offset))
			{
				os::Printer::log("MeshBuffer map vertex buffer overflow!\n",ELL_ERROR);
				return;
			}
		#endif // _IRR_DEBUG
		*/
        indexCount = newIndexCount;
        if (m_indexBuffer)
        {
            switch (indexType)
            {
                case EIT_16BIT:
                    return indexCount*2+m_indexBufferBinding.offset < m_indexBuffer->getSize();
                case EIT_32BIT:
                    return indexCount*4+m_indexBufferBinding.offset < m_indexBuffer->getSize();
                default:
                    return false;
            }
        }

        return true;
	}

	//! Accesses base vertex number.
	/** @returns base vertex number. */
    inline const int32_t& getBaseVertex() const {return baseVertex;}
	//! Sets base vertex.
    inline void setBaseVertex(const int32_t& baseVx)
    {
        baseVertex = baseVx;
    }

	inline const size_t& getInstanceCount() const {return instanceCount;}
	inline void setInstanceCount(const size_t& count)
	{
		instanceCount = count;
	}

	inline const uint32_t& getBaseInstance() const {return baseInstance;}
	inline void setBaseInstance(const uint32_t& base)
	{
		baseInstance = base;
	}


	//! Get the axis aligned bounding box of this meshbuffer.
	/** \return Axis aligned bounding box of this buffer. */
	inline const core::aabbox3df& getBoundingBox() const {return boundingBox;}

	//! Set axis aligned bounding box
	/** \param box User defined axis aligned bounding box to use
	for this buffer. */
	inline void setBoundingBox(const core::aabbox3df& box)
	{
		boundingBox = box;
	}

    uint8_t* getPushConstantsDataPtr() { return m_pushConstantsData; }
    const uint8_t* getPushConstantsDataPtr() const { return m_pushConstantsData; }
};

}}

#endif //__IRR_I_CPU_MESH_BUFFER_H_INCLUDED__