#ifndef __IRR_I_MESH_BUFFER_H_INCLUDED__
#define __IRR_I_MESH_BUFFER_H_INCLUDED__

#include "SMaterial.h"
#include "irr/asset/IRenderpassIndependentPipeline.h"

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
protected:
    _IRR_STATIC_INLINE_CONSTEXPR size_t MAX_VERTEX_ATTRIB_COUNT = SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT;

	virtual ~IMeshBuffer()
	{
        if (leakDebugger)
            leakDebugger->deregisterObj(this);
	}

    core::aabbox3df boundingBox;

    core::smart_refctd_ptr<BufferType> m_vertexBuffers[16];
    core::smart_refctd_ptr<BufferType> m_indexBuffer;

    //! Descriptor set which goes to set=3
    core::smart_refctd_ptr<DescSetType> m_descriptorSet;
    core::smart_refctd_ptr<PipelineType> m_pipeline;

	//indices
	E_INDEX_TYPE indexType;
	int32_t baseVertex;
    uint64_t indexCount;
    size_t indexBufOffset;
    //
    size_t instanceCount;
    uint32_t baseInstance;

    //debug
    core::CLeakDebugger* leakDebugger;
public:
	//! Constructor.
	/**
	@param layout Smart pointer to descriptor of mesh data object.
	@param dbgr Pointer to leak debugger object.
	*/
	IMeshBuffer(core::CLeakDebugger* dbgr=nullptr) :
                        boundingBox(), indexType(EIT_UNKNOWN), baseVertex(0), indexCount(0u), indexBufOffset(0ull),
						instanceCount(1ull), baseInstance(0u), leakDebugger(dbgr)
	{
		if (leakDebugger)
			leakDebugger->registerObj(this);
	}

    inline bool isAttributeEnabled(uint32_t attrId) const
    {
        const auto& vtxInputParams = m_pipeline->getVertexInputParams();
        if (!(vtxInputParams.enabledAttribFlags & (1u<<attrId)))
            return false;
    }
    inline bool isVertexAttribBufferBindingEnabled(uint32_t bndId) const
    {
        const auto& vtxInputParams = m_pipeline->getVertexInputParams();
        if (!(vtxInputParams.enabledBindingFlags & (1u<<bndId)))
            return false;
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
        return vtxInputParams.attributes[attrId].offset;
    }
    inline BufferType* getAttribBoundBuffer(uint32_t attrId) const
    {
        const uint32_t bnd = getBindingNumForAttribute(attrId);
        return m_vertexBuffers[bnd].get();
    }

	//! Get type of index data which is stored in this meshbuffer.
	/** \return Index type of this buffer. */
	inline const E_INDEX_TYPE& getIndexType() const {return indexType;}
	inline void setIndexType(const E_INDEX_TYPE& type) {indexType = type;}

	//! Sets offset in mapped index buffer.
	/** @param byteOffset Offset in bytes. */
	inline void setIndexBufferOffset(const size_t& byteOffset) {indexBufOffset = byteOffset;}
	//! Accesses offset in mapped index buffer.
	/** @returns Offset in bytes. */
	inline const size_t& getIndexBufferOffset() const {return indexBufOffset;}

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
                    return indexCount*2+indexBufOffset < m_indexBuffer->getSize();
                case EIT_32BIT:
                    return indexCount*4+indexBufOffset < m_indexBuffer->getSize();
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
};

}}

#endif //__IRR_I_CPU_MESH_BUFFER_H_INCLUDED__