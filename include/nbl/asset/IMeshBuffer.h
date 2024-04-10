// Copyright (C) 2018-2021 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_MESH_BUFFER_H_INCLUDED_
#define _NBL_ASSET_I_MESH_BUFFER_H_INCLUDED_

#include "nbl/core/shapes/AABB.h"

#include "nbl/asset/IRenderpassIndependentPipeline.h"
#include "nbl/asset/ECommonEnums.h"

#include <algorithm>

namespace nbl::asset
{

template <class BufferType, class DescSetType, class PipelineType>
class IMeshBuffer : public virtual core::IReferenceCounted
{
    public:
        _NBL_STATIC_INLINE_CONSTEXPR size_t MAX_PUSH_CONSTANT_BYTESIZE = 128u;

        _NBL_STATIC_INLINE_CONSTEXPR size_t MAX_VERTEX_ATTRIB_COUNT = SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT;
        _NBL_STATIC_INLINE_CONSTEXPR size_t MAX_ATTR_BUF_BINDING_COUNT = SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT;

    protected:
        virtual ~IMeshBuffer() = default;

        alignas(32) core::aabbox3df boundingBox;

        // TODO: format, stride, input rate
        SBufferBinding<BufferType> m_vertexBufferBindings[MAX_ATTR_BUF_BINDING_COUNT];
        SBufferBinding<BufferType> m_indexBufferBinding;

        //! Skin
        SBufferBinding<BufferType> m_inverseBindPoseBufferBinding,m_jointAABBBufferBinding;

        //! Descriptor set which goes to set=3
        core::smart_refctd_ptr<DescSetType> m_descriptorSet;

        alignas(64) uint8_t m_pushConstantsData[MAX_PUSH_CONSTANT_BYTESIZE]{};//by putting m_pushConstantsData here, alignas(64) takes no extra place

        // TODO: remove descriptor set, pipeline & instancing info
        //! Pipeline for drawing
        core::smart_refctd_ptr<PipelineType> m_pipeline;

	    // draw params
        uint32_t indexCount = 0u;
        uint32_t instanceCount = 1u;
	    int32_t baseVertex = 0;
        uint32_t baseInstance = 0u;

        // others
        uint32_t jointCount : 11;
        uint32_t maxJointsPerVx : 3;
        uint32_t indexType : 2;

    public:
        IMeshBuffer() : jointCount(0u), maxJointsPerVx(0u), indexType(EIT_UNKNOWN) {}
	    //! Constructor.
	    IMeshBuffer(core::smart_refctd_ptr<PipelineType>&& _pipeline,
            core::smart_refctd_ptr<DescSetType>&& _ds,
            SBufferBinding<BufferType> _vtxBindings[MAX_ATTR_BUF_BINDING_COUNT],
            SBufferBinding<BufferType>&& _indexBinding
        ) : boundingBox(), m_indexBufferBinding(std::move(_indexBinding)),
            m_inverseBindPoseBufferBinding(), m_jointAABBBufferBinding(),
            m_descriptorSet(std::move(_ds)), m_pipeline(std::move(_pipeline)),
            indexCount(0u), instanceCount(1u), baseVertex(0), baseInstance(0u),
            jointCount(0u), maxJointsPerVx(0u), indexType(EIT_UNKNOWN)
	    {
            if (_vtxBindings)
                std::copy(_vtxBindings, _vtxBindings+MAX_ATTR_BUF_BINDING_COUNT, m_vertexBufferBindings);
	    }

        inline bool isAttributeEnabled(uint32_t attrId) const
        {
            if (attrId >= MAX_VERTEX_ATTRIB_COUNT)
                return false;
            if (!m_pipeline)
                return false;

            const auto& vtxInputParams = m_pipeline->getCachedCreationParams().vertexInput;
            if (!(vtxInputParams.enabledAttribFlags & (1u<<attrId)))
                return false;
            return true;
        }
        inline bool isVertexAttribBufferBindingEnabled(uint32_t bndId) const
        {
            if (bndId >= MAX_ATTR_BUF_BINDING_COUNT)
                return false;
            if (!m_pipeline)
                return false;

            const auto& vtxInputParams = m_pipeline->getCachedCreationParams().vertexInput;
            if (!(vtxInputParams.enabledBindingFlags & (1u<<bndId)))
                return false;
            return true;
        }
        //! WARNING: does not check whether attribute and binding are enabled!
        inline uint32_t getBindingNumForAttribute(uint32_t attrId) const
        {
            const auto& vtxInputParams = m_pipeline->getCachedCreationParams().vertexInput;
            return vtxInputParams.attributes[attrId].binding;
        }
        inline E_FORMAT getAttribFormat(uint32_t attrId) const
        {
            const auto& vtxInputParams = m_pipeline->getCachedCreationParams().vertexInput;
            return static_cast<E_FORMAT>(vtxInputParams.attributes[attrId].format);
        }
        inline uint32_t getAttribStride(uint32_t attrId) const
        {
            const auto& vtxInputParams = m_pipeline->getCachedCreationParams().vertexInput;
            const uint32_t bnd = getBindingNumForAttribute(attrId);
            return vtxInputParams.bindings[bnd].stride;
        }
        inline uint32_t getAttribOffset(uint32_t attrId) const
        {
            const auto& vtxInputParams = m_pipeline->getCachedCreationParams().vertexInput;
            return vtxInputParams.attributes[attrId].relativeOffset;
        }

        inline SBufferBinding<BufferType>& getAttribBoundBuffer(uint32_t attrId)
        {
            const uint32_t bnd = getBindingNumForAttribute(attrId);
            return m_vertexBufferBindings[bnd];
        }
        inline const SBufferBinding<const BufferType>& getAttribBoundBuffer(uint32_t attrId) const
        {
            const uint32_t bnd = getBindingNumForAttribute(attrId);
            return reinterpret_cast<const SBufferBinding<const BufferType>&>(m_vertexBufferBindings[bnd]);
        }

        inline const SBufferBinding<BufferType>* getVertexBufferBindings()
        {
            return m_vertexBufferBindings;
        }
        inline const SBufferBinding<const BufferType>* getVertexBufferBindings() const
        {
            return reinterpret_cast<const SBufferBinding<const BufferType>*>(m_vertexBufferBindings);
        }

        inline const SBufferBinding<BufferType>& getIndexBufferBinding()
        {
            return m_indexBufferBinding;
        }
        inline const SBufferBinding<const BufferType>& getIndexBufferBinding() const
        {
            return reinterpret_cast<const SBufferBinding<const BufferType>&>(m_indexBufferBinding);
        }
        
        inline const SBufferBinding<BufferType>& getInverseBindPoseBufferBinding()
        {
            return m_inverseBindPoseBufferBinding;
        }
        inline const SBufferBinding<const BufferType>& getInverseBindPoseBufferBinding() const
        {
            return reinterpret_cast<const SBufferBinding<const BufferType>&>(m_inverseBindPoseBufferBinding);
        }
        
        inline const SBufferBinding<BufferType>& getJointAABBBufferBinding()
        {
            return m_jointAABBBufferBinding;
        }
        inline const SBufferBinding<const BufferType>& getJointAABBBufferBinding() const
        {
            return reinterpret_cast<const SBufferBinding<const BufferType>&>(m_jointAABBBufferBinding);
        }

        virtual inline bool setVertexBufferBinding(SBufferBinding<BufferType>&& bufferBinding, uint32_t bindingIndex)
	    {
		    if (bindingIndex >= MAX_ATTR_BUF_BINDING_COUNT)
			    return false;

            m_vertexBufferBindings[bindingIndex] = std::move(bufferBinding);

		    return true;
	    }

        virtual inline void setIndexBufferBinding(SBufferBinding<BufferType>&& bufferBinding)
	    {
            // assert(!isImmutable_debug());

		    m_indexBufferBinding = std::move(bufferBinding);
	    }

        virtual inline void setAttachedDescriptorSet(core::smart_refctd_ptr<DescSetType>&& descriptorSet)
        {
            //assert(!isImmutable_debug());
            m_descriptorSet = std::move(descriptorSet);
        }

        virtual inline void setPipeline(core::smart_refctd_ptr<PipelineType>&& pipeline)
        {
            //assert(!isImmutable_debug());
            m_pipeline = std::move(pipeline);
        }

        inline uint64_t getAttribCombinedOffset(uint32_t attrId) const
        {
            const auto& buf = getAttribBoundBuffer(attrId);
            return buf.offset+static_cast<uint64_t>(getAttribOffset(attrId));
        }

        //! Returns bound on JointID in the vertex attribute
        inline auto getJointCount() const { return jointCount; }

        //! Returns max joint influences
        inline auto getMaxJointsPerVertex() const { return maxJointsPerVx; }

        //!
        virtual inline bool isSkinned() const
        {
            return  jointCount>0u && maxJointsPerVx>0u && m_inverseBindPoseBufferBinding.buffer &&
                    m_inverseBindPoseBufferBinding.offset+jointCount*sizeof(core::matrix3x4SIMD)<=m_inverseBindPoseBufferBinding.buffer->getSize();
        }

        //!
        virtual inline bool setSkin(
            SBufferBinding<BufferType>&& _inverseBindPoseBufferBinding,
            SBufferBinding<BufferType>&& _jointAABBBufferBinding,
            const uint32_t _jointCount, const uint32_t _maxJointsPerVx
        )
        {
            if (!_inverseBindPoseBufferBinding.buffer || !_jointAABBBufferBinding.buffer || _jointCount==0u)
                return false;

            // a very arbitrary constraint
            if (_maxJointsPerVx==0u || _maxJointsPerVx>4u)
                return false;

            if (_inverseBindPoseBufferBinding.offset+_jointCount*sizeof(core::matrix3x4SIMD)>_inverseBindPoseBufferBinding.buffer->getSize())
                return false;

            m_inverseBindPoseBufferBinding = std::move(_inverseBindPoseBufferBinding);
            m_jointAABBBufferBinding = std::move(_jointAABBBufferBinding);
            jointCount = _jointCount;
            maxJointsPerVx = _maxJointsPerVx;
            return true;
        }

        //!
        inline const DescSetType* getAttachedDescriptorSet() const
        {
            return m_descriptorSet.get();
        }

        //!
        inline const PipelineType* getPipeline() const
        {
            return m_pipeline.get();
        }

	    //! Get type of index data which is stored in this meshbuffer.
	    /** \return Index type of this buffer. */
	    inline E_INDEX_TYPE getIndexType() const {return static_cast<E_INDEX_TYPE>(indexType);}
	    inline void setIndexType(const E_INDEX_TYPE type)
	    {
		    indexType = type;
	    }

	    //! Get amount of indices in this meshbuffer.
	    /** \return Number of indices in this buffer. */
	    inline auto getIndexCount() const {return indexCount;}
	    //! It sets amount of indices - value that is being passed to glDrawArrays as vertices amount or to glDrawElements as index amount.
	    /** @returns Whether set amount exceeds mapped buffer's size. Regardless of result the amount is set. */
	    inline bool setIndexCount(const uint32_t newIndexCount)
	    {
            indexCount = newIndexCount;
            if (m_indexBufferBinding.buffer)
            {
                switch (indexType)
                {
                    case EIT_16BIT:
                        return indexCount*sizeof(uint16_t)+m_indexBufferBinding.offset < m_indexBufferBinding.buffer->getSize();
                    case EIT_32BIT:
                        return indexCount*sizeof(uint32_t)+m_indexBufferBinding.offset < m_indexBufferBinding.buffer->getSize();
                    default:
                        return false;
                }
            }

            return true;
	    }

	    //! Accesses base vertex number.
	    /** @returns base vertex number. */
        inline int32_t getBaseVertex() const {return baseVertex;}
	    //! Sets base vertex.
        inline void setBaseVertex(const int32_t baseVx)
        {
            baseVertex = baseVx;
        }

	    inline uint32_t getInstanceCount() const {return instanceCount;}
	    inline void setInstanceCount(const uint32_t count)
	    {
		    instanceCount = count;
	    }

	    inline uint32_t getBaseInstance() const {return baseInstance;}
	    inline void setBaseInstance(const uint32_t base)
	    {
		    baseInstance = base;
	    }


	    //! Get the axis aligned bounding box of this meshbuffer.
	    /** \return Axis aligned bounding box of this buffer. */
	    inline const core::aabbox3df& getBoundingBox() const {return boundingBox;}

	    //! Set axis aligned bounding box
	    /** \param box User defined axis aligned bounding box to use
	    for this buffer. */
	    inline virtual void setBoundingBox(const core::aabbox3df& box)
	    {
		    boundingBox = box;
	    }

        uint8_t* getPushConstantsDataPtr() { return m_pushConstantsData; }
        const uint8_t* getPushConstantsDataPtr() const { return m_pushConstantsData; }
};

}

#endif