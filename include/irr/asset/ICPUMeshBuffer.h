#ifndef __IRR_I_CPU_MESH_BUFFER_H_INCLUDED__
#define __IRR_I_CPU_MESH_BUFFER_H_INCLUDED__

#include "irr/asset/IMeshBuffer.h"
#include "irr/asset/ICPUDescriptorSet.h"
#include "irr/asset/ICPURenderpassIndependentPipeline.h"
#include "irr/asset/bawformat/blobs/MeshBufferBlob.h"
#include "irr/asset/bawformat/BlobSerializable.h"
#include "irr/asset/format/decodePixels.h"
#include "irr/asset/format/encodePixels.h"

namespace irr
{
namespace asset
{

namespace impl
{
    inline E_FORMAT getCorrespondingIntegerFmt(E_FORMAT _scaledFmt)
    {
        switch (_scaledFmt)
        {
        case EF_R8_USCALED: return EF_R8_UINT;
        case EF_R8_SSCALED: return EF_R8_SINT;
        case EF_R8G8_USCALED: return EF_R8G8_UINT;
        case EF_R8G8_SSCALED: return EF_R8G8_SINT;
        case EF_R8G8B8_USCALED: return EF_R8G8B8_UINT;
        case EF_R8G8B8_SSCALED: return EF_R8G8B8_SINT;
        case EF_B8G8R8_USCALED: return EF_B8G8R8_UINT;
        case EF_B8G8R8_SSCALED: return EF_B8G8R8_SINT;
        case EF_R8G8B8A8_USCALED: return EF_R8G8B8A8_UINT;
        case EF_R8G8B8A8_SSCALED: return EF_R8G8B8A8_SINT;
        case EF_B8G8R8A8_USCALED: return EF_B8G8R8A8_UINT;
        case EF_B8G8R8A8_SSCALED: return EF_B8G8R8A8_SINT;
        case EF_A8B8G8R8_USCALED_PACK32: return EF_A8B8G8R8_UINT_PACK32;
        case EF_A8B8G8R8_SSCALED_PACK32: return EF_A8B8G8R8_SINT_PACK32;
        case EF_A2R10G10B10_USCALED_PACK32: return EF_A2R10G10B10_UINT_PACK32;
        case EF_A2R10G10B10_SSCALED_PACK32: return EF_A2R10G10B10_SINT_PACK32;
        case EF_A2B10G10R10_USCALED_PACK32: return EF_A2B10G10R10_UINT_PACK32;
        case EF_A2B10G10R10_SSCALED_PACK32: return EF_A2B10G10R10_SINT_PACK32;
        case EF_R16_USCALED: return EF_R16_UINT;
        case EF_R16_SSCALED: return EF_R16_SINT;
        case EF_R16G16_USCALED: return EF_R16G16_UINT;
        case EF_R16G16_SSCALED: return EF_R16G16_SINT;
        case EF_R16G16B16_USCALED: return EF_R16G16B16_UINT;
        case EF_R16G16B16_SSCALED: return EF_R16G16B16_SINT;
        case EF_R16G16B16A16_USCALED: return EF_R16G16B16A16_UINT;
        case EF_R16G16B16A16_SSCALED: return EF_R16G16B16A16_SINT;

        default: return EF_UNKNOWN;
        }
    }
}

class ICPUMeshBuffer : public IMeshBuffer<ICPUBuffer, ICPUDescriptorSet, ICPURenderpassIndependentPipeline>, public BlobSerializable, public IAsset
{
    using base_t = IMeshBuffer<ICPUBuffer, ICPUDescriptorSet, ICPURenderpassIndependentPipeline>;
    //vertices
    uint32_t posAttrId;
protected:
    virtual ~ICPUMeshBuffer() = default;

    template<typename T>
    core::smart_refctd_ptr<IAsset> clone_template(uint32_t _depth = ~0u) const
    {
        auto cp = core::make_smart_refctd_ptr<T>();
        clone_common(cp.get());
        cp->m_descriptorSet = (_depth > 0u && m_descriptorSet) ? core::smart_refctd_ptr_static_cast<ICPUDescriptorSet>(m_descriptorSet->clone(_depth - 1u)) : m_descriptorSet;
        cp->m_pipeline = (_depth > 0u && m_pipeline) ? core::smart_refctd_ptr_static_cast<ICPURenderpassIndependentPipeline>(m_pipeline->clone(_depth - 1u)) : m_pipeline;

        cp->boundingBox = boundingBox;
        cp->indexType = indexType;
        cp->indexCount = indexCount;
        cp->baseVertex = baseVertex;
        cp->indexCount = indexCount;
        cp->instanceCount = instanceCount;
        cp->baseInstance = baseInstance;
        memcpy(cp->m_pushConstantsData, m_pushConstantsData, sizeof(m_pushConstantsData));
        cp->posAttrId = posAttrId;

        cp->m_indexBufferBinding.offset = m_indexBufferBinding.offset;
        cp->m_indexBufferBinding.buffer = (_depth > 0u && m_indexBufferBinding.buffer) ? 
            core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_indexBufferBinding.buffer->clone(_depth - 1u)) :
            m_indexBufferBinding.buffer;
        for (uint32_t i = 0u; i < MAX_ATTR_BUF_BINDING_COUNT; ++i)
        {
            cp->m_vertexBufferBindings[i].offset = m_vertexBufferBindings[i].offset;
            cp->m_vertexBufferBindings[i].buffer = (_depth > 0u && m_vertexBufferBindings[i].buffer) ?
                core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_vertexBufferBindings[i].buffer->clone(_depth - 1u)) :
                m_vertexBufferBindings[i].buffer;
        }

        return cp;
    }

public:
    //! Default constructor (initializes pipeline, desc set and buffer bindings to nullptr)
    ICPUMeshBuffer() : base_t(nullptr, nullptr, nullptr, SBufferBinding<ICPUBuffer>{}) {}
    using base_t::base_t;

    virtual void* serializeToBlob(void* _stackPtr = nullptr, const size_t& _stackSize = 0) const override
    {
        return CorrespondingBlobTypeFor<ICPUMeshBuffer>::type::createAndTryOnStack(this, _stackPtr, _stackSize);
    }

    core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
    {
        return clone_template<ICPUMeshBuffer>(_depth);
    }

    virtual void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
	{
        if (isDummyObjectForCacheAliasing)
            return;
        convertToDummyObject_common(referenceLevelsBelowToConvert);

		if (referenceLevelsBelowToConvert)
		{
            --referenceLevelsBelowToConvert;
			for (auto i=0u; i<MAX_ATTR_BUF_BINDING_COUNT; i++)
                if (m_vertexBufferBindings[i].buffer)
				    m_vertexBufferBindings[i].buffer->convertToDummyObject(referenceLevelsBelowToConvert);
            if (m_indexBufferBinding.buffer)
			    m_indexBufferBinding.buffer->convertToDummyObject(referenceLevelsBelowToConvert);
            if (m_descriptorSet)
			    m_descriptorSet->convertToDummyObject(referenceLevelsBelowToConvert);
            if (m_pipeline)
			    m_pipeline->convertToDummyObject(referenceLevelsBelowToConvert);
		}
	}
    virtual IAsset::E_TYPE getAssetType() const override { return IAsset::ET_SUB_MESH; }

    virtual size_t conservativeSizeEstimate() const override { return sizeof(base_t) + sizeof(posAttrId); }

    virtual E_MESH_BUFFER_TYPE getMeshBufferType() const { return EMBT_NOT_ANIMATED; }

	inline const SBufferBinding<ICPUBuffer>& getAttribBoundBuffer(uint32_t attrId) const
	{
		return base_t::getAttribBoundBuffer(attrId);
	}
    inline SBufferBinding<ICPUBuffer>* getAttribBoundBuffer(uint32_t attrId)
    {
        const uint32_t bnd = getBindingNumForAttribute(attrId);
        return &m_vertexBufferBindings[bnd];
    }
	inline const SBufferBinding<ICPUBuffer>* getVertexBufferBindings() const
	{
		return m_vertexBufferBindings;
	}
    inline SBufferBinding<ICPUBuffer>* getVertexBufferBindings()
    {
        return m_vertexBufferBindings;
    }
	inline void setIndexBufferBinding(SBufferBinding<ICPUBuffer>&& bufferBinding)
	{
		m_indexBufferBinding = std::move(bufferBinding);
	}
	inline const SBufferBinding<ICPUBuffer>& getIndexBufferBinding() const
	{
		return m_indexBufferBinding;
	}
    inline SBufferBinding<ICPUBuffer>* getIndexBufferBinding()
    {
        return &m_indexBufferBinding;
    }/*
	inline bool setVertexBufferBindingParams(uint32_t bindingIndex, uint32_t stride, E_VERTEX_INPUT_RATE inputRate = E_VERTEX_INPUT_RATE::EVIR_PER_VERTEX)
	{
        if (!m_pipeline)
            return false;
		if (bindingIndex >= MAX_ATTR_BUF_BINDING_COUNT || stride >= 2048ull)
			return false;

		auto& binding(m_pipeline->getVertexInputParams().bindings[bindingIndex]);
		binding.stride = stride;
		binding.inputRate = inputRate;

		return true;
	}*/
	inline bool setVertexBufferBinding(SBufferBinding<ICPUBuffer>&& bufferBinding, uint32_t bindingIndex)
	{
		if (bindingIndex >= MAX_ATTR_BUF_BINDING_COUNT)
			return false;

        m_vertexBufferBindings[bindingIndex] = std::move(bufferBinding);

		return true;
	}/*
	inline bool setVertexAttribFormat(uint32_t attribIndex, uint32_t bindingIndex, E_FORMAT format, uint32_t relativeOffset)
	{
        if (!m_pipeline)
            return false;
		if (bindingIndex >= MAX_ATTR_BUF_BINDING_COUNT || attribIndex >= MAX_VERTEX_ATTRIB_COUNT || relativeOffset >= 2048ull)
			return false;

        auto& attribute = m_pipeline->getVertexInputParams().attributes[attribIndex];
		attribute.binding = bindingIndex;
		attribute.format = format;
		attribute.relativeOffset = relativeOffset;

		return true;
	}*/


	inline void setAttachedDescriptorSet(core::smart_refctd_ptr<ICPUDescriptorSet>&& descriptorSet)
	{
		m_descriptorSet = std::move(descriptorSet);
	}
	inline ICPUDescriptorSet* getAttachedDescriptorSet()
	{
		return m_descriptorSet.get();
	}
	inline const ICPUDescriptorSet* getAttachedDescriptorSet() const
	{
		return m_descriptorSet.get();
	}

	inline void setPipeline(core::smart_refctd_ptr<ICPURenderpassIndependentPipeline>&& pipeline)
	{
		m_pipeline = std::move(pipeline);
	}
	inline ICPURenderpassIndependentPipeline* getPipeline()
	{
		return m_pipeline.get();
	}
    inline const ICPURenderpassIndependentPipeline* getPipeline() const
    {
        return m_pipeline.get();
    }

    inline size_t calcVertexSize() const
    {
        if (!m_pipeline)
            return 0u;

        auto ppln = m_pipeline.get();
        const auto& vtxInputParams = ppln->getVertexInputParams();
        size_t size = 0u;
        for (size_t i = 0; i < MAX_VERTEX_ATTRIB_COUNT; ++i)
            if (vtxInputParams.enabledAttribFlags & (1u<<i))
                size += asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(vtxInputParams.attributes[i].format));
        return size;
    }

    inline size_t calcVertexCount() const
    {
        size_t vertexCount = 0u;
        if (m_indexBufferBinding.buffer)
        {
            if (getIndexType() == EIT_16BIT)
            {
                for (size_t i = 0; i < getIndexCount(); i++)
                {
                    size_t index = reinterpret_cast<const uint16_t*>(getIndices())[i];
                    if (index > vertexCount)
                        vertexCount = index;
                }
                if (getIndexCount())
                    vertexCount++;
            }
            else if (getIndexType() == EIT_32BIT)
            {
                for (size_t i = 0; i < getIndexCount(); i++)
                {
                    size_t index = reinterpret_cast<const uint32_t*>(getIndices())[i];
                    if (index > vertexCount)
                        vertexCount = index;
                }
                if (getIndexCount())
                    vertexCount++;
            }
            else
                vertexCount = getIndexCount();
        }
        else
            vertexCount = getIndexCount();

        return vertexCount;
    }

    //! Returns id of position attribute.
    inline uint32_t getPositionAttributeIx() const { return posAttrId; }
    //! Sets id of position atrribute.
    inline void setPositionAttributeIx(const uint32_t attrId)
    {
        if (attrId >= MAX_VERTEX_ATTRIB_COUNT)
        {
#ifdef _IRR_DEBUG
            //os::Printer::log("MeshBuffer setPositionAttributeIx attribute ID out of range!\n",ELL_ERROR);
#endif // _IRR_DEBUG
            return;
        }

        posAttrId = attrId;
    }

    //! Get access to Indices.
    /** \return Pointer to indices array. */
    inline void* getIndices()
    {
        if (!m_indexBufferBinding.buffer)
            return nullptr;

        return reinterpret_cast<uint8_t*>(m_indexBufferBinding.buffer->getPointer()) + m_indexBufferBinding.offset;
    }

    //! Get access to Indices.
    /** We only keep track of a position attribute, as every vertex needs to have at least a position to be displayed on the screen.
    Certain vertices may not have colors, normals, texture coords, etc. but a position is always present.
    \return Pointer to index array. */
    inline const void* getIndices() const
    {
        if (!m_indexBufferBinding.buffer)
            return nullptr;

        return reinterpret_cast<const uint8_t*>(m_indexBufferBinding.buffer->getPointer()) + m_indexBufferBinding.offset;
    }

    //! Accesses given index of mapped position attribute buffer.
    /** @param ix Index number of vertex which is to be returned.
    @returns `ix`th vertex of mapped attribute buffer or (0, 0, 0, 1) vector if an error occured (e.g. no such vertex).
    @see @ref getAttribute()
    */
    virtual core::vectorSIMDf getPosition(size_t ix) const
    {
        core::vectorSIMDf outPos(0.f, 0.f, 0.f, 1.f);
        bool success = getAttribute(outPos, posAttrId, ix);
#ifdef _IRR_DEBUG
        if (!success)
        {
            //os::Printer::log("SOME DEBUG MESSAGE!\n",ELL_ERROR);
        }
#endif // _IRR_DEBUG
        return outPos;
    }

    //! Accesses data of buffer of attribute of given id
    /** Basically it will get the start of the array at the same point as OpenGL will get upon a glDraw*.
    @param attrId Attribute id.
    @returns Pointer to corresponding buffer's data incremented by `baseVertex` and by `bufferOffset`
    @see @ref getBaseVertex() setBaseVertex() getAttribute()
    */
    virtual uint8_t* getAttribPointer(uint32_t attrId)
    {
        if (!m_pipeline)
            return nullptr;

        const auto& vtxInputParams = m_pipeline->getVertexInputParams();
        if (!isAttributeEnabled(attrId))
            return nullptr;

        const uint32_t bindingNum = vtxInputParams.attributes[attrId].binding;
        if (!isVertexAttribBufferBindingEnabled(bindingNum))
            return nullptr;

        ICPUBuffer* mappedAttrBuf = m_vertexBufferBindings[bindingNum].buffer.get();
        if (!mappedAttrBuf)
            return nullptr;

        int64_t ix = baseVertex;
        ix *= vtxInputParams.bindings[bindingNum].stride;
        ix += (m_vertexBufferBindings[bindingNum].offset + vtxInputParams.attributes[attrId].relativeOffset);
        if (ix < 0 || static_cast<uint64_t>(ix) >= mappedAttrBuf->getSize())
            return nullptr;

        return reinterpret_cast<uint8_t*>(mappedAttrBuf->getPointer()) + ix;
    }
	inline const uint8_t* getAttribPointer(uint32_t attrId) const
	{
		return const_cast<typename std::decay<decltype(*this)>::type&>(*this).getAttribPointer(attrId);
	}

    static inline bool getAttribute(core::vectorSIMDf& output, const void* src, E_FORMAT format)
    {
        if (!src)
            return false;

        bool scaled = false;
        if (!isNormalizedFormat(format) && !isFloatingPointFormat(format) && !(scaled = isScaledFormat(format)))
            return false;

        if (!scaled)
        {
            double output64[4]{ 0., 0., 0., 1. };
            video::decodePixels<double>(format, &src, output64, 0u, 0u);
            std::copy(output64, output64+4, output.pointer);
        }
        else
        {
            if (isSignedFormat(format))
            {
                int64_t output64i[4]{ 0, 0, 0, 1 };
                video::decodePixels<int64_t>(impl::getCorrespondingIntegerFmt(format), &src, output64i, 0u, 0u);
                std::copy(output64i, output64i+4, output.pointer);
            }
            else
            {
                uint64_t output64u[4]{ 0u, 0u, 0u, 1u };
                video::decodePixels<uint64_t>(impl::getCorrespondingIntegerFmt(format), &src, output64u, 0u, 0u);
                std::copy(output64u, output64u+4, output.pointer);
            }
        }

        return true;
    }

    //! Accesses vertex of given index of given vertex attribute. Index number is incremented by `baseVertex`. WARNING: NOT ALL FORMAT CONVERSIONS TO RGBA32F/XYZW32F ARE IMPLEMENTED!
    /** If component count of given attribute is less than 4, only first ones of output vector's members will be written.
    @param[out] output vectorSIMDf object to which index's value will be returned.
    @param[in] attrId Atrribute id.
    @param[in] ix Index which is to be accessed. Will be incremented by `baseVertex`.
    @returns true if successful or false if an error occured (e.g. `ix` out of range, no attribute specified/bound or given attribute's format conversion to vectorSIMDf unsupported).
    @see @ref getBaseVertex() setBaseVertex() getAttribute()
    */
    virtual bool getAttribute(core::vectorSIMDf& output, uint32_t attrId, size_t ix) const
    {
        if (!isAttributeEnabled(attrId))
            return false;

        const uint32_t bindingId = getBindingNumForAttribute(attrId);

        const uint8_t* src = getAttribPointer(attrId);
        src += ix * getAttribStride(attrId);
        if (src >= reinterpret_cast<const uint8_t*>(m_vertexBufferBindings[bindingId].buffer->getPointer()) + m_vertexBufferBindings[bindingId].buffer->getSize())
            return false;

        return getAttribute(output, src, getAttribFormat(attrId));
    }

    static inline bool getAttribute(uint32_t* output, const void* src, E_FORMAT format)
    {
        if (!src)
            return false;

        bool scaled = false;
        if ((scaled = isScaledFormat(format)) || isIntegerFormat(format))
        {
            if (isSignedFormat(format))
            {
                int64_t output64[4]{0, 0, 0, 1};
                video::decodePixels<int64_t>(scaled ? impl::getCorrespondingIntegerFmt(format) : format, &src, output64, 0u, 0u);
                for (uint32_t i = 0u; i < getFormatChannelCount(format); ++i)
                    output[i] = output64[i];
            }
            else
            {
                uint64_t output64[4]{0u, 0u, 0u, 1u};
                video::decodePixels<uint64_t>(scaled ? impl::getCorrespondingIntegerFmt(format) : format, &src, output64, 0u, 0u);
                for (uint32_t i = 0u; i < getFormatChannelCount(format); ++i)
                    output[i] = output64[i];
            }
            return true;
        }

        return false;
    }

    //! Accesses vertex of given index of given vertex attribute. Index number is incremented by `baseVertex`. WARNING: NOT ALL FORMAT CONVERSIONS TO RGBA32F/XYZW32F ARE IMPLEMENTED!
    /** If component count of given attribute is less than 4, only first ones of output vector's members will be written.
    Attributes of integer types smaller than 32 bits are promoted to 32bit integer.
    @param[out] output Pointer to memory to which index's value will be returned.
    @param[in] attrId Atrribute id.
    @param[in] ix Index which is to be accessed. Will be incremented by `baseVertex`.
    @returns true if successful or false if an error occured (e.g. `ix` out of range, no attribute specified/bound or given attribute's format conversion to vectorSIMDf unsupported).
    @see @ref getBaseVertex() setBaseVertex() getAttribute()
    */
    virtual bool getAttribute(uint32_t* output, uint32_t attrId, size_t ix) const
    {
        if (!m_pipeline)
            return false;
        if (!isAttributeEnabled(attrId))
            return false;

        const uint8_t* src = getAttribPointer(attrId);
        src += ix * getAttribStride(attrId);
        const ICPUBuffer* buf = base_t::getAttribBoundBuffer(attrId).buffer.get();
        if (!buf || src >= reinterpret_cast<const uint8_t*>(buf->getPointer()) + buf->getSize())
            return false;

        return getAttribute(output, src, getAttribFormat(attrId));
    }

    static inline bool setAttribute(core::vectorSIMDf input, void* dst, E_FORMAT format)
    {
        bool scaled = false;
        if (!dst || (!isFloatingPointFormat(format) && !isNormalizedFormat(format) && !(scaled = isScaledFormat(format))))
            return false;

        double input64[4];
        for (uint32_t i = 0u; i < 4u; ++i)
            input64[i] = input.pointer[i];

        if (!scaled)
            video::encodePixels<double>(format, dst, input64);
        else
        {
            if (isSignedFormat(format))
            {
                int64_t input64i[4]{ static_cast<int64_t>(input64[0]), static_cast<int64_t>(input64[1]), static_cast<int64_t>(input64[2]), static_cast<int64_t>(input64[3]) };
                video::encodePixels<int64_t>(impl::getCorrespondingIntegerFmt(format), dst, input64i);
            }
            else
            {
                uint64_t input64u[4]{ static_cast<uint64_t>(input64[0]), static_cast<uint64_t>(input64[1]), static_cast<uint64_t>(input64[2]), static_cast<uint64_t>(input64[3]) };
                video::encodePixels<uint64_t>(impl::getCorrespondingIntegerFmt(format), dst, input64u);
            }
        }

        return true;
    }

    //! Sets value of vertex of given index of given attribute. WARNING: NOT ALL FORMAT CONVERSIONS FROM RGBA32F/XYZW32F (vectorSIMDf) ARE IMPLEMENTED!
    /** @param input Value which is to be set.
    @param attrId Atrribute id.
    @param ix Index of vertex which is to be set. Will be incremented by `baseVertex`.
    @returns true if successful or false if an error occured (e.g. no such index).
    @see @ref getBaseVertex() setBaseVertex() getAttribute()
    */
    virtual bool setAttribute(core::vectorSIMDf input, uint32_t attrId, size_t ix)
    {
        if (!m_pipeline)
            return false;
        if (!isAttributeEnabled(attrId))
            return false;

        uint8_t* dst = getAttribPointer(attrId);
        dst += ix * getAttribStride(attrId);
        ICPUBuffer* buf = getAttribBoundBuffer(attrId)->buffer.get();
        if (!buf || dst >= ((const uint8_t*)(buf->getPointer())) + buf->getSize())
            return false;

        return setAttribute(input, dst, getAttribFormat(attrId));
    }

    static inline bool setAttribute(const uint32_t* _input, void* dst, E_FORMAT format)
    {
        const bool scaled = isScaledFormat(format);
        if (!dst || !(scaled || isIntegerFormat(format)))
            return false;
        uint8_t* vxPtr = (uint8_t*)dst;

        if (isSignedFormat(format))
        {
            int64_t input[4];
            for (uint32_t i = 0u; i < 4u; ++i)
                input[i] = reinterpret_cast<const int32_t*>(_input)[i];
            video::encodePixels<int64_t>(scaled ? impl::getCorrespondingIntegerFmt(format) : format, dst, input);
        }
        else
        {
            uint64_t input[4];
            for (uint32_t i = 0u; i < 4u; ++i)
                input[i] = _input[i];
            video::encodePixels<uint64_t>(scaled ? impl::getCorrespondingIntegerFmt(format) : format, dst, input);
        }
        return true;
    }

    //! @copydoc setAttribute(core::vectorSIMDf, const E_VERTEX_ATTRIBUTE_ID&, size_t)
    virtual bool setAttribute(const uint32_t* _input, uint32_t attrId, size_t ix)
    {
        if (!m_pipeline)
            return false;
        if (!isAttributeEnabled(attrId))
            return false;

        uint8_t* dst = getAttribPointer(attrId);
        dst += ix * getAttribStride(attrId);
        ICPUBuffer* buf = getAttribBoundBuffer(attrId)->buffer.get();
        if (dst >= ((const uint8_t*)(buf->getPointer())) + buf->getSize())
            return false;

        return setAttribute(_input, dst, getAttribFormat(attrId));
    }


    //! Recalculates the bounding box. Should be called if the mesh changed.
    virtual void recalculateBoundingBox()
    {
		setBoundingBox(calculateBoundingBox(this));
    }

	//! Utility function
    static core::aabbox3df calculateBoundingBox(const ICPUMeshBuffer* mb)
    {
		core::aabbox3df retval;
		retval.reset(core::vector3df(0.f));
        if (!mb->getPipeline())
            return retval;

		auto posAttrId = mb->getPositionAttributeIx();
        const ICPUBuffer* mappedAttrBuf = mb->getAttribBoundBuffer(posAttrId).buffer.get();
        if (posAttrId >= MAX_VERTEX_ATTRIB_COUNT || !mappedAttrBuf)
            return retval;

		const void* indices = mb->getIndices();
        for (size_t j=0ull; j<mb->getIndexCount(); j++)
        {
            size_t ix;
            if (indices)
            {
                switch (mb->getIndexType())
                {
					case EIT_32BIT:
						ix = reinterpret_cast<const uint32_t*>(indices)[j];
						break;
					case EIT_16BIT:
						ix = reinterpret_cast<const uint16_t*>(indices)[j];
						break;
					default:
						return retval;
                }
            }
            else
                ix = j;


            if (j)
				retval.addInternalPoint(mb->getPosition(ix).getAsVector3df());
            else
				retval.reset(mb->getPosition(ix).getAsVector3df());
        }
		return retval;
    }
};

}}

#endif //__IRR_I_CPU_MESH_BUFFER_H_INCLUDED__