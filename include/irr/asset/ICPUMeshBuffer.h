#ifndef __IRR_I_CPU_MESH_BUFFER_H_INCLUDED__
#define __IRR_I_CPU_MESH_BUFFER_H_INCLUDED__

#include "IMeshBuffer.h"
#include "irr/asset/IAsset.h"
#include "irr/video/decodePixels.h"
#include "irr/video/encodePixels.h"

namespace irr { namespace asset
{

namespace impl
{
    inline video::E_FORMAT getCorrespondingIntegerFmt(video::E_FORMAT _scaledFmt)
    {
        switch (_scaledFmt)
        {
        case video::EF_R8_USCALED: return video::EF_R8_UINT;
        case video::EF_R8_SSCALED: return video::EF_R8_SINT;
        case video::EF_R8G8_USCALED: return video::EF_R8G8_UINT;
        case video::EF_R8G8_SSCALED: return video::EF_R8G8_SINT;
        case video::EF_R8G8B8_USCALED: return video::EF_R8G8B8_UINT;
        case video::EF_R8G8B8_SSCALED: return video::EF_R8G8B8_SINT;
        case video::EF_B8G8R8_USCALED: return video::EF_B8G8R8_UINT;
        case video::EF_B8G8R8_SSCALED: return video::EF_B8G8R8_SINT;
        case video::EF_R8G8B8A8_USCALED: return video::EF_R8G8B8A8_UINT;
        case video::EF_R8G8B8A8_SSCALED: return video::EF_R8G8B8A8_SINT;
        case video::EF_B8G8R8A8_USCALED: return video::EF_B8G8R8A8_UINT;
        case video::EF_B8G8R8A8_SSCALED: return video::EF_B8G8R8A8_SINT;
        case video::EF_A8B8G8R8_USCALED_PACK32: return video::EF_A8B8G8R8_UINT_PACK32;
        case video::EF_A8B8G8R8_SSCALED_PACK32: return video::EF_A8B8G8R8_SINT_PACK32;
        case video::EF_A2R10G10B10_USCALED_PACK32: return video::EF_A2R10G10B10_UINT_PACK32;
        case video::EF_A2R10G10B10_SSCALED_PACK32: return video::EF_A2R10G10B10_SINT_PACK32;
        case video::EF_A2B10G10R10_USCALED_PACK32: return video::EF_A2B10G10R10_UINT_PACK32;
        case video::EF_A2B10G10R10_SSCALED_PACK32: return video::EF_A2B10G10R10_SINT_PACK32;
        case video::EF_R16_USCALED: return video::EF_R16_UINT;
        case video::EF_R16_SSCALED: return video::EF_R16_SINT;
        case video::EF_R16G16_USCALED: return video::EF_R16G16_UINT;
        case video::EF_R16G16_SSCALED: return video::EF_R16G16_SINT;
        case video::EF_R16G16B16_USCALED: return video::EF_R16G16B16_UINT;
        case video::EF_R16G16B16_SSCALED: return video::EF_R16G16B16_SINT;
        case video::EF_R16G16B16A16_USCALED: return video::EF_R16G16B16A16_UINT;
        case video::EF_R16G16B16A16_SSCALED: return video::EF_R16G16B16A16_SINT;

        default: return video::EF_UNKNOWN;
        }
    }
}

class ICPUMeshBuffer : public scene::IMeshBuffer<asset::ICPUBuffer>, public asset::BlobSerializable, public asset::IAsset
{
    //vertices
    scene::E_VERTEX_ATTRIBUTE_ID posAttrId;
public:
    ICPUMeshBuffer(core::LeakDebugger* dbgr = NULL) : IMeshBuffer<asset::ICPUBuffer>(NULL, dbgr), posAttrId(scene::EVAI_ATTR0) {}

    virtual void* serializeToBlob(void* _stackPtr = NULL, const size_t& _stackSize = 0) const
    {
        return asset::CorrespondingBlobTypeFor<ICPUMeshBuffer>::type::createAndTryOnStack(this, _stackPtr, _stackSize);
    }

    virtual void convertToDummyObject() override {}
    virtual asset::IAsset::E_TYPE getAssetType() const override { return asset::IAsset::ET_SUB_MESH; }

    virtual size_t conservativeSizeEstimate() const override { return sizeof(IMeshBuffer<asset::ICPUBuffer>) + sizeof(posAttrId); }

    virtual scene::E_MESH_BUFFER_TYPE getMeshBufferType() const { return scene::EMBT_NOT_ANIMATED; }

    size_t calcVertexSize() const
    {
        if (!meshLayout)
            return 0u;

        size_t size = 0u;
        for (size_t i = 0; i < scene::EVAI_COUNT; ++i)
            if (meshLayout->getMappedBuffer((scene::E_VERTEX_ATTRIBUTE_ID)i))
                size += video::getTexelOrBlockSize(meshLayout->getAttribFormat((scene::E_VERTEX_ATTRIBUTE_ID)i));
        return size;
    }

    size_t calcVertexCount() const
    {
        size_t vertexCount = 0u;
        if (meshLayout && meshLayout->getIndexBuffer())
        {
            if (getIndexType() == scene::EIT_16BIT)
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
            else if (getIndexType() == scene::EIT_32BIT)
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
    inline const scene::E_VERTEX_ATTRIBUTE_ID& getPositionAttributeIx() const { return posAttrId; }
    //! Sets id of position atrribute.
    inline void setPositionAttributeIx(const scene::E_VERTEX_ATTRIBUTE_ID& attrId)
    {
        if (attrId >= scene::EVAI_COUNT)
        {
#ifdef _DEBUG
            //os::Printer::log("MeshBuffer setPositionAttributeIx attribute ID out of range!\n",ELL_ERROR);
#endif // _DEBUG
            return;
        }

        posAttrId = attrId;
    }

    //! Get access to Indices.
    /** \return Pointer to indices array. */
    inline void* getIndices()
    {
        if (!meshLayout)
            return NULL;
        if (!meshLayout->getIndexBuffer())
            return NULL;

        return ((uint8_t*)meshLayout->getIndexBuffer()->getPointer()) + indexBufOffset;
    }

    //! Get access to Indices.
    /** We only keep track of a position attribute, as every vertex needs to have at least a position to be displayed on the screen.
    Certain vertices may not have colors, normals, texture coords, etc. but a position is always present.
    \return Pointer to index array. */
    inline const void* getIndices() const
    {
        if (!meshLayout)
            return NULL;
        if (!meshLayout->getIndexBuffer())
            return NULL;

        return ((const uint8_t*)meshLayout->getIndexBuffer()->getPointer()) + indexBufOffset;
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
#ifdef _DEBUG
        if (!success)
        {
            //os::Printer::log("SOME DEBUG MESSAGE!\n",ELL_ERROR);
        }
#endif // _DEBUG
        return outPos;
    }

    //! Accesses data of buffer of attribute of given id
    /** Basically it will get the start of the array at the same point as OpenGL will get upon a glDraw*.
    @param attrId Attribute id.
    @returns Pointer to corresponding buffer's data incremented by `baseVertex` and by `bufferOffset`
    @see @ref getBaseVertex() setBaseVertex() getAttribute()
    */
    virtual uint8_t* getAttribPointer(const scene::E_VERTEX_ATTRIBUTE_ID& attrId) const
    {
        if (!meshLayout)
            return NULL;

        const asset::ICPUBuffer* mappedAttrBuf = meshLayout->getMappedBuffer(attrId);
        if (attrId >= scene::EVAI_COUNT || !mappedAttrBuf)
            return NULL;

        int64_t ix = baseVertex;
        ix *= meshLayout->getMappedBufferStride(attrId);
        ix += meshLayout->getMappedBufferOffset(attrId);
        if (ix < 0 || static_cast<uint64_t>(ix) >= mappedAttrBuf->getSize())
            return NULL;

        return ((uint8_t*)mappedAttrBuf->getPointer()) + ix;
    }

    static bool getAttribute(core::vectorSIMDf& output, const void* src, video::E_FORMAT format)
    {
        if (!src)
            return false;

        bool scaled = false;
        if (!video::isNormalizedFormat(format) && !video::isFloatingPointFormat(format) && !(scaled = video::isScaledFormat(format)))
            return false;

        if (!scaled)
        {
            double output64[4];
            video::decodePixels<double>(format, &src, output64, 0u, 0u);
            std::copy(output64, output64+4, output.pointer);
        }
        else
        {
            if (video::isSignedFormat(format))
            {
                int64_t output64i[4];
                video::decodePixels<int64_t>(impl::getCorrespondingIntegerFmt(format), &src, output64i, 0u, 0u);
                std::copy(output64i, output64i+4, output.pointer);
            }
            else
            {
                uint64_t output64u[4];
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
    virtual bool getAttribute(core::vectorSIMDf& output, const scene::E_VERTEX_ATTRIBUTE_ID& attrId, size_t ix) const
    {
        if (!meshLayout)
            return false;
        const asset::ICPUBuffer* mappedAttrBuf = meshLayout->getMappedBuffer(attrId);
        if (!mappedAttrBuf)
            return false;

        uint8_t* src = getAttribPointer(attrId);
        src += ix * meshLayout->getMappedBufferStride(attrId);
        if (src >= ((const uint8_t*)(mappedAttrBuf->getPointer())) + mappedAttrBuf->getSize())
            return false;

        return getAttribute(output, src, meshLayout->getAttribFormat(attrId));
    }

    static bool getAttribute(uint32_t* output, const void* src, video::E_FORMAT format)
    {
        if (!src)
            return false;

        bool scaled = false;
        if ((scaled = video::isScaledFormat(format)) || video::isIntegerFormat(format))
        {
            if (video::isSignedFormat(format))
            {
                int64_t output64[4];
                video::decodePixels<int64_t>(scaled ? impl::getCorrespondingIntegerFmt(format) : format, &src, output64, 0u, 0u);
                for (uint32_t i = 0u; i < video::getFormatChannelCount(format); ++i)
                    output[i] = output64[i];
            }
            else
            {
                uint64_t output64[4];
                video::decodePixels<uint64_t>(scaled ? impl::getCorrespondingIntegerFmt(format) : format, &src, output64, 0u, 0u);
                for (uint32_t i = 0u; i < video::getFormatChannelCount(format); ++i)
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
    virtual bool getAttribute(uint32_t* output, const scene::E_VERTEX_ATTRIBUTE_ID& attrId, size_t ix) const
    {
        if (!meshLayout)
            return false;
        const asset::ICPUBuffer* mappedAttrBuf = meshLayout->getMappedBuffer(attrId);
        if (!mappedAttrBuf)
            return false;

        uint8_t* src = getAttribPointer(attrId);
        src += ix * meshLayout->getMappedBufferStride(attrId);
        if (src >= ((const uint8_t*)(mappedAttrBuf->getPointer())) + mappedAttrBuf->getSize())
            return false;

        return getAttribute(output, src, meshLayout->getAttribFormat(attrId));
    }

    static bool setAttribute(core::vectorSIMDf input, void* dst, video::E_FORMAT format)
    {
        bool scaled = false;
        if (!dst || (!video::isFloatingPointFormat(format) && !video::isNormalizedFormat(format) && !(scaled = video::isScaledFormat(format))))
            return false;

        double input64[4];
        for (uint32_t i = 0u; i < 4u; ++i)
            input64[i] = input.pointer[i];

        if (!scaled)
            video::encodePixels<double>(format, dst, input64);
        else
        {
            if (video::isSignedFormat(format))
            {
                int64_t input64i[4]{ input64[0], input64[1], input64[2], input64[3] };
                video::encodePixels<int64_t>(impl::getCorrespondingIntegerFmt(format), dst, input64i);
            }
            else
            {
                uint64_t input64u[4]{ input64[0], input64[1], input64[2], input64[3] };
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
    virtual bool setAttribute(core::vectorSIMDf input, const scene::E_VERTEX_ATTRIBUTE_ID& attrId, size_t ix) const
    {
        if (!meshLayout)
            return false;
        const asset::ICPUBuffer* mappedBuffer = meshLayout->getMappedBuffer(attrId);
        if (!mappedBuffer)
            return false;

        uint8_t* dst = getAttribPointer(attrId);
        dst += ix * meshLayout->getMappedBufferStride(attrId);
        if (dst >= ((const uint8_t*)(mappedBuffer->getPointer())) + mappedBuffer->getSize())
            return false;

        return setAttribute(input, dst, meshLayout->getAttribFormat(attrId));
    }

    static bool setAttribute(const uint32_t* _input, void* dst, video::E_FORMAT format)
    {
        const bool scaled = video::isScaledFormat(format);
        if (!dst || !(scaled || video::isIntegerFormat(format)))
            return false;
        uint8_t* vxPtr = (uint8_t*)dst;

        if (video::isSignedFormat(format))
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

    //! @copydoc setAttribute(core::vectorSIMDf, const scene::E_VERTEX_ATTRIBUTE_ID&, size_t)
    virtual bool setAttribute(const uint32_t* _input, const scene::E_VERTEX_ATTRIBUTE_ID& attrId, size_t ix) const
    {
        if (!meshLayout)
            return false;
        const asset::ICPUBuffer* mappedBuffer = meshLayout->getMappedBuffer(attrId);
        if (!mappedBuffer)
            return false;

        uint8_t* dst = getAttribPointer(attrId);
        dst += ix * meshLayout->getMappedBufferStride(attrId);
        if (dst >= ((const uint8_t*)(mappedBuffer->getPointer())) + mappedBuffer->getSize())
            return false;

        return setAttribute(_input, dst, meshLayout->getAttribFormat(attrId));
    }


    //! Recalculates the bounding box. Should be called if the mesh changed.
    virtual void recalculateBoundingBox()
    {
        if (!meshLayout)
        {
            boundingBox.reset(core::vector3df(0.f));
            return;
        }

        const asset::ICPUBuffer* mappedAttrBuf = meshLayout->getMappedBuffer(posAttrId);
        if (posAttrId >= scene::EVAI_COUNT || !mappedAttrBuf)
        {
            boundingBox.reset(core::vector3df(0.f));
            return;
        }

        for (size_t j = 0; j < indexCount; j++)
        {
            size_t ix;
            void* indices = getIndices();
            if (indices)
            {
                switch (indexType)
                {
                case scene::EIT_32BIT:
                    ix = ((uint32_t*)indices)[j];
                    break;
                case scene::EIT_16BIT:
                    ix = ((uint16_t*)indices)[j];
                    break;
                default:
                    return;
                }
            }
            else
                ix = j;


            if (j)
                boundingBox.addInternalPoint(getPosition(ix).getAsVector3df());
            else
                boundingBox.reset(getPosition(ix).getAsVector3df());
        }
    }
};

}}

#endif //__IRR_I_CPU_MESH_BUFFER_H_INCLUDED__