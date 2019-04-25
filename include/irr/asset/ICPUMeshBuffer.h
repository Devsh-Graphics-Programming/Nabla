#ifndef __IRR_I_CPU_MESH_BUFFER_H_INCLUDED__
#define __IRR_I_CPU_MESH_BUFFER_H_INCLUDED__

#include "irr/asset/IAsset.h"
#include "irr/video/decodePixels.h"
#include "irr/video/encodePixels.h"
#include "irr/asset/bawformat/CBAWFile.h"

namespace irr { namespace asset
{

namespace impl
{
    inline asset::E_FORMAT getCorrespondingIntegerFmt(asset::E_FORMAT _scaledFmt)
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

	//! Enumeration for all primitive types there are.
	enum E_PRIMITIVE_TYPE
	{
		//! All vertices are non-connected points.
		EPT_POINTS=0,

		//! All vertices form a single connected line.
		EPT_LINE_STRIP,

		//! Just as LINE_STRIP, but the last and the first vertex is also connected.
		EPT_LINE_LOOP,

		//! Every two vertices are connected creating n/2 lines.
		EPT_LINES,

		//! After the first two vertices each vertex defines a new triangle.
		//! Always the two last and the new one form a new triangle.
		EPT_TRIANGLE_STRIP,

		//! After the first two vertices each vertex defines a new triangle.
		//! All around the common first vertex.
		EPT_TRIANGLE_FAN,

		//! Explicitly set all vertices for each triangle.
		EPT_TRIANGLES

		// missing adjacency types and patches
	};

//!
enum E_INDEX_TYPE
{
    EIT_16BIT = 0,
    EIT_32BIT,
    EIT_UNKNOWN
};

//! Available vertex attribute ids
/** As of 2018 most OpenGL implementations support 16 attributes (some CAD GPUs more) */
enum E_VERTEX_ATTRIBUTE_ID
{
    EVAI_ATTR0=0,
    EVAI_ATTR1,
    EVAI_ATTR2,
    EVAI_ATTR3,
    EVAI_ATTR4,
    EVAI_ATTR5,
    EVAI_ATTR6,
    EVAI_ATTR7,
    EVAI_ATTR8,
    EVAI_ATTR9,
    EVAI_ATTR10,
    EVAI_ATTR11,
    EVAI_ATTR12,
    EVAI_ATTR13,
    EVAI_ATTR14,
    EVAI_ATTR15,
    EVAI_COUNT
};

enum E_MESH_BUFFER_TYPE
{
    EMBT_UNKNOWN = 0,
    EMBT_NOT_ANIMATED,
    EMBT_ANIMATED_FRAME_BASED,
    EMBT_ANIMATED_SKINNED
};

template <class T>
class IMeshDataFormatDesc : public virtual core::IReferenceCounted
{
    protected:
		//! Read https://www.khronos.org/opengl/wiki/Vertex_Specification for understanding of attribute IDs, indices, attribute formats etc.
        asset::E_FORMAT attrFormat[EVAI_COUNT];
        uint32_t attrStride[EVAI_COUNT];
        size_t attrOffset[EVAI_COUNT];
        uint32_t attrDivisor;

        //vertices
        T* mappedAttrBuf[EVAI_COUNT];
        //indices
        T* mappedIndexBuf;

        virtual ~IMeshDataFormatDesc()
        {
            for (size_t i=0; i<EVAI_COUNT; i++)
            {
                if (mappedAttrBuf[i])
                    mappedAttrBuf[i]->drop();
            }

            if (getIndexBuffer())
                getIndexBuffer()->drop();
        }
    public:
        //! Default constructor.
        IMeshDataFormatDesc()
        {
            for (size_t i=0; i<EVAI_COUNT; i++)
            {
                attrFormat[i] = EF_R32G32B32A32_SFLOAT;
                attrStride[i] = 16;
                attrOffset[i] = 0;
                mappedAttrBuf[i] = nullptr;
            }
            attrDivisor = 0u;
            mappedIndexBuf = nullptr;
        }

        inline bool formatCanBeAppended(const IMeshDataFormatDesc<T>* other) const
        {
            bool retVal = true;
            for (size_t i=0; retVal&&i<EVAI_COUNT; i++)
            {
                if (this->getMappedBuffer(static_cast<E_VERTEX_ATTRIBUTE_ID>(i)))
                {
                    if (other->getMappedBuffer(static_cast<E_VERTEX_ATTRIBUTE_ID>(i)))
                        retVal = retVal && attrFormat[i] == other->attrFormat[i];
                    else
                        return false;
                }
                else
                {
                    if (other->getMappedBuffer(static_cast<E_VERTEX_ATTRIBUTE_ID>(i)))
                        return false;
                    else
                        retVal = retVal && attrFormat[i] == other->attrFormat[i];
                }
            }
            return retVal;
        }

        inline void setIndexBuffer(T* ixbuf)
        {
    /*
    #ifdef _IRR_DEBUG
            if (size<0x7fffffffffffffffuLL&&ixbuf&&(ixbuf->getSize()>size+offset)) //not that easy to check
            {
                os::Printer::log("MeshBuffer map index buffer overflow!\n",ELL_ERROR);
                return;
            }
    #endif // _IRR_DEBUG
    */
            if (ixbuf)
                ixbuf->grab();

            if (mappedIndexBuf)
                mappedIndexBuf->drop();
            mappedIndexBuf = ixbuf;
        }

        inline const T* getIndexBuffer() const
        {
            return mappedIndexBuf;
        }


        //! remember that the divisor needs to be <=0x1u<<_IRR_VAO_MAX_ATTRIB_DIVISOR_BITS
        virtual void setVertexAttrBuffer(T* attrBuf, E_VERTEX_ATTRIBUTE_ID attrId, asset::E_FORMAT format, size_t stride=0, size_t offset=0, uint32_t divisor=0) = 0;

        inline const T* getMappedBuffer(E_VERTEX_ATTRIBUTE_ID attrId) const
        {
            assert(attrId<EVAI_COUNT);
            return mappedAttrBuf[attrId];
        }

        inline asset::E_FORMAT getAttribFormat(E_VERTEX_ATTRIBUTE_ID attrId) const
        {
            assert(attrId < EVAI_COUNT);
            return attrFormat[attrId];
        }

        inline void setMappedBufferOffset(E_VERTEX_ATTRIBUTE_ID attrId, size_t offset)
        {
            assert(attrId<EVAI_COUNT);

            if (!mappedAttrBuf[attrId])
                return;

            attrOffset[attrId] = offset;
        }

        inline const size_t& getMappedBufferOffset(E_VERTEX_ATTRIBUTE_ID attrId) const
        {
            assert(attrId<EVAI_COUNT);
            return attrOffset[attrId];
        }

        inline const uint32_t& getMappedBufferStride(E_VERTEX_ATTRIBUTE_ID attrId) const
        {
            assert(attrId<EVAI_COUNT);
            return attrStride[attrId];
        }

        inline uint32_t getAttribDivisor(E_VERTEX_ATTRIBUTE_ID attrId) const
        {
            assert(attrId<EVAI_COUNT);
            return (attrDivisor>>attrId)&1u;
        }

        inline void swapVertexAttrBuffer(T* attrBuf, E_VERTEX_ATTRIBUTE_ID attrId)
        {
            swapVertexAttrBuffer(attrBuf, attrId, attrOffset[attrId], attrStride[attrId]);
        }

        inline void swapVertexAttrBuffer(T* attrBuf, E_VERTEX_ATTRIBUTE_ID attrId, size_t newOffset)
        {
            swapVertexAttrBuffer(attrBuf, attrId, newOffset, attrStride[attrId]);
        }

        inline void swapVertexAttrBuffer(T* attrBuf, E_VERTEX_ATTRIBUTE_ID attrId, size_t newOffset, size_t newStride)
        {
            if (!mappedAttrBuf[attrId] || !attrBuf)
                return;

            attrBuf->grab();
            mappedAttrBuf[attrId]->drop();

            mappedAttrBuf[attrId] = attrBuf;
            attrOffset[attrId] = newOffset;
            attrStride[attrId] = newStride;
        }
};


class ICPUMeshDataFormatDesc : public IMeshDataFormatDesc<asset::ICPUBuffer>, public asset::BlobSerializable, public asset::IAsset
{
    protected:
	    ~ICPUMeshDataFormatDesc()
	    {
	    }
	public:
		virtual void* serializeToBlob(void* _stackPtr = NULL, const size_t& _stackSize = 0) const
		{
			return asset::CorrespondingBlobTypeFor<IMeshDataFormatDesc<asset::ICPUBuffer> >::type::createAndTryOnStack(static_cast<const IMeshDataFormatDesc<asset::ICPUBuffer>*>(this), _stackPtr, _stackSize);
		}

        size_t conservativeSizeEstimate() const override
        {
            return asset::CorrespondingBlobTypeFor<IMeshDataFormatDesc<asset::ICPUBuffer>>::type::calcBlobSizeForObj(this);
        }

        void convertToDummyObject() override
        {
        }

        asset::IAsset::E_TYPE getAssetType() const override { return asset::IAsset::ET_MESH_DATA_DESCRIPTOR; }

        //! remember that the divisor must be 0 or 1
        void setVertexAttrBuffer(asset::ICPUBuffer* attrBuf, E_VERTEX_ATTRIBUTE_ID attrId, asset::E_FORMAT format, size_t stride=0, size_t offset=0, uint32_t divisor=0) override
        {
            assert(attrId<EVAI_COUNT);
            assert(divisor<=1u);

            attrDivisor &= ~(divisor<<attrId);

            if (attrBuf)
            {
                attrBuf->grab();

                attrFormat[attrId] = format;
                // Don't get confused by `getTexelOrBlockSize` name. All vertex attrib, color, etc. are maintained with single enum asset::E_FORMAT and its naming conventions is color-like, and so are related functions. Whole story began from Vulkan's VkFormat.
                attrStride[attrId] = stride!=0 ? stride : asset::getTexelOrBlockSize(format);
                attrOffset[attrId] = offset;
                attrDivisor |= (divisor<<attrId);
            }
            else
            {
                attrFormat[attrId] = EF_R32G32B32A32_SFLOAT;
                attrStride[attrId] = 16;
                attrOffset[attrId] = 0;
                //attrDivisor &= ~(1u<<attrId); //cleared before if
            }

            if (mappedAttrBuf[attrId])
                mappedAttrBuf[attrId]->drop();
            mappedAttrBuf[attrId] = attrBuf;
        }
};

template <class T>
class IMeshBuffer : public virtual core::IReferenceCounted
{
    using MaterialType = typename std::conditional<std::is_same<T, asset::ICPUBuffer>::value, video::SCPUMaterial, video::SGPUMaterial>::type;

protected:
	virtual ~IMeshBuffer()
	{
        if (leakDebugger)
            leakDebugger->deregisterObj(this);

        if (meshLayout)
            meshLayout->drop();
	}

    MaterialType Material;
    core::aabbox3df boundingBox;
    IMeshDataFormatDesc<T>* meshLayout;
	//indices
	E_INDEX_TYPE indexType;
	int32_t baseVertex;
    uint64_t indexCount;
    size_t indexBufOffset;
    //
    size_t instanceCount;
    uint32_t baseInstance;
    //primitives
    E_PRIMITIVE_TYPE primitiveType;

    //debug
    core::LeakDebugger* leakDebugger;
public:
	//! Constructor.
	/**
	@param layout Pointer to descriptor of mesh data object. Will be grabbed.
	@param dbgr Pointer to leak debugger object.
	*/
	IMeshBuffer(IMeshDataFormatDesc<T>* layout=NULL, core::LeakDebugger* dbgr=NULL) : leakDebugger(dbgr)
	{
        if (leakDebugger)
            leakDebugger->registerObj(this);

	    meshLayout = layout;
	    if (meshLayout)
            meshLayout->grab();

        indexType = EIT_UNKNOWN;
        baseVertex = 0;
        indexCount = 0;
        indexBufOffset = 0;

	    primitiveType = EPT_TRIANGLES;

        instanceCount = 1;
        baseInstance = 0;
	}

	//! Access data descriptor objects.
	/** @returns data descriptor object. */
	inline IMeshDataFormatDesc<T>* getMeshDataAndFormat() {return meshLayout;}
	//! @copydoc getMeshDataAndFormat()
	inline const IMeshDataFormatDesc<T>* getMeshDataAndFormat() const {return meshLayout;}
	//! Sets data descriptor object.
	/**
	Grabs the new object and drops the old one.
	@param layout new descriptor object.
	*/
	inline void setMeshDataAndFormat(IMeshDataFormatDesc<T>* layout)
	{
	    if (layout)
            layout->grab();

	    if (meshLayout)
            meshLayout->drop();
        meshLayout = layout;
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
        if (meshLayout)
        {
            const T* mappedIndexBuf = meshLayout->getIndexBuffer();
            if (mappedIndexBuf)
            {
                switch (indexType)
                {
                    case EIT_16BIT:
                        return indexCount*2+indexBufOffset<mappedIndexBuf->getSize();
                    case EIT_32BIT:
                        return indexCount*4+indexBufOffset<mappedIndexBuf->getSize();
                    default:
                        return false;
                }
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


	inline const E_PRIMITIVE_TYPE& getPrimitiveType() const {return primitiveType;}
	inline void setPrimitiveType(const E_PRIMITIVE_TYPE& type)
	{
		primitiveType = type;
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

	//! Get material of this meshbuffer
	/** \return Material of this buffer */
	inline const MaterialType& getMaterial() const
	{
		return Material;
	}


	//! Get material of this meshbuffer
	/** \return Material of this buffer */
	inline MaterialType& getMaterial()
	{
		return Material;
	}
};

class ICPUMeshBuffer : public IMeshBuffer<asset::ICPUBuffer>, public asset::BlobSerializable, public asset::IAsset
{
    //vertices
    E_VERTEX_ATTRIBUTE_ID posAttrId;
public:
    ICPUMeshBuffer(core::LeakDebugger* dbgr = NULL) : IMeshBuffer<asset::ICPUBuffer>(NULL, dbgr), posAttrId(EVAI_ATTR0) {}

    virtual void* serializeToBlob(void* _stackPtr = NULL, const size_t& _stackSize = 0) const
    {
        return asset::CorrespondingBlobTypeFor<ICPUMeshBuffer>::type::createAndTryOnStack(this, _stackPtr, _stackSize);
    }

    virtual void convertToDummyObject() override {}
    virtual asset::IAsset::E_TYPE getAssetType() const override { return asset::IAsset::ET_SUB_MESH; }

    virtual size_t conservativeSizeEstimate() const override { return sizeof(IMeshBuffer<asset::ICPUBuffer>) + sizeof(posAttrId); }

    virtual E_MESH_BUFFER_TYPE getMeshBufferType() const { return EMBT_NOT_ANIMATED; }

    size_t calcVertexSize() const
    {
        if (!meshLayout)
            return 0u;

        size_t size = 0u;
        for (size_t i = 0; i < EVAI_COUNT; ++i)
            if (meshLayout->getMappedBuffer((E_VERTEX_ATTRIBUTE_ID)i))
                size += asset::getTexelOrBlockSize(meshLayout->getAttribFormat((E_VERTEX_ATTRIBUTE_ID)i));
        return size;
    }

    size_t calcVertexCount() const
    {
        size_t vertexCount = 0u;
        if (meshLayout && meshLayout->getIndexBuffer())
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
    inline const E_VERTEX_ATTRIBUTE_ID& getPositionAttributeIx() const { return posAttrId; }
    //! Sets id of position atrribute.
    inline void setPositionAttributeIx(const E_VERTEX_ATTRIBUTE_ID& attrId)
    {
        if (attrId >= EVAI_COUNT)
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
    virtual uint8_t* getAttribPointer(const E_VERTEX_ATTRIBUTE_ID& attrId) const
    {
        if (!meshLayout)
            return NULL;

        const asset::ICPUBuffer* mappedAttrBuf = meshLayout->getMappedBuffer(attrId);
        if (attrId >= EVAI_COUNT || !mappedAttrBuf)
            return NULL;

        int64_t ix = baseVertex;
        ix *= meshLayout->getMappedBufferStride(attrId);
        ix += meshLayout->getMappedBufferOffset(attrId);
        if (ix < 0 || static_cast<uint64_t>(ix) >= mappedAttrBuf->getSize())
            return NULL;

        return ((uint8_t*)mappedAttrBuf->getPointer()) + ix;
    }

    static bool getAttribute(core::vectorSIMDf& output, const void* src, asset::E_FORMAT format)
    {
        if (!src)
            return false;

        bool scaled = false;
        if (!asset::isNormalizedFormat(format) && !asset::isFloatingPointFormat(format) && !(scaled = asset::isScaledFormat(format)))
            return false;

        if (!scaled)
        {
            double output64[4]{ 0., 0., 0., 1. };
            video::decodePixels<double>(format, &src, output64, 0u, 0u);
            std::copy(output64, output64+4, output.pointer);
        }
        else
        {
            if (asset::isSignedFormat(format))
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
    virtual bool getAttribute(core::vectorSIMDf& output, const E_VERTEX_ATTRIBUTE_ID& attrId, size_t ix) const
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

    static bool getAttribute(uint32_t* output, const void* src, asset::E_FORMAT format)
    {
        if (!src)
            return false;

        bool scaled = false;
        if ((scaled = asset::isScaledFormat(format)) || asset::isIntegerFormat(format))
        {
            if (asset::isSignedFormat(format))
            {
                int64_t output64[4]{0, 0, 0, 1};
                video::decodePixels<int64_t>(scaled ? impl::getCorrespondingIntegerFmt(format) : format, &src, output64, 0u, 0u);
                for (uint32_t i = 0u; i < asset::getFormatChannelCount(format); ++i)
                    output[i] = output64[i];
            }
            else
            {
                uint64_t output64[4]{0u, 0u, 0u, 1u};
                video::decodePixels<uint64_t>(scaled ? impl::getCorrespondingIntegerFmt(format) : format, &src, output64, 0u, 0u);
                for (uint32_t i = 0u; i < asset::getFormatChannelCount(format); ++i)
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
    virtual bool getAttribute(uint32_t* output, const E_VERTEX_ATTRIBUTE_ID& attrId, size_t ix) const
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

    static bool setAttribute(core::vectorSIMDf input, void* dst, asset::E_FORMAT format)
    {
        bool scaled = false;
        if (!dst || (!asset::isFloatingPointFormat(format) && !asset::isNormalizedFormat(format) && !(scaled = asset::isScaledFormat(format))))
            return false;

        double input64[4];
        for (uint32_t i = 0u; i < 4u; ++i)
            input64[i] = input.pointer[i];

        if (!scaled)
            video::encodePixels<double>(format, dst, input64);
        else
        {
            if (asset::isSignedFormat(format))
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
    virtual bool setAttribute(core::vectorSIMDf input, const E_VERTEX_ATTRIBUTE_ID& attrId, size_t ix) const
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

    static bool setAttribute(const uint32_t* _input, void* dst, asset::E_FORMAT format)
    {
        const bool scaled = asset::isScaledFormat(format);
        if (!dst || !(scaled || asset::isIntegerFormat(format)))
            return false;
        uint8_t* vxPtr = (uint8_t*)dst;

        if (asset::isSignedFormat(format))
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
    virtual bool setAttribute(const uint32_t* _input, const E_VERTEX_ATTRIBUTE_ID& attrId, size_t ix) const
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
        if (posAttrId >= EVAI_COUNT || !mappedAttrBuf)
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
                case EIT_32BIT:
                    ix = ((uint32_t*)indices)[j];
                    break;
                case EIT_16BIT:
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

namespace std
{
    template <>
    struct hash<irr::asset::E_VERTEX_ATTRIBUTE_ID>
    {
        std::size_t operator()(const irr::asset::E_VERTEX_ATTRIBUTE_ID& k) const noexcept { return k; }
    };
}

#endif //__IRR_I_CPU_MESH_BUFFER_H_INCLUDED__