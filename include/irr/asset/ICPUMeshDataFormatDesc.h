#ifndef __IRR_I_CPU_MESH_DATA_FORMAT_DESC_H_INCLUDED__
#define __IRR_I_CPU_MESH_DATA_FORMAT_DESC_H_INCLUDED__

//! THIS FILE WILL GET KILLED AND REPLACED BY DESCRIPTOR LAYOUT AND GRAPHICS PIPELINES

#include "irr/core/core.h"
#include "irr/asset/ICPUBuffer.h"
#include "irr/asset/format/decodePixels.h"
#include "irr/asset/format/encodePixels.h"
#include "irr/asset/bawformat/Blob.h"
#include "irr/asset/bawformat/BlobSerializable.h"
#include "irr/asset/bawformat/blobs/MeshDataFormatBlob.h"

namespace irr
{
namespace asset
{

//! Available vertex attribute ids
/** As of 2018 most OpenGL and Vulkan implementations support 16 attributes (some CAD GPUs more) */
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

template <class T>
class IMeshDataFormatDesc : public virtual core::IReferenceCounted
{
    protected:
		//! Read https://www.khronos.org/opengl/wiki/Vertex_Specification for understanding of attribute IDs, indices, attribute formats etc.
        E_FORMAT attrFormat[EVAI_COUNT];
        uint32_t attrStride[EVAI_COUNT];
        size_t attrOffset[EVAI_COUNT];
        uint32_t attrDivisor;

        // vertices
        core::smart_refctd_ptr<T> mappedAttrBuf[EVAI_COUNT];
        // indices
        core::smart_refctd_ptr<T> mappedIndexBuf;

        virtual ~IMeshDataFormatDesc() {}
    public:
        //! Default constructor.
        IMeshDataFormatDesc()
        {
            for (size_t i=0; i<EVAI_COUNT; i++)
            {
                attrFormat[i] = EF_R32G32B32A32_SFLOAT;
                attrStride[i] = 16;
                attrOffset[i] = 0;
            }
            attrDivisor = 0u;
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

        inline void setIndexBuffer(core::smart_refctd_ptr<T>&& ixbuf)
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
            mappedIndexBuf = std::move(ixbuf);
        }

        inline const T* getIndexBuffer() const
        {
			return mappedIndexBuf.get();
        }


        //! remember that the divisor needs to be <=0x1u<<_IRR_VAO_maxATTRIB_DIVISOR_BITS
        virtual void setVertexAttrBuffer(core::smart_refctd_ptr<T>&& attrBuf, E_VERTEX_ATTRIBUTE_ID attrId, E_FORMAT format, size_t stride=0, size_t offset=0, uint32_t divisor=0) = 0;

        inline const T* getMappedBuffer(E_VERTEX_ATTRIBUTE_ID attrId) const
        {
			assert(attrId < EVAI_COUNT);
			return mappedAttrBuf[attrId].get();
        }

        inline E_FORMAT getAttribFormat(E_VERTEX_ATTRIBUTE_ID attrId) const
        {
            assert(attrId<EVAI_COUNT);
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

        inline void swapVertexAttrBuffer(core::smart_refctd_ptr<T>&& attrBuf, E_VERTEX_ATTRIBUTE_ID attrId)
        {
            swapVertexAttrBuffer(std::move(attrBuf), attrId, attrOffset[attrId], attrStride[attrId]);
        }

        inline void swapVertexAttrBuffer(core::smart_refctd_ptr<T>&& attrBuf, E_VERTEX_ATTRIBUTE_ID attrId, size_t newOffset)
        {
            swapVertexAttrBuffer(std::move(attrBuf), attrId, newOffset, attrStride[attrId]);
        }

        inline void swapVertexAttrBuffer(core::smart_refctd_ptr<T>&& attrBuf, E_VERTEX_ATTRIBUTE_ID attrId, size_t newOffset, size_t newStride)
        {
            if (!attrBuf)
                return;

            mappedAttrBuf[attrId] = std::move(attrBuf);
            attrOffset[attrId] = newOffset;
            attrStride[attrId] = newStride;
        }
};


class ICPUMeshDataFormatDesc final : public IMeshDataFormatDesc<ICPUBuffer>, public BlobSerializable, public IAsset
{
    protected:
		friend class ICPUMeshBuffer; // only for access to `getIndexBuffer` and `getMappedBuffer`
		inline ICPUBuffer* getIndexBuffer()
		{
			return mappedIndexBuf.get();
		}
		inline ICPUBuffer* getMappedBuffer(E_VERTEX_ATTRIBUTE_ID attrId)
		{
			assert(attrId < EVAI_COUNT);
			return mappedAttrBuf[attrId].get();
		}

	    ~ICPUMeshDataFormatDesc()
	    {
	    }
	public:
		inline void* serializeToBlob(void* _stackPtr = nullptr, const size_t& _stackSize = 0) const override
		{
			// @crisspl sure that IMeshDataFormatDesc<ICPUBuffer> is the right type to query here?
			return CorrespondingBlobTypeFor<IMeshDataFormatDesc<ICPUBuffer> >::type::createAndTryOnStack(static_cast<const IMeshDataFormatDesc<ICPUBuffer>*>(this), _stackPtr, _stackSize);
		}

        size_t conservativeSizeEstimate() const override
        {
			// @crisspl sure that IMeshDataFormatDesc<ICPUBuffer> is the right type to query here?
            return CorrespondingBlobTypeFor<IMeshDataFormatDesc<ICPUBuffer>>::type::calcBlobSizeForObj(this);
        }

        inline void convertToDummyObject() override
        {
        }

        inline IAsset::E_TYPE getAssetType() const override { return IAsset::ET_MESH_DATA_DESCRIPTOR; }


		inline const ICPUBuffer* getIndexBuffer() const
		{
			return IMeshDataFormatDesc<ICPUBuffer>::getIndexBuffer();
		}
		inline const ICPUBuffer* getMappedBuffer(E_VERTEX_ATTRIBUTE_ID attrId) const
		{
			return IMeshDataFormatDesc<ICPUBuffer>::getMappedBuffer(attrId);
		}

        //! remember that the divisor must be 0 or 1
        inline void setVertexAttrBuffer(core::smart_refctd_ptr<ICPUBuffer>&& attrBuf, E_VERTEX_ATTRIBUTE_ID attrId, E_FORMAT format, size_t stride=0, size_t offset=0, uint32_t divisor=0) override
        {
            assert(attrId<EVAI_COUNT);
            assert(divisor<=1u);

            attrDivisor &= ~(divisor<<attrId);

            if (attrBuf)
            {
                attrFormat[attrId] = format;
                // Don't get confused by `getTexelOrBlockBytesize` name. All vertex attrib, color, etc. are maintained with single enum E_FORMAT and its naming conventions is color-like, and so are related functions. Whole story began from Vulkan's VkFormat.
                attrStride[attrId] = stride!=0 ? stride : getTexelOrBlockBytesize(format);
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

            mappedAttrBuf[attrId] = std::move(attrBuf);
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

#endif //__IRR_I_CPU_MESH_DATA_FORMAT_DESC_H_INCLUDED__