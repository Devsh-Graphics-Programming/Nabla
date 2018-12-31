// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_MESH_BUFFER_H_INCLUDED__
#define __I_MESH_BUFFER_H_INCLUDED__

#include <algorithm>

#include "ITransformFeedback.h"
#include "SMaterial.h"
#include "aabbox3d.h"
#include "irr/asset/ICPUBuffer.h"
#include "IGPUBuffer.h"
#include "vectorSIMD.h"
#include "coreutil.h"
#include "irr/asset/bawformat/CBAWFile.h"
#include "assert.h"
#include "irr/video/EColorFormat.h"

namespace irr
{
namespace scene
{

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
            video::E_FORMAT attrFormat[EVAI_COUNT];
            size_t attrStride[EVAI_COUNT];
            size_t attrOffset[EVAI_COUNT];
            uint32_t attrDivisor;

            //vertices
            T* mappedAttrBuf[scene::EVAI_COUNT];
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
                    attrFormat[i] = video::EF_R32G32B32A32_SFLOAT;
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
        #ifdef _DEBUG
                if (size<0x7fffffffffffffffuLL&&ixbuf&&(ixbuf->getSize()>size+offset)) //not that easy to check
                {
                    os::Printer::log("MeshBuffer map index buffer overflow!\n",ELL_ERROR);
                    return;
                }
        #endif // _DEBUG
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
            virtual void setVertexAttrBuffer(T* attrBuf, E_VERTEX_ATTRIBUTE_ID attrId, video::E_FORMAT format, size_t stride=0, size_t offset=0, uint32_t divisor=0) = 0;

            inline const T* getMappedBuffer(scene::E_VERTEX_ATTRIBUTE_ID attrId) const
            {
                assert(attrId<EVAI_COUNT);
                return mappedAttrBuf[attrId];
            }

            inline video::E_FORMAT getAttribFormat(E_VERTEX_ATTRIBUTE_ID attrId) const
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

            inline const size_t& getMappedBufferStride(E_VERTEX_ATTRIBUTE_ID attrId) const
            {
                assert(attrId<EVAI_COUNT);
                return attrStride[attrId];
            }

            inline uint32_t getAttribDivisor(E_VERTEX_ATTRIBUTE_ID attrId) const
            {
                assert(attrId<EVAI_COUNT);
                return (attrDivisor>>attrId)&1u;
            }

            inline void swapVertexAttrBuffer(T* attrBuf, scene::E_VERTEX_ATTRIBUTE_ID attrId)
            {
                swapVertexAttrBuffer(attrBuf, attrId, attrOffset[attrId], attrStride[attrId]);
            }

            inline void swapVertexAttrBuffer(T* attrBuf, scene::E_VERTEX_ATTRIBUTE_ID attrId, size_t newOffset)
            {
                swapVertexAttrBuffer(attrBuf, attrId, newOffset, attrStride[attrId]);
            }

            inline void swapVertexAttrBuffer(T* attrBuf, scene::E_VERTEX_ATTRIBUTE_ID attrId, size_t newOffset, size_t newStride)
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


	class ICPUMeshDataFormatDesc : public IMeshDataFormatDesc<asset::ICPUBuffer>, asset::BlobSerializable
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

            //! remember that the divisor must be 0 or 1
            void setVertexAttrBuffer(asset::ICPUBuffer* attrBuf, E_VERTEX_ATTRIBUTE_ID attrId, video::E_FORMAT format, size_t stride=0, size_t offset=0, uint32_t divisor=0) override
            {
                assert(attrId<EVAI_COUNT);
                assert(divisor<=1u);

                attrDivisor &= ~(divisor<<attrId);

                if (attrBuf)
                {
                    attrBuf->grab();

                    attrFormat[attrId] = format;
                    // Don't get confused by `getTexelOrBlockSize` name. All vertex attrib, color, etc. are maintained with single enum E_FORMAT and its naming conventions is color-like, and so are related functions. Whole story began from Vulkan's VkFormat.
                    attrStride[attrId] = stride!=0 ? stride : video::getTexelOrBlockSize(format);
                    attrOffset[attrId] = offset;
                    attrDivisor |= (divisor<<attrId);
                }
                else
                {
                    attrFormat[attrId] = video::EF_R32G32B32A32_SFLOAT;
                    attrStride[attrId] = 16;
                    attrOffset[attrId] = 0;
                    //attrDivisor &= ~(1u<<attrId); //cleared before if
                }

                if (mappedAttrBuf[attrId])
                    mappedAttrBuf[attrId]->drop();
                mappedAttrBuf[attrId] = attrBuf;
            }
	};

	class IGPUMeshDataFormatDesc : public IMeshDataFormatDesc<video::IGPUBuffer>
	{
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
#ifdef _DEBUG
            if (size<0x7fffffffffffffffuLL&&ixbuf&&(ixbuf->getSize()>size+offset))
            {
                os::Printer::log("MeshBuffer map vertex buffer overflow!\n",ELL_ERROR);
                return;
            }
#endif // _DEBUG
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

	class IGPUMeshBuffer : public IMeshBuffer<video::IGPUBuffer>
	{
            video::ITransformFeedback* attachedXFormFeedback;
            uint32_t attachedXFormFeedbackStream;
        protected:
            virtual ~IGPUMeshBuffer()
            {
                if (attachedXFormFeedback)
                    attachedXFormFeedback->drop();
            }
        public:
            IGPUMeshBuffer(core::LeakDebugger* dbgr=NULL) : IMeshBuffer<video::IGPUBuffer>(NULL,dbgr), attachedXFormFeedback(NULL), attachedXFormFeedbackStream(0) {}

            inline void setIndexCountFromXFormFeedback(video::ITransformFeedback* xformFeedback, const uint32_t & stream)
            {
                attachedXFormFeedbackStream = stream;


                if (xformFeedback==attachedXFormFeedback)
                    return;

                if (!xformFeedback)
                {
                    if (attachedXFormFeedback)
                        attachedXFormFeedback->drop();

                    attachedXFormFeedback = NULL;
                    return;
                }

                xformFeedback->grab();
                if (attachedXFormFeedback)
                    attachedXFormFeedback->drop();
                attachedXFormFeedback = xformFeedback;

                indexType = EIT_UNKNOWN;
                indexCount = 0;
            }

            inline video::ITransformFeedback* getXFormFeedback() const {return attachedXFormFeedback;}

            inline const uint32_t& getXFormFeedbackStream() const {return attachedXFormFeedbackStream;}

            bool isIndexCountGivenByXFormFeedback() const {return attachedXFormFeedback!=NULL;}
	};

#include "irr/irrpack.h"
    struct SkinnedVertexIntermediateData
    {
        SkinnedVertexIntermediateData()
        {
            memset(this,0,20);
        }
        uint8_t boneIDs[4];
        float boneWeights[4];
    } PACK_STRUCT;

    struct SkinnedVertexFinalData
    {
        public:
            uint8_t boneIDs[4];
            uint32_t boneWeights; //ECT_NORMALIZED_UNSIGNED_INT_2_10_10_10_REV
    } PACK_STRUCT;
#include "irr/irrunpack.h"

} // end namespace scene
} // end namespace irr

namespace std
{
	template <>
	struct hash<irr::scene::E_VERTEX_ATTRIBUTE_ID>
	{
		std::size_t operator()(const irr::scene::E_VERTEX_ATTRIBUTE_ID& k) const noexcept {return k;}
	};
}

#endif


