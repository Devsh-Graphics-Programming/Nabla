#ifndef __IRR_I_MESH_BUFFER_H_INCLUDED__
#define __IRR_I_MESH_BUFFER_H_INCLUDED__

#include "irr/asset/ICPUMeshDataFormatDesc.h"
#include "SMaterial.h"

namespace irr
{
namespace asset
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

enum E_MESH_BUFFER_TYPE
{
    EMBT_UNKNOWN = 0,
    EMBT_NOT_ANIMATED,
    EMBT_ANIMATED_FRAME_BASED,
    EMBT_ANIMATED_SKINNED
};

template <class T>
class IMeshBuffer : public virtual core::IReferenceCounted
{
    using MaterialType = typename std::conditional<std::is_same<T, ICPUBuffer>::value, video::SCPUMaterial, video::SGPUMaterial>::type;

protected:
	virtual ~IMeshBuffer()
	{
        if (leakDebugger)
            leakDebugger->deregisterObj(this);
	}

    MaterialType Material;
    core::aabbox3df boundingBox;
	core::smart_refctd_ptr<IMeshDataFormatDesc<T>> meshLayout;
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
    core::CLeakDebugger* leakDebugger;
public:
	//! Constructor.
	/**
	@param layout Smart pointer to descriptor of mesh data object.
	@param dbgr Pointer to leak debugger object.
	*/
	IMeshBuffer(IMeshDataFormatDesc<T>* layout=nullptr, core::CLeakDebugger* dbgr=nullptr) : Material(), boundingBox(),
						meshLayout(layout), indexType(EIT_UNKNOWN), baseVertex(0), indexCount(0u), indexBufOffset(0ull),
						instanceCount(1ull), baseInstance(0u), primitiveType(EPT_TRIANGLES), leakDebugger(dbgr)
	{
		if (leakDebugger)
			leakDebugger->registerObj(this);
	}

	//! Access data descriptor objects.
	/** @returns data descriptor object. */
	inline IMeshDataFormatDesc<T>* getMeshDataAndFormat() {return meshLayout.get();}
	//! @copydoc getMeshDataAndFormat()
	inline const IMeshDataFormatDesc<T>* getMeshDataAndFormat() const {return meshLayout.get();}

	//! Sets data descriptor object.
	/**
	@param layout new descriptor object.
	*/
	inline void setMeshDataAndFormat(core::smart_refctd_ptr<IMeshDataFormatDesc<T>>&& layout)
	{
        meshLayout = std::move(layout);
	}

	//! Get type of index data which is stored in this meshbuffer.
	/** \return Index type of this buffer. */
	inline const E_INDEX_TYPE& getIndexType() const {return indexType;}
	inline void setIndexType(const E_INDEX_TYPE& type)
	{
		indexType = type;
	}

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

}}

#endif //__IRR_I_CPU_MESH_BUFFER_H_INCLUDED__