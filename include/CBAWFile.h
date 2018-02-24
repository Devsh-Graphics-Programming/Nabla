// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_BAW_FILE_H_INCLUDED__
#define __IRR_BAW_FILE_H_INCLUDED__

#include <map>
#include <vector>

#include "stdint.h"
#include "irrTypes.h"
#include "irrArray.h"
#include "aabbox3d.h"
#include "SMaterial.h"

namespace irr {

namespace core
{
	class ICPUBuffer;
}

namespace scene
{
	class ICPUMeshBuffer;
	class ICPUMesh;
	class ISceneManager; 
	class ICPUSkinnedMesh;
	class SCPUSkinMeshBuffer;
	template<typename> class IMeshDataFormatDesc;
	class CFinalBoneHierarchy;
}
namespace io
{
	class IFileSystem;
}
namespace video
{
	class IVirtualTexture;
}

namespace core
{
	struct BlobLoadingParams;

#include "irrpack.h"
	//! Cast pointer to block of blob-headers to BlobHeader* and easily iterate and/or access members
	struct BlobHeaderV0
	{
		uint32_t blobSize;
		uint32_t blobSizeDecompr;

		uint8_t compressionType;
		uint8_t dummy[3];
		uint32_t blobType;
		uint64_t handle;

		uint64_t blobHash[4];

		//! Assigns sizes and calculates hash of data.
		void finalize(const void* _notCompressedData, const void* _data, size_t _sizeDecompr, size_t _sizeCompr, uint8_t _comprType);
		//! Calculates hash from `_data` and compares to current one (`blobHash` member).
		bool validate(const void* _decomprData) const;
	} PACK_STRUCT;

	//! Cast pointer to (first byte of) file buffer to BAWFile*. 256bit header must be first member (start of file).
	struct BAWFileV0 {
		//! 32-byte BaW binary format header, currently equal to "IrrlichtBaW BinaryFile" (and the rest filled with zeroes).
		//! Also: last 8 bytes of file header is file-version number.
		uint64_t fileHeader[4];

		//! Number of internal blobs
		uint32_t numOfInternalBlobs;
		//! Blobs offsets counted from after blob-headers block
		uint32_t blobOffsets[1];

		size_t calcOffsetsOffset() const { return sizeof(fileHeader) + sizeof(numOfInternalBlobs); }
		size_t calcHeadersOffset() const { return calcOffsetsOffset() + numOfInternalBlobs*sizeof(blobOffsets[0]); }
		size_t calcBlobsOffset() const { return calcHeadersOffset() + numOfInternalBlobs*sizeof(BlobHeaderV0); }
	} PACK_STRUCT;
#include "irrunpack.h"

	struct Blob
	{
		//! Coding method of blob's data enumeration
		enum E_BLOB_CODING_TYPE
		{
			EBCT_RAW = 0,
			EBCT_AES128_GCM,
			EBCT_LZ4,
			EBCT_LZ4_AES128_GCM,
			EBCT_LZMA,
			EBCT_LZMA_AES128_GCM,
			EBCT_COUNT
		};
		//! Type of blob enumeration
		enum E_BLOB_TYPE
		{
			EBT_MESH = 0,
			EBT_SKINNED_MESH,
			EBT_MESH_BUFFER,
			EBT_SKINNED_MESH_BUFFER,
			EBT_RAW_DATA_BUFFER,
			EBT_DATA_FORMAT_DESC,
			EBT_FINAL_BONE_HIERARCHY,
			EBT_TEXTURE_PATH,
			EBT_COUNT
		};

		void* getData() { return this; }
		const void* getData() const { return this; }
	};

	template<template<typename, typename> class SizingT, typename B, typename T>
	struct SizedBlob
	{
	protected: // not intended for direct usage
		SizedBlob() {}
		~SizedBlob() {}

	public:
		static size_t calcBlobSizeForObj(const T*);// { return sizeof(B); }

		//! Utility function for making blobs
		/**
		@param _obj Pointer to the object for which the blob will be made.
		@param _stackPtr Pointer to stack memory, usually you'd declare it as `uint8_t _stackPtr[_size]`.
		@param _size The size of the stack memory available.
		@return Pointer to created blob, if it does not equal _stackPtr then new memory was malloc'd which needs to be free'd.
		*/
		static B* createAndTryOnStack(const T* _obj, void* _stackPtr=NULL, const size_t& _size=0)
		{
			const size_t actualObjSize = calcBlobSizeForObj(_obj);
			void* mem;
			if (!_stackPtr || actualObjSize > _size)
				mem = malloc(actualObjSize);
			else if (_stackPtr && _size >= actualObjSize)
				mem = _stackPtr;
			else
				mem = NULL;

			if (!mem)
				return (B*)mem;
			new (mem) B(_obj);
			return (B*)mem;
		}
	};

	template<typename B, typename T>
	struct VariableSizeBlob : SizedBlob<VariableSizeBlob, B, T>
	{
	protected: // not intended for direct usage
        VariableSizeBlob() {}
        ~VariableSizeBlob() {}
	};

	template<typename B, typename T>
	struct FixedSizeBlob : SizedBlob<FixedSizeBlob, B, T>
	{
	protected: // not intended for direct usage
		FixedSizeBlob() {}
		~FixedSizeBlob() {}
	};

	template<typename B, typename T>
	struct TypedBlob : Blob
	{
		static std::vector<uint64_t> getNeededDeps(const void* _blob);
		static void* instantiateEmpty(const void* _blob, size_t _blobSize, const BlobLoadingParams& _params);
		static void* finalize(void* _obj, const void* _blob, size_t _blobSize, std::map<uint64_t, void*>& _deps, const BlobLoadingParams& _params);
		static void releaseObj(const void* _obj);
	};

	struct RawBufferBlobV0 : TypedBlob<RawBufferBlobV0, ICPUBuffer>, FixedSizeBlob<RawBufferBlobV0, ICPUBuffer>
	{};

	struct TexturePathBlobV0 : TypedBlob<TexturePathBlobV0, video::IVirtualTexture>, FixedSizeBlob<TexturePathBlobV0, video::IVirtualTexture>
	{};

#include "irrpack.h"
	//! Utility struct. Cast blob pointer to MeshBlob* to make life easier.
	struct MeshBlobV0 : VariableSizeBlob<MeshBlobV0,scene::ICPUMesh>, TypedBlob<MeshBlobV0, scene::ICPUMesh>
	{
		friend struct SizedBlob<core::VariableSizeBlob, MeshBlobV0, scene::ICPUMesh>;
	private:
            //! WARNING: Constructor saves only bounding box and mesh buffer count (not mesh buffer pointers)
		explicit MeshBlobV0(const scene::ICPUMesh* _mesh);

	public:
        aabbox3df box;
        uint32_t meshBufCnt;
        uint64_t meshBufPtrs[1];
	} PACK_STRUCT;

	//! Utility struct. Cast blob pointer to MeshBlob* to make life easier.
	struct SkinnedMeshBlobV0 : VariableSizeBlob<SkinnedMeshBlobV0,scene::ICPUSkinnedMesh>, TypedBlob<SkinnedMeshBlobV0, scene::ICPUSkinnedMesh>
	{
		friend struct SizedBlob<core::VariableSizeBlob, SkinnedMeshBlobV0, scene::ICPUSkinnedMesh>;
	private:
        explicit SkinnedMeshBlobV0(const scene::ICPUSkinnedMesh* _sm);

	public:
        uint64_t boneHierarchyPtr;
        aabbox3df box;
        uint32_t meshBufCnt;
        uint64_t meshBufPtrs[1];
	} PACK_STRUCT;

	//! Simple struct of essential data of ICPUMeshBuffer that has to be exported
	struct MeshBufferBlobV0 : TypedBlob<MeshBufferBlobV0, scene::ICPUMeshBuffer>, FixedSizeBlob<MeshBufferBlobV0, scene::ICPUMeshBuffer>
	{
		//! Constructor filling all members
		explicit MeshBufferBlobV0(const scene::ICPUMeshBuffer*);

		video::SMaterial mat;
		core::aabbox3df box;
		uint64_t descPtr;
		uint32_t indexType;
		uint32_t baseVertex;
		uint64_t indexCount;
		size_t indexBufOffset;
		size_t instanceCount;
		uint32_t baseInstance;
		uint32_t primitiveType;
		uint32_t posAttrId;
	} PACK_STRUCT;

	struct SkinnedMeshBufferBlobV0 : TypedBlob<SkinnedMeshBufferBlobV0, scene::SCPUSkinMeshBuffer>, FixedSizeBlob<SkinnedMeshBufferBlobV0, scene::SCPUSkinMeshBuffer>
	{
		//! Constructor filling all members
		explicit SkinnedMeshBufferBlobV0(const scene::SCPUSkinMeshBuffer*);

		video::SMaterial mat;
		core::aabbox3df box;
		uint64_t descPtr;
		uint32_t indexType;
		uint32_t baseVertex;
		uint64_t indexCount;
		size_t indexBufOffset;
		size_t instanceCount;
		uint32_t baseInstance;
		uint32_t primitiveType;
		uint32_t posAttrId;
		uint32_t indexValMin;
		uint32_t indexValMax;
		uint32_t maxVertexBoneInfluences;
	} PACK_STRUCT;

	//! Simple struct of essential data of ICPUMeshDataFormatDesc that has to be exported
	struct MeshDataFormatDescBlobV0 : TypedBlob<MeshDataFormatDescBlobV0, scene::IMeshDataFormatDesc<core::ICPUBuffer> >, FixedSizeBlob<MeshDataFormatDescBlobV0, scene::IMeshDataFormatDesc<core::ICPUBuffer> >
	{
	private:
		enum { VERTEX_ATTRIB_CNT = 16 };
	public:
		//! Constructor filling all members
		explicit MeshDataFormatDescBlobV0(const scene::IMeshDataFormatDesc<core::ICPUBuffer>*);

		uint32_t cpa[VERTEX_ATTRIB_CNT];
		uint32_t attrType[VERTEX_ATTRIB_CNT];
		size_t attrStride[VERTEX_ATTRIB_CNT];
		size_t attrOffset[VERTEX_ATTRIB_CNT];
		uint32_t attrDivisor[VERTEX_ATTRIB_CNT];
		uint64_t attrBufPtrs[VERTEX_ATTRIB_CNT];
		uint64_t idxBufPtr;
	} PACK_STRUCT;

	struct FinalBoneHierarchyBlobV0 : VariableSizeBlob<FinalBoneHierarchyBlobV0,scene::CFinalBoneHierarchy>, TypedBlob<FinalBoneHierarchyBlobV0, scene::CFinalBoneHierarchy>
	{
		friend struct SizedBlob<core::VariableSizeBlob, FinalBoneHierarchyBlobV0, scene::CFinalBoneHierarchy>;
	private:
		FinalBoneHierarchyBlobV0(const scene::CFinalBoneHierarchy* _fbh);

	public:
		//! Used for creating a blob. Calculates offset of the block of blob resulting from exporting `*_fbh` object.
		/** @param _fbh Pointer to object on the basis of which offset of the block will be calculated.
		@return Offset where the block must begin (used while writing blob data (exporting)).
		*/
        static size_t calcBonesOffset(const scene::CFinalBoneHierarchy* _fbh);
		//! @copydoc calcBonesOffset(const scene::CFinalBoneHierarchy*)
        static size_t calcLevelsOffset(const scene::CFinalBoneHierarchy* _fbh);
		//! @copydoc calcBonesOffset(const scene::CFinalBoneHierarchy*)
        static size_t calcKeyFramesOffset(const scene::CFinalBoneHierarchy* _fbh);
		//! @copydoc calcBonesOffset(const scene::CFinalBoneHierarchy*)
        static size_t calcInterpolatedAnimsOffset(const scene::CFinalBoneHierarchy* _fbh);
		//! @copydoc calcBonesOffset(const scene::CFinalBoneHierarchy*)
        static size_t calcNonInterpolatedAnimsOffset(const scene::CFinalBoneHierarchy* _fbh);
		//! @copydoc calcBonesOffset(const scene::CFinalBoneHierarchy*)
        static size_t calcBoneNamesOffset(const scene::CFinalBoneHierarchy* _fbh);

		//! Used for creating a blob. Calculates size (in bytes) of the block of blob resulting from exporting `*_fbh` object.
		/** @param _fbh Pointer to object on the basis of which size of the block will be calculated.
		@return Size of the block calculated on the basis of data containted by *_fbh object.
		*/
        static size_t calcBonesByteSize(const scene::CFinalBoneHierarchy* _fbh);
		//! @copydoc calcBonesByteSize(const scene::CFinalBoneHierarchy*)
        static size_t calcLevelsByteSize(const scene::CFinalBoneHierarchy* _fbh);
		//! @copydoc calcBonesByteSize(const scene::CFinalBoneHierarchy*)
        static size_t calcKeyFramesByteSize(const scene::CFinalBoneHierarchy* _fbh);
		//! @copydoc calcBonesByteSize(const scene::CFinalBoneHierarchy*)
        static size_t calcInterpolatedAnimsByteSize(const scene::CFinalBoneHierarchy* _fbh);
		//! @copydoc calcBonesByteSize(const scene::CFinalBoneHierarchy*)
        static size_t calcNonInterpolatedAnimsByteSize(const scene::CFinalBoneHierarchy* _fbh);
		//! @copydoc calcBonesByteSize(const scene::CFinalBoneHierarchy*)
        static size_t calcBoneNamesByteSize(const scene::CFinalBoneHierarchy* _fbh);

		//! Used for importing (unpacking) blob. Calculates offset of the block.
		/** @returns Offset of the block based on corresponding member of the blob object.
		*/
		size_t calcBonesOffset() const;
		//! @copydoc calcBonesOffset()
		size_t calcLevelsOffset() const;
		//! @copydoc calcBonesOffset()
		size_t calcKeyFramesOffset() const;
		//! @copydoc calcBonesOffset()
		size_t calcInterpolatedAnimsOffset() const;
		//! @copydoc calcBonesOffset()
		size_t calcNonInterpolatedAnimsOffset() const;
		//! @copydoc calcBonesOffset()
		size_t calcBoneNamesOffset() const;

		//! Used for importing (unpacking) blob. Calculates size (in bytes) of the block.
		/** @returns Size of the block based on corresponding member of the blob object.
		*/
		size_t calcBonesByteSize() const;
		//! @copydoc calcBonesByteSize()
		size_t calcLevelsByteSize() const;
		//! @copydoc calcBonesByteSize()
		size_t calcKeyFramesByteSize() const;
		//! @copydoc calcBonesByteSize()
		size_t calcInterpolatedAnimsByteSize() const;
		//! @copydoc calcBonesByteSize()
		size_t calcNonInterpolatedAnimsByteSize() const;
		// size of bone names is not dependent of any of 'count variables'. Since it's the last block its size can be calculated by {blobSize - boneNamesOffset}.

        size_t boneCount;
        size_t numLevelsInHierarchy;
        size_t keyframeCount;
	} PACK_STRUCT;
#include "irrunpack.h"

	template<typename>
	struct CorrespondingBlobTypeFor;
	template<>
	struct CorrespondingBlobTypeFor<ICPUBuffer> { typedef RawBufferBlobV0 type; };
	template<>
	struct CorrespondingBlobTypeFor<scene::ICPUMesh> { typedef MeshBlobV0 type; };
	template<>
	struct CorrespondingBlobTypeFor<scene::ICPUSkinnedMesh> { typedef SkinnedMeshBlobV0 type; };
	template<>
	struct CorrespondingBlobTypeFor<scene::ICPUMeshBuffer> { typedef MeshBufferBlobV0 type; };
	template<>
	struct CorrespondingBlobTypeFor<scene::SCPUSkinMeshBuffer> { typedef SkinnedMeshBufferBlobV0 type; };
	template<>
	struct CorrespondingBlobTypeFor<scene::IMeshDataFormatDesc<core::ICPUBuffer> > { typedef MeshDataFormatDescBlobV0 type; };
	template<>
	struct CorrespondingBlobTypeFor<scene::CFinalBoneHierarchy> { typedef FinalBoneHierarchyBlobV0 type; };
	template<>
	struct CorrespondingBlobTypeFor<video::IVirtualTexture> { typedef TexturePathBlobV0 type; };

	template<typename T>
	typename CorrespondingBlobTypeFor<T>::type* toBlobPtr(const void* _blob)
	{
		return (typename CorrespondingBlobTypeFor<T>::type*)_blob;
	}

	class BlobSerializable
	{
	public:
		virtual ~BlobSerializable() {}

		virtual void* serializeToBlob(void* _stackPtr=NULL, const size_t& _stackSize=0) const = 0;
	};

}} // irr::core

#endif
