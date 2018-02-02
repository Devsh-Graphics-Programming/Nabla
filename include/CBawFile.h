// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_BAW_FILE_H_INCLUDED__
#define __IRR_BAW_FILE_H_INCLUDED__

#include "stdint.h"
#include "IMesh.h"

namespace irr { 

namespace scene 
{
	class ICPUSkinnedMesh;
	class SCPUSkinMeshBuffer;
	class CFinalBoneHierarchy;
}

namespace core
{

#include "irrpack.h"
	//! Cast pointer to block of blob-headers to BlobHeader* and easily iterate and/or access members
	struct BlobHeaderV1
	{
		uint32_t blobSize;
		uint32_t blobSizeDecompr;

		uint8_t compressionType;
		uint8_t dummy[3];
		uint32_t blobType;
		uint64_t handle;

		uint64_t blobHash[4];

		//! Assigns size and calculates hash of data
		void finalize(const void* _data, size_t _size);
	} PACK_STRUCT;

	//! Cast pointer to (first byte of) file buffer to BaWFile*. 256bit header must be first member (start of file).
	struct BawFileV1 {
		//! 32-byte BaW binary format header, currently equal to "IrrlichtBaW BinaryFile" (and the rest filled with zeroes)
		uint64_t fileHeader[4];

		//! Number of internal blobs
		uint32_t numOfInternalBlobs;
		//! Blobs offsets counted from after blob-headers block
		uint32_t blobOffsets[1];

		size_t calcOffsetsOffset() const { return sizeof(fileHeader) + sizeof(numOfInternalBlobs); }
		size_t calcHeadersOffset() const { return calcOffsetsOffset() + numOfInternalBlobs*sizeof(blobOffsets[0]); }
		size_t calcBlobsOffset() const { return calcHeadersOffset() + numOfInternalBlobs*sizeof(BlobHeaderV1); }
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
	};

	template<typename T>
	struct VariableSizeBlob;

	template<>
	struct VariableSizeBlob<scene::ICPUMesh> : Blob
	{
		static size_t calcBlobSizeForObj(scene::ICPUMesh* _obj);
		static void* allocMemForBlob(scene::ICPUMesh* _obj);
	};
	template<>
	struct VariableSizeBlob<scene::ICPUSkinnedMesh> : Blob
	{
		static size_t calcBlobSizeForObj(scene::ICPUSkinnedMesh* _obj);
		static void* allocMemForBlob(scene::ICPUSkinnedMesh* _obj);
	};
	template<>
	struct VariableSizeBlob<scene::CFinalBoneHierarchy> : Blob
	{
		static size_t calcBlobSizeForObj(scene::CFinalBoneHierarchy* _obj);
		static void* allocMemForBlob(scene::CFinalBoneHierarchy* _obj);
	};

#include "irrpack.h"
	//! Utility struct. Cast blob pointer to MeshBlob* to make life easier.
	struct MeshBlobV1 : VariableSizeBlob<scene::ICPUMesh>
	{
		//! WARNING: Constructor saves only bounding box and mesh buffer count (not mesh buffer pointers)
		explicit MeshBlobV1(const aabbox3df & _box, uint32_t _cnt);

		aabbox3df box;
		uint32_t meshBufCnt;
		uint64_t meshBufPtrs[1];
	} PACK_STRUCT;

	//! Utility struct. Cast blob pointer to MeshBlob* to make life easier.
	struct SkinnedMeshBlobV1 : VariableSizeBlob<scene::ICPUSkinnedMesh>
	{
		//! WARNING: Constructor saves only bone hierarchy, bounding box and mesh buffer count (not mesh buffer pointers)
		explicit SkinnedMeshBlobV1(scene::CFinalBoneHierarchy* _fbh, const aabbox3df & _box, uint32_t _cnt);

		uint64_t boneHierarchyPtr;
		aabbox3df box;
		uint32_t meshBufCnt;
		uint64_t meshBufPtrs[1];
	} PACK_STRUCT;

	//! Simple struct of essential data of ICPUMeshBuffer that has to be exported
	struct MeshBufferBlobV1 : Blob
	{
		//! Constructor filling all members
		explicit MeshBufferBlobV1(const scene::ICPUMeshBuffer*);

		video::SMaterial mat;
		core::aabbox3df box;
		uint64_t descPtr;
		video::E_INDEX_TYPE indexType;
		uint32_t baseVertex;
		uint64_t indexCount;
		size_t indexBufOffset;
		size_t instanceCount;
		uint32_t baseInstance;
		scene::E_PRIMITIVE_TYPE primitiveType;
		scene::E_VERTEX_ATTRIBUTE_ID posAttrId;
	} PACK_STRUCT;

	struct SkinnedMeshBufferBlobV1 : MeshBufferBlobV1
	{
		//! Constructor filling all members
		explicit SkinnedMeshBufferBlobV1(const scene::SCPUSkinMeshBuffer*);

		uint32_t indexValMin;
		uint32_t indexValMax;
		uint32_t maxVertexBoneInfluences;
	} PACK_STRUCT;

	//! Simple struct of essential data of ICPUMeshDataFormatDesc that has to be exported
	struct MeshDataFormatDescBlobV1 : Blob
	{
		//! Constructor filling all members
		explicit MeshDataFormatDescBlobV1(const scene::IMeshDataFormatDesc<core::ICPUBuffer>*);

		scene::E_COMPONENTS_PER_ATTRIBUTE cpa[scene::EVAI_COUNT];
		scene::E_COMPONENT_TYPE attrType[scene::EVAI_COUNT];
		size_t attrStride[scene::EVAI_COUNT];
		size_t attrOffset[scene::EVAI_COUNT];
		uint32_t attrDivisor[scene::EVAI_COUNT];
		uint64_t attrBufPtrs[scene::EVAI_COUNT];
		uint64_t idxBufPtr;
	} PACK_STRUCT;

	struct FinalBoneHierarchyBlobV1 : VariableSizeBlob<scene::CFinalBoneHierarchy>
	{
		friend class scene::CFinalBoneHierarchy;
	private:
		//! For filling you are supposed to use scene::CFinalBoneHierarchy::fillExportBlob()
		FinalBoneHierarchyBlobV1(size_t _bCnt, size_t _numLvls, size_t _kfCnt);

	public:
		static size_t calcBonesOffset(scene::CFinalBoneHierarchy* _fbh);
		static size_t calcBoneNamesOffset(scene::CFinalBoneHierarchy* _fbh);
		static size_t calcLevelsOffset(scene::CFinalBoneHierarchy* _fbh);
		static size_t calcKeyFramesOffset(scene::CFinalBoneHierarchy* _fbh);
		static size_t calcInterpolatedAnimsOffset(scene::CFinalBoneHierarchy* _fbh);
		static size_t calcNonInterpolatedAnimsOffset(scene::CFinalBoneHierarchy* _fbh);

		static size_t calcBonesByteSize(scene::CFinalBoneHierarchy* _fbh);
		static size_t calcBoneNamesByteSize(scene::CFinalBoneHierarchy* _fbh);
		static size_t calcLevelsByteSize(scene::CFinalBoneHierarchy* _fbh);
		static size_t calcKeyFramesByteSize(scene::CFinalBoneHierarchy* _fbh);
		static size_t calcInterpolatedAnimsByteSize(scene::CFinalBoneHierarchy* _fbh);
		static size_t calcNonInterpolatedAnimsByteSize(scene::CFinalBoneHierarchy* _fbh);

		size_t boneCount;
		size_t numLevelsInHierarchy;
		size_t keyframeCount;
	} PACK_STRUCT;
#include "irrunpack.h"

	template<typename>
	struct CorrespondingBlobTypeFor;
	template<>
	struct CorrespondingBlobTypeFor<ICPUBuffer> { typedef Blob type; };
	template<>
	struct CorrespondingBlobTypeFor<scene::ICPUMesh> { typedef MeshBlobV1 type; };
	template<>
	struct CorrespondingBlobTypeFor<scene::ICPUSkinnedMesh> { typedef SkinnedMeshBlobV1 type; };
	template<>
	struct CorrespondingBlobTypeFor<scene::ICPUMeshBuffer> { typedef MeshBufferBlobV1 type; };
	template<>
	struct CorrespondingBlobTypeFor<scene::SCPUSkinMeshBuffer> { typedef SkinnedMeshBufferBlobV1 type; };
	template<>
	struct CorrespondingBlobTypeFor<scene::IMeshDataFormatDesc<core::ICPUBuffer> > { typedef MeshDataFormatDescBlobV1 type; };
	template<>
	struct CorrespondingBlobTypeFor<scene::CFinalBoneHierarchy> { typedef FinalBoneHierarchyBlobV1 type; };
	template<>
	struct CorrespondingBlobTypeFor<video::IVirtualTexture> { typedef Blob type; };

	template<typename T> 
	typename CorrespondingBlobTypeFor<T>::type* toBlobPtr(const void* _blob)
	{
		return (typename CorrespondingBlobTypeFor<T>::type*)_blob;
	}

}} // irr::core

#endif
