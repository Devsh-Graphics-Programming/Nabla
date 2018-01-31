// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_BAW_FILE_H_INCLUDED__
#define __IRR_BAW_FILE_H_INCLUDED__

#include <cstdint>
#include "IMeshBuffer.h"

namespace irr { namespace core
{

#include "irrpack.h"
	//! Cast pointer to block of blob-headers to BlobHeader* and easily iterate and/or access members
	struct BlobHeaderV1
	{
		uint32_t blobSize;
		uint32_t blobSizeDecompr;

		uint8_t compressionType;
		uint8_t blobType;
		uint8_t dummy[6];
		uint64_t handle;

		uint64_t blobHash[4];
	} PACK_STRUCT;

	//! Cast pointer to (first byte of) file buffer to BaWFile*. 256bit header must be first member (start of file).
	struct BawFileV1 {
		//! 32-byte BaW binary format header, currently equal to "IrrlichtBaW BinaryFile" (and the rest filled with zeroes)
		uint64_t fileHeader[4];

		//! Number of internal blobs
		uint32_t numOfInternalBlobs;
		//! Blobs offsets counted from after blob-headers block
		uint32_t blobOffsets[1];
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
			EBT_MESH_BUFFER,
			EBT_RAW_DATA_BUFFER,
			EBT_DATA_FORMAT_DESC,
			EBT_TEXTURE_PATH,
			EBT_COUNT
		};

		void* getData() { return this; }
	};

#include "irrpack.h"
	//! Utility struct. Used only while loading (CBAWMeshLoader). Cast blob pointer to MeshBlob* to make life easier.
	struct MeshBlobV1 : Blob
	{
		//! WARNING: Constructor saves only bounding box and mesh buffer count
		explicit MeshBlobV1(const aabbox3df & _box, uint32_t _cnt);

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
#include "irrunpack.h"

}} // irr::core

#endif
