// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_BAW_FILE_H_INCLUDED__
#define __IRR_BAW_FILE_H_INCLUDED__

#include <cstdint>

namespace irr { namespace core
{

#include "irrpack.h"
	//! Cast pointer to block of blob-headers to BlobHeader* and easily iterate and/or access members
	struct BlobHeader
	{
		uint32_t blobSize;
		uint32_t blobSizeDecompr;

		uint64_t compressionType : 3;
		uint64_t blobType : 7;
		uint64_t handle;

		uint64_t blobHash[4];
	} PACK_STRUCT;

	//! Cast pointer to (first byte of) file buffer to BaWFile*
	struct BawFile {
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
			EBT_MESH_BUFFER = 0,
			EBT_RAW_DATA_BUFFER,
			EBT_DATA_FORMAT_DESC,
			EBT_TEXTURE_PATH,
			EBT_COUNT
		};

		//! Meant to be used as pointer to blob's data
		uint8_t data[1];
	};

}} // irr::core

#endif
