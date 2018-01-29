#ifndef __IRR_BAW_FILE_H_INCLUDED__
#define __IRR_BAW_FILE_H_INCLUDED__

#include <cstdint>

namespace irr { namespace core
{

#include "irrpack.h"
	struct BlobHeader
	{
		uint32_t blobSize;
		uint32_t blobSizeDecompr;

		uint64_t compressionType : 2;
		uint64_t blobType : 7;
		uint64_t handle;

		uint64_t blobHash[4];
	} PACK_STRUCT;

	struct BawFile {
		uint64_t fileHeader[4];

		uint32_t numOfInternalBlobs;
		uint32_t blobOffsetsFromHere[1];
	} PACK_STRUCT;
#include "irrunpack.h"

	struct Blob
	{
		enum E_BLOB_COMPRESSION_TYPE
		{
			EBCT_NONE = 0,
			EBCT_LZ4,
			EBCT_LZMA,
			EBCT_COUNT
		};
		enum E_BLOB_TYPE
		{
			EBT_MESH_BUFFER = 0,
			EBT_RAW_DATA_BUFFER,
			EBT_DATA_FORMAT_DESC,
			EBT_TEXTURE_PATH,
			EBT_COUNT
		};

		uint8_t data[1];
	};

}} // irr::core

#endif
