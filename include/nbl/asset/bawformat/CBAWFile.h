// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_BAW_FILE_H_INCLUDED__
#define __NBL_ASSET_BAW_FILE_H_INCLUDED__


#include "aabbox3d.h"
#include "nbl/core/xxHash256.h"
#include "nbl/asset/bawformat/Blob.h"
#include "nbl/asset/bawformat/BlobSerializable.h"
#include "nbl/asset/bawformat/blobs/RawBufferBlob.h"
#include "nbl/asset/bawformat/blobs/MeshDataFormatBlob.h"
#include "nbl/asset/bawformat/blobs/TexturePathBlob.h"
#include "nbl/asset/bawformat/blobs/MeshBufferBlob.h"
#include "nbl/asset/bawformat/blobs/MeshBlob.h"

namespace nbl
{
namespace asset
{

#include "nbl/nblpack.h"
	//! Cast pointer to block of blob-headers to BlobHeader* and easily iterate and/or access members
    template<uint64_t Version>
	struct NBL_API BlobHeaderVn
	{
		uint32_t blobSize;
		uint32_t blobSizeDecompr;

		uint8_t compressionType;
		uint8_t dummy[3];
		uint32_t blobType;
		uint64_t handle;

		union {
			uint64_t blobHash[4];
			uint8_t gcmTag[16];
		} PACK_STRUCT;

		//! Assigns sizes and calculates hash of data.
		void finalize(const void* _data, size_t _sizeDecompr, size_t _sizeCompr, uint8_t _comprType);
		//! Calculates hash from `_data` and compares to current one (`blobHash` member).
		bool validate(const void* _data) const;
		//! Calculates size of blob along with required padding
		static uint32_t calcEncSize(uint32_t _size) { return (_size+15) & uint32_t(-16); }
		uint32_t calcEncSize() const { return calcEncSize(blobSize); }
		uint32_t effectiveSize() const { return (compressionType & Blob::EBCT_AES128_GCM) ? calcEncSize() : blobSize; }
	} PACK_STRUCT;
    template<uint64_t Version>
    void BlobHeaderVn<Version>::finalize(const void* _data, size_t _sizeDecompr, size_t _sizeCompr, uint8_t _comprType)
    {
	    blobSizeDecompr = _sizeDecompr;
	    blobSize = _sizeCompr;
	    compressionType = _comprType;

	    if (!(compressionType & Blob::EBCT_AES128_GCM)) // use gcmTag instead (set while encrypting).
		    core::XXHash_256(_data, blobSize, blobHash);
    }
    template<uint64_t Version>
    bool BlobHeaderVn<Version>::validate(const void* _data) const
    {
	    if (compressionType & Blob::EBCT_AES128_GCM) // use gcm authentication instead. Decryption will fail if data is corrupted.
		    return true;
        uint64_t tmpHash[4];
	    core::XXHash_256(_data, blobSize, tmpHash);
	    for (size_t i=0; i<4; i++)
		    if (tmpHash[i] != blobHash[i])
			    return false;
        return true;
    }

	//! Cast pointer to (first byte of) file buffer to BAWFile*. 256bit header must be first member (start of file).
    //! If something changes in basic format structure, this should go to asset::legacyv0 namespace
    template<uint64_t Version>
	struct NBL_API NBL_FORCE_EBO BAWFileVn {
        static constexpr const char* HEADER_STRING = "IrrlichtBaW BinaryFile";
        static constexpr uint64_t version = Version;

		//! 32-byte BaW binary format header, currently equal to "IrrlichtBaW BinaryFile" (and the rest filled with zeroes).
		//! Also: last 8 bytes of file header is file-version number.
		uint64_t fileHeader[4];

		//! Number of internal blobs
		uint32_t numOfInternalBlobs;
		//! Init vector
		unsigned char iv[16];
		//! Blobs offsets counted from after blob-headers block
		uint32_t blobOffsets[1];

		size_t calcOffsetsOffset() const { return sizeof(fileHeader) + sizeof(numOfInternalBlobs) + sizeof(iv); }
		size_t calcHeadersOffset() const { return calcOffsetsOffset() + numOfInternalBlobs*sizeof(blobOffsets[0]); }
		size_t calcBlobsOffset() const { return calcHeadersOffset() + numOfInternalBlobs*sizeof(BlobHeaderVn<Version>); }
	} PACK_STRUCT;
#include "nbl/nblunpack.h"


	// ===============
	// .baw VERSION 
	// ===============
	constexpr uint32_t CurrentBAWFormatVersion = 3u;
	using BlobHeaderV3 = BlobHeaderVn<CurrentBAWFormatVersion>;
	using BAWFileV3 = BAWFileVn<CurrentBAWFormatVersion>;

	using BlobHeaderLatest = BlobHeaderV3;

	bool encAes128gcm(const void* _input, size_t _inSize, void* _output, size_t _outSize, const unsigned char* _key, const unsigned char* _iv, void* _tag);
	bool decAes128gcm(const void* _input, size_t _inSize, void* _output, size_t _outSize, const unsigned char* _key, const unsigned char* _iv, void* _tag);

}} // nbl::asset

#endif
