// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_BLOB_H_INCLUDED__
#define __IRR_BLOB_H_INCLUDED__


namespace irr
{
namespace asset
{
	struct BlobLoadingParams;

#include "irr/irrpack.h"
	struct IRR_FORCE_EBO Blob
	{
		//! Coding method of blob's data enumeration
		enum E_BLOB_CODING_TYPE
		{
			EBCT_RAW = 0x00,
			EBCT_AES128_GCM = 0x01,
			EBCT_LZ4 = 0x02,
			EBCT_LZ4_AES128_GCM = 0x03,
			EBCT_LZMA = 0x04,
			EBCT_LZMA_AES128_GCM = 0x05
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
	struct IRR_FORCE_EBO SizedBlob
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
		@return Pointer to created blob, if it does not equal _stackPtr then new memory was dynamically allocated which needs to be freed.
		*/
		static B* createAndTryOnStack(const T* _obj, void* _stackPtr = NULL, const size_t& _size = 0)
		{
			const size_t actualObjSize = calcBlobSizeForObj(_obj);
			void* mem;
			if (!_stackPtr || actualObjSize > _size)
				mem = _IRR_ALIGNED_MALLOC(actualObjSize, _IRR_SIMD_ALIGNMENT);
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
	struct IRR_FORCE_EBO VariableSizeBlob : SizedBlob<VariableSizeBlob, B, T>
	{
	protected: // not intended for direct usage
		VariableSizeBlob() {}
		~VariableSizeBlob() {}
	};

	template<typename B, typename T>
	struct IRR_FORCE_EBO FixedSizeBlob : SizedBlob<FixedSizeBlob, B, T>
	{
	protected: // not intended for direct usage
		FixedSizeBlob() {}
		~FixedSizeBlob() {}
	};

	template<typename B, typename T>
	struct IRR_FORCE_EBO TypedBlob : Blob
	{
		static core::unordered_set<uint64_t> getNeededDeps(const void* _blob);
		static void* instantiateEmpty(const void* _blob, size_t _blobSize, BlobLoadingParams& _params);
		static void* finalize(void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, BlobLoadingParams& _params);
		static void releaseObj(const void* _obj);

		//static core::string printMemberPackingDebug();
	};
#include "irr/irrunpack.h"

	template<typename>
	struct CorrespondingBlobTypeFor;

	template<typename T>
	typename CorrespondingBlobTypeFor<T>::type* toBlobPtr(const void* _blob)
	{
		return (typename CorrespondingBlobTypeFor<T>::type*)_blob;
	}

}
} // irr::asset

#endif
