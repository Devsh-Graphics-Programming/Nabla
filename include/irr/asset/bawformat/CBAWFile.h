// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_BAW_FILE_H_INCLUDED__
#define __IRR_BAW_FILE_H_INCLUDED__


#include "aabbox3d.h"
#include "SMaterial.h"
#include "irr/asset/ICPUTexture.h"
#include "irr/asset/ICPUBuffer.h"
#include "coreutil.h"

namespace irr {

namespace asset
{
    class ICPUMeshBuffer;
    class ICPUMesh;
    class ICPUSkinnedMesh;
    class ICPUSkinnedMeshBuffer;
	template<typename> class IMeshDataFormatDesc;
    namespace legacyv0
    {
        struct MeshDataFormatDescBlobV0;
    }
}

namespace scene
{
	class ISceneManager;
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

	//! Cast pointer to block of blob-headers to BlobHeader* and easily iterate and/or access members
    template<uint64_t Version>
	struct BlobHeaderVn
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
	struct IRR_FORCE_EBO BAWFileVn {
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
		static B* createAndTryOnStack(const T* _obj, void* _stackPtr=NULL, const size_t& _size=0)
		{
			const size_t actualObjSize = calcBlobSizeForObj(_obj);
			void* mem;
			if (!_stackPtr || actualObjSize > _size)
				mem = _IRR_ALIGNED_MALLOC(actualObjSize,_IRR_SIMD_ALIGNMENT);
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
		static void* instantiateEmpty(const void* _blob, size_t _blobSize, const BlobLoadingParams& _params);
		static void* finalize(void* _obj, const void* _blob, size_t _blobSize, core::unordered_map<uint64_t, void*>& _deps, const BlobLoadingParams& _params);
		static void releaseObj(const void* _obj);

		//static core::string printMemberPackingDebug();
	};

	struct IRR_FORCE_EBO RawBufferBlobV0 : TypedBlob<RawBufferBlobV0, asset::ICPUBuffer>, FixedSizeBlob<RawBufferBlobV0, asset::ICPUBuffer>
	{};

	struct IRR_FORCE_EBO TexturePathBlobV0 : TypedBlob<TexturePathBlobV0, asset::ICPUTexture>, FixedSizeBlob<TexturePathBlobV0, asset::ICPUTexture>
	{};

	//! Utility struct. Cast blob pointer to MeshBlob* to make life easier.
	struct IRR_FORCE_EBO MeshBlobV0 : VariableSizeBlob<MeshBlobV0,asset::ICPUMesh>, TypedBlob<MeshBlobV0, asset::ICPUMesh>
	{
		//friend struct SizedBlob<VariableSizeBlob, MeshBlobV0, asset::ICPUMesh>;
	public:
            //! WARNING: Constructor saves only bounding box and mesh buffer count (not mesh buffer pointers)
		explicit MeshBlobV0(const asset::ICPUMesh* _mesh);

	public:
        core::aabbox3df box;
        uint32_t meshBufCnt;
        uint64_t meshBufPtrs[1];
	} PACK_STRUCT;
    static_assert(sizeof(core::aabbox3df)==24, "sizeof(core::aabbox3df) must be 24");
    static_assert(sizeof(MeshBlobV0::meshBufPtrs)==8, "sizeof(MeshBlobV0::meshBufPtrs) must be 8");
    static_assert(
        sizeof(MeshBlobV0) ==
        sizeof(MeshBlobV0::box) + sizeof(MeshBlobV0::meshBufCnt) + sizeof(MeshBlobV0::meshBufPtrs),
        "MeshBlobV0: Size of blob is not sum of its contents!"
    );

	//! Utility struct. Cast blob pointer to MeshBlob* to make life easier.
	struct IRR_FORCE_EBO SkinnedMeshBlobV0 : VariableSizeBlob<SkinnedMeshBlobV0,asset::ICPUSkinnedMesh>, TypedBlob<SkinnedMeshBlobV0, asset::ICPUSkinnedMesh>
	{
		//friend struct SizedBlob<VariableSizeBlob, SkinnedMeshBlobV0, asset::ICPUSkinnedMesh>;
	public:
        explicit SkinnedMeshBlobV0(const asset::ICPUSkinnedMesh* _sm);

	public:
        uint64_t boneHierarchyPtr;
        core::aabbox3df box;
        uint32_t meshBufCnt;
        uint64_t meshBufPtrs[1];
	} PACK_STRUCT;
    static_assert(sizeof(SkinnedMeshBlobV0::meshBufPtrs)==8, "sizeof(SkinnedMeshBlobV0::meshBufPtrs) must be 8");
    static_assert(
        sizeof(SkinnedMeshBlobV0) ==
        sizeof(SkinnedMeshBlobV0::boneHierarchyPtr) + sizeof(SkinnedMeshBlobV0::box) + sizeof(SkinnedMeshBlobV0::meshBufCnt) + sizeof(SkinnedMeshBlobV0::meshBufPtrs),
        "SkinnedMeshBlobV0: Size of blob is not sum of its contents!"
    );

	//! Simple struct of essential data of ICPUMeshBuffer that has to be exported
	struct IRR_FORCE_EBO MeshBufferBlobV0 : TypedBlob<MeshBufferBlobV0, asset::ICPUMeshBuffer>, FixedSizeBlob<MeshBufferBlobV0, asset::ICPUMeshBuffer>
	{
		//! Constructor filling all members
		explicit MeshBufferBlobV0(const asset::ICPUMeshBuffer*);

		video::SGPUMaterial mat;
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
    static_assert(sizeof(MeshBufferBlobV0::mat)==197, "sizeof(MeshBufferBlobV0::mat) must be 197");
    static_assert(
        sizeof(MeshBufferBlobV0) ==
        sizeof(MeshBufferBlobV0::mat) + sizeof(MeshBufferBlobV0::box) + sizeof(MeshBufferBlobV0::descPtr) + sizeof(MeshBufferBlobV0::indexType) + sizeof(MeshBufferBlobV0::baseVertex)
        + sizeof(MeshBufferBlobV0::indexCount) + sizeof(MeshBufferBlobV0::indexBufOffset) + sizeof(MeshBufferBlobV0::instanceCount) + sizeof(MeshBufferBlobV0::baseInstance)
        + sizeof(MeshBufferBlobV0::primitiveType) + sizeof(MeshBufferBlobV0::posAttrId),
        "MeshBufferBlobV0: Size of blob is not sum of its contents!"
    );

	struct IRR_FORCE_EBO SkinnedMeshBufferBlobV0 : TypedBlob<SkinnedMeshBufferBlobV0, asset::ICPUSkinnedMeshBuffer>, FixedSizeBlob<SkinnedMeshBufferBlobV0, asset::ICPUSkinnedMeshBuffer>
	{
		//! Constructor filling all members
		explicit SkinnedMeshBufferBlobV0(const asset::ICPUSkinnedMeshBuffer*);

		video::SGPUMaterial mat;
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
    static_assert(sizeof(SkinnedMeshBufferBlobV0::mat)==197, "sizeof(MeshBufferBlobV0::mat) must be 197");
    static_assert(
        sizeof(SkinnedMeshBufferBlobV0) ==
        sizeof(SkinnedMeshBufferBlobV0::mat) + sizeof(SkinnedMeshBufferBlobV0::box) + sizeof(SkinnedMeshBufferBlobV0::descPtr) + sizeof(SkinnedMeshBufferBlobV0::indexType) + sizeof(SkinnedMeshBufferBlobV0::baseVertex)
        + sizeof(SkinnedMeshBufferBlobV0::indexCount) + sizeof(SkinnedMeshBufferBlobV0::indexBufOffset) + sizeof(SkinnedMeshBufferBlobV0::instanceCount) + sizeof(SkinnedMeshBufferBlobV0::baseInstance)
        + sizeof(SkinnedMeshBufferBlobV0::primitiveType) + sizeof(SkinnedMeshBufferBlobV0::posAttrId) + sizeof(SkinnedMeshBufferBlobV0::indexValMin) + sizeof(SkinnedMeshBufferBlobV0::indexValMax) + sizeof(SkinnedMeshBufferBlobV0::maxVertexBoneInfluences),
        "SkinnedMeshBufferBlobV0: Size of blob is not sum of its contents!"
    );

	struct IRR_FORCE_EBO FinalBoneHierarchyBlobV0 : VariableSizeBlob<FinalBoneHierarchyBlobV0,scene::CFinalBoneHierarchy>, TypedBlob<FinalBoneHierarchyBlobV0, scene::CFinalBoneHierarchy>
	{
		//friend struct SizedBlob<VariableSizeBlob, FinalBoneHierarchyBlobV0, scene::CFinalBoneHierarchy>;
	public:
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
#include "irr/irrunpack.h"
    static_assert(
        sizeof(FinalBoneHierarchyBlobV0) ==
        sizeof(FinalBoneHierarchyBlobV0::boneCount) + sizeof(FinalBoneHierarchyBlobV0::numLevelsInHierarchy) + sizeof(FinalBoneHierarchyBlobV0::keyframeCount),
        "FinalBoneHierarchyBlobV0: Size of blob is not sum of its contents!"
    );

    // ===============
    // .baw VERSION 1
    // ===============
    using BlobHeaderV1 = BlobHeaderVn<1>;
    using BAWFileV1 = BAWFileVn<1>;
    using RawBufferBlobV1 = RawBufferBlobV0;
    using TexturePathBlobV1 = TexturePathBlobV0;
    using MeshBlobV1 = MeshBlobV0;
    using SkinnedMeshBlobV1 = SkinnedMeshBlobV0;
    using MeshBufferBlobV1 = MeshBufferBlobV0;
    using SkinnedMeshBufferBlobV1 = SkinnedMeshBufferBlobV0;
    using FinalBoneHierarchyBlobV1 = FinalBoneHierarchyBlobV0;

#include "irr/irrpack.h"
    struct MeshDataFormatDescBlobV1 : TypedBlob<MeshDataFormatDescBlobV1, asset::IMeshDataFormatDesc<asset::ICPUBuffer> >, FixedSizeBlob<MeshDataFormatDescBlobV1, asset::IMeshDataFormatDesc<asset::ICPUBuffer> >
    {
    private:
        enum { VERTEX_ATTRIB_CNT = 16 };
    public:
        //! Constructor filling all members
        explicit MeshDataFormatDescBlobV1(const asset::IMeshDataFormatDesc<asset::ICPUBuffer>*);
        //! Backward compatibility constructor
        explicit MeshDataFormatDescBlobV1(const asset::legacyv0::MeshDataFormatDescBlobV0&);

        uint32_t attrFormat[VERTEX_ATTRIB_CNT];
        uint32_t attrStride[VERTEX_ATTRIB_CNT];
        size_t attrOffset[VERTEX_ATTRIB_CNT];
        uint32_t attrDivisor;
        uint32_t padding;
        uint64_t attrBufPtrs[VERTEX_ATTRIB_CNT];
        uint64_t idxBufPtr;
    } PACK_STRUCT;
#include "irr/irrunpack.h"
    static_assert(
        sizeof(MeshDataFormatDescBlobV1) ==
        sizeof(MeshDataFormatDescBlobV1::attrFormat) + sizeof(MeshDataFormatDescBlobV1::attrStride) + sizeof(MeshDataFormatDescBlobV1::attrOffset) + sizeof(MeshDataFormatDescBlobV1::attrDivisor) + sizeof(MeshDataFormatDescBlobV1::padding) + sizeof(MeshDataFormatDescBlobV1::attrBufPtrs) + sizeof(MeshDataFormatDescBlobV1::idxBufPtr),
        "MeshDataFormatDescBlobV1: Size of blob is not sum of its contents!"
    );
    template<>
    inline size_t SizedBlob<FixedSizeBlob, MeshDataFormatDescBlobV1, asset::IMeshDataFormatDesc<asset::ICPUBuffer> >::calcBlobSizeForObj(const asset::IMeshDataFormatDesc<asset::ICPUBuffer>* _obj)
    {
        return sizeof(MeshDataFormatDescBlobV1);
    }

	template<typename>
	struct CorrespondingBlobTypeFor;
	template<>
	struct CorrespondingBlobTypeFor<asset::ICPUBuffer> { typedef RawBufferBlobV1 type; };
	template<>
	struct CorrespondingBlobTypeFor<asset::ICPUMesh> { typedef MeshBlobV1 type; };
	template<>
	struct CorrespondingBlobTypeFor<asset::ICPUSkinnedMesh> { typedef SkinnedMeshBlobV1 type; };
	template<>
	struct CorrespondingBlobTypeFor<asset::ICPUMeshBuffer> { typedef MeshBufferBlobV1 type; };
	template<>
	struct CorrespondingBlobTypeFor<asset::ICPUSkinnedMeshBuffer> { typedef SkinnedMeshBufferBlobV1 type; };
	template<>
	struct CorrespondingBlobTypeFor<asset::IMeshDataFormatDesc<asset::ICPUBuffer> > { typedef MeshDataFormatDescBlobV1 type; };
	template<>
	struct CorrespondingBlobTypeFor<scene::CFinalBoneHierarchy> { typedef FinalBoneHierarchyBlobV1 type; };
	template<>
	struct CorrespondingBlobTypeFor<video::IVirtualTexture> { typedef TexturePathBlobV1 type; };

	template<typename T>
	typename CorrespondingBlobTypeFor<T>::type* toBlobPtr(const void* _blob)
	{
		return (typename CorrespondingBlobTypeFor<T>::type*)_blob;
	}

	class IRR_FORCE_EBO BlobSerializable
	{
	public:
		virtual ~BlobSerializable() {}

		virtual void* serializeToBlob(void* _stackPtr=NULL, const size_t& _stackSize=0) const = 0;
	};

	bool encAes128gcm(const void* _input, size_t _inSize, void* _output, size_t _outSize, const unsigned char* _key, const unsigned char* _iv, void* _tag);
	bool decAes128gcm(const void* _input, size_t _inSize, void* _output, size_t _outSize, const unsigned char* _key, const unsigned char* _iv, void* _tag);

}} // irr::asset

#endif
