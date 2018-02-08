// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_BAW_FILE_H_INCLUDED__
#define __IRR_BAW_FILE_H_INCLUDED__

#include <deque>

#include "stdint.h"
#include "IMesh.h"
#include "irrArray.h"
#include "path.h"

namespace irr {

namespace scene
{
	class ISceneManager;
	class ICPUSkinnedMesh;
	class SCPUSkinMeshBuffer;
	class CFinalBoneHierarchy;
}
namespace io
{
	class IFileSystem;
}

namespace core
{

	struct BlobLoadingParams
	{
		scene::ISceneManager* sm;
		io::IFileSystem* fs;
		io::path filePath;
	};

	//! Class abstracting blobs version from process of loading them.
	/** 
	If you wish to extend .baw format by you own blob types, here's what you have to do:
		- add a corresponding to your blob value to Blob::E_BLOB_TYPE enum
		- make a class (or struct, a matter of keyword) representing your blob
		- your class must contain following static member functions:
			-# `uint32_t getNeededDeps(void* _blob, std::deque<uint64_t>& _q);` - pushes (push_back) to the end of `_q` 'handles' (i.e. pointers/addresses) of other blobs that `_blob` depends on (i.e. are its members)
			-# `void* tryMake(void* _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams& _params);` - creates an object from blob. `_deps` is map of mentioned dependencies (key is 'handle' and value is pointer to existing object)
		- Let `BlobsLoadingManager` know about your blob type, by using _IRR_ADD_BLOB_SUPPORT_ inside definition of the manager's constructor. See existing code for example.

	Feature not ready yet. (only loading actually)
	*/
	class BlobsLoadingManager
	{
	public:
		explicit BlobsLoadingManager(uint64_t _fileVer = 0);

		uint32_t getNeededDeps(uint32_t _blobType, void* _blob, std::deque<uint64_t>& _q);
		void* tryMake(uint32_t _blobType, void* _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams& _params);

		const uint64_t & getFileVer() const { return m_fileVer; }
		void setFileVer(uint64_t _fver) { m_fileVer = _fver; }

	private:
		typedef uint32_t(*EvalFunc_t)(void*, std::deque<uint64_t>&);
		typedef void*(*MakeFunc_t)(void*, size_t, std::map<uint64_t, void*>, const BlobLoadingParams&);

		uint64_t m_fileVer;
		std::map<uint32_t, std::map<uint64_t, EvalFunc_t>> m_evalFunctions;
		std::map<uint32_t, std::map<uint64_t, MakeFunc_t>> m_makeFunctions;
	};

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

		//! Assigns size and calculates hash of data.
		void finalize(const void* _data, size_t _size);
		//! Calculates hash from `_data` and compares to current one (`blobHash` member).
		bool validate(const void* _data) const;
	} PACK_STRUCT;

	//! Cast pointer to (first byte of) file buffer to BaWFile*. 256bit header must be first member (start of file).
	struct BawFileV0 {
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

		static uint32_t getLoadPriorityForType(E_BLOB_TYPE _t)
		{
			switch (_t)
			{
			case EBT_MESH:
			case EBT_SKINNED_MESH:
				return 1e6;
			case EBT_MESH_BUFFER:
			case EBT_SKINNED_MESH_BUFFER:
			case EBT_FINAL_BONE_HIERARCHY:
				return 2e6;
			case EBT_DATA_FORMAT_DESC:
			case EBT_TEXTURE_PATH:
				return 3e6;
			case EBT_RAW_DATA_BUFFER:
				return 4e6;
			}
		}

		void* getData() { return this; }
		const void* getData() const { return this; }
	};

	template<typename B, typename T>
	struct VariableSizeBlob : Blob
	{
	    protected: //not intended for direct usage
            VariableSizeBlob() {}
            ~VariableSizeBlob() {}
	    public:
            static size_t calcBlobSizeForObj(const T* _obj);
            static B* allocMemForBlob(const T* _obj) // get rid of this in the future
            {
                return (B*)malloc(calcBlobSizeForObj(_obj));
            }

			//! Utility function for making VariableSizeBlobs
			/**
			@param _obj Pointer to the object for which the blob will be made.
			@param _stackPtr Pointer to stack memory, usually you'd declare it as `uint8_t _stackPtr[_size]`.
			@param _size The size of the stack memory available.
			@return Pointer to created blob, if it does not equal _stackPtr then new memory was malloc'd which needs to be free'd
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

	struct RawBufferBlobV0 : Blob
	{
		static uint32_t getNeededDeps(void* _blob, std::deque<uint64_t>& _q) { return 0; }
		static void* tryMake(void* _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams& _params);
	};

	struct TexturePathBlobV0 : Blob
	{
		static uint32_t getNeededDeps(void* _blob, std::deque<uint64_t>& _q) { return 0; }
		static void* tryMake(void* _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams& _params);
	};

#include "irrpack.h"
	//! Utility struct. Cast blob pointer to MeshBlob* to make life easier.
	struct MeshBlobV0 : VariableSizeBlob<MeshBlobV0,scene::ICPUMesh>
	{
		friend struct VariableSizeBlob<MeshBlobV0, scene::ICPUMesh>;
	private:
            //! WARNING: Constructor saves only bounding box and mesh buffer count (not mesh buffer pointers)
		explicit MeshBlobV0(const scene::ICPUMesh* _mesh);

	public:
		static uint32_t getNeededDeps(void* _blob, std::deque<uint64_t>& _q);
		static void* tryMake(void* _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams& _params);

	public:
        aabbox3df box;
        uint32_t meshBufCnt;
        uint64_t meshBufPtrs[1];
	} PACK_STRUCT;

	//! Utility struct. Cast blob pointer to MeshBlob* to make life easier.
	struct SkinnedMeshBlobV0 : VariableSizeBlob<SkinnedMeshBlobV0,scene::ICPUSkinnedMesh>
	{
		friend struct VariableSizeBlob<SkinnedMeshBlobV0, scene::ICPUSkinnedMesh>;
	private:
            //! WARNING: Constructor saves only bone hierarchy, bounding box and mesh buffer count (not mesh buffer pointers)
        explicit SkinnedMeshBlobV0(const scene::ICPUSkinnedMesh* _sm);

	public:
		static uint32_t getNeededDeps(void* _blob, std::deque<uint64_t>& _q);
		static void* tryMake(void* _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams& _params);

	public:
        uint64_t boneHierarchyPtr;
        aabbox3df box;
        uint32_t meshBufCnt;
        uint64_t meshBufPtrs[1];
	} PACK_STRUCT;

	//! Simple struct of essential data of ICPUMeshBuffer that has to be exported
	struct MeshBufferBlobV0 : Blob
	{
		//! Constructor filling all members
		explicit MeshBufferBlobV0(const scene::ICPUMeshBuffer*);

		static uint32_t getNeededDeps(void* _blob, std::deque<uint64_t>& _q);
		static void* tryMake(void* _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams& _params);

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

	struct SkinnedMeshBufferBlobV0 : MeshBufferBlobV0
	{
		//! Constructor filling all members
		explicit SkinnedMeshBufferBlobV0(const scene::SCPUSkinMeshBuffer*);

		static uint32_t getNeededDeps(void* _blob, std::deque<uint64_t>& _q);
		static void* tryMake(void* _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams& _params);

		uint32_t indexValMin;
		uint32_t indexValMax;
		uint32_t maxVertexBoneInfluences;
	} PACK_STRUCT;

	//! Simple struct of essential data of ICPUMeshDataFormatDesc that has to be exported
	struct MeshDataFormatDescBlobV0 : Blob
	{
		//! Constructor filling all members
		explicit MeshDataFormatDescBlobV0(const scene::IMeshDataFormatDesc<core::ICPUBuffer>*);

		static uint32_t getNeededDeps(void* _blob, std::deque<uint64_t>& _q);
		static void* tryMake(void* _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams& _params);

		scene::E_COMPONENTS_PER_ATTRIBUTE cpa[scene::EVAI_COUNT];
		scene::E_COMPONENT_TYPE attrType[scene::EVAI_COUNT];
		size_t attrStride[scene::EVAI_COUNT];
		size_t attrOffset[scene::EVAI_COUNT];
		uint32_t attrDivisor[scene::EVAI_COUNT];
		uint64_t attrBufPtrs[scene::EVAI_COUNT];
		uint64_t idxBufPtr;
	} PACK_STRUCT;

	struct FinalBoneHierarchyBlobV0 : VariableSizeBlob<FinalBoneHierarchyBlobV0,scene::CFinalBoneHierarchy>
	{
		friend struct VariableSizeBlob<FinalBoneHierarchyBlobV0, scene::CFinalBoneHierarchy>;
	private:
		FinalBoneHierarchyBlobV0(const scene::CFinalBoneHierarchy* _fbh);

	public:
		static uint32_t getNeededDeps(void* _blob, std::deque<uint64_t>& _q) { return 0; }
		static void* tryMake(void* _blob, size_t _blobSize, std::map<uint64_t, void*> _deps, const BlobLoadingParams& _params);

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
	struct CorrespondingBlobTypeFor<ICPUBuffer> { typedef Blob type; };
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
	struct CorrespondingBlobTypeFor<video::IVirtualTexture> { typedef Blob type; };

	template<typename T>
	typename CorrespondingBlobTypeFor<T>::type* toBlobPtr(const void* _blob)
	{
		return (typename CorrespondingBlobTypeFor<T>::type*)_blob;
	}

}} // irr::core

#endif
