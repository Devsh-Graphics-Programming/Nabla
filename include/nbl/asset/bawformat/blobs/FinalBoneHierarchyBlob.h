// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#ifndef __NBL_ASSET_FINAL_BONE_HIERARCHY_BLOB_H_INCLUDED__
#define __NBL_ASSET_FINAL_BONE_HIERARCHY_BLOB_H_INCLUDED__

#include "nbl/core/declarations.h"

namespace nbl
{
namespace asset
{

#include "nbl/nblpack.h"
struct NBL_FORCE_EBO FinalBoneHierarchyBlobV3 : VariableSizeBlob<FinalBoneHierarchyBlobV3,CFinalBoneHierarchy>, TypedBlob<FinalBoneHierarchyBlobV3, CFinalBoneHierarchy>
{
public:
	enum E_BLOB_FINAL_BONE_HIERARCHY_FLAG : uint32_t
	{
		EBFBHF_RIGHT_HANDED = 0x1u
	};

	FinalBoneHierarchyBlobV3(const CFinalBoneHierarchy* _fbh);

public:
	//! Used for creating a blob. Calculates offset of the block of blob resulting from exporting `*_fbh` object.
	/** @param _fbh Pointer to object on the basis of which offset of the block will be calculated.
	@return Offset where the block must begin (used while writing blob data (exporting)).
	*/
    static size_t calcBonesOffset(const CFinalBoneHierarchy* _fbh);
	//! @copydoc calcBonesOffset(const CFinalBoneHierarchy*)
    static size_t calcLevelsOffset(const CFinalBoneHierarchy* _fbh);
	//! @copydoc calcBonesOffset(const CFinalBoneHierarchy*)
    static size_t calcKeyFramesOffset(const CFinalBoneHierarchy* _fbh);
	//! @copydoc calcBonesOffset(const CFinalBoneHierarchy*)
    static size_t calcInterpolatedAnimsOffset(const CFinalBoneHierarchy* _fbh);
	//! @copydoc calcBonesOffset(const CFinalBoneHierarchy*)
    static size_t calcNonInterpolatedAnimsOffset(const CFinalBoneHierarchy* _fbh);
	//! @copydoc calcBonesOffset(const CFinalBoneHierarchy*)
    static size_t calcBoneNamesOffset(const CFinalBoneHierarchy* _fbh);

	//! Used for creating a blob. Calculates size (in bytes) of the block of blob resulting from exporting `*_fbh` object.
	/** @param _fbh Pointer to object on the basis of which size of the block will be calculated.
	@return Size of the block calculated on the basis of data containted by *_fbh object.
	*/
    static size_t calcBonesByteSize(const CFinalBoneHierarchy* _fbh);
	//! @copydoc calcBonesByteSize(const CFinalBoneHierarchy*)
    static size_t calcLevelsByteSize(const CFinalBoneHierarchy* _fbh);
	//! @copydoc calcBonesByteSize(const CFinalBoneHierarchy*)
    static size_t calcKeyFramesByteSize(const CFinalBoneHierarchy* _fbh);
	//! @copydoc calcBonesByteSize(const CFinalBoneHierarchy*)
    static size_t calcInterpolatedAnimsByteSize(const CFinalBoneHierarchy* _fbh);
	//! @copydoc calcBonesByteSize(const CFinalBoneHierarchy*)
    static size_t calcNonInterpolatedAnimsByteSize(const CFinalBoneHierarchy* _fbh);
	//! @copydoc calcBonesByteSize(const CFinalBoneHierarchy*)
    static size_t calcBoneNamesByteSize(const CFinalBoneHierarchy* _fbh);

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

	uint32_t finalBoneHierarchyFlags;
    size_t boneCount;
    size_t numLevelsInHierarchy;
    size_t keyframeCount;
} PACK_STRUCT;
#include "nbl/nblunpack.h"
static_assert(
    sizeof(FinalBoneHierarchyBlobV3) ==
    sizeof(FinalBoneHierarchyBlobV3::boneCount) + sizeof(FinalBoneHierarchyBlobV3::numLevelsInHierarchy) + sizeof(FinalBoneHierarchyBlobV3::keyframeCount) + sizeof(FinalBoneHierarchyBlobV3::finalBoneHierarchyFlags),
    "FinalBoneHierarchyBlobV3: Size of blob is not sum of its contents!"
);

template<>
struct CorrespondingBlobTypeFor<CFinalBoneHierarchy> { typedef FinalBoneHierarchyBlobV3 type; };

}
} // nbl::asset

#endif
