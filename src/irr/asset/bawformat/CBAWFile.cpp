// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "irr/asset/bawformat/CBAWFile.h"

#include "irr/asset/ICPUBuffer.h"
#include "irr/asset/ICPUSkinnedMesh.h"
#include "irr/asset/ICPUSkinnedMeshBuffer.h"
#include "irr/asset/bawformat/legacy/CBAWLegacy.h"
#include "CFinalBoneHierarchy.h"

#ifdef _NBL_COMPILE_WITH_OPENSSL_
#include "openssl/evp.h"
#endif


namespace irr
{
namespace asset
{
	

template<>
size_t SizedBlob<VariableSizeBlob, RawBufferBlobV0, ICPUBuffer>::calcBlobSizeForObj(const ICPUBuffer* _obj)
{
	return _obj->getSize();
}

MeshBlobV3::MeshBlobV3(const asset::ICPUMesh* _mesh) : box(_mesh->getBoundingBox()), meshBufCnt(_mesh->getMeshBufferCount())
{
	for (uint32_t i = 0; i < meshBufCnt; ++i)
		meshBufPtrs[i] = reinterpret_cast<uint64_t>(_mesh->getMeshBuffer(i));

	meshFlags = 0; // default initialization for proper usage of bit operators later on
}

template<>
size_t SizedBlob<VariableSizeBlob, MeshBlobV3, asset::ICPUMesh>::calcBlobSizeForObj(const asset::ICPUMesh* _obj)
{
	return sizeof(MeshBlobV3) + (_obj->getMeshBufferCount()-1) * sizeof(uint64_t);
}

SkinnedMeshBlobV3::SkinnedMeshBlobV3(const asset::ICPUSkinnedMesh* _sm)
	: boneHierarchyPtr(reinterpret_cast<uint64_t>(_sm->getBoneReferenceHierarchy())), box(_sm->getBoundingBox()), meshBufCnt(_sm->getMeshBufferCount())
{
	for (uint32_t i = 0; i < meshBufCnt; ++i)
	{
		meshBufPtrs[i] = reinterpret_cast<uint64_t>(_sm->getMeshBuffer(i));
	}

	meshFlags = 0; // default initialization for proper usage of bit operators later on
}

template<>
size_t SizedBlob<VariableSizeBlob, SkinnedMeshBlobV3, asset::ICPUSkinnedMesh>::calcBlobSizeForObj(const asset::ICPUSkinnedMesh* _obj)
{
	return sizeof(SkinnedMeshBlobV3) + (_obj->getMeshBufferCount() - 1) * sizeof(uint64_t);
}

MeshBufferBlobV3::MeshBufferBlobV3(const asset::ICPUMeshBuffer* _mb)
{
}

template<>
size_t SizedBlob<FixedSizeBlob, MeshBufferBlobV3, asset::ICPUMeshBuffer>::calcBlobSizeForObj(const asset::ICPUMeshBuffer* _obj)
{
	return sizeof(MeshBufferBlobV3);
}

SkinnedMeshBufferBlobV3::SkinnedMeshBufferBlobV3(const asset::ICPUSkinnedMeshBuffer* _smb)
{
}

template<>
size_t SizedBlob<FixedSizeBlob, SkinnedMeshBufferBlobV3, asset::ICPUSkinnedMeshBuffer>::calcBlobSizeForObj(const asset::ICPUSkinnedMeshBuffer* _obj)
{
	return sizeof(SkinnedMeshBufferBlobV3);
}

FinalBoneHierarchyBlobV3::FinalBoneHierarchyBlobV3(const CFinalBoneHierarchy* _fbh)
{
	boneCount = _fbh->getBoneCount();
	numLevelsInHierarchy = _fbh->getHierarchyLevels();
	keyframeCount = _fbh->getKeyFrameCount();

	uint8_t* const ptr = ((uint8_t*)this);
	memcpy(ptr + calcBonesOffset(_fbh), _fbh->getBoneData(), calcBonesByteSize(_fbh));
	memcpy(ptr + calcLevelsOffset(_fbh), _fbh->getBoneTreeLevelEnd(), calcLevelsByteSize(_fbh));
	memcpy(ptr + calcKeyFramesOffset(_fbh), _fbh->getKeys(), calcKeyFramesByteSize(_fbh));
	memcpy(ptr + calcInterpolatedAnimsOffset(_fbh), _fbh->getInterpolatedAnimationData(), calcInterpolatedAnimsByteSize(_fbh));
	memcpy(ptr + calcNonInterpolatedAnimsOffset(_fbh), _fbh->getNonInterpolatedAnimationData(), calcNonInterpolatedAnimsByteSize(_fbh));
	uint8_t* strPtr = ptr + calcBoneNamesOffset(_fbh);
	for (size_t i = 0; i < boneCount; ++i)
	{
		memcpy(strPtr, _fbh->getBoneName(i).c_str(), _fbh->getBoneName(i).size());
		strPtr += _fbh->getBoneName(i).size();
		*strPtr = 0;
		++strPtr;
	}

	finalBoneHierarchyFlags = 0; //default initialization for proper usage of bit operators later on
}

template<>
size_t SizedBlob<VariableSizeBlob, FinalBoneHierarchyBlobV3,CFinalBoneHierarchy>::calcBlobSizeForObj(const CFinalBoneHierarchy* _obj)
{
	return
		sizeof(FinalBoneHierarchyBlobV3) +
		FinalBoneHierarchyBlobV3::calcBonesByteSize(_obj) +
		FinalBoneHierarchyBlobV3::calcLevelsByteSize(_obj) +
		FinalBoneHierarchyBlobV3::calcKeyFramesByteSize(_obj) +
		FinalBoneHierarchyBlobV3::calcInterpolatedAnimsByteSize(_obj) +
		FinalBoneHierarchyBlobV3::calcNonInterpolatedAnimsByteSize(_obj) +
		FinalBoneHierarchyBlobV3::calcBoneNamesByteSize(_obj);
}

size_t FinalBoneHierarchyBlobV3::calcBonesOffset(const CFinalBoneHierarchy* _fbh)
{
	return sizeof(FinalBoneHierarchyBlobV3);
}
size_t FinalBoneHierarchyBlobV3::calcLevelsOffset(const CFinalBoneHierarchy * _fbh)
{
	return calcBonesOffset(_fbh) + calcBonesByteSize(_fbh);
}
size_t FinalBoneHierarchyBlobV3::calcKeyFramesOffset(const CFinalBoneHierarchy * _fbh)
{
	return calcLevelsOffset(_fbh) + calcLevelsByteSize(_fbh);
}
size_t FinalBoneHierarchyBlobV3::calcInterpolatedAnimsOffset(const CFinalBoneHierarchy * _fbh)
{
	return calcKeyFramesOffset(_fbh) + calcKeyFramesByteSize(_fbh);
}
size_t FinalBoneHierarchyBlobV3::calcNonInterpolatedAnimsOffset(const CFinalBoneHierarchy * _fbh)
{
	return calcInterpolatedAnimsOffset(_fbh) + calcInterpolatedAnimsByteSize(_fbh);
}
size_t FinalBoneHierarchyBlobV3::calcBoneNamesOffset(const CFinalBoneHierarchy* _fbh)
{
	return calcNonInterpolatedAnimsOffset(_fbh) + calcNonInterpolatedAnimsByteSize(_fbh);
}

size_t FinalBoneHierarchyBlobV3::calcBonesByteSize(const CFinalBoneHierarchy * _fbh)
{
	return _fbh->getBoneCount()*sizeof(*_fbh->getBoneData());
}
size_t FinalBoneHierarchyBlobV3::calcLevelsByteSize(const CFinalBoneHierarchy * _fbh)
{
	return _fbh->getHierarchyLevels()*sizeof(*_fbh->getBoneTreeLevelEnd());
}
size_t FinalBoneHierarchyBlobV3::calcKeyFramesByteSize(const CFinalBoneHierarchy * _fbh)
{
	return _fbh->getKeyFrameCount()*sizeof(*_fbh->getKeys());
}
size_t FinalBoneHierarchyBlobV3::calcInterpolatedAnimsByteSize(const CFinalBoneHierarchy * _fbh)
{
	return _fbh->getAnimationCount()*sizeof(*_fbh->getInterpolatedAnimationData());
}
size_t FinalBoneHierarchyBlobV3::calcNonInterpolatedAnimsByteSize(const CFinalBoneHierarchy * _fbh)
{
	return _fbh->getAnimationCount()*sizeof(*_fbh->getNonInterpolatedAnimationData());
}
size_t FinalBoneHierarchyBlobV3::calcBoneNamesByteSize(const CFinalBoneHierarchy * _fbh)
{
	return _fbh->getSizeOfAllBoneNames();
}

size_t FinalBoneHierarchyBlobV3::calcBonesOffset() const
{
	return sizeof(FinalBoneHierarchyBlobV3);
}
size_t FinalBoneHierarchyBlobV3::calcLevelsOffset() const
{
	return calcBonesOffset() + calcBonesByteSize();
}
size_t FinalBoneHierarchyBlobV3::calcKeyFramesOffset() const
{
	return calcLevelsOffset() + calcLevelsByteSize();
}
size_t FinalBoneHierarchyBlobV3::calcInterpolatedAnimsOffset() const
{
	return calcKeyFramesOffset() + calcKeyFramesByteSize();
}
size_t FinalBoneHierarchyBlobV3::calcNonInterpolatedAnimsOffset() const
{
	return calcInterpolatedAnimsOffset() + calcInterpolatedAnimsByteSize();
}
size_t FinalBoneHierarchyBlobV3::calcBoneNamesOffset() const
{
	return calcNonInterpolatedAnimsOffset() + calcNonInterpolatedAnimsByteSize();
}

size_t FinalBoneHierarchyBlobV3::calcBonesByteSize() const
{
	return boneCount * CFinalBoneHierarchy::getSizeOfSingleBone();
}
size_t FinalBoneHierarchyBlobV3::calcLevelsByteSize() const
{
	return numLevelsInHierarchy * sizeof(size_t);
}
size_t FinalBoneHierarchyBlobV3::calcKeyFramesByteSize() const
{
	return keyframeCount * sizeof(float);
}
size_t FinalBoneHierarchyBlobV3::calcInterpolatedAnimsByteSize() const
{
	return keyframeCount * boneCount * CFinalBoneHierarchy::getSizeOfSingleAnimationData();
}
size_t FinalBoneHierarchyBlobV3::calcNonInterpolatedAnimsByteSize() const
{
	return keyframeCount * boneCount * CFinalBoneHierarchy::getSizeOfSingleAnimationData();
}

bool encAes128gcm(const void* _input, size_t _inSize, void* _output, size_t _outSize, const unsigned char* _key, const unsigned char* _iv, void* _tag)
{
#ifdef _NBL_COMPILE_WITH_OPENSSL_
	EVP_CIPHER_CTX *ctx;
	int outlen;

	if (!(ctx = EVP_CIPHER_CTX_new()))
		return false;

	EVP_EncryptInit_ex(ctx, EVP_aes_128_gcm(), NULL, NULL, NULL);
	EVP_CIPHER_CTX_set_padding(ctx, 0); // disable padding
	EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, 16, NULL);
	EVP_EncryptInit_ex(ctx, NULL, NULL, _key, _iv);
	EVP_EncryptUpdate(ctx, (unsigned char*)_output, &outlen, (const unsigned char*)_input, int(_inSize));
	EVP_EncryptFinal_ex(ctx, (unsigned char*)_output, &outlen);
	EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, _tag); // save tag

	EVP_CIPHER_CTX_free(ctx);
	return true;
#else
	return false;
#endif
}
bool decAes128gcm(const void* _input, size_t _inSize, void* _output, size_t _outSize, const unsigned char* _key, const unsigned char* _iv, void* _tag)
{
#ifdef _NBL_COMPILE_WITH_OPENSSL_
	EVP_CIPHER_CTX *ctx;
	int outlen;

	if (!(ctx = EVP_CIPHER_CTX_new()))
		return false;

	EVP_DecryptInit_ex(ctx, EVP_aes_128_gcm(), NULL, NULL, NULL);
	EVP_CIPHER_CTX_set_padding(ctx, 0); // disable apdding
	EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, 16, NULL);
	EVP_DecryptInit_ex(ctx, NULL, NULL, _key, _iv);
	EVP_DecryptUpdate(ctx, (unsigned char*)_output, &outlen, (const unsigned char*)_input, int(_inSize));
	EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, 16, _tag); // set expected tag value

	int retval = EVP_DecryptFinal_ex(ctx, (unsigned char*)_output, &outlen);

	EVP_CIPHER_CTX_free(ctx);
	return retval > 0;
#else
	return false;
#endif
}

}} // irr::core
