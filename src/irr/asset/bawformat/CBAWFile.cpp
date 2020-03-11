// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#include "irr/asset/bawformat/CBAWFile.h"

#include "irr/asset/ICPUBuffer.h"
#include "irr/asset/ICPUTexture.h"
#include "irr/asset/ICPUSkinnedMesh.h"
#include "irr/asset/ICPUSkinnedMeshBuffer.h"
#include "irr/asset/bawformat/legacy/CBAWLegacy.h"
#include "CFinalBoneHierarchy.h"

#ifdef _IRR_COMPILE_WITH_OPENSSL_
#include "openssl/evp.h"
#pragma comment(lib, "libeay32.lib")
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

template<>
size_t SizedBlob<VariableSizeBlob, TexturePathBlobV0, ICPUTexture>::calcBlobSizeForObj(const ICPUTexture* _obj)
{
	return _obj->getSourceFilename().size();
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
	memcpy(&mat, &_mb->getMaterial(), sizeof(video::SCPUMaterial));
	_mb->getMaterial().serializeBitfields(mat.bitfieldsPtr());
	for (size_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
		_mb->getMaterial().TextureLayer[i].SamplingParams.serializeBitfields(mat.TextureLayer[i].SamplingParams.bitfieldsPtr());

	memcpy(&box, &_mb->getBoundingBox(), sizeof(core::aabbox3df));
	descPtr = reinterpret_cast<uint64_t>(_mb->getMeshDataAndFormat());
	indexType = _mb->getIndexType();
	baseVertex = _mb->getBaseVertex();
	indexCount = _mb->getIndexCount();
	indexBufOffset = _mb->getIndexBufferOffset();
	instanceCount = _mb->getInstanceCount();
	baseInstance = _mb->getBaseInstance();
	primitiveType = _mb->getPrimitiveType();
	posAttrId = _mb->getPositionAttributeIx();
	normalAttrId = _mb->getNormalAttributeIx();
}

template<>
size_t SizedBlob<FixedSizeBlob, MeshBufferBlobV3, asset::ICPUMeshBuffer>::calcBlobSizeForObj(const asset::ICPUMeshBuffer* _obj)
{
	return sizeof(MeshBufferBlobV3);
}

SkinnedMeshBufferBlobV3::SkinnedMeshBufferBlobV3(const asset::ICPUSkinnedMeshBuffer* _smb)
{
	memcpy(&mat, &_smb->getMaterial(), sizeof(video::SCPUMaterial));
	_smb->getMaterial().serializeBitfields(mat.bitfieldsPtr());
	for (size_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
		_smb->getMaterial().TextureLayer[i].SamplingParams.serializeBitfields(mat.TextureLayer[i].SamplingParams.bitfieldsPtr());

	memcpy(&box, &_smb->getBoundingBox(), sizeof(core::aabbox3df));
	descPtr = reinterpret_cast<uint64_t>(_smb->getMeshDataAndFormat());
	indexType = _smb->getIndexType();
	baseVertex = _smb->getBaseVertex();
	indexCount = _smb->getIndexCount();
	indexBufOffset = _smb->getIndexBufferOffset();
	instanceCount = _smb->getInstanceCount();
	baseInstance = _smb->getBaseInstance();
	primitiveType = _smb->getPrimitiveType();
	posAttrId = _smb->getPositionAttributeIx();
	normalAttrId = _smb->getNormalAttributeIx();
	indexValMin = _smb->getIndexMinBound();
	indexValMax = _smb->getIndexMaxBound();
	maxVertexBoneInfluences = _smb->getMaxVertexBoneInfluences();
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


// .baw VERSION 1

MeshDataFormatDescBlobV1::MeshDataFormatDescBlobV1(const asset::IMeshDataFormatDesc<asset::ICPUBuffer>* _desc) : attrDivisor{0u}
{
    static_assert(VERTEX_ATTRIB_CNT == EVAI_COUNT, "VERTEX_ATTRIB_CNT != EVAI_COUNT");

    for (E_VERTEX_ATTRIBUTE_ID i = EVAI_ATTR0; i < EVAI_COUNT; i = E_VERTEX_ATTRIBUTE_ID((int)i + 1))
        attrFormat[(int)i] = _desc->getAttribFormat(i);
    for (E_VERTEX_ATTRIBUTE_ID i = EVAI_ATTR0; i < EVAI_COUNT; i = E_VERTEX_ATTRIBUTE_ID((int)i + 1))
        attrStride[(int)i] = _desc->getMappedBufferStride(i);
    for (E_VERTEX_ATTRIBUTE_ID i = EVAI_ATTR0; i < EVAI_COUNT; i = E_VERTEX_ATTRIBUTE_ID((int)i + 1))
        attrOffset[(int)i] = _desc->getMappedBufferOffset(i);
    for (E_VERTEX_ATTRIBUTE_ID i = EVAI_ATTR0; i < EVAI_COUNT; i = E_VERTEX_ATTRIBUTE_ID((int)i + 1))
        attrDivisor |= (_desc->getAttribDivisor(i)<<i);
    for (E_VERTEX_ATTRIBUTE_ID i = EVAI_ATTR0; i < EVAI_COUNT; i = E_VERTEX_ATTRIBUTE_ID((int)i + 1))
        attrBufPtrs[(int)i] = reinterpret_cast<uint64_t>(_desc->getMappedBuffer(i));

    idxBufPtr = reinterpret_cast<uint64_t>(_desc->getIndexBuffer());
}

MeshDataFormatDescBlobV1::MeshDataFormatDescBlobV1(const asset::legacyv0::MeshDataFormatDescBlobV0& _v0blob) : attrDivisor{0u}
{
    for (uint32_t i = 0u; i < EVAI_COUNT; ++i)
    {
        attrFormat[i] =
            asset::legacyv0::mapECT_plus_ECPA_onto_E_FORMAT(
                static_cast<asset::legacyv0::E_COMPONENT_TYPE>(_v0blob.attrType[i]),
                static_cast<asset::legacyv0::E_COMPONENTS_PER_ATTRIBUTE>(_v0blob.cpa[i])
            );
    }
    for (uint32_t i = 0u; i < EVAI_COUNT; ++i)
        attrStride[i] = static_cast<uint32_t>(_v0blob.attrStride[i]); // calm the compiler down, this is old code.
    for (uint32_t i = 0u; i < EVAI_COUNT; ++i)
        attrOffset[i] = _v0blob.attrOffset[i];
    for (uint32_t i = 0u; i < EVAI_COUNT; ++i)
        attrDivisor |= (std::min(_v0blob.attrDivisor[i], 1u) << i); // attribute divisor can be equal max 1 now
    for (uint32_t i = 0u; i < EVAI_COUNT; ++i)
        attrBufPtrs[i] = _v0blob.attrBufPtrs[i];

    idxBufPtr = _v0blob.idxBufPtr;
}

bool encAes128gcm(const void* _input, size_t _inSize, void* _output, size_t _outSize, const unsigned char* _key, const unsigned char* _iv, void* _tag)
{
#ifdef _IRR_COMPILE_WITH_OPENSSL_
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
#ifdef _IRR_COMPILE_WITH_OPENSSL_
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
