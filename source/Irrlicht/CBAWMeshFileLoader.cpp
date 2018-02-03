// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#include "CBAWMeshFileLoader.h"

#include "CFinalBoneHierarchy.h"
#include "SMesh.h"
#include "CSkinnedMesh.h"

namespace irr { namespace scene
{
CBAWMeshFileLoader::~CBAWMeshFileLoader()
{
	if (m_fileSystem)
		m_fileSystem->drop();
}

CBAWMeshFileLoader::CBAWMeshFileLoader(scene::ISceneManager* _sm, io::IFileSystem* _fs) : m_sceneMgr(_sm), m_fileSystem(_fs)
{
#ifdef _DEBUG
	setDebugName("CBAWMeshFileLoader");
#endif
	if (m_fileSystem)
		m_fileSystem->grab();
}

template<>
core::ICPUBuffer* CBAWMeshFileLoader::make<core::ICPUBuffer>(SBlobData& _data, SContext& _ctx) const
{
	typename core::CorrespondingBlobTypeFor<core::ICPUBuffer>::type* blob = core::toBlobPtr<core::ICPUBuffer>(_data.blob);

	void* blobData = malloc(_data.header->blobSizeDecompr);
	memcpy(blobData, blob->getData(), _data.header->blobSizeDecompr);
	core::ICPUBuffer* buf = new core::ICPUBuffer(_data.header->blobSizeDecompr, blobData);

	return buf;
}
template<>
IMeshDataFormatDesc<core::ICPUBuffer>* CBAWMeshFileLoader::make<IMeshDataFormatDesc<core::ICPUBuffer> >(SBlobData& _data, SContext& _ctx) const
{
	typename core::CorrespondingBlobTypeFor<IMeshDataFormatDesc<core::ICPUBuffer> >::type* blob = core::toBlobPtr<IMeshDataFormatDesc<core::ICPUBuffer> >(_data.blob);

	IMeshDataFormatDesc<core::ICPUBuffer>* desc = new scene::ICPUMeshDataFormatDesc();
	for (E_VERTEX_ATTRIBUTE_ID i = EVAI_ATTR0; i < EVAI_COUNT; i = E_VERTEX_ATTRIBUTE_ID((int)i + 1))
	{
		if (blob->attrBufPtrs[(int)i])
			desc->mapVertexAttrBuffer(
				loadBlob<core::ICPUBuffer>(_ctx.blobs[blob->attrBufPtrs[(int)i]], _ctx),
				i,
				blob->cpa[(int)i],
				blob->attrType[(int)i],
				blob->attrStride[(int)i],
				blob->attrOffset[(int)i],
				blob->attrDivisor[(int)i]
			);
	}
	if (blob->idxBufPtr)
		desc->mapIndexBuffer(loadBlob<core::ICPUBuffer>(_ctx.blobs[blob->idxBufPtr], _ctx));

	return desc;
}
template<>
video::IVirtualTexture* CBAWMeshFileLoader::make<video::IVirtualTexture>(SBlobData& _data, SContext& _ctx) const
{
	typename core::CorrespondingBlobTypeFor<video::IVirtualTexture>::type* blob = core::toBlobPtr<video::IVirtualTexture>(_data.blob);

	video::ITexture* texture;
	const char* const texname = (const char*)blob->getData();
	bool newTexture;
	if (m_fileSystem->existFile(texname))
	{
		newTexture = m_sceneMgr->getVideoDriver()->findTexture(texname) == 0;
		texture = m_sceneMgr->getVideoDriver()->getTexture(texname);
	}
	else
	{
		const io::path path = _ctx.filePath + texname;
		newTexture = m_sceneMgr->getVideoDriver()->findTexture(path) == 0;
		// try to read from the path relative to where the .baw is loaded from
		texture = m_sceneMgr->getVideoDriver()->getTexture(path);
	}
	//! @todo @bug Do somemthing with `newTexture`? In obj loader something happens in case where newTexture is true

	return texture;
}
template<>
ICPUMeshBuffer* CBAWMeshFileLoader::make<ICPUMeshBuffer>(SBlobData& _data, SContext& _ctx) const
{
	typename core::CorrespondingBlobTypeFor<ICPUMeshBuffer>::type* blob = core::toBlobPtr<ICPUMeshBuffer>(_data.blob);

	ICPUMeshBuffer* buf = new scene::ICPUMeshBuffer();
	memcpy(&buf->getMaterial(), &blob->mat, sizeof(video::SMaterial));
	buf->setBoundingBox(blob->box);
	buf->setMeshDataAndFormat(loadBlob<IMeshDataFormatDesc<core::ICPUBuffer> >(_ctx.blobs[blob->descPtr], _ctx));
	buf->setIndexType(blob->indexType);
	buf->setBaseVertex(blob->baseVertex);
	buf->setIndexCount(blob->indexCount);
	buf->setIndexBufferOffset(blob->indexBufOffset);
	buf->setInstanceCount(blob->instanceCount);
	buf->setBaseInstance(blob->baseInstance);
	buf->setPrimitiveType(blob->primitiveType);
	buf->setPositionAttributeIx(blob->posAttrId);
	for (uint32_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
	{
		uint64_t tex = reinterpret_cast<uint64_t>(buf->getMaterial().getTexture(i));
		if (tex)
			buf->getMaterial().setTexture(i, loadBlob<video::IVirtualTexture>(_ctx.blobs[tex], _ctx));
	}

	return buf;
}
template<>
SCPUSkinMeshBuffer* CBAWMeshFileLoader::make<SCPUSkinMeshBuffer>(SBlobData& _data, SContext& _ctx) const
{
	typename core::CorrespondingBlobTypeFor<SCPUSkinMeshBuffer>::type* blob = core::toBlobPtr<SCPUSkinMeshBuffer>(_data.blob);

	SCPUSkinMeshBuffer* buf = new scene::SCPUSkinMeshBuffer();
	memcpy(&buf->getMaterial(), &blob->mat, sizeof(video::SMaterial));
	buf->setBoundingBox(blob->box);
	buf->setMeshDataAndFormat(loadBlob<IMeshDataFormatDesc<core::ICPUBuffer> >(_ctx.blobs[blob->descPtr], _ctx));
	buf->setIndexType(blob->indexType);
	buf->setBaseVertex(blob->baseVertex);
	buf->setIndexCount(blob->indexCount);
	buf->setIndexBufferOffset(blob->indexBufOffset);
	buf->setInstanceCount(blob->instanceCount);
	buf->setBaseInstance(blob->baseInstance);
	buf->setPrimitiveType(blob->primitiveType);
	buf->setPositionAttributeIx(blob->posAttrId);
	buf->setIndexRange(blob->indexValMin, blob->indexValMax);
	buf->setMaxVertexBoneInfluences(blob->maxVertexBoneInfluences);
	for (uint32_t i = 0; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
	{
		uint64_t tex = reinterpret_cast<uint64_t>(buf->getMaterial().getTexture(i));
		if (tex)
			buf->getMaterial().setTexture(i, loadBlob<video::IVirtualTexture>(_ctx.blobs[tex], _ctx));
	}

	return buf;
}
template<>
CFinalBoneHierarchy* CBAWMeshFileLoader::make<CFinalBoneHierarchy>(SBlobData& _data, SContext& _ctx) const
{	typename core::CorrespondingBlobTypeFor<CFinalBoneHierarchy>::type* blob = core::toBlobPtr<CFinalBoneHierarchy>(_data.blob);

	CFinalBoneHierarchy* fbh = new CFinalBoneHierarchy(blob);

	return new CFinalBoneHierarchy(blob);
}
template<>
ICPUMesh* CBAWMeshFileLoader::make<ICPUMesh>(SBlobData& _data, SContext& _ctx) const
{
	typename core::CorrespondingBlobTypeFor<ICPUMesh>::type* blob = core::toBlobPtr<ICPUMesh>(_data.blob);

	SCPUMesh* mesh = new scene::SCPUMesh();
	mesh->setBoundingBox(blob->box);
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		mesh->addMeshBuffer(loadBlob<ICPUMeshBuffer>(_ctx.blobs[blob->meshBufPtrs[i]], _ctx));

	return mesh;
}
template<>
ICPUSkinnedMesh* CBAWMeshFileLoader::make<ICPUSkinnedMesh>(SBlobData& _data, SContext& _ctx) const
{
	typename core::CorrespondingBlobTypeFor<ICPUSkinnedMesh>::type* blob = core::toBlobPtr<ICPUSkinnedMesh>(_data.blob);

	CCPUSkinnedMesh* mesh = new scene::CCPUSkinnedMesh();
	mesh->setBoneReferenceHierarchy(loadBlob<CFinalBoneHierarchy>(_ctx.blobs[blob->boneHierarchyPtr], _ctx));
	mesh->setBoundingBox(blob->box);
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		mesh->addMeshBuffer(loadBlob<SCPUSkinMeshBuffer>(_ctx.blobs[blob->meshBufPtrs[i]], _ctx));
	// shall i call mesh->finalize()?

	return mesh;
}

ICPUMesh* CBAWMeshFileLoader::createMesh(io::IReadFile* _file)
{
	SContext ctx;
	if (!verifyFile(_file, ctx))
		return 0;

	const uint32_t filesize = _file->getSize();
	uint8_t* const fileBuffer = (uint8_t*)malloc(filesize);
	_file->read(fileBuffer, filesize);
	const core::BawFileV1* const bawFile = (core::BawFileV1*)fileBuffer;
	const uint32_t HEADERS_FILE_OFFSET = bawFile->calcHeadersOffset();
	const uint32_t BLOBS_FILE_OFFSET = bawFile->calcBlobsOffset();

	ctx.filePath = _file->getFileName();
	if (ctx.filePath[ctx.filePath.size()-1] != '/')
		ctx.filePath += "/";
	std::map<uint64_t, SBlobData>::iterator meshBlobData;

	const core::BlobHeaderV1* const headers = (core::BlobHeaderV1*)(fileBuffer + HEADERS_FILE_OFFSET);
	for (int i = 0; i < bawFile->numOfInternalBlobs; ++i)
	{
		SBlobData pair = SBlobData{headers+i, fileBuffer + BLOBS_FILE_OFFSET + bawFile->blobOffsets[i], NULL};
		const std::map<uint64_t, SBlobData>::iterator it = ctx.blobs.insert(std::make_pair(headers[i].handle, pair)).first;
		if (pair.header->blobType == core::Blob::EBT_MESH || pair.header->blobType == core::Blob::EBT_SKINNED_MESH)
			meshBlobData = it;
	}

	/** Validate your headers here, that:
    A-their blobs do not run over each other
    B-that their data's offsets are in ascending order
    C-that the last blob's data does not run past the end of the array
    **/

	ICPUMesh* mesh;
	switch (meshBlobData->second.header->blobType)
	{
	case core::Blob::EBT_MESH:
		mesh = loadBlob<ICPUMesh>(meshBlobData->second, ctx);
		break;
	case core::Blob::EBT_SKINNED_MESH:
		mesh = loadBlob<ICPUSkinnedMesh>(meshBlobData->second, ctx);
		break;
	}

	free(fileBuffer);
	return mesh;
}

template<typename T>
T* CBAWMeshFileLoader::loadBlob(SBlobData& _data, SContext& _ctx) const
{
    void* const specialDeadPtr = reinterpret_cast<void* const>(0xdeadbeefDEADBEEFull);

	if (_data.createdObj)
		return reinterpret_cast<T*>(_data.createdObj==specialDeadPtr ? NULL:_data.createdObj); //faster this way

    //! Here you should actually seek, and load from the file

    T* retval = NULL;
    if (_data.header->validate(_data.blob))
        retval = make<T>(_data,_ctx);

	_data.createdObj = retval ? retval:specialDeadPtr;
	return retval;
}

bool CBAWMeshFileLoader::verifyFile(io::IReadFile * _file, SContext& _ctx) const
{
	if (_file->getSize() < sizeof(core::BawFileV1::fileHeader))
		return false;

	const char * const headerStrPattern = "IrrlichtBaW BinaryFile";
	char headerStr[sizeof(core::BawFileV1::fileHeader)];
	_file->read(headerStr, sizeof(headerStr));

	if (strcmp(headerStr, headerStrPattern) != 0)
		return false;
	_ctx.fileVersion = ((uint64_t*)headerStr)[3];
	if (_ctx.fileVersion>=1)
        return false;

	_file->seek(0);

	return true;
}

}} // irr::scene
