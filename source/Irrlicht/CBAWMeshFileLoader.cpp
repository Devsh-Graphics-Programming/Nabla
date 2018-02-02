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
core::ICPUBuffer* CBAWMeshFileLoader::make<core::ICPUBuffer>(SPair& _data, SContext&) const
{
	typename core::CorrespondingBlobTypeFor<core::ICPUBuffer>::type* blob = core::toBlobPtr<core::ICPUBuffer>(_data.blob);

	void* blobData = malloc(_data.header->blobSizeDecompr);
	memcpy(blobData, blob->getData(), _data.header->blobSizeDecompr);
	return new core::ICPUBuffer(_data.header->blobSizeDecompr, blobData);
}
template<>
IMeshDataFormatDesc<core::ICPUBuffer>* CBAWMeshFileLoader::make<IMeshDataFormatDesc<core::ICPUBuffer> >(SPair& _data, SContext& _ctx) const
{
	typename core::CorrespondingBlobTypeFor<IMeshDataFormatDesc<core::ICPUBuffer> >::type* blob = core::toBlobPtr<IMeshDataFormatDesc<core::ICPUBuffer> >(_data.blob);

	IMeshDataFormatDesc<core::ICPUBuffer>* desc = new scene::ICPUMeshDataFormatDesc();
	for (E_VERTEX_ATTRIBUTE_ID i = EVAI_ATTR0; i < EVAI_COUNT; i = E_VERTEX_ATTRIBUTE_ID((int)i + 1))
	{
		if (blob->attrBufPtrs[(int)i])
			desc->mapVertexAttrBuffer(
				make<core::ICPUBuffer>(_ctx.blobs[blob->attrBufPtrs[(int)i]], _ctx),
				i,
				blob->cpa[(int)i],
				blob->attrType[(int)i],
				blob->attrStride[(int)i],
				blob->attrOffset[(int)i],
				blob->attrDivisor[(int)i]
			);
	}
	if (blob->idxBufPtr)
		desc->mapIndexBuffer(make<core::ICPUBuffer>(_ctx.blobs[blob->idxBufPtr], _ctx));

	return desc;
}
template<>
video::IVirtualTexture* CBAWMeshFileLoader::make<video::IVirtualTexture>(SPair& _data, SContext& _ctx) const
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
ICPUMeshBuffer* CBAWMeshFileLoader::make<ICPUMeshBuffer>(SPair& _data, SContext& _ctx) const
{
	typename core::CorrespondingBlobTypeFor<ICPUMeshBuffer>::type* blob = core::toBlobPtr<ICPUMeshBuffer>(_data.blob);

	ICPUMeshBuffer* buf = new scene::ICPUMeshBuffer();
	memcpy(&buf->getMaterial(), &blob->mat, sizeof(video::SMaterial));
	buf->setBoundingBox(blob->box);
	buf->setMeshDataAndFormat(make<IMeshDataFormatDesc<core::ICPUBuffer> >(_ctx.blobs[blob->descPtr], _ctx));
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
			buf->getMaterial().setTexture(i, make<video::IVirtualTexture>(_ctx.blobs[tex], _ctx));
	}

	return buf;
}
template<>
SCPUSkinMeshBuffer* CBAWMeshFileLoader::make<SCPUSkinMeshBuffer>(SPair& _data, SContext& _ctx) const
{
	typename core::CorrespondingBlobTypeFor<SCPUSkinMeshBuffer>::type* blob = core::toBlobPtr<SCPUSkinMeshBuffer>(_data.blob);

	SCPUSkinMeshBuffer* buf = new scene::SCPUSkinMeshBuffer();
	memcpy(&buf->getMaterial(), &blob->mat, sizeof(video::SMaterial));
	buf->setBoundingBox(blob->box);
	buf->setMeshDataAndFormat(make<IMeshDataFormatDesc<core::ICPUBuffer> >(_ctx.blobs[blob->descPtr], _ctx));
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
			buf->getMaterial().setTexture(i, make<video::IVirtualTexture>(_ctx.blobs[tex], _ctx));
	}

	return buf;
}
template<>
CFinalBoneHierarchy* CBAWMeshFileLoader::make<CFinalBoneHierarchy>(SPair& _data, SContext& _ctx) const
{
	typename core::CorrespondingBlobTypeFor<CFinalBoneHierarchy>::type* blob = core::toBlobPtr<CFinalBoneHierarchy>(_data.blob);

	return new CFinalBoneHierarchy(blob);
}
template<>
ICPUMesh* CBAWMeshFileLoader::make<ICPUMesh>(SPair& _data, SContext& _ctx) const
{
	typename core::CorrespondingBlobTypeFor<ICPUMesh>::type* blob = core::toBlobPtr<ICPUMesh>(_data.blob);

	SCPUMesh* mesh = new scene::SCPUMesh();
	mesh->setBoundingBox(blob->box);
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		mesh->addMeshBuffer(make<ICPUMeshBuffer>(_ctx.blobs[blob->meshBufPtrs[i]], _ctx));

	return mesh;
}
template<>
ICPUSkinnedMesh* CBAWMeshFileLoader::make<ICPUSkinnedMesh>(SPair& _data, SContext& _ctx) const
{
	typename core::CorrespondingBlobTypeFor<ICPUSkinnedMesh>::type* blob = core::toBlobPtr<ICPUSkinnedMesh>(_data.blob);

	CCPUSkinnedMesh* mesh = new scene::CCPUSkinnedMesh();
	mesh->setBoneReferenceHierarchy(make<CFinalBoneHierarchy>(_ctx.blobs[blob->boneHierarchyPtr], _ctx));
	mesh->setBoundingBox(blob->box);
	for (uint32_t i = 0; i < blob->meshBufCnt; ++i)
		mesh->addMeshBuffer(make<SCPUSkinMeshBuffer>(_ctx.blobs[blob->meshBufPtrs[i]], _ctx));
	// shall i call mesh->finalize()?
	return mesh;
}

ICPUMesh* CBAWMeshFileLoader::createMesh(io::IReadFile* _file)
{
	const uint32_t filesize = _file->getSize();
	uint8_t* const fileBuffer = (uint8_t*)malloc(filesize);
	_file->read(fileBuffer, filesize);
	printf("XDDDDDDDDDDDDD\n");
	core::BawFileV1* const bawFile = (core::BawFileV1*)fileBuffer;
	const uint32_t HEADERS_FILE_OFFSET = sizeof(core::BawFileV1::fileHeader) + sizeof(core::BawFileV1::numOfInternalBlobs) + sizeof(core::BawFileV1::blobOffsets[0])*bawFile->numOfInternalBlobs;

	SContext ctx;
	ctx.filePath = _file->getFileName();
	if (ctx.filePath[ctx.filePath.size()-1] != '/')
		ctx.filePath += "/";
	std::map<uint64_t, SPair>::iterator meshBlobData;

	core::BlobHeaderV1* const headers = (core::BlobHeaderV1*)(fileBuffer + HEADERS_FILE_OFFSET);
	for (int i = 0; i < bawFile->numOfInternalBlobs; ++i)
	{
		SPair pair = SPair{headers+i, (uint8_t*)(headers) + bawFile->numOfInternalBlobs*sizeof(core::BlobHeaderV1) + bawFile->blobOffsets[i]};
		const std::map<uint64_t, SPair>::iterator it = ctx.blobs.insert(std::make_pair(headers[i].handle, pair)).first;
		if (pair.header->blobType == core::Blob::EBT_MESH || pair.header->blobType == core::Blob::EBT_SKINNED_MESH)
			meshBlobData = it;
	}

	ICPUMesh* mesh;
	switch (meshBlobData->second.header->blobType)
	{
	case core::Blob::EBT_MESH:
		mesh = make<ICPUMesh>(meshBlobData->second, ctx);
		break;
	case core::Blob::EBT_SKINNED_MESH:
		mesh = make<ICPUSkinnedMesh>(meshBlobData->second, ctx);
		break;
	}

	free(fileBuffer);
	return mesh;
}

}} // irr::scene