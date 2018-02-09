// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#include "CBAWMeshWriter.h"

#include "IFileSystem.h"
#include "IWriteFile.h"
#include "irrArray.h"
#include "ITexture.h"
#include "irrTypes.h"
#include "ISkinnedMesh.h"
#include "CFinalBoneHierarchy.h"

#define BAW_FILE_VERSION 0

namespace irr {
	namespace scene
	{

		const char * const CBAWMeshWriter::BAW_FILE_HEADER = "IrrlichtBaW BinaryFile\0\0\0\0\0\0\0\0\0";

		CBAWMeshWriter::CBAWMeshWriter(io::IFileSystem* _fs) : m_fileSystem(_fs)
		{
#ifdef _DEBUG
			setDebugName("CBAWMeshWriter");
#endif
		}

		template<>
		void CBAWMeshWriter::exportAsBlob<ICPUMesh>(ICPUMesh* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx)
		{
			uint8_t stackData[1u<<14];
			core::MeshBlobV0* data = core::MeshBlobV0::createAndTryOnStack(_obj, stackData, sizeof(stackData));

			tryWrite(data, _file, _ctx, core::MeshBlobV0::calcBlobSizeForObj(_obj), _headerIdx);

			if ((uint8_t*)data != stackData)
				free(data);
		}
		template<>
		void CBAWMeshWriter::exportAsBlob<ICPUSkinnedMesh>(ICPUSkinnedMesh* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx)
		{
			uint8_t stackData[1u << 14];
			core::SkinnedMeshBlobV0* data = core::SkinnedMeshBlobV0::createAndTryOnStack(_obj,stackData,sizeof(stackData));

			tryWrite(data, _file, _ctx, core::SkinnedMeshBlobV0::calcBlobSizeForObj(_obj), _headerIdx);

			if ((uint8_t*)data != stackData)
				free(data);
		}
		template<>
		void CBAWMeshWriter::exportAsBlob<ICPUMeshBuffer>(ICPUMeshBuffer* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx)
		{
			const core::MeshBufferBlobV0 data(_obj);

			tryWrite(&data, _file, _ctx, sizeof(data), _headerIdx);
		}
		template<>
		void CBAWMeshWriter::exportAsBlob<SCPUSkinMeshBuffer>(SCPUSkinMeshBuffer* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx)
		{
			const core::SkinnedMeshBufferBlobV0 data(_obj);

			tryWrite(&data, _file, _ctx, sizeof(data), _headerIdx);
		}
		template<>
		void CBAWMeshWriter::exportAsBlob<video::IVirtualTexture>(video::IVirtualTexture* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx)
		{
			video::ITexture* tex;
			if (!(tex = dynamic_cast<video::ITexture*>(_obj)))
				return pushCorruptedOffset(_ctx);

			const io::path fileDir = io::IFileSystem::getFileDir(m_fileSystem->getAbsolutePath(_file->getFileName())); // get out-file directory
			const io::path path = m_fileSystem->getRelativeFilename(tex->getName().getInternalName(), fileDir); // get texture-file path relative to out-file's directory
			const uint32_t len = strlen(path.c_str()) + 1;

			tryWrite(path.c_str(), _file, _ctx, len, _headerIdx);
		}
		template<>
		void CBAWMeshWriter::exportAsBlob<scene::CFinalBoneHierarchy>(scene::CFinalBoneHierarchy* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx)
		{
			uint8_t stackData[1u<<14]; // 16kB
			core::FinalBoneHierarchyBlobV0* data = core::FinalBoneHierarchyBlobV0::createAndTryOnStack(_obj,stackData,sizeof(stackData));

			tryWrite(data, _file, _ctx, core::FinalBoneHierarchyBlobV0::calcBlobSizeForObj(_obj), _headerIdx);

			if ((uint8_t*)data != stackData)
				free(data);
		}
		template<>
		void CBAWMeshWriter::exportAsBlob<IMeshDataFormatDesc<core::ICPUBuffer> >(IMeshDataFormatDesc<core::ICPUBuffer>* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx)
		{
			const core::MeshDataFormatDescBlobV0 data(_obj);

			tryWrite(&data, _file, _ctx, sizeof(data), _headerIdx);
		}
		template<>
		void CBAWMeshWriter::exportAsBlob<core::ICPUBuffer>(core::ICPUBuffer* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx)
		{
			tryWrite(_obj->getPointer(), _file, _ctx, _obj->getSize(), _headerIdx);
		}

		bool CBAWMeshWriter::writeMesh(io::IWriteFile* _file, ICPUMesh* _mesh, int32_t _flags)
		{
			if (!_mesh || !_file)
				return false;

			const uint32_t FILE_HEADER_SIZE = 32;
			_IRR_DEBUG_BREAK_IF(FILE_HEADER_SIZE != sizeof(core::BAWFileV0::fileHeader))

			uint64_t header[4];
			memcpy(header, BAW_FILE_HEADER, FILE_HEADER_SIZE);
			header[3] = BAW_FILE_VERSION;

			_file->write(header, FILE_HEADER_SIZE);

			SContext ctx; // context of this call of `writeMesh`

			const uint32_t numOfInternalBlobs = genHeaders(_mesh, ctx);
			const uint32_t OFFSETS_FILE_OFFSET = FILE_HEADER_SIZE + sizeof(uint32_t);
			const uint32_t HEADERS_FILE_OFFSET = OFFSETS_FILE_OFFSET + numOfInternalBlobs*sizeof(ctx.offsets[0]);

			ctx.offsets.set_used(numOfInternalBlobs);

			_file->write(&numOfInternalBlobs, sizeof(numOfInternalBlobs));
			// will be overwritten after actually calculating offsets
			_file->write(ctx.offsets.const_pointer(), ctx.offsets.size() * sizeof(ctx.offsets[0]));

			// will be overwritten after calculating not known yet data (hash and size for texture paths)
			_file->write(ctx.headers.const_pointer(), ctx.headers.size() * sizeof(core::BlobHeaderV0));

			ctx.offsets.set_used(0); // set `used` to 0, to allow push starting from 0 index
			for (int i = 0; i < ctx.headers.size(); ++i)
			{
				switch (ctx.headers[i].blobType)
				{
				case core::Blob::EBT_MESH:
					exportAsBlob(reinterpret_cast<ICPUMesh*>(ctx.headers[i].handle), i, _file, ctx);
					break;
				case core::Blob::EBT_SKINNED_MESH:
					exportAsBlob(reinterpret_cast<ICPUSkinnedMesh*>(ctx.headers[i].handle), i, _file, ctx);
					break;
				case core::Blob::EBT_MESH_BUFFER:
					exportAsBlob(reinterpret_cast<ICPUMeshBuffer*>(ctx.headers[i].handle), i, _file, ctx);
					break;
				case core::Blob::EBT_SKINNED_MESH_BUFFER:
					exportAsBlob(reinterpret_cast<SCPUSkinMeshBuffer*>(ctx.headers[i].handle), i, _file, ctx);
					break;
				case core::Blob::EBT_RAW_DATA_BUFFER:
					exportAsBlob(reinterpret_cast<core::ICPUBuffer*>(ctx.headers[i].handle), i, _file, ctx);
					break;
				case core::Blob::EBT_DATA_FORMAT_DESC:
					exportAsBlob(reinterpret_cast<IMeshDataFormatDesc<core::ICPUBuffer>*>(ctx.headers[i].handle), i, _file, ctx);
					break;
				case core::Blob::EBT_FINAL_BONE_HIERARCHY:
					exportAsBlob(reinterpret_cast<CFinalBoneHierarchy*>(ctx.headers[i].handle), i, _file, ctx);
					break;
				case core::Blob::EBT_TEXTURE_PATH:
					exportAsBlob(reinterpret_cast<video::IVirtualTexture*>(ctx.headers[i].handle), i, _file, ctx);
					break;
				}
			}

			const size_t prevPos = _file->getPos();

			// overwrite offsets
			_file->seek(OFFSETS_FILE_OFFSET);
			_file->write(ctx.offsets.const_pointer(), ctx.offsets.size() * sizeof(ctx.offsets[0]));
			// overwrite headers
			_file->seek(HEADERS_FILE_OFFSET);
			_file->write(ctx.headers.const_pointer(), ctx.headers.size() * sizeof(core::BlobHeaderV0));

			_file->seek(prevPos);

			return true;
		}

		uint32_t CBAWMeshWriter::genHeaders(ICPUMesh* _mesh, SContext& _ctx)
		{
			_ctx.headers.clear();

			bool isMeshAnimated = true;
			ICPUSkinnedMesh* skinnedMesh = 0;

			if (_mesh)
			{
				skinnedMesh = _mesh->getMeshType()!=EMT_ANIMATED_SKINNED ? NULL:reinterpret_cast<ICPUSkinnedMesh*>(_mesh); //ICPUSkinnedMesh is a direct non-virtual inheritor
				if (!skinnedMesh || (skinnedMesh && skinnedMesh->isStatic()))
					isMeshAnimated = false;

				core::BlobHeaderV0 bh;
				bh.handle = reinterpret_cast<uint64_t>(_mesh);
				bh.compressionType = core::Blob::EBCT_RAW;
				bh.blobType = isMeshAnimated ? core::Blob::EBT_SKINNED_MESH : core::Blob::EBT_MESH;
				_ctx.headers.push_back(bh);
				// no need to add to `countedObjects` set since there's only one mesh
			}
			else return 0;

			if (isMeshAnimated)
			{
				core::BlobHeaderV0 bh;
				bh.handle = reinterpret_cast<uint64_t>(skinnedMesh->getBoneReferenceHierarchy());
				bh.compressionType = core::Blob::EBCT_RAW;
				bh.blobType = core::Blob::EBT_FINAL_BONE_HIERARCHY;
				_ctx.headers.push_back(bh);
				// no need to add to `countedObjects` set since there's only one bone hierarchy
			}

			std::set<const IReferenceCounted*> countedObjects;
			for (uint32_t i = 0; i < _mesh->getMeshBufferCount(); ++i)
			{
				const ICPUMeshBuffer* const meshBuffer = _mesh->getMeshBuffer(i);
				const IMeshDataFormatDesc<core::ICPUBuffer>* const desc = meshBuffer->getMeshDataAndFormat();

				if (!meshBuffer || !desc)
					continue;

				if (countedObjects.find(meshBuffer) == countedObjects.end())
				{
					core::BlobHeaderV0 bh;
					bh.handle = reinterpret_cast<uint64_t>(meshBuffer);
					bh.compressionType = core::Blob::EBCT_RAW;
					bh.blobType = isMeshAnimated ? core::Blob::EBT_SKINNED_MESH_BUFFER : core::Blob::EBT_MESH_BUFFER;
					_ctx.headers.push_back(bh);
					countedObjects.insert(meshBuffer);

					const video::SMaterial & mat = meshBuffer->getMaterial();
					for (int tid = 0; tid < _IRR_MATERIAL_MAX_TEXTURES_; ++tid) // texture path blob headers
					{
						video::IVirtualTexture* texture = mat.getTexture(tid);
						if (mat.getTexture(tid) && countedObjects.find(texture) == countedObjects.end())
						{
							bh.handle = reinterpret_cast<uint64_t>(texture);
							bh.compressionType = core::Blob::EBCT_RAW;
							bh.blobType = core::Blob::EBT_TEXTURE_PATH;
							_ctx.headers.push_back(bh);
							countedObjects.insert(texture);
						}
						else continue;
					}
				}

				if (countedObjects.find(desc) == countedObjects.end())
				{
					core::BlobHeaderV0 bh;
					bh.handle = reinterpret_cast<uint64_t>(desc);
					bh.compressionType = core::Blob::EBCT_RAW;
					bh.blobType = core::Blob::EBT_DATA_FORMAT_DESC;
					_ctx.headers.push_back(bh);
					countedObjects.insert(desc);
				}

				const core::ICPUBuffer* idxBuffer = desc->getIndexBuffer();
				if (idxBuffer && countedObjects.find(idxBuffer) == countedObjects.end())
				{
					core::BlobHeaderV0 bh;
					bh.handle = reinterpret_cast<uint64_t>(idxBuffer);
					bh.compressionType = core::Blob::EBCT_RAW;
					bh.blobType = core::Blob::EBT_RAW_DATA_BUFFER;
					_ctx.headers.push_back(bh);
					countedObjects.insert(desc->getIndexBuffer());
				}

				for (int attId = 0; attId < EVAI_COUNT; ++attId)
				{
					const core::ICPUBuffer* attBuffer = desc->getMappedBuffer((E_VERTEX_ATTRIBUTE_ID)attId);
					if (attBuffer && countedObjects.find(attBuffer) == countedObjects.end())
					{
						core::BlobHeaderV0 bh;
						bh.handle = reinterpret_cast<uint64_t>(attBuffer);
						bh.compressionType = core::Blob::EBCT_RAW;
						bh.blobType = core::Blob::EBT_RAW_DATA_BUFFER;
						bh.blobSize = bh.blobSizeDecompr = attBuffer->getSize();
						_ctx.headers.push_back(bh);
						countedObjects.insert(attBuffer);
					}
				}
			}
			return _ctx.headers.size();
		}

		void CBAWMeshWriter::calcAndPushNextOffset(uint32_t _blobSize, SContext& _ctx) const
		{
			_ctx.offsets.push_back(!_ctx.offsets.size() ? 0 : _ctx.offsets.getLast() + _blobSize);
		}

		void CBAWMeshWriter::tryWrite(const void * _data, io::IWriteFile * _file, SContext & _ctx, size_t _size, uint32_t _headerIdx) const
		{
			if (!_data)
				return pushCorruptedOffset(_ctx);
			_ctx.headers[_headerIdx].finalize(_data, _size);
			_file->write(_data, _ctx.headers[_headerIdx].blobSizeDecompr);
			calcAndPushNextOffset(!_headerIdx ? 0 : _ctx.headers[_headerIdx - 1].blobSizeDecompr, _ctx);
		}

}} // end ns irr::scene
