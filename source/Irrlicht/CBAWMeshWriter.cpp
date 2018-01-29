// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#include "CBAWMeshWriter.h"

#include "IFileSystem.h"
#include "IWriteFile.h"
#include "irrArray.h"
#include "ITexture.h"
#include "coreutil.h"
#include "irrTypes.h"

namespace irr {
	namespace scene
	{

		const char * const CBAWMeshWriter::BAW_FILE_HEADER = "IrrlichtBaW BinaryFile\0\0\0\0\0\0\0\0\0";

		CBAWMeshWriter::~CBAWMeshWriter() {}

		CBAWMeshWriter::CBAWMeshWriter()
		{
#ifdef _DEBUG
			setDebugName("CBAWMeshWriter");
#endif
		}

		template<>
		void CBAWMeshWriter::exportAsBlob<ICPUMeshBuffer>(ICPUMeshBuffer* _obj, uint32_t _headerIdx, io::IWriteFile* _file)
		{
			uint8_t* const data = (uint8_t*)std::malloc(m_headers[_headerIdx].blobSizeDecompr);
			uint8_t* nearestFree = data;
			std::memcpy(nearestFree, &_obj->getMaterial(), sizeof(video::SMaterial));
			nearestFree += sizeof(video::SMaterial);
			std::memcpy(nearestFree, &_obj->getBoundingBox(), sizeof(core::aabbox3df));
			nearestFree += sizeof(core::aabbox3df);
			*((uint64_t *)nearestFree) = reinterpret_cast<uint64_t>(_obj->getMeshDataAndFormat());
			nearestFree += sizeof(uint64_t);
			*((video::E_INDEX_TYPE *)nearestFree) = _obj->getIndexType();
			nearestFree += sizeof(video::E_INDEX_TYPE);
			*((int32_t *)nearestFree) = _obj->getBaseVertex();
			nearestFree += sizeof(int32_t);
			*((uint64_t *)nearestFree) = _obj->getIndexCount();
			nearestFree += sizeof(uint64_t);
			*((size_t *)nearestFree) = _obj->getIndexBufferOffset();
			nearestFree += sizeof(size_t);
			*((size_t *)nearestFree) = _obj->getInstanceCount();
			nearestFree += sizeof(size_t);
			*((uint32_t *)nearestFree) = _obj->getBaseInstance();
			nearestFree += sizeof(uint32_t);
			*((E_PRIMITIVE_TYPE *)nearestFree) = _obj->getPrimitiveType();
			nearestFree += sizeof(E_PRIMITIVE_TYPE);
			*((E_VERTEX_ATTRIBUTE_ID *)nearestFree) = _obj->getPositionAttributeIx();
			nearestFree += sizeof(E_VERTEX_ATTRIBUTE_ID);

			core::XXHash_256(data, m_headers[_headerIdx].blobSizeDecompr, m_headers[_headerIdx].blobHash);
			_file->write(data, m_headers[_headerIdx].blobSizeDecompr);
			calcAndPushNextOffset(!_headerIdx ? 0 : m_headers[_headerIdx-1].blobSizeDecompr);

			std::free(data);
		}
		template<>
		void CBAWMeshWriter::exportAsBlob<video::IVirtualTexture>(video::IVirtualTexture* _obj, uint32_t _headerIdx, io::IWriteFile* _file)
		{
			video::ITexture* tex;
			if (!(tex = dynamic_cast<video::ITexture*>(_obj)))
				return;
			//! @todo @bug here we get absolute path and we want relative
			const io::path path = tex->getName().getInternalName();
			const uint32_t len = std::strlen(path.c_str()) + 1;

			m_headers[_headerIdx].blobSize = m_headers[_headerIdx].blobSizeDecompr = len;
			core::XXHash_256(path.c_str(), len, m_headers[_headerIdx].blobHash);
			calcAndPushNextOffset(!_headerIdx ? 0 : m_headers[_headerIdx-1].blobSizeDecompr);

			_file->write(path.c_str(), len);
		}
		template<>
		void CBAWMeshWriter::exportAsBlob<IMeshDataFormatDesc<core::ICPUBuffer> >(IMeshDataFormatDesc<core::ICPUBuffer>* _obj, uint32_t _headerIdx, io::IWriteFile* _file)
		{
			uint8_t* const data = (uint8_t*)std::malloc(m_headers[_headerIdx].blobSizeDecompr);

			E_COMPONENTS_PER_ATTRIBUTE* cpa = (E_COMPONENTS_PER_ATTRIBUTE*)data;
			E_COMPONENT_TYPE* attType = (E_COMPONENT_TYPE*)((uint8_t*)cpa + EVAI_COUNT * sizeof(E_COMPONENTS_PER_ATTRIBUTE));
			size_t* attStride = (size_t*)((uint8_t*)attType + EVAI_COUNT * sizeof(E_COMPONENT_TYPE));
			size_t* attOffset = (size_t*)((uint8_t*)attStride + EVAI_COUNT * sizeof(size_t));
			uint32_t* attDiv = (uint32_t*)((uint8_t*)attOffset + EVAI_COUNT * sizeof(size_t));
			uint64_t* attBufs = (uint64_t*)((uint8_t*)attDiv + EVAI_COUNT * sizeof(uint32_t));

			uint64_t* idxBuf = (uint64_t*)((uint8_t*)attBufs + EVAI_COUNT * sizeof(uint64_t));

			for (int i = 0; i < EVAI_COUNT; ++i)
			{
				E_VERTEX_ATTRIBUTE_ID vaid = (E_VERTEX_ATTRIBUTE_ID)i;
				cpa[i] = _obj->getAttribComponentCount(vaid);
				attType[i] = _obj->getAttribType(vaid);
				attStride[i] = _obj->getMappedBufferStride(vaid);
				attOffset[i] = _obj->getMappedBufferOffset(vaid);
				attDiv[i] = _obj->getAttribDivisor(vaid);
				attBufs[i] = reinterpret_cast<uint64_t>(_obj->getMappedBuffer(vaid));
			}
			idxBuf[0] = reinterpret_cast<uint64_t>(_obj->getIndexBuffer());

			core::XXHash_256(data, m_headers[_headerIdx].blobSizeDecompr, m_headers[_headerIdx].blobHash);
			_file->write(data, m_headers[_headerIdx].blobSizeDecompr);
			calcAndPushNextOffset(!_headerIdx ? 0 : m_headers[_headerIdx-1].blobSizeDecompr);

			std::free(data);
		}
		template<>
		void CBAWMeshWriter::exportAsBlob<core::ICPUBuffer>(core::ICPUBuffer* _obj, uint32_t _headerIdx, io::IWriteFile* _file)
		{
			core::XXHash_256(_obj->getPointer(), _obj->getSize(), m_headers[_headerIdx].blobHash);
			_file->write(_obj->getPointer(), _obj->getSize());
			calcAndPushNextOffset(!_headerIdx ? 0 : m_headers[_headerIdx-1].blobSizeDecompr);
		}

		bool CBAWMeshWriter::writeMesh(io::IWriteFile* _file, ICPUMesh* _mesh, int32_t _flags)
		{
			if (!_mesh || !_file)
				return false;

			const uint32_t FILE_HEADER_SIZE = 32;
			_IRR_DEBUG_BREAK_IF(FILE_HEADER_SIZE != sizeof(core::BawFile::fileHeader))

			_file->write(BAW_FILE_HEADER, FILE_HEADER_SIZE);

			const uint32_t numOfInternalBlobs = genHeaders(_mesh);
			const uint32_t OFFSETS_FILE_OFFSET = FILE_HEADER_SIZE + sizeof(uint32_t);
			const uint32_t HEADERS_FILE_OFFSET = OFFSETS_FILE_OFFSET + numOfInternalBlobs*sizeof(m_offsets[0]);

			m_offsets.set_used(numOfInternalBlobs);

			_file->write(&numOfInternalBlobs, sizeof(numOfInternalBlobs));
			// will be overwritten after actually calculating offsets
			_file->write(m_offsets.const_pointer(), m_offsets.size() * sizeof(m_offsets[0]));

			// will be overwritten after calculating not known yet data (hash and size for texture paths)
			_file->write(m_headers.const_pointer(), m_headers.size() * sizeof(core::BlobHeader));

			m_offsets.set_used(0); // set `used` to 0, to allow push starting from 0 index
			for (int i = 0; i < m_headers.size(); ++i)
			{
				switch (m_headers[i].blobType)
				{
				case core::Blob::EBT_MESH_BUFFER:
					exportAsBlob(reinterpret_cast<ICPUMeshBuffer*>(m_headers[i].handle), i, _file);
					break;
				case core::Blob::EBT_RAW_DATA_BUFFER:
					exportAsBlob(reinterpret_cast<core::ICPUBuffer*>(m_headers[i].handle), i, _file);
					break;
				case core::Blob::EBT_DATA_FORMAT_DESC:
					exportAsBlob(reinterpret_cast<IMeshDataFormatDesc<core::ICPUBuffer>*>(m_headers[i].handle), i, _file);
					break;
				case core::Blob::EBT_TEXTURE_PATH:
					exportAsBlob(reinterpret_cast<video::IVirtualTexture*>(m_headers[i].handle), i, _file);
					break;
				}
			}

			const size_t prevPos = _file->getPos();

			// overwrite offsets
			_file->seek(OFFSETS_FILE_OFFSET);
			_file->write(m_offsets.const_pointer(), m_offsets.size() * sizeof(m_offsets[0]));
			// overwrite headers
			_file->seek(HEADERS_FILE_OFFSET);
			_file->write(m_headers.const_pointer(), m_headers.size() * sizeof(core::BlobHeader));

			_file->seek(prevPos);

			return true;
		}

		uint32_t CBAWMeshWriter::genHeaders(ICPUMesh * _mesh)
		{
			m_headers.clear();
			std::set<const IReferenceCounted*> countedObjects;
			for (uint32_t i = 0; i < _mesh->getMeshBufferCount(); ++i)
			{
				const ICPUMeshBuffer * const meshBuffer = _mesh->getMeshBuffer(i);
				const IMeshDataFormatDesc<core::ICPUBuffer> * const desc = meshBuffer->getMeshDataAndFormat();

				if (!meshBuffer || !desc)
					continue;

				if (countedObjects.find(meshBuffer) == countedObjects.end())
				{
					core::BlobHeader bh;
					bh.handle = reinterpret_cast<uint64_t>(meshBuffer);
					bh.compressionType = core::Blob::EBCT_RAW;
					bh.blobType = core::Blob::EBT_MESH_BUFFER;
					bh.blobSize = bh.blobSizeDecompr = // size of significant members of ICPUMeshBuffer
						sizeof(uint64_t) + sizeof(video::E_INDEX_TYPE) + 2 * sizeof(uint32_t) + 2 * sizeof(size_t) + sizeof(uint64_t) + sizeof(E_PRIMITIVE_TYPE) + sizeof(E_VERTEX_ATTRIBUTE_ID)
						+ sizeof(video::SMaterial)
						+ sizeof(core::aabbox3df);
					m_headers.push_back(bh);
					countedObjects.emplace(meshBuffer);

					const video::SMaterial & mat = meshBuffer->getMaterial();
					for (int tid = 0; tid < _IRR_MATERIAL_MAX_TEXTURES_; ++tid) // texture path blob headers
					{
						video::IVirtualTexture* texture = mat.getTexture(tid);
						if (mat.getTexture(tid) && countedObjects.find(texture) == countedObjects.end())
						{
							bh.handle = reinterpret_cast<uint64_t>(texture);
							bh.compressionType = core::Blob::EBCT_RAW;
							bh.blobType = core::Blob::EBT_TEXTURE_PATH;
							bh.blobSize = bh.blobSizeDecompr = 0; // must be overwritten while exporting blobs
							m_headers.push_back(bh);
							countedObjects.emplace(texture);
						}
						else continue;
					}
				}

				if (countedObjects.find(desc) == countedObjects.end())
				{
					core::BlobHeader bh;
					bh.handle = reinterpret_cast<uint64_t>(desc);
					bh.compressionType = core::Blob::EBCT_RAW;
					bh.blobType = core::Blob::EBT_DATA_FORMAT_DESC;
					bh.blobSize = bh.blobSizeDecompr = // size of members of ICPUMeshDataFormatDesc
						EVAI_COUNT
						* (sizeof(E_COMPONENTS_PER_ATTRIBUTE) + sizeof(E_COMPONENT_TYPE) + 2 * sizeof(size_t) + sizeof(uint32_t) + sizeof(uint64_t))
						+ sizeof(uint64_t);
					m_headers.push_back(bh);
					countedObjects.emplace(desc);
				}

				const core::ICPUBuffer* idxBuffer = desc->getIndexBuffer();
				if (idxBuffer && countedObjects.find(idxBuffer) == countedObjects.end())
				{
					core::BlobHeader bh;
					bh.handle = reinterpret_cast<uint64_t>(idxBuffer);
					bh.compressionType = core::Blob::EBCT_RAW;
					bh.blobType = core::Blob::EBT_RAW_DATA_BUFFER;
					bh.blobSize = bh.blobSizeDecompr = idxBuffer->getSize();
					m_headers.push_back(bh);
					countedObjects.emplace(desc->getIndexBuffer());
				}

				for (int attId = 0; attId < EVAI_COUNT; ++attId)
				{
					const core::ICPUBuffer* attBuffer = desc->getMappedBuffer((E_VERTEX_ATTRIBUTE_ID)attId);
					if (attBuffer && countedObjects.find(attBuffer) == countedObjects.end())
					{
						core::BlobHeader bh;
						bh.handle = reinterpret_cast<uint64_t>(attBuffer);
						bh.compressionType = core::Blob::EBCT_RAW;
						bh.blobType = core::Blob::EBT_RAW_DATA_BUFFER;
						bh.blobSize = bh.blobSizeDecompr = attBuffer->getSize();
						m_headers.push_back(bh);
						countedObjects.emplace(attBuffer);
					}
				}
			}
			return m_headers.size();
		}

		void CBAWMeshWriter::calcAndPushNextOffset(uint32_t _blobSize)
		{
			m_offsets.push_back(!m_offsets.size() ? 0 : m_offsets.getLast() + _blobSize);
		}

	}
} // end ns irr::scene