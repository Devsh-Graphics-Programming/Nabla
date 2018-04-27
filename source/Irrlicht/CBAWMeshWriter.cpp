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
#include "irrMacros.h"
#include "ISkinnedMesh.h"
#include "CFinalBoneHierarchy.h"
#include "os.h"
#include "lz4/lz4.h"
#include "lzma/LzmaEnc.h"

#define BAW_FILE_VERSION 0


namespace irr {namespace scene {

	const char * const CBAWMeshWriter::BAW_FILE_HEADER = "IrrlichtBaW BinaryFile\0\0\0\0\0\0\0\0\0";

	CBAWMeshWriter::CBAWMeshWriter(io::IFileSystem* _fs) : m_fileSystem(_fs)
	{
#ifdef _DEBUG
		setDebugName("CBAWMeshWriter");
#endif
	}

	template<>
	void CBAWMeshWriter::exportAsBlob<ICPUMesh>(ICPUMesh* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx, bool _compress)
	{
		uint8_t stackData[1u<<14];
		core::MeshBlobV0* data = core::MeshBlobV0::createAndTryOnStack(_obj, stackData, sizeof(stackData));

		tryWrite(data, _file, _ctx, core::MeshBlobV0::calcBlobSizeForObj(_obj), _headerIdx, _compress);

		if ((uint8_t*)data != stackData)
			free(data);
	}
	template<>
	void CBAWMeshWriter::exportAsBlob<ICPUSkinnedMesh>(ICPUSkinnedMesh* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx, bool _compress)
	{
		uint8_t stackData[1u << 14];
		core::SkinnedMeshBlobV0* data = core::SkinnedMeshBlobV0::createAndTryOnStack(_obj,stackData,sizeof(stackData));

		tryWrite(data, _file, _ctx, core::SkinnedMeshBlobV0::calcBlobSizeForObj(_obj), _headerIdx, _compress);

		if ((uint8_t*)data != stackData)
			free(data);
	}
	template<>
	void CBAWMeshWriter::exportAsBlob<ICPUMeshBuffer>(ICPUMeshBuffer* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx, bool _compress)
	{
		core::MeshBufferBlobV0 data(_obj);

		tryWrite(&data, _file, _ctx, sizeof(data), _headerIdx, _compress);
	}
	template<>
	void CBAWMeshWriter::exportAsBlob<SCPUSkinMeshBuffer>(SCPUSkinMeshBuffer* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx, bool _compress)
	{
		core::SkinnedMeshBufferBlobV0 data(_obj);

		tryWrite(&data, _file, _ctx, sizeof(data), _headerIdx, _compress);
	}
	template<>
	void CBAWMeshWriter::exportAsBlob<video::IVirtualTexture>(video::IVirtualTexture* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx, bool _compress)
	{
		video::ITexture* tex;
		if (!(tex = dynamic_cast<video::ITexture*>(_obj)))
			return pushCorruptedOffset(_ctx);

		const io::path fileDir = _ctx.props->relPath.size() ? _ctx.props->relPath : io::IFileSystem::getFileDir(m_fileSystem->getAbsolutePath(_file->getFileName())); // get relative-file's directory
		io::path path = m_fileSystem->getRelativeFilename(tex->getName().getInternalName(), fileDir); // get texture-file path relative to the file's directory
		const uint32_t len = strlen(path.c_str()) + 1;

		tryWrite(&path[0], _file, _ctx, len, _headerIdx, _compress);
	}
	template<>
	void CBAWMeshWriter::exportAsBlob<scene::CFinalBoneHierarchy>(scene::CFinalBoneHierarchy* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx, bool _compress)
	{
		uint8_t stackData[1u<<14]; // 16kB
		core::FinalBoneHierarchyBlobV0* data = core::FinalBoneHierarchyBlobV0::createAndTryOnStack(_obj,stackData,sizeof(stackData));

		tryWrite(data, _file, _ctx, core::FinalBoneHierarchyBlobV0::calcBlobSizeForObj(_obj), _headerIdx, _compress);

		if ((uint8_t*)data != stackData)
			free(data);
	}
	template<>
	void CBAWMeshWriter::exportAsBlob<IMeshDataFormatDesc<core::ICPUBuffer> >(IMeshDataFormatDesc<core::ICPUBuffer>* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx, bool _compress)
	{
		core::MeshDataFormatDescBlobV0 data(_obj);

		tryWrite(&data, _file, _ctx, sizeof(data), _headerIdx, _compress);
	}
	template<>
	void CBAWMeshWriter::exportAsBlob<core::ICPUBuffer>(core::ICPUBuffer* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx, bool _compress)
	{
		tryWrite(_obj->getPointer(), _file, _ctx, _obj->getSize(), _headerIdx, _compress);
	}

	bool CBAWMeshWriter::writeMesh(io::IWriteFile* _file, ICPUMesh* _mesh, int32_t _flags)
	{
		WriteProperties wp;
		if (!(_flags & EMWF_WRITE_COMPRESSED))
			wp.blobLz4ComprThresh = wp.blobLzmaComprThresh = 0xffffffffFFFFFFFFU;
		wp.encryptBlobBitField = EET_NOTHING;
		return writeMesh(_file, _mesh, wp);
	}

	bool CBAWMeshWriter::writeMesh(io::IWriteFile* _file, scene::ICPUMesh* _mesh, WriteProperties& _propsStruct)
	{
		if (!_mesh || !_file || _propsStruct.blobLz4ComprThresh > _propsStruct.blobLzmaComprThresh)
		{
#ifdef _DEBUG
			if (_propsStruct.blobLz4ComprThresh > _propsStruct.blobLzmaComprThresh)
				os::Printer::log("LZMA threshold must be greater or equal LZ4 threshold!", ELL_ERROR);
#endif
			return false;
		}

		const uint32_t FILE_HEADER_SIZE = 32;
		_IRR_DEBUG_BREAK_IF(FILE_HEADER_SIZE != sizeof(core::BAWFileV0::fileHeader))

		uint64_t header[4];
		memcpy(header, BAW_FILE_HEADER, FILE_HEADER_SIZE);
		header[3] = BAW_FILE_VERSION;

		_file->write(header, FILE_HEADER_SIZE);

		SContext ctx; // context of this call of `writeMesh`
		ctx.props = &_propsStruct;

		const uint32_t numOfInternalBlobs = genHeaders(_mesh, ctx);
		const uint32_t OFFSETS_FILE_OFFSET = FILE_HEADER_SIZE + sizeof(uint32_t) + sizeof(core::BAWFileV0::iv);
		const uint32_t HEADERS_FILE_OFFSET = OFFSETS_FILE_OFFSET + numOfInternalBlobs * sizeof(ctx.offsets[0]);

		ctx.offsets.set_used(numOfInternalBlobs);

		_file->write(&numOfInternalBlobs, sizeof(numOfInternalBlobs));
		//_file->write(ctx.pwdVer, 2);
		_file->write(_propsStruct.initializationVector, 16);
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
				exportAsBlob(reinterpret_cast<ICPUMesh*>(ctx.headers[i].handle), i, _file, ctx, toEncrypt(_propsStruct, EET_MESHES));
				break;
			case core::Blob::EBT_SKINNED_MESH:
				exportAsBlob(reinterpret_cast<ICPUSkinnedMesh*>(ctx.headers[i].handle), i, _file, ctx, toEncrypt(_propsStruct, EET_MESHES));
				break;
			case core::Blob::EBT_MESH_BUFFER:
				exportAsBlob(reinterpret_cast<ICPUMeshBuffer*>(ctx.headers[i].handle), i, _file, ctx, toEncrypt(_propsStruct, EET_MESH_BUFFERS));
				break;
			case core::Blob::EBT_SKINNED_MESH_BUFFER:
				exportAsBlob(reinterpret_cast<SCPUSkinMeshBuffer*>(ctx.headers[i].handle), i, _file, ctx, toEncrypt(_propsStruct, EET_MESH_BUFFERS));
				break;
			case core::Blob::EBT_RAW_DATA_BUFFER:
				exportAsBlob(reinterpret_cast<core::ICPUBuffer*>(ctx.headers[i].handle), i, _file, ctx, toEncrypt(_propsStruct, EET_RAW_BUFFERS));
				break;
			case core::Blob::EBT_DATA_FORMAT_DESC:
				exportAsBlob(reinterpret_cast<IMeshDataFormatDesc<core::ICPUBuffer>*>(ctx.headers[i].handle), i, _file, ctx, toEncrypt(_propsStruct, EET_DATA_FORMAT_DESC));
				break;
			case core::Blob::EBT_FINAL_BONE_HIERARCHY:
				exportAsBlob(reinterpret_cast<CFinalBoneHierarchy*>(ctx.headers[i].handle), i, _file, ctx, toEncrypt(_propsStruct, EET_ANIMATION_DATA));
				break;
			case core::Blob::EBT_TEXTURE_PATH:
				exportAsBlob(reinterpret_cast<video::IVirtualTexture*>(ctx.headers[i].handle), i, _file, ctx, toEncrypt(_propsStruct, EET_TEXTURE_PATHS));
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
			skinnedMesh = _mesh->getMeshType()!=EMT_ANIMATED_SKINNED ? NULL:dynamic_cast<ICPUSkinnedMesh*>(_mesh); //ICPUSkinnedMesh is a direct non-virtual inheritor
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

		std::unordered_set<const IReferenceCounted*> countedObjects;
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

	void CBAWMeshWriter::tryWrite(void* _data, io::IWriteFile * _file, SContext & _ctx, size_t _size, uint32_t _headerIdx, bool _encrypt) const
	{
		if (!_data)
			return pushCorruptedOffset(_ctx);

		uint8_t stack[1u<<14];

		size_t compressedSize = _size;
		void* data = _data;
		uint8_t comprType = core::Blob::EBCT_RAW;

		if (_size >= _ctx.props->blobLzmaComprThresh)
		{
			data = compressWithLzma(data, _size, compressedSize);
			if (data != _data)
				comprType |= core::Blob::EBCT_LZMA;
		}
		else if (_size >= _ctx.props->blobLz4ComprThresh)
		{
			data = compressWithLz4AndTryOnStack(data, _size, stack, sizeof(stack), compressedSize);
			if (data != _data)
				comprType |= core::Blob::EBCT_LZ4;
		}

		if (_encrypt)
		{
			const size_t encrSize = core::BlobHeaderV0::calcEncSize(compressedSize);
			void* in = malloc(encrSize);
			memset(((uint8_t*)in) + (compressedSize-16), 0, 16);
			memcpy(in, data, compressedSize);
			void* out = malloc(encrSize);
			if (core::encAes128gcm(data, encrSize, out, encrSize, _ctx.props->encryptionPassPhrase, _ctx.props->initializationVector, _ctx.headers[_headerIdx].gcmTag))
			{
				if (data != _data && data != stack) // allocated in compressing functions?
					free(data);
				data = out;
				free(in);
				comprType |= core::Blob::EBCT_AES128_GCM;
			}
			else
			{
#ifdef _DEBUG
				os::Printer::log("Failed to encrypt! Blob exported without encryption.", ELL_WARNING);
#endif
				free(in);
				free(out);
			}
		}

		_ctx.headers[_headerIdx].finalize(data, _size, compressedSize, comprType);
		const size_t writeSize = (comprType & core::Blob::EBCT_AES128_GCM) ? core::BlobHeaderV0::calcEncSize(compressedSize) : compressedSize;
		_file->write(data, writeSize);
		calcAndPushNextOffset(!_headerIdx ? 0 : _ctx.headers[_headerIdx - 1].effectiveSize(), _ctx);

		if (data != stack && data != _data)
			free(const_cast<void*>(data)); // safe const_cast since the only case when this executes is when `data` points to malloc'd memory
	}

	bool CBAWMeshWriter::toEncrypt(const WriteProperties& _wp, E_ENCRYPTION_TARGETS _req) const
	{
		return (_wp.encryptBlobBitField & _req);
	}

	void* CBAWMeshWriter::compressWithLz4AndTryOnStack(const void* _input, size_t _inputSize, void* _stack, size_t _stackSize, size_t& _outComprSize) const
	{
		void* data = _stack;
		size_t dstSize = _stackSize;
		size_t compressedSize = 0;
		const int lz4CompressBound = LZ4_compressBound(_inputSize);

		if (lz4CompressBound) // if input is not too large
		{
			if (lz4CompressBound > _stackSize)
			{
				dstSize = core::BlobHeaderV0::calcEncSize(lz4CompressBound);
				data = malloc(dstSize);
			}
			compressedSize = LZ4_compress_default((const char*)_input, (char*)data, _inputSize, dstSize);
		}
		if (!compressedSize) // if compression did not succeed
		{
			if (data != _stack)
				free(data);
			compressedSize = _inputSize;
			data = const_cast<void*>(_input);
#ifdef _DEBUG
			os::Printer::log("Failed to compress (lz4). Blob exported without compression.", ELL_WARNING);
#endif
		}
		_outComprSize = compressedSize;
		return data;
	}

	void* CBAWMeshWriter::compressWithLzma(const void* _input, size_t _inputSize, size_t& _outComprSize) const
	{
		ISzAlloc alloc{&core::LzmaMemMngmnt::alloc, &core::LzmaMemMngmnt::release};
		SizeT propsSize = LZMA_PROPS_SIZE;

		UInt32 dictSize = _inputSize; // next nearest (to input size) power of two times two
		--dictSize;
		for (uint32_t p = 1; p < 32; p <<= 1)
			dictSize |= dictSize>>p;
		++dictSize;
		dictSize <<= 1;

		// Lzma props: https://stackoverflow.com/a/21384797/5538150
		CLzmaEncProps props;
		LzmaEncProps_Init(&props);
		props.dictSize = dictSize;
		props.level = 5; // compression level [0;9]
		props.algo = 0; // fast algo: a little worse compression, a little less loading time
		props.lp = 2; // 2^2==sizeof(float)

		const SizeT heapSize = _inputSize + LZMA_PROPS_SIZE;
		uint8_t* data = (uint8_t*)malloc(heapSize);
		SizeT destSize = heapSize;

		SRes res = LzmaEncode(data+propsSize, &destSize, (const Byte*)_input, _inputSize, &props, data, &propsSize, props.writeEndMark, NULL, &alloc, &alloc);
		if (res != SZ_OK)
		{
			free(data);
			data = (uint8_t*)const_cast<void*>(_input);
			destSize = _inputSize;
#ifdef _DEBUG
			os::Printer::log("Failed to compress (lzma). Blob exported without compression.", ELL_WARNING);
#endif
		}
		_outComprSize = destSize + propsSize;
		return data;
	}

}} // end ns irr::scene
