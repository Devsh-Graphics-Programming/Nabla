// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "CBAWMeshWriter.h"

#include "irr/core/core.h"
#include "irr/asset/asset.h"
#include "os.h"

#include "IFileSystem.h"
#include "IWriteFile.h"


#include "lz4/lib/lz4.h"
#undef Bool
#include "lzma/C/LzmaEnc.h"

namespace irr
{
namespace asset
{

struct LzmaMemMngmnt
{
        static void *alloc(ISzAllocPtr, size_t _size) { return _NBL_ALIGNED_MALLOC(_size,_NBL_SIMD_ALIGNMENT); }
        static void release(ISzAllocPtr, void* _addr) { _NBL_ALIGNED_FREE(_addr); }
    private:
        LzmaMemMngmnt() {}
};

	const char * const CBAWMeshWriter::BAW_FILE_HEADER = "IrrlichtBaW BinaryFile\0\0\0\0\0\0\0\0\0";

	CBAWMeshWriter::CBAWMeshWriter(io::IFileSystem* _fs) : m_fileSystem(_fs)
	{
#ifdef _NBL_DEBUG
		setDebugName("CBAWMeshWriter");
#endif
	}

	template<>
	void CBAWMeshWriter::exportAsBlob<ICPUMesh>(ICPUMesh* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx)
	{
		uint8_t stackData[1u<<14];
        auto data = MeshBlobV3::createAndTryOnStack(_obj, stackData, sizeof(stackData));

        const E_WRITER_FLAGS flags = _ctx.writerOverride->getAssetWritingFlags(_ctx.inner, _obj, 0u);
		if (flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED)
			data->meshFlags |= MeshBlobV3::EBMF_RIGHT_HANDED;

        const uint8_t* encrPwd = nullptr;
        _ctx.writerOverride->getEncryptionKey(encrPwd, _ctx.inner, _obj, 0u);
        const float comprLvl = _ctx.writerOverride->getAssetCompressionLevel(_ctx.inner, _obj, 0u);
		tryWrite(data, _file, _ctx, MeshBlobV3::calcBlobSizeForObj(_obj), _headerIdx, flags, encrPwd, comprLvl);

		if ((uint8_t*)data != stackData)
			_NBL_ALIGNED_FREE(data);
	}
	template<>
	void CBAWMeshWriter::exportAsBlob<ICPUSkinnedMesh>(ICPUSkinnedMesh* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx)
	{
		uint8_t stackData[1u << 14];
        SkinnedMeshBlobV3* data = SkinnedMeshBlobV3::createAndTryOnStack(_obj,stackData,sizeof(stackData));

        const E_WRITER_FLAGS flags = _ctx.writerOverride->getAssetWritingFlags(_ctx.inner, _obj, 0u);
		if (flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED)
			data->meshFlags |= SkinnedMeshBlobV3::EBMF_RIGHT_HANDED;

        const uint8_t* encrPwd = nullptr;
        _ctx.writerOverride->getEncryptionKey(encrPwd, _ctx.inner, _obj, 0u);
        const float comprLvl = _ctx.writerOverride->getAssetCompressionLevel(_ctx.inner, _obj, 0u);
		tryWrite(data, _file, _ctx, SkinnedMeshBlobV3::calcBlobSizeForObj(_obj), _headerIdx, flags, encrPwd, comprLvl);

		if ((uint8_t*)data != stackData)
			_NBL_ALIGNED_FREE(data);
	}
	template<>
	void CBAWMeshWriter::exportAsBlob<ICPUMeshBuffer>(ICPUMeshBuffer* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx)
	{
        MeshBufferBlobV3 data(_obj);

        const E_WRITER_FLAGS flags = _ctx.writerOverride->getAssetWritingFlags(_ctx.inner, _obj, 1u);
        const uint8_t* encrPwd = nullptr;
        _ctx.writerOverride->getEncryptionKey(encrPwd, _ctx.inner, _obj, 1u);
        const float comprLvl = _ctx.writerOverride->getAssetCompressionLevel(_ctx.inner, _obj, 1u);
		tryWrite(&data, _file, _ctx, sizeof(data), _headerIdx, flags, encrPwd, comprLvl);
	}
	template<>
	void CBAWMeshWriter::exportAsBlob<ICPUSkinnedMeshBuffer>(ICPUSkinnedMeshBuffer* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx)
	{
        SkinnedMeshBufferBlobV3 data(_obj);

        const E_WRITER_FLAGS flags = _ctx.writerOverride->getAssetWritingFlags(_ctx.inner, _obj, 1u);
        const uint8_t* encrPwd = nullptr;
        _ctx.writerOverride->getEncryptionKey(encrPwd, _ctx.inner, _obj, 1u);
        const float comprLvl = _ctx.writerOverride->getAssetCompressionLevel(_ctx.inner, _obj, 1u);
		tryWrite(&data, _file, _ctx, sizeof(data), _headerIdx, flags, encrPwd, comprLvl);
	}
	template<>
	void CBAWMeshWriter::exportAsBlob<CFinalBoneHierarchy>(CFinalBoneHierarchy* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx)
	{
		uint8_t stackData[1u<<14]; // 16kB
        auto data = FinalBoneHierarchyBlobV3::createAndTryOnStack(_obj,stackData,sizeof(stackData));

		tryWrite(data, _file, _ctx, FinalBoneHierarchyBlobV3::calcBlobSizeForObj(_obj), _headerIdx, EWF_NONE);

		if ((uint8_t*)data != stackData)
			_NBL_ALIGNED_FREE(data);
	}
	template<>
	void CBAWMeshWriter::exportAsBlob<ICPUBuffer>(ICPUBuffer* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx)
	{
        const E_WRITER_FLAGS flags = _ctx.writerOverride->getAssetWritingFlags(_ctx.inner, _obj, 3u);
        const uint8_t* encrPwd = nullptr;
        _ctx.writerOverride->getEncryptionKey(encrPwd, _ctx.inner, _obj, 3u);
        const float comprLvl = _ctx.writerOverride->getAssetCompressionLevel(_ctx.inner, _obj, 3u);
		tryWrite(_obj->getPointer(), _file, _ctx, _obj->getSize(), _headerIdx, flags, encrPwd, comprLvl);
	}

	bool CBAWMeshWriter::writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
	{
        return false;
	}

	uint32_t CBAWMeshWriter::genHeaders(const ICPUMesh* _mesh, SContext& _ctx)
	{
    return 0u;
	}

	void CBAWMeshWriter::calcAndPushNextOffset(uint32_t _blobSize, SContext& _ctx) const
	{
		_ctx.offsets.push_back(!_ctx.offsets.size() ? 0 : _ctx.offsets.back() + _blobSize);
	}

	void CBAWMeshWriter::tryWrite(void* _data, io::IWriteFile * _file, SContext & _ctx, size_t _size, uint32_t _headerIdx, E_WRITER_FLAGS _flags, const uint8_t* _encrPwd, float _comprLvl) const
	{
		if (!_data)
			return pushCorruptedOffset(_ctx);

#ifndef _NBL_COMPILE_WITH_OPENSSL_
		_encrypt = false;
#endif // _NBL_COMPILE_WITH_OPENSSL_

		uint8_t stack[1u<<14];

		size_t compressedSize = _size;
		void* data = _data;
		uint8_t comprType = Blob::EBCT_RAW;

        if (_flags & EWF_COMPRESSED)
        {
            if (_comprLvl > 0.3f)
            {
                data = compressWithLzma(data, _size, compressedSize);
                if (data != _data)
                    comprType |= Blob::EBCT_LZMA;
            }
            else if (_comprLvl == 0.3f && _size<=0xffffffffull)
            {
                data = compressWithLz4AndTryOnStack(data, static_cast<uint32_t>(_size), stack, static_cast<uint32_t>(sizeof(stack)), compressedSize);
                if (data != _data)
                    comprType |= Blob::EBCT_LZ4;
            }
        }

		if (_flags & EWF_ENCRYPTED)
		{
			const size_t encrSize = BlobHeaderLatest::calcEncSize(compressedSize);
			void* in = _NBL_ALIGNED_MALLOC(encrSize,_NBL_SIMD_ALIGNMENT);
			memset(((uint8_t*)in) + (compressedSize-16), 0, 16);
			memcpy(in, data, compressedSize);

			void* out = _NBL_ALIGNED_MALLOC(encrSize, _NBL_SIMD_ALIGNMENT);

            const WriteProperties* props = reinterpret_cast<const WriteProperties*>(_ctx.inner.params.userData);
			if (encAes128gcm(data, encrSize, out, encrSize, _encrPwd, props->initializationVector, _ctx.headers[_headerIdx].gcmTag))
			{
				if (data != _data && data != stack) // allocated in compressing functions?
					_NBL_ALIGNED_FREE(data);
				data = out;
				_NBL_ALIGNED_FREE(in);
				comprType |= Blob::EBCT_AES128_GCM;
			}
			else
			{
#ifdef _NBL_DEBUG
				os::Printer::log("Failed to encrypt! Blob exported without encryption.", ELL_WARNING);
#endif
				_NBL_ALIGNED_FREE(in);
				_NBL_ALIGNED_FREE(out);
			}
		}

		_ctx.headers[_headerIdx].finalize(data, _size, compressedSize, comprType);
		const size_t writeSize = (comprType & Blob::EBCT_AES128_GCM) ? BlobHeaderLatest::calcEncSize(compressedSize) : compressedSize;
		_file->write(data, writeSize);
		calcAndPushNextOffset(!_headerIdx ? 0 : _ctx.headers[_headerIdx - 1].effectiveSize(), _ctx);

		if (data != stack && data != _data)
			_NBL_ALIGNED_FREE(const_cast<void*>(data)); // safe const_cast since the only case when this executes is when `data` points to _NBL_ALIGNED_MALLOC'd memory
	}

	void* CBAWMeshWriter::compressWithLz4AndTryOnStack(const void* _input, uint32_t _inputSize, void* _stack, uint32_t _stackSize, size_t& _outComprSize) const
	{
		void* data = _stack;
		size_t dstSize = _stackSize;
		size_t compressedSize = 0;
		const int lz4CompressBound = LZ4_compressBound(_inputSize);

		if (lz4CompressBound) // if input is not too large
		{
			if (lz4CompressBound > _stackSize)
			{
				dstSize = BlobHeaderLatest::calcEncSize(lz4CompressBound);
				data = _NBL_ALIGNED_MALLOC(dstSize,_NBL_SIMD_ALIGNMENT);
			}
			compressedSize = LZ4_compress_default((const char*)_input, (char*)data, _inputSize, dstSize);
		}
		if (!compressedSize) // if compression did not succeed
		{
			if (data != _stack)
				_NBL_ALIGNED_FREE(data);
			compressedSize = _inputSize;
			data = const_cast<void*>(_input);
#ifdef _NBL_DEBUG
			os::Printer::log("Failed to compress (lz4). Blob exported without compression.", ELL_WARNING);
#endif
		}
		_outComprSize = compressedSize;
		return data;
	}

	void* CBAWMeshWriter::compressWithLzma(const void* _input, size_t _inputSize, size_t& _outComprSize) const
	{
		ISzAlloc alloc{&LzmaMemMngmnt::alloc, &LzmaMemMngmnt::release};
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
		uint8_t* data = (uint8_t*)_NBL_ALIGNED_MALLOC(heapSize,_NBL_SIMD_ALIGNMENT);
		SizeT destSize = heapSize;

		SRes res = LzmaEncode(data+propsSize, &destSize, (const Byte*)_input, _inputSize, &props, data, &propsSize, props.writeEndMark, NULL, &alloc, &alloc);
		if (res != SZ_OK)
		{
			_NBL_ALIGNED_FREE(data);
			data = (uint8_t*)const_cast<void*>(_input);
			destSize = _inputSize;
#ifdef _NBL_DEBUG
			os::Printer::log("Failed to compress (lzma). Blob exported without compression.", ELL_WARNING);
#endif
		}
		_outComprSize = destSize + propsSize;
		return data;
	}

}} // end ns irr::scene
