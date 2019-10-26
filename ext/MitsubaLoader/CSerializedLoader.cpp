#include "IrrCompileConfig.h"

#include "irr/core/core.h"
#include "IReadFile.h"
#include "os.h"

#include "../../ext/MitsubaLoader/CSerializedLoader.h"

#ifndef _IRR_COMPILE_WITH_ZLIB_
#error "Need zlib for this loader"
#endif
#include "zlib/zlib.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

enum MESH_FLAGS
{
	MF_PER_VERTEX_NORMALS	= 0x0001u,
	MF_TEXTURE_COORDINATES	= 0x0002u,
	MF_VERTEX_COLORS		= 0x0008u,
	MF_FACE_NORMALS			= 0x0010u,
	MF_SINGLE_FLOAT			= 0x1000u,
	MF_DOUBLE_FLOAT			= 0x2000u
};

// maybe move to core
#define PAGE_SIZE 4096
struct alignas(PAGE_SIZE) Page_t
{
	uint8_t data[PAGE_SIZE];
};
#undef PAGE_SIZE


//! creates/loads an animated mesh from the file.
asset::SAssetBundle CSerializedLoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	if (!_file)
        return {};

	SContext ctx = {_file,0,nullptr};
	size_t maxSize = 0u;
	{
		FileHeader header;
		ctx.file->seek(0u);
		ctx.file->read(&header, sizeof(header));
		if (header!=FileHeader())
		{
			os::Printer::log("Not a valid `.serialized` file", ctx.file->getFileName().c_str(), ELL_ERROR);
			return {};
		}

		size_t backPos = ctx.file->getSize() - sizeof(uint32_t);
		ctx.file->seek(backPos);
		ctx.file->read(&ctx.meshCount,sizeof(uint32_t));
		if (ctx.meshCount==0u)
			return {};

		ctx.meshOffsets = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint64_t> >(ctx.meshCount*2u);
		backPos -= sizeof(uint64_t)*ctx.meshCount;
		ctx.file->seek(backPos);
		ctx.file->read(ctx.meshOffsets->data(),sizeof(uint64_t)*ctx.meshCount);
		for (uint32_t i=0; i<ctx.meshCount; i++)
		{
			size_t localSize;
			if (i == ctx.meshCount-1u)
				localSize = backPos;
			else
				localSize = ctx.meshOffsets->operator[](i+1u);
			localSize -= ctx.meshOffsets->operator[](i);
			ctx.meshOffsets->operator[](i+ctx.meshCount) = localSize;
			if (localSize > maxSize)
				maxSize = localSize;
		}
	}
	if (maxSize==0u)
		return {};


	core::vector<core::smart_refctd_ptr<asset::CCPUMesh> > meshes;
	meshes.reserve(ctx.meshCount);

	uint8_t* data = reinterpret_cast<uint8_t*>(_IRR_ALIGNED_MALLOC(maxSize,alignof(double)));
	constexpr size_t CHUNK = 256ull*1024ull;
	core::vector<Page_t> decompressed(CHUNK/sizeof(Page_t));
	for (uint32_t i=0; i<ctx.meshCount; i++)
	{
		auto localSize = ctx.meshOffsets->operator[](i+ctx.meshCount);
		ctx.file->seek(sizeof(FileHeader)+ctx.meshOffsets->operator[](i));
		ctx.file->read(data,localSize);
		// decompress
		size_t decompressSize;
		{
			// Setup the inflate stream.
			z_stream stream;
			stream.next_in = (Bytef*)data;
			stream.avail_in = (uInt)localSize;
			stream.total_in = 0;
			stream.next_out = (Bytef*)decompressed.data();
			stream.avail_out = CHUNK;
			stream.total_out = 0u;
			stream.zalloc = (alloc_func)0;
			stream.zfree = (free_func)0;

			int32_t err = inflateInit(&stream, -MAX_WBITS);
			if (err == Z_OK)
			{
				while (err == Z_OK && err != Z_STREAM_END)
				{
					err = inflate(&stream, Z_SYNC_FLUSH);
					if (err!=Z_OK || err==Z_STREAM_END || stream.avail_out)
						continue;

					if (stream.total_out+CHUNK>decompressed.size()*sizeof(Page_t))
						decompressed.resize(decompressed.size()+CHUNK/sizeof(Page_t));
					stream.next_out = reinterpret_cast<Bytef*>(decompressed.data())+stream.total_out;
					stream.avail_out = CHUNK;
				}
			}
			decompressSize = stream.total_out;
			int32_t err2 = inflateEnd(&stream);

			if (err == Z_OK || err == Z_STREAM_END)
				err = err2;
			if (err != Z_OK)
			{
				std::wstring msg(L"Error decompressing mesh ix ");
				msg += std::to_wstring(i);
				os::Printer::log(msg, ELL_ERROR);
				continue;
			}
		}
		// too small to hold anything
		if (decompressSize < sizeof(uint8_t)+sizeof(uint64_t)*2ull)
			continue;

		// some tracking
		uint8_t* ptr = reinterpret_cast<uint8_t*>(decompressed.data());
		uint8_t* streamEnd = ptr+decompressSize;
		// vertex size determination
		auto flags = *(reinterpret_cast<uint32_t*&>(ptr)++);
		size_t typeSize;
		size_t vertexAttributeCount = 3u;
		size_t vertexSize;
		{
			if (flags & MF_SINGLE_FLOAT)
				typeSize = sizeof(float);
			else if (flags & MF_DOUBLE_FLOAT)
				typeSize = sizeof(double);
			else
				continue;

			if ((flags & MF_PER_VERTEX_NORMALS) || (flags & MF_FACE_NORMALS))
				vertexAttributeCount += 3ull;
			if (flags & MF_TEXTURE_COORDINATES)
				vertexAttributeCount += 2ull;
			if (flags & MF_VERTEX_COLORS)
				vertexAttributeCount += 3ull;

			vertexSize = vertexAttributeCount*typeSize;
		}

		// get name
		char* stringPtr = reinterpret_cast<char*>(ptr);
		while (ptr < streamEnd)
		if (! *(ptr++))
				break;
		// name too long
		size_t stringLen = reinterpret_cast<char*>(ptr)-stringPtr;
		if (ptr+sizeof(uint64_t)*2ull > streamEnd)
			continue;

		// 
		uint64_t vertexCount = *(reinterpret_cast<uint64_t*&>(ptr)++);
		if (vertexCount<3ull || vertexCount>0xFFFFFFFFull)
			continue;
		uint64_t triangleCount = *(reinterpret_cast<uint64_t*&>(ptr)++);
		if (triangleCount<1ull)
			continue;
		size_t vertexDataSize = vertexCount*vertexSize;
		if (ptr+vertexDataSize > streamEnd)
			continue;
		size_t indexDataSize = sizeof(uint32_t)*3ull*triangleCount;
		size_t totalDataSize = vertexDataSize+indexDataSize;
		if (ptr+totalDataSize > streamEnd)
			continue;

		auto buf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(totalDataSize);
		void* outPtr = buf->getPointer();
		auto readAttributes = [&](auto* outPtr, size_t attrOffset, auto* &inPtr, size_t attrCount) -> void
		{
			for (uint64_t j=0ull; j<vertexCount; j++)
			for (uint64_t k=0ull; k<attrCount; k++)
				outPtr[j*vertexAttributeCount+attrOffset+k] = *(inPtr++);
		};

		auto desc = core::make_smart_refctd_ptr<asset::ICPUMeshDataFormatDesc>();
		size_t attrOffset = 0ull;
		auto readAttributeDispatch = [&](asset::E_VERTEX_ATTRIBUTE_ID attrId, size_t attrCount, bool read = true) -> void
		{
			asset::E_FORMAT format = asset::EF_UNKNOWN;
			switch (attrCount)
			{
				case 2ull:
					format = typeSize==sizeof(double) ? asset::EF_R64G64_SFLOAT:asset::EF_R32G32_SFLOAT;
					break;
				case 3ull:
					format = typeSize==sizeof(double) ? asset::EF_R64G64B64_SFLOAT:asset::EF_R32G32B32_SFLOAT;
					break;
				default:
					assert(false);
					break;
			}
			desc->setVertexAttrBuffer(core::smart_refctd_ptr(buf), attrId, format, vertexSize, attrOffset*typeSize);
			if (read)
			{
				if (flags & MF_SINGLE_FLOAT)
					readAttributes(reinterpret_cast<float*>(outPtr), attrOffset, reinterpret_cast<float*&>(ptr), attrCount);
				else if (flags & MF_DOUBLE_FLOAT)
					readAttributes(reinterpret_cast<double*>(outPtr), attrOffset, reinterpret_cast<double*&>(ptr), attrCount);
			}
			attrOffset += attrCount;
		};

		// pos
		readAttributeDispatch(asset::EVAI_ATTR0, 3ull);
		// normal
		if ((flags & MF_PER_VERTEX_NORMALS) || (flags & MF_FACE_NORMALS))
			readAttributeDispatch(asset::EVAI_ATTR3, 3ull, flags&MF_PER_VERTEX_NORMALS); // TODO: normal quantization and optimization
		if (flags & MF_TEXTURE_COORDINATES) // TODO: UV quantization and optimization
			readAttributeDispatch(asset::EVAI_ATTR2, 2ull);
		if (flags & MF_VERTEX_COLORS) // TODO: quantize to 32bit format like RGB9E5
			readAttributeDispatch(asset::EVAI_ATTR1, 3ull);

		// create mesh buffer
		desc->setIndexBuffer(std::move(buf));
		auto mb = core::make_smart_refctd_ptr<asset::ICPUMeshBuffer>();
		mb->setMeshDataAndFormat(std::move(desc));
		mb->setIndexBufferOffset(vertexDataSize);
        mb->setIndexCount(triangleCount*3u);
        mb->setIndexType(asset::EIT_32BIT);
        mb->setPrimitiveType(asset::EPT_TRIANGLES);

		// read indices and possibly create per-face normals
		auto readIndices = [&]() -> bool
		{
			uint32_t* indexPtr = reinterpret_cast<uint32_t*>(outPtr)+vertexDataSize/sizeof(uint32_t);
			for (uint64_t j=0ull; j<triangleCount; j++)
			{
				uint32_t* triangleIndices = indexPtr;
				for (uint64_t k=0ull; k<3ull; k++)
				{
					triangleIndices[k] = *(reinterpret_cast<uint32_t*&>(ptr)++);
					if (triangleIndices[k] >= static_cast<uint32_t>(vertexCount))
						return false;
				}
				indexPtr += 3u;

				if (flags & MF_FACE_NORMALS)
				{
					core::vectorSIMDf pos[3];
					for (uint64_t k=0ull; k<3ull; k++)
						pos[k] = mb->getPosition(triangleIndices[k]);
					auto normal = core::cross(pos[1]-pos[0],pos[2]-pos[0]);
					for (uint64_t k=0ull; k<3ull; k++)
						mb->setAttribute(normal,asset::EVAI_ATTR3,k);
				}
			}
			return true;
		};
		if (!readIndices())
			continue;
		mb->recalculateBoundingBox();

		// create mesh
		auto mesh = core::make_smart_refctd_ptr<asset::CCPUMesh>();
		mesh->addMeshBuffer(std::move(mb));
		mesh->recalculateBoundingBox();

		manager->setAssetMetadata(mesh.get(), core::make_smart_refctd_ptr<CSerializedMetadata>(std::string(stringPtr,stringLen),i) );
		meshes.push_back(std::move(mesh));
	}
	_IRR_ALIGNED_FREE(data);

	return meshes;
}


}
}
}

