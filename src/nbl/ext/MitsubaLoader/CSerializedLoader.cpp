// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/asset/compile_config.h"

#include "nbl/ext/MitsubaLoader/CSerializedLoader.h"
#include "nbl/ext/MitsubaLoader/CMitsubaSerializedMetadata.h"

// need Zlib to get this loader
#ifdef _NBL_COMPILE_WITH_ZLIB_
#include "zlib/zlib.h"


namespace nbl::ext::MitsubaLoader
{

// maybe move to core
template<size_t PageSize=4096>
struct alignas(PageSize) Page
{
	uint8_t data[PageSize];
};

//! creates/loads an animated mesh from the file.
asset::SAssetBundle CSerializedLoader::loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	using namespace nbl::core;
	using namespace nbl::system;
	using namespace nbl::asset;
	if (!_file)
        return {};

	SContext ctx = {
		IAssetLoader::SAssetLoadContext(_params,_file),
		0,
		nullptr
	};

	size_t backPos;
	{
		FileHeader header;
		{
			IFile::success_t success;
			_file->read(success,&header,0,sizeof(header));
			if (!success)
				return false;
		}
		if (header!=FileHeader())
		{
			_params.logger.log("\"%s\" is not a valid `.serialized` file",ILogger::E_LOG_LEVEL::ELL_ERROR,ctx.inner.mainFile->getFileName().string().c_str());
			return {};
		}

		backPos = ctx.inner.mainFile->getSize() - sizeof(ctx.meshCount);
		{
			IFile::success_t success;
			ctx.inner.mainFile->read(success,&ctx.meshCount,backPos,sizeof(ctx.meshCount));
			if (!success || ctx.meshCount==0u)
				return {};
		}

		ctx.meshOffsets = make_refctd_dynamic_array<smart_refctd_dynamic_array<uint64_t> >(ctx.meshCount*2u);
		backPos -= sizeof(uint64_t)*ctx.meshCount;
		{
			IFile::success_t success;
			ctx.inner.mainFile->read(success,ctx.meshOffsets->data(),backPos,sizeof(uint64_t)*ctx.meshCount);
			if (!success)
				return {};
		}
	}
	size_t maxSize = 0u;
	{
		uint64_t* const sizes = ctx.meshOffsets->data()+ctx.meshCount;
		for (uint32_t i=0; i<ctx.meshCount; i++)
		{
			size_t localSize;
			if (i == ctx.meshCount-1u)
				localSize = backPos;
			else
				localSize = ctx.meshOffsets->operator[](i+1u);
			localSize -= ctx.meshOffsets->operator[](i);
			sizes[i] = localSize;
			if (localSize > maxSize)
				maxSize = localSize;
		}
	}
	if (maxSize==0u)
		return {};

	auto meta = make_smart_refctd_ptr<CMitsubaSerializedMetadata>(ctx.meshCount);
	core::vector<smart_refctd_ptr<ICPUPolygonGeometry>> geoms; geoms.reserve(ctx.meshCount);
	{
		auto data = std::make_unique<uint8_t[]>(maxSize);
		assert(is_aligned_to(data.get(), alignof(double)));

		enum MESH_FLAGS : uint32_t
		{
			MF_PER_VERTEX_NORMALS = 0x0001u,
			MF_TEXTURE_COORDINATES = 0x0002u,
			MF_VERTEX_COLORS = 0x0008u,
			MF_FACE_NORMALS = 0x0010u,
			MF_SINGLE_FLOAT = 0x1000u,
			MF_DOUBLE_FLOAT = 0x2000u
		};

		constexpr size_t CHUNK = 256<<10;
		using page_t = Page<>;
		core::vector<page_t> decompressed(CHUNK/sizeof(page_t));
		for (uint32_t i=0; i<ctx.meshCount; i++)
		{
			auto localSize = ctx.meshOffsets->operator[](i+ctx.meshCount);
			{
				IFile::success_t success;
				ctx.inner.mainFile->read(success,data.get(),sizeof(FileHeader)+ctx.meshOffsets->operator[](i),localSize);
				if (!success)
					continue;
			}
			// decompress
			size_t decompressSize;
			{
				// Setup the inflate stream.
				z_stream stream;
				stream.next_in = (Bytef*)data.get();
				stream.avail_in = (uInt)localSize;
				stream.total_in = 0;
				stream.next_out = (Bytef*)decompressed.data();
				stream.avail_out = CHUNK;
				stream.total_out = 0u;
				stream.zalloc = (alloc_func)0;
				stream.zfree = (free_func)0;

				int32_t err = inflateInit2(&stream, -MAX_WBITS);
				if (err == Z_OK)
				{
					while (err == Z_OK && err != Z_STREAM_END)
					{
						err = inflate(&stream, Z_SYNC_FLUSH);
						if (err!=Z_OK || err==Z_STREAM_END || stream.avail_out)
							continue;

						if (stream.total_out+CHUNK>decompressed.size()*sizeof(page_t))
							decompressed.resize(decompressed.size()+CHUNK/sizeof(page_t));
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
					_params.logger.log("Error decompressing mesh ix %d",ILogger::E_LOG_LEVEL::ELL_ERROR,i);
					continue;
				}
			}
			// too small to hold anything (flags, empty name, zero vertex and index count)
			constexpr size_t MinMeshSize = sizeof(MESH_FLAGS)+sizeof(char)+sizeof(uint64_t)*2;
			if (decompressSize<MinMeshSize)
				continue;
			// some tracking
			uint8_t* ptr = reinterpret_cast<uint8_t*>(decompressed.data());
			uint8_t* const streamEnd = ptr+decompressSize;
			// vertex size determination
			const auto flags = *(reinterpret_cast<bitflag<MESH_FLAGS>*&>(ptr)++);
			const auto typeSize = [&]()->size_t
			{
				if (flags.hasFlags(MF_SINGLE_FLOAT))
					return sizeof(float);
				else if (flags.hasFlags(MF_DOUBLE_FLOAT))
					return sizeof(double);
				return 0;
			}();
			if (!typeSize)
				continue;
			const bool sourceIsDoubles = typeSize==sizeof(double);

			// get name
			const char* const stringPtr = reinterpret_cast<const char*>(ptr);
			while (ptr<streamEnd)
			if (! *(ptr++))
				break;
			// name too long
			const size_t stringLen = reinterpret_cast<char*>(ptr)-stringPtr;
			if (ptr+sizeof(uint64_t)*2>streamEnd)
				continue;

			// 
			const uint64_t vertexCount = *(reinterpret_cast<uint64_t*&>(ptr)++);
			if (vertexCount<3ull || vertexCount>0xFFFFFFFFull)
				continue;
			const uint64_t triangleCount = *(reinterpret_cast<uint64_t*&>(ptr)++);
			if (triangleCount<1ull)
				continue;

			const bool requiresNormals = flags.hasFlags(MF_PER_VERTEX_NORMALS);
			const bool hasUVs = flags.hasFlags(MF_TEXTURE_COORDINATES);
			const bool hasColors = flags.hasFlags(MF_VERTEX_COLORS);
			const size_t indexDataSize = sizeof(uint32_t)*3*triangleCount;
			{
				size_t vertexDataSize = 3;
				if (requiresNormals)
					vertexDataSize += 3;
				if (hasUVs)
					vertexDataSize += 2;
				if (hasColors)
					vertexDataSize += 3;
				vertexDataSize *= typeSize*vertexCount;
				if (ptr+vertexDataSize > streamEnd)
					continue;
				const size_t totalDataSize = vertexDataSize+indexDataSize;
				if (ptr+totalDataSize > streamEnd)
					continue;
			}

			auto geo = make_smart_refctd_ptr<ICPUPolygonGeometry>();
			geo->setIndexing(IPolygonGeometryBase::TriangleList());

// TODO: adopted view
			auto indexbuf = ICPUBuffer::create({ indexDataSize });
			const uint32_t posAttrSize = typeSize*3u;
			// TODO: can adopt mapped file to avoid moving data in RAM
			auto posbuf = ICPUBuffer::create({ vertexCount*posAttrSize });

			// cannot adopt mapped file, because these can be different formats (64bit not needed no matter what)
			// we let everyone outside compress our vertex attributes as they please
			using normal_t = hlsl::float32_t3;
			if (requiresNormals)
				geo->setNormalView(createView(EF_R32G32B32_SFLOAT,vertexCount));
// TODO: name the attributes!
			auto* const auxViews = geo->getAuxAttributeViews();
			// do not EVER get tempted by using half floats for UVs, T-junction meshes will f-u-^
			using uv_t = hlsl::float32_t2;
			if (hasUVs)
				auxViews->push_back(createView(EF_R32G32B32_SFLOAT,vertexCount));
			using color_t = hlsl::float16_t4;
			if (hasColors)
				auxViews->push_back(createView(EF_R32G32B32_SFLOAT,vertexCount));
#if 0
		void* posPtr = posbuf->getPointer();
		auto* normalPtr = !normalbuf ? nullptr:reinterpret_cast<CQuantNormalCache::value_type_t<EF_A2B10G10R10_SNORM_PACK32>*>(normalbuf->getPointer());
		unaligned_vec2* uvPtr = !uvbuf ? nullptr:reinterpret_cast<unaligned_vec2*>(uvbuf->getPointer());
		uint32_t* colorPtr = !colorbuf ? nullptr:reinterpret_cast<uint32_t*>(colorbuf->getPointer());


		enableAttribute(POSITION_ATTRIBUTE,sourceIsDoubles ? EF_R64G64B64_SFLOAT:EF_R32G32B32_SFLOAT,posbuf);
		{
			core::aabbox3df aabb;
			auto readPositions = [&aabb,ptr,posPtr](const auto& pos) -> void
			{
				size_t vertexIx = std::distance(reinterpret_cast<decltype(&pos)>(ptr),&pos);
				const auto* coords = pos.pointer;
				if (vertexIx)
					aabb.addInternalPoint(coords[0],coords[1],coords[2]);
				else
					aabb.reset(coords[0],coords[1],coords[2]);
				reinterpret_cast<std::remove_const_t<std::remove_reference_t<decltype(pos)>>*>(posPtr)[vertexIx] = pos;
			};
			if (sourceIsDoubles)
			{
				auto*& typedPtr = reinterpret_cast<unaligned_dvec3*&>(ptr);
				std::for_each_n(core::execution::seq,typedPtr,vertexCount,readPositions);
				typedPtr += vertexCount;
			}
			else
			{
				auto*& typedPtr = reinterpret_cast<unaligned_vec3*&>(ptr);
				std::for_each_n(core::execution::seq,typedPtr,vertexCount,readPositions);
				typedPtr += vertexCount;
			}
			meshBuffer->setBoundingBox(aabb);
		}
		if (requiresNormals)
		{
			enableAttribute(NORMAL_ATTRIBUTE,EF_A2B10G10R10_SNORM_PACK32,normalbuf);
			auto readNormals = [quantNormalCache,ptr,normalPtr](const auto& nml) -> void
			{
				size_t vertexIx = std::distance(reinterpret_cast<decltype(&nml)>(ptr),&nml);
				core::vectorSIMDf simdNormal(nml.pointer[0],nml.pointer[1],nml.pointer[2]);
				normalPtr[vertexIx] = quantNormalCache->quantize<EF_A2B10G10R10_SNORM_PACK32>(simdNormal);
			};
			const bool read = flags&MF_PER_VERTEX_NORMALS;
			if (sourceIsDoubles)
			{
				auto*& typedPtr = reinterpret_cast<unaligned_dvec3*&>(ptr);
				if (read)
					std::for_each_n(core::execution::seq,typedPtr,vertexCount,readNormals);
				typedPtr += vertexCount;
			}
			else
			{
				auto*& typedPtr = reinterpret_cast<unaligned_vec3*&>(ptr);
				if (read)
					std::for_each_n(core::execution::seq,typedPtr,vertexCount,readNormals);
				typedPtr += vertexCount;
			}
			meshBuffer->setNormalAttributeIx(NORMAL_ATTRIBUTE);
		}
		if (hasUVs)
		{
			enableAttribute(UV_ATTRIBUTE,EF_R32G32_SFLOAT,uvbuf);
			auto readUVs = [ptr,uvPtr](const auto& uv) -> void
			{
				size_t vertexIx = std::distance(reinterpret_cast<decltype(&uv)>(ptr),&uv);
				for (auto k=0u; k<2u; k++)
					uvPtr[vertexIx].pointer[k] = uv.pointer[k];
			};
			if (sourceIsDoubles)
			{
				auto*& typedPtr = reinterpret_cast<unaligned_dvec2*&>(ptr);
				std::for_each_n(core::execution::seq,typedPtr,vertexCount,readUVs);
				typedPtr += vertexCount;
			}
			else
			{
				auto*& typedPtr = reinterpret_cast<unaligned_vec2*&>(ptr);
				std::for_each_n(core::execution::seq,typedPtr,vertexCount,readUVs);
				typedPtr += vertexCount;
			}
		}
		if (hasColors)
		{
			enableAttribute(COLOR_ATTRIBUTE,EF_B10G11R11_UFLOAT_PACK32,colorbuf);
			auto readColors = [ptr,colorPtr](const auto& color) -> void
			{
				size_t vertexIx = std::distance(reinterpret_cast<decltype(&color)>(ptr),&color);
				const double colors[3] = {color.pointer[0],color.pointer[1],color.pointer[2]};
				encodePixels<EF_B10G11R11_UFLOAT_PACK32,double>(colorPtr+vertexIx,colors);
			};
			if (sourceIsDoubles)
			{
				auto*& typedPtr = reinterpret_cast<unaligned_dvec3*&>(ptr);
				std::for_each_n(core::execution::seq,typedPtr,vertexCount,readColors);
				typedPtr += vertexCount;
			}
			else
			{
				auto*& typedPtr = reinterpret_cast<unaligned_vec3*&>(ptr);
				std::for_each_n(core::execution::seq,typedPtr,vertexCount,readColors);
				typedPtr += vertexCount;
			}
		}

		meshBuffer->setIndexBufferBinding({0u,indexbuf});
		meshBuffer->setIndexCount(triangleCount * 3u);
		meshBuffer->setIndexType(EIT_32BIT);

		// read indices and possibly create per-face normals
		auto readIndices = [&]() -> bool
		{
			uint32_t* indexPtr = reinterpret_cast<uint32_t*>(indexbuf->getPointer());
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
			}
			return true;
		};
		if (!readIndices())
			continue;
#endif

			meta->placeMeta(geoms.size(),geo.get(),{std::string(stringPtr,stringLen),i});
			geoms.push_back(std::move(geo));
		}
	}

	return SAssetBundle(std::move(meta),std::move(geoms));
}

}
#endif