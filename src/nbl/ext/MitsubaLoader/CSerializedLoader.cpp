// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/compile_config.h"

#include "nbl/ext/MitsubaLoader/CSerializedLoader.h"
#include "nbl/ext/MitsubaLoader/CMitsubaSerializedMetadata.h"

#ifndef _NBL_COMPILE_WITH_ZLIB_
#error "Need zlib for this loader"
#endif
#include "zlib/zlib.h"

namespace nbl
{

using namespace asset;

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

constexpr auto POSITION_ATTRIBUTE = 0;
constexpr auto COLOR_ATTRIBUTE = 1;
constexpr auto UV_ATTRIBUTE = 2;
constexpr auto NORMAL_ATTRIBUTE = 3;

// maybe move to core
#define PAGE_SIZE 4096
struct alignas(PAGE_SIZE) Page_t
{
	uint8_t data[PAGE_SIZE];
};
#undef PAGE_SIZE


template<typename T, size_t N>
struct alignas(T) unaligned_gvecN
{
	T pointer[N];
};

using unaligned_vec2 = unaligned_gvecN<float,2ull>;
using unaligned_vec3 = unaligned_gvecN<float,3ull>;

using unaligned_dvec2 = unaligned_gvecN<double,2ull>;
using unaligned_dvec3 = unaligned_gvecN<double,3ull>;


//! creates/loads an animated mesh from the file.
asset::SAssetBundle CSerializedLoader::loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	if (!_file)
        return {};

	SContext ctx = {
		IAssetLoader::SAssetLoadContext(_params,_file),
		0,
		nullptr
	};

	if (_params.meshManipulatorOverride == nullptr)
	{
		_NBL_DEBUG_BREAK_IF(true);
		assert(false);
	}
	CQuantNormalCache* const quantNormalCache = _params.meshManipulatorOverride->getQuantNormalCache();

	size_t maxSize = 0u;
	{
		FileHeader header;
		system::future<size_t> future;
		ctx.inner.mainFile->read(future, &header, 0u, sizeof(header));
		future.get();
		if (header!=FileHeader())
		{
			_params.logger.log("Not a valid `.serialized` file", system::ILogger::E_LOG_LEVEL::ELL_ERROR, ctx.inner.mainFile->getFileName().string().c_str());
			return {};
		}

		size_t backPos = ctx.inner.mainFile->getSize() - sizeof(uint32_t);
		ctx.inner.mainFile->read(future,&ctx.meshCount,backPos,sizeof(uint32_t));
		future.get();
		if (ctx.meshCount==0u)
			return {};

		ctx.meshOffsets = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint64_t> >(ctx.meshCount*2u);
		backPos -= sizeof(uint64_t)*ctx.meshCount;
		ctx.inner.mainFile->read(future, ctx.meshOffsets->data(),backPos,sizeof(uint64_t)*ctx.meshCount);
		future.get();
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

	auto meta = core::make_smart_refctd_ptr<CMitsubaSerializedMetadata>(ctx.meshCount,core::smart_refctd_ptr(IRenderpassIndependentPipelineLoader::m_basicViewParamsSemantics));
	core::vector<core::smart_refctd_ptr<ICPUMesh>> meshes; meshes.reserve(ctx.meshCount);

	uint8_t* data = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(maxSize,alignof(double)));
	constexpr size_t CHUNK = 256ull*1024ull;
	core::vector<Page_t> decompressed(CHUNK/sizeof(Page_t));
	system::future<size_t> future;
	for (uint32_t i=0; i<ctx.meshCount; i++)
	{
		auto localSize = ctx.meshOffsets->operator[](i+ctx.meshCount);
		ctx.inner.mainFile->read(future,data,sizeof(FileHeader)+ctx.meshOffsets->operator[](i),localSize);
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
				std::string msg("Error decompressing mesh ix ");
				msg += std::to_string(i);
				_params.logger.log(msg, system::ILogger::E_LOG_LEVEL::ELL_ERROR);
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
		{
			if (flags & MF_SINGLE_FLOAT)
				typeSize = sizeof(float);
			else if (flags & MF_DOUBLE_FLOAT)
				typeSize = sizeof(double);
			else
				continue;
		}
		const bool sourceIsDoubles = typeSize==sizeof(double);
		const bool requiresNormals = (flags&MF_PER_VERTEX_NORMALS) || (flags&MF_FACE_NORMALS);
		const bool hasUVs = flags&MF_TEXTURE_COORDINATES;
		const bool hasColors = flags&MF_VERTEX_COLORS;

		// get name
		char* stringPtr = reinterpret_cast<char*>(ptr);
		while (ptr < streamEnd)
		if (! *(ptr++))
				break;
		// name too long
		const size_t stringLen = reinterpret_cast<char*>(ptr)-stringPtr;
		if (ptr+sizeof(uint64_t)*2ull > streamEnd)
			continue;

		// 
		const uint64_t vertexCount = *(reinterpret_cast<uint64_t*&>(ptr)++);
		if (vertexCount<3ull || vertexCount>0xFFFFFFFFull)
			continue;
		const uint64_t triangleCount = *(reinterpret_cast<uint64_t*&>(ptr)++);
		if (triangleCount<1ull)
			continue;
		const size_t indexDataSize = sizeof(uint32_t)*3ull*triangleCount;
		{
			size_t vertexDataSize = 3ull;
			if (requiresNormals)
				vertexDataSize += 3ull;
			if (hasUVs)
				vertexDataSize += 2ull;
			if (hasColors)
				vertexDataSize += 3ull;
			vertexDataSize *= typeSize*vertexCount;
			if (ptr+vertexDataSize > streamEnd)
				continue;
			size_t totalDataSize = vertexDataSize+indexDataSize;
			if (ptr+totalDataSize > streamEnd)
				continue;
		}

		auto indexbuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(indexDataSize);
		const uint32_t posAttrSize = typeSize*3u;
		auto posbuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(vertexCount*posAttrSize);
		core::smart_refctd_ptr<asset::ICPUBuffer> normalbuf,uvbuf,colorbuf;
		if (requiresNormals)
			normalbuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t)*vertexCount);
		// TODO: UV quantization and optimization (maybe lets just always use half floats?)
		constexpr size_t uvAttrSize = sizeof(float)*2u;
		if (hasUVs)
			uvbuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(uvAttrSize*vertexCount);
		if (hasColors)
			colorbuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t)*vertexCount);

		void* posPtr = posbuf->getPointer();
		CQuantNormalCache::value_type_t<EF_A2B10G10R10_SNORM_PACK32>* normalPtr = !normalbuf ? nullptr:reinterpret_cast<CQuantNormalCache::value_type_t<EF_A2B10G10R10_SNORM_PACK32>*>(normalbuf->getPointer());
		unaligned_vec2* uvPtr = !uvbuf ? nullptr:reinterpret_cast<unaligned_vec2*>(uvbuf->getPointer());
		uint32_t* colorPtr = !colorbuf ? nullptr:reinterpret_cast<uint32_t*>(colorbuf->getPointer());


		auto meshBuffer = core::make_smart_refctd_ptr<asset::ICPUMeshBuffer>();
		meshBuffer->setPositionAttributeIx(POSITION_ATTRIBUTE);

		auto chooseShaderPath = [&]() -> std::string
		{
			if (!hasColors)
			{
				if (hasUVs)
					return "nbl/builtin/material/debug/vertex_uv/specialized_shader";
				if (requiresNormals)
					return "nbl/builtin/material/debug/vertex_normal/specialized_shader";
			}
			return "nbl/builtin/material/debug/vertex_color/specialized_shader"; // if only positions are present, shaders with debug vertex colors are assumed
		};
		
		core::smart_refctd_ptr<ICPUSpecializedShader> mbVertexShader;
		core::smart_refctd_ptr<ICPUSpecializedShader> mbFragmentShader;
		{
			const IAsset::E_TYPE types[]{ IAsset::E_TYPE::ET_SPECIALIZED_SHADER, IAsset::E_TYPE::ET_SPECIALIZED_SHADER, static_cast<IAsset::E_TYPE>(0u) };
			const std::string basepath = chooseShaderPath();

			auto bundle = m_assetMgr->findAssets(basepath+".vert", types);
			mbVertexShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(bundle->begin()->getContents().begin()[0]);
			bundle = m_assetMgr->findAssets(basepath+".frag", types);
			mbFragmentShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(bundle->begin()->getContents().begin()[0]);
		}
		auto mbPipelineLayout = _override->findDefaultAsset<ICPUPipelineLayout>("nbl/builtin/material/lambertian/no_texture/pipeline_layout",ctx.inner,_hierarchyLevel+ICPUMesh::PIPELINE_LAYOUT_HIERARCHYLEVELS_BELOW).first;


		asset::SBlendParams blendParams;
		asset::SRasterizationParams rastarizationParams;
		asset::SPrimitiveAssemblyParams primitiveAssemblyParams;
		primitiveAssemblyParams.primitiveType = asset::EPT_TRIANGLE_LIST;

		asset::SVertexInputParams inputParams;
		auto enableAttribute = [&meshBuffer,&inputParams](uint16_t attrId, asset::E_FORMAT format, const core::smart_refctd_ptr<asset::ICPUBuffer>& buf) -> void
		{
			inputParams.enabledBindingFlags |= core::createBitmask({ attrId });
			inputParams.bindings[attrId].inputRate = asset::EVIR_PER_VERTEX;
			inputParams.bindings[attrId].stride = asset::getTexelOrBlockBytesize(format);
			inputParams.enabledAttribFlags |= core::createBitmask({ attrId });
			inputParams.attributes[attrId].binding = attrId;
			inputParams.attributes[attrId].format = format;
			meshBuffer->setVertexBufferBinding({0,buf},attrId);
		};

		meshBuffer->setPositionAttributeIx(POSITION_ATTRIBUTE);
		enableAttribute(POSITION_ATTRIBUTE,sourceIsDoubles ? asset::EF_R64G64B64_SFLOAT:asset::EF_R32G32B32_SFLOAT,posbuf);
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
			enableAttribute(NORMAL_ATTRIBUTE,asset::EF_A2B10G10R10_SNORM_PACK32,normalbuf);
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
			enableAttribute(UV_ATTRIBUTE,asset::EF_R32G32_SFLOAT,uvbuf);
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
			enableAttribute(COLOR_ATTRIBUTE,asset::EF_B10G11R11_UFLOAT_PACK32,colorbuf);
			auto readColors = [ptr,colorPtr](const auto& color) -> void
			{
				size_t vertexIx = std::distance(reinterpret_cast<decltype(&color)>(ptr),&color);
				const double colors[3] = {color.pointer[0],color.pointer[1],color.pointer[2]};
				asset::encodePixels<asset::EF_B10G11R11_UFLOAT_PACK32,double>(colorPtr+vertexIx,colors);
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

		auto mbPipeline = core::make_smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>(std::move(mbPipelineLayout), nullptr, nullptr, inputParams, blendParams, primitiveAssemblyParams, rastarizationParams);
		mbPipeline->setShaderAtStage(asset::ISpecializedShader::E_SHADER_STAGE::ESS_VERTEX, mbVertexShader.get());
		mbPipeline->setShaderAtStage(asset::ISpecializedShader::E_SHADER_STAGE::ESS_FRAGMENT, mbFragmentShader.get());

		meshBuffer->setIndexBufferBinding({0u,indexbuf});
		meshBuffer->setIndexCount(triangleCount * 3u);
		meshBuffer->setIndexType(asset::EIT_32BIT);

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

				if (flags & MF_FACE_NORMALS)
				{
					core::vectorSIMDf pos[3];
					for (uint64_t k=0ull; k<3ull; k++)
						pos[k] = meshBuffer->getPosition(triangleIndices[k]);
					auto normal = core::cross(pos[1]-pos[0],pos[2]-pos[0]);
					for (uint64_t k=0ull; k<3ull; k++)
						meshBuffer->setAttribute(normal,NORMAL_ATTRIBUTE,k);
				}
			}
			return true;
		};
		if (!readIndices())
			continue;


		auto mesh = core::make_smart_refctd_ptr<asset::ICPUMesh>();

		meta->placeMeta(meshes.size(),mbPipeline.get(),mesh.get(),{std::string(stringPtr,stringLen),i});

		meshBuffer->setPipeline(std::move(mbPipeline));

		mesh->setBoundingBox(meshBuffer->getBoundingBox());
		mesh->getMeshBufferVector().emplace_back(std::move(meshBuffer));
		meshes.push_back(std::move(mesh));
	}
	_NBL_ALIGNED_FREE(data);

	return SAssetBundle(std::move(meta),std::move(meshes));
}


}
}
}

