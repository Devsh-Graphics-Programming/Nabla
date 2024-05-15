// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "nbl/core/declarations.h"

#include "nbl/asset/IAssetManager.h"
#include "nbl/asset/utils/IMeshManipulator.h"

#ifdef _NBL_COMPILE_WITH_OBJ_LOADER_

#include "nbl/system/ISystem.h"
#include "nbl/system/IFile.h"

#include "nbl/asset/metadata/COBJMetadata.h"
#include "nbl/asset/utils/CQuantNormalCache.h"

#include "COBJMeshFileLoader.h"

#include <filesystem>

namespace nbl
{
namespace asset
{

//#ifdef _NBL_DEBUG
#define _NBL_DEBUG_OBJ_LOADER_
//#endif

static const uint32_t WORD_BUFFER_LENGTH = 512;

constexpr uint32_t POSITION = 0u;
constexpr uint32_t UV = 2u;
constexpr uint32_t NORMAL = 3u;
constexpr uint32_t BND_NUM = 0u;

//! Constructor
COBJMeshFileLoader::COBJMeshFileLoader(IAssetManager* _manager) : AssetManager(_manager), System(_manager->getSystem())
{
}


//! destructor
COBJMeshFileLoader::~COBJMeshFileLoader()
{
}

asset::SAssetBundle COBJMeshFileLoader::loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
    SContext ctx(
        asset::IAssetLoader::SAssetLoadContext{
            _params,
            _file
        },
		_hierarchyLevel,
        _override
    );

	if (_params.meshManipulatorOverride == nullptr)
	{
		_NBL_DEBUG_BREAK_IF(true);
		assert(false);
	}

	CQuantNormalCache* const quantNormalCache = _params.meshManipulatorOverride->getQuantNormalCache();

	const long filesize = _file->getSize();
	if (!filesize)
        return {};

	const uint32_t WORD_BUFFER_LENGTH = 512u;
    char tmpbuf[WORD_BUFFER_LENGTH]{};

	uint32_t smoothingGroup=0;

	const std::filesystem::path fullName = _file->getFileName();
	const std::string relPath = [&fullName]() -> std::string
	{
		auto dir = fullName.parent_path().string();
		return dir;
	}();

    //value_type: directory from which .mtl (pipeline) was loaded and the pipeline
	using pipeline_meta_pair_t = std::pair<core::smart_refctd_ptr<ICPURenderpassIndependentPipeline>,const CMTLMetadata::CRenderpassIndependentPipeline*>;
	struct hash_t
	{
		inline auto operator()(const pipeline_meta_pair_t& item) const
		{
			return std::hash<std::string>()(item.second->m_name);
		}
	};
	struct key_equal_t
	{
		inline bool operator()(const pipeline_meta_pair_t& lhs, const pipeline_meta_pair_t& rhs) const
		{
			return lhs.second->m_name==rhs.second->m_name;
		}
	};
    core::unordered_multiset<pipeline_meta_pair_t,hash_t,key_equal_t> pipelines;

	// TODO: map the file whenever possible
    std::string fileContents;
    fileContents.resize(filesize);
	char* const buf = fileContents.data();

	system::IFile::success_t success;
	_file->read(success, buf, 0, filesize);
	if (!success)
		return {};

	const char* const bufEnd = buf+filesize;
	// Process obj information
	const char* bufPtr = buf;
	std::string grpName, mtlName;

	auto performActionBasedOnOrientationSystem = [&](auto performOnRightHanded, auto performOnLeftHanded)
	{
		if (_params.loaderFlags & E_LOADER_PARAMETER_FLAGS::ELPF_RIGHT_HANDED_MESHES)
			performOnRightHanded();
		else
			performOnLeftHanded();
	};


    struct vec3 {
        float data[3];
    };
    struct vec2 {
        float data[2];
    };
    core::vector<vec3> vertexBuffer;
    core::vector<vec3> normalsBuffer;
    core::vector<vec2> textureCoordBuffer;

    core::vector<core::smart_refctd_ptr<ICPUMeshBuffer>> submeshes;
    core::vector<core::vector<uint32_t>> indices;
    core::vector<SObjVertex> vertices;
    core::map<SObjVertex, uint32_t> map_vtx2ix;
    core::vector<bool> recalcNormals;
    core::vector<bool> submeshWasLoadedFromCache;
    core::vector<std::string> submeshCacheKeys;
    core::vector<std::string> submeshMaterialNames;
    core::vector<uint32_t> vtxSmoothGrp;

	// TODO: handle failures much better!
	constexpr const char* NO_MATERIAL_MTL_NAME = "#";
	bool noMaterial = true;
	bool dummyMaterialCreated = false;
	while(bufPtr != bufEnd)
	{
		switch(bufPtr[0])
		{
		case 'm':	// mtllib (material)
		{
			if (ctx.useMaterials)
			{
				bufPtr = goAndCopyNextWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
				_params.logger.log("Reading material _file %s", system::ILogger::ELL_DEBUG, tmpbuf);

                std::string mtllib = tmpbuf;
                std::replace(mtllib.begin(), mtllib.end(), '\\', '/');
                SAssetLoadParams loadParams(_params);
				loadParams.workingDirectory = _file->getFileName().parent_path();
                auto bundle = interm_getAssetInHierarchy(AssetManager, mtllib, loadParams, _hierarchyLevel+ICPUMesh::PIPELINE_HIERARCHYLEVELS_BELOW, _override);
                
				if (bundle.getContents().empty())
					break;

				if (bundle.getMetadata())
				{
					auto meta = bundle.getMetadata()->selfCast<const CMTLMetadata>();
					if (bundle.getAssetType()==IAsset::ET_RENDERPASS_INDEPENDENT_PIPELINE)
					for (auto ass : bundle.getContents())
					{
						auto ppln = core::smart_refctd_ptr_static_cast<ICPURenderpassIndependentPipeline>(ass);
						const auto pplnMeta = meta->getAssetSpecificMetadata(ppln.get());
						if (!pplnMeta)
							continue;

						pipelines.emplace(std::move(ppln),pplnMeta);
					}
				}
			}
		}
			break;

		case 'v':               // v, vn, vt
			//reset flags
			noMaterial = true;
			dummyMaterialCreated = false;
			switch(bufPtr[1])
			{
			case ' ':          // vertex
				{
					vec3 vec;
					bufPtr = readVec3(bufPtr, vec.data, bufEnd);
					performActionBasedOnOrientationSystem([&]() {vec.data[0] = -vec.data[0];}, [&]() {});
					vertexBuffer.push_back(vec);
				}
				break;

			case 'n':       // normal
				{
					vec3 vec;
					bufPtr = readVec3(bufPtr, vec.data, bufEnd);
					performActionBasedOnOrientationSystem([&]() {vec.data[0] = -vec.data[0]; }, [&]() {});
					normalsBuffer.push_back(vec);
				}
				break;

			case 't':       // texcoord
				{
					vec2 vec;
					bufPtr = readUV(bufPtr, vec.data, bufEnd);
					textureCoordBuffer.push_back(vec);
				}
				break;
			}
			break;

		case 'g': // group name
            bufPtr = goAndCopyNextWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
            grpName = tmpbuf;
			break;
		case 's': // smoothing can be a group or off (equiv. to 0)
			{
				bufPtr = goAndCopyNextWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
				_params.logger.log("Loaded smoothing group start %s",system::ILogger::ELL_DEBUG, tmpbuf);
				if (strcmp("off", tmpbuf)==0)
					smoothingGroup=0u;
				else
                    sscanf(tmpbuf,"%u",&smoothingGroup);
			}
			break;

		case 'u': // usemtl
			// get name of material
			{
				noMaterial = false;
				bufPtr = goAndCopyNextWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
				_params.logger.log("Loaded material start %s", system::ILogger::ELL_DEBUG, tmpbuf);
				mtlName=tmpbuf;

                if (ctx.useMaterials && !ctx.useGroups)
                {
                    asset::IAsset::E_TYPE types[] {asset::IAsset::ET_SUB_MESH, (asset::IAsset::E_TYPE)0u };
                    auto mb_bundle = _override->findCachedAsset(genKeyForMeshBuf(ctx, _file->getFileName().string(), mtlName, grpName), types, ctx.inner, _hierarchyLevel+ICPUMesh::MESHBUFFER_HIERARCHYLEVELS_BELOW);
                    auto mbs = mb_bundle.getContents();
					bool notempty = mbs.size()!=0ull;
                    {
                        auto mb = notempty ? core::smart_refctd_ptr_static_cast<ICPUMeshBuffer>(*mbs.begin()) : core::make_smart_refctd_ptr<ICPUMeshBuffer>();
                        submeshes.push_back(std::move(mb));
                    }
                    indices.emplace_back();
                    recalcNormals.push_back(false);
                    submeshWasLoadedFromCache.push_back(notempty);
                    //if submesh was loaded from cache - insert empty "cache key" (submesh loaded from cache won't be added to cache again)
                    submeshCacheKeys.push_back(submeshWasLoadedFromCache.back() ? "" : genKeyForMeshBuf(ctx, _file->getFileName().string(), mtlName, grpName));
                    submeshMaterialNames.push_back(mtlName);
                }
			}
			break;
		case 'f':               // face
		{
			if (noMaterial && !dummyMaterialCreated)
			{
				dummyMaterialCreated = true;

				submeshes.push_back(core::make_smart_refctd_ptr<ICPUMeshBuffer>());
				indices.emplace_back();
				recalcNormals.push_back(false);
				submeshWasLoadedFromCache.push_back(false);
				submeshCacheKeys.push_back(genKeyForMeshBuf(ctx, _file->getFileName().string(), NO_MATERIAL_MTL_NAME, grpName));
				submeshMaterialNames.push_back(NO_MATERIAL_MTL_NAME);
			}

			SObjVertex v;

			// get all vertices data in this face (current line of obj _file)
			const std::string wordBuffer = copyLine(bufPtr, bufEnd);
			const char* linePtr = wordBuffer.c_str();
			const char* const endPtr = linePtr + wordBuffer.size();

			core::vector<uint32_t> faceCorners;
			faceCorners.reserve(32ull);

			// read in all vertices
			linePtr = goNextWord(linePtr, endPtr);
			while (0 != linePtr[0])
			{
				// Array to communicate with retrieveVertexIndices()
				// sends the buffer sizes and gets the actual indices
				// if index not set returns -1
				int32_t Idx[3];
				Idx[1] = Idx[2] = -1;

				// read in next vertex's data
				uint32_t wlength = copyWord(tmpbuf, linePtr, WORD_BUFFER_LENGTH, endPtr);
				// this function will also convert obj's 1-based index to c++'s 0-based index
				retrieveVertexIndices(tmpbuf, Idx, tmpbuf+wlength+1, vertexBuffer.size(), textureCoordBuffer.size(), normalsBuffer.size());
				v.pos[0] = vertexBuffer[Idx[0]].data[0];
				v.pos[1] = vertexBuffer[Idx[0]].data[1];
				v.pos[2] = vertexBuffer[Idx[0]].data[2];
				//set texcoord
				if ( -1 != Idx[1] )
                {
					v.uv[0] = textureCoordBuffer[Idx[1]].data[0];
					v.uv[1] = textureCoordBuffer[Idx[1]].data[1];
                }
				else
                {
					v.uv[0] = core::nan<float>();
					v.uv[1] = core::nan<float>();
                }
                //set normal
				if ( -1 != Idx[2] )
                {
					core::vectorSIMDf simdNormal;
					simdNormal.set(normalsBuffer[Idx[2]].data);
                    simdNormal.makeSafe3D();
					v.normal32bit = quantNormalCache->quantize<EF_A2B10G10R10_SNORM_PACK32>(simdNormal);
                }
				else
				{
					v.normal32bit = core::vectorSIMDu32(0u);
                    recalcNormals.back() = true;
				}

				uint32_t ix;
				auto vtx_ix = map_vtx2ix.find(v);
				if (vtx_ix != map_vtx2ix.end() && smoothingGroup==vtxSmoothGrp[vtx_ix->second])
					ix = vtx_ix->second;
				else
				{
					ix = vertices.size();
					vertices.push_back(v);
                    vtxSmoothGrp.push_back(smoothingGroup);
					map_vtx2ix.insert({v, ix});
				}

				faceCorners.push_back(ix);

				// go to next vertex
				linePtr = goNextWord(linePtr, endPtr);
			}

            // triangulate the face
            for (uint32_t i = 1u; i < faceCorners.size()-1u; ++i)
            {
                // Add a triangle
                performActionBasedOnOrientationSystem
                (
                [&]()
                {
                    indices.back().push_back(faceCorners[0]);
                    indices.back().push_back(faceCorners[i]);
                    indices.back().push_back(faceCorners[i + 1]);
                },
                [&]()
                {
                    indices.back().push_back(faceCorners[i + 1]);
                    indices.back().push_back(faceCorners[i]);
                    indices.back().push_back(faceCorners[0]);
                }
                );
            }
		}
		break;

		case '#': // comment
		default:
			break;
		}	// end switch(bufPtr[0])
		// eat up rest of line
		bufPtr = goNextLine(bufPtr, bufEnd);
	}	// end while(bufPtr && (bufPtr-buf<filesize))

	// prune out invalid empty shape groups (TODO: convert to AoS and use an erase_if)
	for (size_t i = 0ull; i < submeshes.size(); ++i)
	if (indices[i].size())
		i++;
	else
	{
		submeshes.erase(submeshes.begin()+i);
		indices.erase(indices.begin()+i);
		recalcNormals.erase(recalcNormals.begin()+i);
		submeshWasLoadedFromCache.erase(submeshWasLoadedFromCache.begin()+i);
		submeshCacheKeys.erase(submeshCacheKeys.begin()+i);
		submeshMaterialNames.erase(submeshMaterialNames.begin()+i);
	}
	
    core::unordered_set<pipeline_meta_pair_t,hash_t,key_equal_t> usedPipelines;
    {
        uint64_t ixBufOffset = 0ull;
        for (size_t i = 0ull; i < submeshes.size(); ++i)
        {
            if (submeshWasLoadedFromCache[i])
                continue;                

            submeshes[i]->setIndexCount(indices[i].size());
            submeshes[i]->setIndexType(EIT_32BIT);
			submeshes[i]->setIndexBufferBinding({ixBufOffset,nullptr});
            ixBufOffset += indices[i].size()*4ull;

            const uint32_t hasUV = !core::isnan(vertices[indices[i][0]].uv[0]);
			using namespace std::string_literals;
			_params.logger.log("Has UV: "s + (hasUV ? "YES":"NO"), system::ILogger::ELL_DEBUG);
			// search in loaded
			pipeline_meta_pair_t pipeline;
			{
				CMTLMetadata::CRenderpassIndependentPipeline dummyKey;
				dummyKey.m_name = submeshCacheKeys[i].substr(submeshCacheKeys[i].find_last_of('?')+1u);
				pipeline_meta_pair_t dummy{nullptr,&dummyKey};

				auto rng = pipelines.equal_range(dummy);
				for (auto it=rng.first; it!=rng.second; it++)
				if (it->second->m_hash==hasUV)
				{
					pipeline = *it;
					break;
				}
			}
			//if there's no pipeline for this meshbuffer, set dummy one
			if (!pipeline.first)
			{
				const IAsset::E_TYPE searchTypes[] = {IAsset::ET_RENDERPASS_INDEPENDENT_PIPELINE,static_cast<IAsset::E_TYPE>(0u)};
				auto bundle = _override->findCachedAsset("nbl/builtin/renderpass_independent_pipeline/loader/mtl/missing_material_pipeline",searchTypes,ctx.inner,_hierarchyLevel+ICPUMesh::PIPELINE_HIERARCHYLEVELS_BELOW);
				const auto* meta = bundle.getMetadata()->selfCast<CMTLMetadata>();
				const auto contents = bundle.getContents();
				for (auto pplnIt=contents.begin(); pplnIt!=contents.end(); pplnIt++)
				{
					auto ppln = core::smart_refctd_ptr_static_cast<ICPURenderpassIndependentPipeline>(*pplnIt);
					auto pplnMeta = meta->getAssetSpecificMetadata(ppln.get());
					if (pplnMeta && pplnMeta->m_hash==hasUV)
					{
						pipeline = { std::move(ppln),pplnMeta };
						break;
					}
				}
			}
			// do some checks
			assert(pipeline.first && pipeline.second);
			const auto* cPpln = pipeline.first.get();
            if (hasUV)
            {
                const auto& vtxParams = cPpln->getCachedCreationParams().vertexInput;
                assert(vtxParams.attributes[POSITION].relativeOffset==offsetof(SObjVertex,pos));
                assert(vtxParams.attributes[NORMAL].relativeOffset==offsetof(SObjVertex,normal32bit));
                assert(vtxParams.attributes[UV].relativeOffset==offsetof(SObjVertex,uv));
                assert(vtxParams.enabledAttribFlags&(1u<<UV));
                assert(vtxParams.enabledBindingFlags==(1u<<BND_NUM));
            }

			const uint32_t pcoffset = cPpln->getLayout()->getPushConstantRanges().begin()[0].offset;
			submeshes[i]->setAttachedDescriptorSet(core::smart_refctd_ptr<ICPUDescriptorSet>(pipeline.second->m_descriptorSet3));
			memcpy(
				submeshes[i]->getPushConstantsDataPtr()+pcoffset,
				&pipeline.second->m_materialParams,
				sizeof(CMTLMetadata::CRenderpassIndependentPipeline::SMaterialParameters)
			);

			usedPipelines.insert(pipeline);
			submeshes[i]->setPipeline(std::move(pipeline.first));
        }

        core::smart_refctd_ptr<ICPUBuffer> vtxBuf = core::make_smart_refctd_ptr<ICPUBuffer>(vertices.size() * sizeof(SObjVertex));
        memcpy(vtxBuf->getPointer(), vertices.data(), vtxBuf->getSize());

        auto ixBuf = core::make_smart_refctd_ptr<ICPUBuffer>(ixBufOffset);
        for (size_t i = 0ull; i < submeshes.size(); ++i)
        {
            if (submeshWasLoadedFromCache[i])
                continue;

            submeshes[i]->setPositionAttributeIx(POSITION);
			submeshes[i]->setNormalAttributeIx(NORMAL);
			
			submeshes[i]->setIndexBufferBinding({submeshes[i]->getIndexBufferBinding().offset,ixBuf});
            const uint64_t offset = submeshes[i]->getIndexBufferBinding().offset;
            memcpy(reinterpret_cast<uint8_t*>(ixBuf->getPointer())+offset, indices[i].data(), indices[i].size()*4ull);

            SBufferBinding<ICPUBuffer> vtxBufBnd;
            vtxBufBnd.offset = 0ull;
            vtxBufBnd.buffer = vtxBuf;
            submeshes[i]->setVertexBufferBinding(std::move(vtxBufBnd), BND_NUM);

			if (recalcNormals[i])
			{
				auto vtxcmp = [&vtxSmoothGrp](const IMeshManipulator::SSNGVertexData& v0, const IMeshManipulator::SSNGVertexData& v1, ICPUMeshBuffer* buffer)
				{
					return vtxSmoothGrp[v0.indexOffset]==vtxSmoothGrp[v1.indexOffset];
				};

				auto* meshManipulator = AssetManager->getMeshManipulator();
				meshManipulator->calculateSmoothNormals(submeshes[i].get(), false, 1.52e-5f, NORMAL, vtxcmp);
			}
        }
    }

    auto mesh = core::make_smart_refctd_ptr<ICPUMesh>();
    for (auto& submesh : submeshes)
    {
		IMeshManipulator::recalculateBoundingBox(submesh.get());
        mesh->getMeshBufferVector().emplace_back(std::move(submesh));
    }

	IMeshManipulator::recalculateBoundingBox(mesh.get());
	if (mesh->getMeshBuffers().empty())
        return {};
    
	//
	auto meta = core::make_smart_refctd_ptr<COBJMetadata>(usedPipelines.size());
	uint32_t metaOffset = 0u;
	for (auto pipeAndMeta : usedPipelines)
		meta->placeMeta(metaOffset++,pipeAndMeta.first.get(),*pipeAndMeta.second);

    //at the very end, insert submeshes into cache
	uint32_t i = 0u;
	for (auto meshbuffer : mesh->getMeshBuffers())
	{
		auto bundle = SAssetBundle(meta,{ core::smart_refctd_ptr<ICPUMeshBuffer>(meshbuffer) });
        _override->insertAssetIntoCache(bundle, submeshCacheKeys[i++], ctx.inner, _hierarchyLevel+ICPUMesh::MESHBUFFER_HIERARCHYLEVELS_BELOW);
	}

	return SAssetBundle(std::move(meta),{std::move(mesh)});
}


//! Read 3d vector of floats
const char* COBJMeshFileLoader::readVec3(const char* bufPtr, float vec[3], const char* const bufEnd)
{
	const uint32_t WORD_BUFFER_LENGTH = 256;
	char wordBuffer[WORD_BUFFER_LENGTH];

	bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
	sscanf(wordBuffer,"%f",vec);
	bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
	sscanf(wordBuffer,"%f",vec+1);
	bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
	sscanf(wordBuffer,"%f",vec+2);

    vec[0] = -vec[0]; // change handedness
	return bufPtr;
}


//! Read 2d vector of floats
const char* COBJMeshFileLoader::readUV(const char* bufPtr, float vec[2], const char* const bufEnd)
{
	const uint32_t WORD_BUFFER_LENGTH = 256;
	char wordBuffer[WORD_BUFFER_LENGTH];

	bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
	sscanf(wordBuffer,"%f",vec);
	bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
	sscanf(wordBuffer,"%f",vec+1);

	vec[1] = 1.f-vec[1]; // change handedness
	return bufPtr;
}


//! Read boolean value represented as 'on' or 'off'
const char* COBJMeshFileLoader::readBool(const char* bufPtr, bool& tf, const char* const bufEnd)
{
	const uint32_t BUFFER_LENGTH = 8;
	char tfStr[BUFFER_LENGTH];

	bufPtr = goAndCopyNextWord(tfStr, bufPtr, BUFFER_LENGTH, bufEnd);
	tf = strcmp(tfStr, "off") != 0;
	return bufPtr;
}

//! skip space characters and stop on first non-space
const char* COBJMeshFileLoader::goFirstWord(const char* buf, const char* const bufEnd, bool acrossNewlines)
{
	// skip space characters
	if (acrossNewlines)
		while((buf != bufEnd) && core::isspace(*buf))
			++buf;
	else
		while((buf != bufEnd) && core::isspace(*buf) && (*buf != '\n'))
			++buf;

	return buf;
}


//! skip current word and stop at beginning of next one
const char* COBJMeshFileLoader::goNextWord(const char* buf, const char* const bufEnd, bool acrossNewlines)
{
	// skip current word
	while(( buf != bufEnd ) && !core::isspace(*buf))
		++buf;

	return goFirstWord(buf, bufEnd, acrossNewlines);
}


//! Read until line break is reached and stop at the next non-space character
const char* COBJMeshFileLoader::goNextLine(const char* buf, const char* const bufEnd)
{
	// look for newline characters
	while(buf != bufEnd)
	{
		// found it, so leave
		if (*buf=='\n' || *buf=='\r')
			break;
		++buf;
	}
	return goFirstWord(buf, bufEnd);
}


uint32_t COBJMeshFileLoader::copyWord(char* outBuf, const char* const inBuf, uint32_t outBufLength, const char* const bufEnd)
{
	if (!outBufLength)
		return 0;
	if (!inBuf)
	{
		*outBuf = 0;
		return 0;
	}

	uint32_t i = 0;
	while(inBuf[i])
	{
		if (core::isspace(inBuf[i]) || &(inBuf[i]) == bufEnd)
			break;
		++i;
	}

	uint32_t length = core::min(i, outBufLength-1);
	for (uint32_t j=0; j<length; ++j)
		outBuf[j] = inBuf[j];

	outBuf[length] = 0;
	return length;
}


std::string COBJMeshFileLoader::copyLine(const char* inBuf, const char* bufEnd)
{
	if (!inBuf)
		return std::string();

	const char* ptr = inBuf;
	while (ptr<bufEnd)
	{
		if (*ptr=='\n' || *ptr=='\r')
			break;
		++ptr;
	}
	// we must avoid the +1 in case the array is used up
	return std::string(inBuf, (uint32_t)(ptr-inBuf+((ptr < bufEnd) ? 1 : 0)));
}


const char* COBJMeshFileLoader::goAndCopyNextWord(char* outBuf, const char* inBuf, uint32_t outBufLength, const char* bufEnd)
{
	inBuf = goNextWord(inBuf, bufEnd, false);
	copyWord(outBuf, inBuf, outBufLength, bufEnd);
	return inBuf;
}


bool COBJMeshFileLoader::retrieveVertexIndices(char* vertexData, int32_t* idx, const char* bufEnd, uint32_t vbsize, uint32_t vtsize, uint32_t vnsize)
{
	char word[16] = "";
	const char* p = goFirstWord(vertexData, bufEnd);
	uint32_t idxType = 0;	// 0 = posIdx, 1 = texcoordIdx, 2 = normalIdx

	uint32_t i = 0;
	while ( p != bufEnd )
	{
		if ( ( core::isdigit(*p)) || (*p == '-') )
		{
			// build up the number
			word[i++] = *p;
		}
		else if ( *p == '/' || *p == ' ' || *p == '\0' )
		{
			// number is completed. Convert and store it
			word[i] = '\0';
			// if no number was found index will become 0 and later on -1 by decrement
			sscanf(word,"%d",idx+idxType);
			if (idx[idxType]<0)
			{
				switch (idxType)
				{
					case 0:
						idx[idxType] += vbsize;
						break;
					case 1:
						idx[idxType] += vtsize;
						break;
					case 2:
						idx[idxType] += vnsize;
						break;
				}
			}
			else
				idx[idxType]-=1;

			// reset the word
			word[0] = '\0';
			i = 0;

			// go to the next kind of index type
			if (*p == '/')
			{
				if ( ++idxType > 2 )
				{
					// error checking, shouldn't reach here unless file is wrong
					idxType = 0;
				}
			}
			else
			{
				// set all missing values to disable (=-1)
				while (++idxType < 3)
					idx[idxType]=-1;
				++p;
				break; // while
			}
		}

		// go to the next char
		++p;
	}

	return true;
}

std::string COBJMeshFileLoader::genKeyForMeshBuf(const SContext& _ctx, const std::string& _baseKey, const std::string& _mtlName, const std::string& _grpName) const
{
    return _baseKey + "?" + _grpName + "?" + _mtlName;
}




} // end namespace scene
} // end namespace nbl

#endif // _NBL_COMPILE_WITH_OBJ_LOADER_
