// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "irr/core/core.h"

#ifdef _IRR_COMPILE_WITH_OBJ_LOADER_

#include "IFileSystem.h"
#include "COBJMeshFileLoader.h"
#include "irr/asset/IMeshManipulator.h"
#include "IVideoDriver.h"
#include "irr/video/CGPUMesh.h"
#include "irr/asset/normal_quantization.h"
#include "IReadFile.h"
#include "os.h"
#include "irr/asset/IAssetManager.h"

/*
namespace std
{
    template <>
    struct hash<irr::asset::SObjVertex>
    {
        std::size_t operator()(const irr::asset::SObjVertex& k) const
        {
            using std::size_t;
            using std::hash;

            return hash(k.normal32bit)^
                    (reinterpret_cast<const uint32_t&>(k.pos[0])*4996156539000000107ull)^
                    (reinterpret_cast<const uint32_t&>(k.pos[1])*620612627000000023ull)^
                    (reinterpret_cast<const uint32_t&>(k.pos[2])*1231379668000000199ull)^
                    (reinterpret_cast<const uint32_t&>(k.uv[0])*1099543332000000001ull)^
                    (reinterpret_cast<const uint32_t&>(k.uv[1])*1123461104000000009ull);
        }
    };

}
*/

namespace irr
{
namespace asset
{

static void insertShaderIntoCache(core::smart_refctd_ptr<ICPUSpecializedShader>& asset, const char* path, IAssetManager* _assetMgr)
{
    asset::SAssetBundle bundle({ asset });
    _assetMgr->changeAssetKey(bundle, path);
    _assetMgr->insertAssetIntoCache(bundle);
};

//#ifdef _IRR_DEBUG
#define _IRR_DEBUG_OBJ_LOADER_
//#endif

static const uint32_t WORD_BUFFER_LENGTH = 512;


//! Constructor
COBJMeshFileLoader::COBJMeshFileLoader(IAssetManager* _manager) : AssetManager(_manager), FileSystem(_manager->getFileSystem())
{
}


//! destructor
COBJMeshFileLoader::~COBJMeshFileLoader()
{
}

asset::SAssetBundle COBJMeshFileLoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
    SContext ctx(
        asset::IAssetLoader::SAssetLoadContext{
            _params,
            _file
        },
		_hierarchyLevel,
        _override
    );

	const long filesize = _file->getSize();
	if (!filesize)
        return {};

	const uint32_t WORD_BUFFER_LENGTH = 512u;
    char tmpbuf[WORD_BUFFER_LENGTH]{};

	uint32_t smoothingGroup=0;

	const io::path fullName = _file->getFileName();
	const std::string relPath = (io::IFileSystem::getFileDir(fullName)+"/").c_str();

    //value_type: directory from which .mtl (pipeline) was loaded and the pipeline
    core::unordered_multimap<std::string, std::pair<std::string, core::smart_refctd_ptr<ICPURenderpassIndependentPipeline>>> pipelines;

    std::string fileContents;
    fileContents.resize(filesize);
	char* const buf = fileContents.data();
	_file->read(buf, filesize);
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

	while(bufPtr != bufEnd)
	{
		switch(bufPtr[0])
		{
		case 'm':	// mtllib (material)
		{
			if (ctx.useMaterials)
			{
				bufPtr = goAndCopyNextWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
#ifdef _IRR_DEBUG_OBJ_LOADER_
				os::Printer::log("Reading material _file",tmpbuf);
#endif

                std::string mtllib = relPath+tmpbuf;
                std::replace(mtllib.begin(), mtllib.end(), '\\', '/');
                SAssetLoadParams loadParams;
                auto bundle = interm_getAssetInHierarchy(AssetManager, mtllib, loadParams, _hierarchyLevel+ICPUMesh::PIPELINE_HIERARCHYLEVELS_BELOW, _override);
                for (auto it = bundle.getContents().first; it != bundle.getContents().second; ++it)
                {
                    auto pipeln = core::smart_refctd_ptr_static_cast<ICPURenderpassIndependentPipeline>(*it);
                    auto metadata = static_cast<const CMTLPipelineMetadata*>(pipeln->getMetadata());
                    std::string mtlfilepath = relPath+tmpbuf;

                    decltype(pipelines)::value_type::second_type val{std::move(mtlfilepath), std::move(pipeln)};
                    pipelines.insert({metadata->getMaterialName(), std::move(val)});
                }
			}
		}
			break;

		case 'v':               // v, vn, vt
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
#ifdef _IRR_DEBUG_OBJ_LOADER_
	os::Printer::log("Loaded smoothing group start",tmpbuf, ELL_DEBUG);
#endif
				if (strcmp("off", tmpbuf)==0)
					smoothingGroup=0u;
				else
                    sscanf(tmpbuf,"%u",&smoothingGroup);
			}
			break;

		case 'u': // usemtl
			// get name of material
			{
				bufPtr = goAndCopyNextWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
#ifdef _IRR_DEBUG_OBJ_LOADER_
	os::Printer::log("Loaded material start",tmpbuf, ELL_DEBUG);
#endif
				mtlName=tmpbuf;

                if (ctx.useMaterials && !ctx.useGroups)
                {
                    asset::IAsset::E_TYPE types[] {asset::IAsset::ET_SUB_MESH, (asset::IAsset::E_TYPE)0u };
                    auto mb_bundle = _override->findCachedAsset(genKeyForMeshBuf(ctx, _file->getFileName().c_str(), mtlName, grpName), types, ctx.inner, _hierarchyLevel+ICPUMesh::MESHBUFFER_HIERARCHYLEVELS_BELOW);
                    auto mbs = mb_bundle.getContents();
                    {
                        auto mb = (mbs.first != mbs.second) ? core::smart_refctd_ptr_static_cast<ICPUMeshBuffer>(*mbs.first) : core::make_smart_refctd_ptr<ICPUMeshBuffer>();
                        submeshes.push_back(std::move(mb));
                    }
                    indices.emplace_back();
                    recalcNormals.push_back(false);
                    submeshWasLoadedFromCache.push_back(mbs.first!=mbs.second);
                    //if submesh was loaded from cache - insert empty "cache key" (submesh loaded from cache won't be added to cache again)
                    submeshCacheKeys.push_back(submeshWasLoadedFromCache.back() ? "" : genKeyForMeshBuf(ctx, _file->getFileName().c_str(), mtlName, grpName));
                    submeshMaterialNames.push_back(mtlName);
                }
			}
			break;
		case 'f':               // face
		{
			SObjVertex v;

			// get all vertices data in this face (current line of obj _file)
			const core::stringc wordBuffer = copyLine(bufPtr, bufEnd);
			const char* linePtr = wordBuffer.c_str();
			const char* const endPtr = linePtr+wordBuffer.size();

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
					v.normal32bit = asset::quantizeNormal2_10_10_10(simdNormal);
                }
				else
				{
					v.normal32bit = 0;
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

    constexpr uint32_t POSITION = 0u;
    constexpr uint32_t UV       = 2u;
    constexpr uint32_t NORMAL   = 3u;
    constexpr uint32_t BND_NUM  = 0u;
    {
        uint64_t ixBufOffset = 0ull;
        for (size_t i = 0ull; i < submeshes.size(); ++i)
        {
            if (submeshWasLoadedFromCache[i])
                continue;                

            submeshes[i]->setIndexCount(indices[i].size());
            submeshes[i]->setIndexType(EIT_32BIT);
            submeshes[i]->getIndexBufferBinding()->offset = ixBufOffset;
            ixBufOffset += indices[i].size()*4ull;

            const uint32_t hasUV = !core::isnan(vertices[indices[i][0]].uv[0]);
            auto rng = pipelines.equal_range(submeshMaterialNames[i]);
            for (auto it = rng.first; it != rng.second; ++it)
            {
                auto& pipeline = it->second.second;
                const CMTLPipelineMetadata* metadata = static_cast<const CMTLPipelineMetadata*>(pipeline->getMetadata());
                if (metadata->getHashVal()==hasUV)
                {
                    submeshes[i]->setPipeline(core::smart_refctd_ptr(pipeline));
                    const auto& vtxParams = pipeline->getVertexInputParams();
                    assert(vtxParams.attributes[POSITION].relativeOffset==offsetof(SObjVertex,pos));
                    assert(vtxParams.attributes[NORMAL].relativeOffset==offsetof(SObjVertex,normal32bit));
                    assert(vtxParams.attributes[UV].relativeOffset==offsetof(SObjVertex,uv));
                    assert(vtxParams.enabledAttribFlags&(1u<<UV));
                    assert(vtxParams.enabledBindingFlags==(1u<<BND_NUM));

                    auto ds3 = core::smart_refctd_ptr<ICPUDescriptorSet>(metadata->getDescriptorSet());
                    submeshes[i]->setAttachedDescriptorSet(std::move(ds3));

                    const uint32_t pcoffset = pipeline->getLayout()->getPushConstantRanges().begin()[0].offset;
                    memcpy(
                        submeshes[i]->getPushConstantsDataPtr()+pcoffset,
                        &metadata->getMaterialParams(),
                        sizeof(CMTLPipelineMetadata::SMTLMaterialParameters)
                    );

                    break;
                }
            }
        }

        core::smart_refctd_ptr<ICPUBuffer> vtxBuf = core::make_smart_refctd_ptr<ICPUBuffer>(vertices.size() * sizeof(SObjVertex));
        memcpy(vtxBuf->getPointer(), vertices.data(), vtxBuf->getSize());

        auto ixBuf = core::make_smart_refctd_ptr<ICPUBuffer>(ixBufOffset);
        for (size_t i = 0ull; i < submeshes.size(); ++i)
        {
            if (submeshWasLoadedFromCache[i])
                continue;

            submeshes[i]->setPositionAttributeIx(POSITION);

            submeshes[i]->getIndexBufferBinding()->buffer = ixBuf;
            const uint64_t offset = submeshes[i]->getIndexBufferBinding()->offset;
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

    auto mesh = core::make_smart_refctd_ptr<CCPUMesh>();
    for (auto& submesh : submeshes)
    {
        mesh->addMeshBuffer(std::move(submesh));
    }

	if (mesh->getMeshBufferCount())
		mesh->recalculateBoundingBox(true);
	else
        return {};
    
    //at the very end, insert submeshes into cache
    for (uint32_t i = 0u; i < mesh->getMeshBufferCount(); ++i)
    {
        SAssetBundle bundle{ core::smart_refctd_ptr<ICPUMeshBuffer>(mesh->getMeshBuffer(i)) };
        _override->insertAssetIntoCache(bundle, submeshCacheKeys[i], ctx.inner, _hierarchyLevel+ICPUMesh::MESHBUFFER_HIERARCHYLEVELS_BELOW);
    }

	return SAssetBundle({std::move(mesh)});
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


core::stringc COBJMeshFileLoader::copyLine(const char* inBuf, const char* bufEnd)
{
	if (!inBuf)
		return core::stringc();

	const char* ptr = inBuf;
	while (ptr<bufEnd)
	{
		if (*ptr=='\n' || *ptr=='\r')
			break;
		++ptr;
	}
	// we must avoid the +1 in case the array is used up
	return core::stringc(inBuf, (uint32_t)(ptr-inBuf+((ptr < bufEnd) ? 1 : 0)));
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
} // end namespace irr

#endif // _IRR_COMPILE_WITH_OBJ_LOADER_
