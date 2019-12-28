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
#include "irr/asset/CMTLPipelineMetadata.h"

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

#define NEW_SHADERS

namespace irr
{
namespace asset
{

//#ifdef _IRR_DEBUG
#define _IRR_DEBUG_OBJ_LOADER_
//#endif

static const uint32_t WORD_BUFFER_LENGTH = 512;


//! Constructor
COBJMeshFileLoader::COBJMeshFileLoader(IAssetManager* _manager) : AssetManager(_manager), FileSystem(_manager->getFileSystem())
{
#ifdef _IRR_DEBUG
	setDebugName("COBJMeshFileLoader");
#endif
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

    core::unordered_map<std::string, core::smart_refctd_ptr<ICPURenderpassIndependentPipeline>> pipelines;

    std::string fileContents;
    fileContents.resize(filesize);
	char* buf = fileContents.data();
	_file->read(buf, filesize);
	const char* const bufEnd = buf+filesize;

	// Process obj information
	const char* bufPtr = buf;
	std::string grpName, mtlName;
    bool submeshLoadedFromCache = false;

	auto performActionBasedOnOrientationSystem = [&](auto performOnRightHanded, auto performOnLeftHanded)
	{
		if (_params.loaderFlags & E_LOADER_PARAMETER_FLAGS::ELPF_RIGHT_HANDED_MESHES)
			performOnRightHanded();
		else
			performOnLeftHanded();
	};


    core::vector<core::vector3df> vertexBuffer;
    core::vector<core::vector3df> normalsBuffer;
    core::vector<core::vector2df> textureCoordBuffer;

    core::vector<core::smart_refctd_ptr<ICPUMeshBuffer>> submeshes;
    core::vector<core::vector<uint32_t>> indices;
    core::vector<SObjVertex> vertices;
    core::map<SObjVertex, uint32_t> map_vtx2ix;
    core::vector<bool> recalcNormals;
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

                SAssetLoadParams loadParams;
                auto bundle = AssetManager->getAsset(tmpbuf, loadParams);
                for (auto it = bundle.getContents().first; it != bundle.getContents().second; ++it)
                {
                    auto pipeln = core::smart_refctd_ptr_static_cast<ICPURenderpassIndependentPipeline>(*it);
                    auto metadata = static_cast<CMTLPipelineMetadata*>(pipeln->getMetadata());
                    pipelines.insert({metadata->getMaterial().name, pipeln});
                }
			}
		}
			break;

		case 'v':               // v, vn, vt
            if (submeshLoadedFromCache)
                break;
			switch(bufPtr[1])
			{
			case ' ':          // vertex
				{
					core::vector3df vec;
					bufPtr = readVec3(bufPtr, vec, bufEnd);
					performActionBasedOnOrientationSystem([&]() {vec.X = -vec.X;}, [&]() {});
					vertexBuffer.push_back(vec);
				}
				break;

			case 'n':       // normal
				{
					core::vector3df vec;
					bufPtr = readVec3(bufPtr, vec, bufEnd);
					performActionBasedOnOrientationSystem([&]() {vec.X = -vec.X; }, [&]() {});
					normalsBuffer.push_back(vec);
				}
				break;

			case 't':       // texcoord
				{
					core::vector2df vec;
					bufPtr = readUV(bufPtr, vec, bufEnd);
					textureCoordBuffer.push_back(vec);
				}
				break;
			}
			break;

		case 'g': // group name
            bufPtr = goNextWord(bufPtr, bufEnd);
			break;
		case 's': // smoothing can be a group or off (equiv. to 0)
			{
				bufPtr = goAndCopyNextWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
#ifdef _IRR_DEBUG_OBJ_LOADER_
	os::Printer::log("Loaded smoothing group start",tmpbuf, ELL_DEBUG);
#endif
				if (core::stringc("off")==tmpbuf)
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
                    auto mb_bundle = _override->findCachedAsset(genKeyForMeshBuf(ctx, _file->getFileName().c_str(), mtlName, grpName), types, ctx.inner, 1u).getContents();
                    if (mb_bundle.first!=mb_bundle.second)
                    {
                        submeshes.push_back(*mb_bundle.first);
                    }
                    else
                    {
                        submeshes.push_back(core::make_smart_refctd_ptr<ICPUMeshBuffer>());
                        auto found = pipelines.find(mtlName);
                        assert(found != pipelines.end());
#ifndef _IRR_DEBUG
                        if (found != pipelines.end())
                        {
#endif
                            auto pipeln = found->second;
                            //cloning pipeline because it will be edited (vertex input params)
                            //note shallow copy (depth=0), i.e. only pipeline is cloned, but all its sub-assets are taken from original object
                            submeshes.back()->setPipeline(pipeln->clone(0u));
#ifndef _IRR_DEBUG
                        }
#endif
                    }
                    indices.emplace_back();
                    recalcNormals.push_back(false);

                    submeshLoadedFromCache = (mb_bundle.first!=mb_bundle.second);
                }
			}
			break;

		case 'f':               // face
		{
            if (submeshLoadedFromCache)
                break;

			SObjVertex v;

			// get all vertices data in this face (current line of obj _file)
			const core::stringc wordBuffer = copyLine(bufPtr, bufEnd);
			const char* linePtr = wordBuffer.c_str();
			const char* const endPtr = linePtr+wordBuffer.size();

			core::vector<uint32_t> faceCorners;
			faceCorners.reserve(3ull);

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
				v.pos[0] = vertexBuffer[Idx[0]].X;
				v.pos[1] = vertexBuffer[Idx[0]].Y;
				v.pos[2] = vertexBuffer[Idx[0]].Z;
				//set texcoord
				if ( -1 != Idx[1] )
                {
					v.uv[0] = textureCoordBuffer[Idx[1]].X;
					v.uv[1] = textureCoordBuffer[Idx[1]].Y;
                }
				else
                {
					v.uv[0] = std::numeric_limits<float>::quiet_NaN();
					v.uv[1] = std::numeric_limits<float>::quiet_NaN();
                }
                //set normal
				if ( -1 != Idx[2] )
                {
					core::vectorSIMDf simdNormal;
					simdNormal.set(normalsBuffer[Idx[2]]);
					v.normal32bit = asset::quantizeNormal2_10_10_10(simdNormal);
                }
				else
				{
					v.normal32bit = 0;
                    recalcNormals.back() = true;
				}

				uint32_t ix;
				auto vtx_ix = map_vtx2ix.find(v);
				if (vtx_ix != map_vtx2ix.end())
					ix = vtx_ix->second;
				else
				{
					ix = vertices.size();
					vertices.push_back(v);
					map_vtx2ix.insert({v, ix});
				}

				faceCorners.push_back(ix);

				// go to next vertex
				linePtr = goNextWord(linePtr, endPtr);
			}

            assert(faceCorners.size()==3ull);
			performActionBasedOnOrientationSystem
			(
				[](){},
				[&]() 
				{
                    std::swap(faceCorners.front(), faceCorners.back());
				}
			);
            indices.back().insert(indices.back().end(), faceCorners.begin(), faceCorners.end());
		}
		break;

		case '#': // comment
		default:
			break;
		}	// end switch(bufPtr[0])
		// eat up rest of line
		bufPtr = goNextLine(bufPtr, bufEnd);
	}	// end while(bufPtr && (bufPtr-buf<filesize))

    core::vector<core::vectorSIMDf> newNormals;
    auto doRecalcNormals = [&vertices,&newNormals](const core::vector<uint32_t>& _ixs) {
        memset(newNormals.data(), 0, sizeof(core::vectorSIMDf)*newNormals.size());

        auto minmax = std::minmax_element(_ixs.begin(), _ixs.end());
        const uint32_t maxsz = *minmax.second - *minmax.first;
        const uint32_t min = *minmax.first;
        
        newNormals.resize(maxsz, core::vectorSIMDf(0.f));
        for (size_t i = 0ull; i < _ixs.size(); i += 3ull)
        {
            core::vectorSIMDf v1, v2, v3;
            v1.set(vertices[_ixs[i-min+0u]].pos);
            v2.set(vertices[_ixs[i-min+1u]].pos);
            v3.set(vertices[_ixs[i-min+2u]].pos);
            v1.makeSafe3D();
            v2.makeSafe3D();
            v3.makeSafe3D();
            core::vectorSIMDf normal(core::plane3dSIMDf(v1, v2, v3).getNormal());
            newNormals[_ixs[i-min+0u]] += normal;
            newNormals[_ixs[i-min+1u]] += normal;
            newNormals[_ixs[i-min+2u]] += normal;
        }
        for (uint32_t ix : _ixs)
            vertices[ix].normal32bit = asset::quantizeNormal2_10_10_10(newNormals[ix-min]);
    };

    {
        uint64_t ixBufOffset = 0ull;
        for (size_t i = 0ull; i < submeshes.size(); ++i)
        {
            if (recalcNormals[i])
                doRecalcNormals(indices[i]);

            submeshes[i]->setIndexCount(indices[i].size());
            submeshes[i]->setIndexType(EIT_32BIT);
            submeshes[i]->getIndexBufferBinding()->offset = ixBufOffset;
            ixBufOffset += indices[i].size()*4ull;

            const bool hasUV = !std::isnan(vertices[indices[i][0]]);
            SVertexInputParams vtxParams;
            vtxParams.enabledAttribFlags = 0b1001u | (hasUV*0b0100u);
            vtxParams.enabledBindingFlags = vtxParams.enabledAttribFlags;
            //position
            vtxParams.attributes[0].binding = 0u;
            vtxParams.attributes[0].format = EF_R32G32B32_SFLOAT;
            vtxParams.attributes[0].relativeOffset = 0u;
            //normal
            vtxParams.attributes[3].binding = 3u;
            vtxParams.attributes[3].format = EF_A2B10G10R10_SNORM_PACK32;
            vtxParams.attributes[3].relativeOffset = 20u;
            //uv
            if (hasUV)
            {
                vtxParams.attributes[2].binding = 2u;
                vtxParams.attributes[2].format = EF_R32G32_SFLOAT;
                vtxParams.attributes[2].relativeOffset = 12u;
            }
            submeshes[i]->getPipeline()->getVertexInputParams() = vtxParams;
        }

        core::smart_refctd_ptr<ICPUBuffer> vtxBuf = core::make_smart_refctd_ptr<ICPUBuffer>(vertices.size() * sizeof(SObjVertex));
        memcpy(vtxBuf->getPointer(), vertices.data(), vtxBuf->getSize());

        auto ixBuf = core::make_smart_refctd_ptr<ICPUBuffer>(ixBufOffset);
        for (size_t i = 0ull; i < submeshes.size(); ++i)
        {
            submeshes[i]->getIndexBufferBinding()->buffer = ixBuf;
            uint64_t offset = submeshes[i]->getIndexBufferBinding()->offset;
            memcpy(reinterpret_cast<uint8_t*>(ixBuf->getPointer())+offset, indices[i].data(), indices[i].size()*4ull);

            SBufferBinding<ICPUBuffer> vtxBufBnd;
            vtxBufBnd.offset = 0ull;
            vtxBufBnd.buffer = vtxBuf;

            submeshes[i]->getVertexBufferBindings()[0] = vtxBufBnd;
            submeshes[i]->getVertexBufferBindings()[3] = vtxBufBnd;
            if (submeshes[i]->getPipeline()->getVertexInputParams().enabledAttribFlags & 0b0100)
                submeshes[i]->getVertexBufferBindings()[2] = vtxBufBnd;
        }
    }
	asset::CCPUMesh* mesh = new asset::CCPUMesh();

	// Combine all the groups (meshbuffers) into the mesh
	for ( uint32_t m = 0; m < ctx.Materials.size(); ++m )
	{
        {//arbitrary scope
            auto preloadedMbItr = ctx.preloadedSubmeshes.find(ctx.Materials[m]);
            if (preloadedMbItr != ctx.preloadedSubmeshes.end())
            {
                mesh->addMeshBuffer(core::smart_refctd_ptr<ICPUMeshBuffer>(preloadedMbItr->second,core::dont_grab));
                preloadedMbItr->second->drop(); // after grab when we got it from cache
                continue;
            }
        }

		if ( ctx.Materials[m]->Indices.size() == 0 )
        {
            continue;
        }

        if (ctx.Materials[m]->RecalculateNormals)
        {
            core::allocator<core::vectorSIMDf> alctr;
            core::vectorSIMDf* newNormals = alctr.allocate(ctx.Materials[m]->Vertices.size());
            memset(newNormals,0,sizeof(core::vectorSIMDf)*ctx.Materials[m]->Vertices.size());
            for (size_t i=0; i<ctx.Materials[m]->Indices.size(); i+=3)
            {
                core::vectorSIMDf v1,v2,v3;
                v1.set(ctx.Materials[m]->Vertices[ctx.Materials[m]->Indices[i+0]].pos);
                v2.set(ctx.Materials[m]->Vertices[ctx.Materials[m]->Indices[i+1]].pos);
                v3.set(ctx.Materials[m]->Vertices[ctx.Materials[m]->Indices[i+2]].pos);
                v1.makeSafe3D();
                v2.makeSafe3D();
                v3.makeSafe3D();
                core::vectorSIMDf normal(core::plane3dSIMDf(v1, v2, v3).getNormal());
                newNormals[ctx.Materials[m]->Indices[i+0]] += normal;
                newNormals[ctx.Materials[m]->Indices[i+1]] += normal;
                newNormals[ctx.Materials[m]->Indices[i+2]] += normal;
            }
            for (size_t i=0; i<ctx.Materials[m]->Vertices.size(); i++)
            {
                ctx.Materials[m]->Vertices[i].normal32bit = asset::quantizeNormal2_10_10_10(newNormals[i]);
            }
            alctr.deallocate(newNormals,ctx.Materials[m]->Vertices.size());
        }
        //if (ctx.Materials[m]->Material.MaterialType == -1)
            //os::Printer::log("Loading OBJ Models with normal maps and tangents not supported!\n",ELL_ERROR);
        //TODO ^^^^

            /*
        {
            SMesh tmp;
            tmp.addMeshBuffer(ctx.Materials[m]->Meshbuffer);
            IMesh* tangentMesh = SceneManager->getMeshManipulator()->createMeshWithTangents(&tmp);
            mesh->addMeshBuffer(tangentMesh->getMeshBuffer(0));
            tangentMesh->drop();
        }
        else*/

        auto meshbuffer = core::make_smart_refctd_ptr<asset::ICPUMeshBuffer>();
        mesh->addMeshBuffer(core::smart_refctd_ptr(meshbuffer));

#ifndef NEW_SHADERS
        meshbuffer->getMaterial() = ctx.Materials[m]->Material;

        auto desc = core::make_smart_refctd_ptr<asset::ICPUMeshDataFormatDesc>();
#endif
        bool doesntNeedIndices = true;
        size_t baseVertex = ctx.Materials[m]->Indices[0];
        for (size_t i=1; i<ctx.Materials[m]->Indices.size(); i++)
        {
            if (baseVertex+i!=ctx.Materials[m]->Indices[i])
            {
                doesntNeedIndices = false;
                break;
            }
        }

        size_t actualVertexCount;
        if (doesntNeedIndices)
        {
            meshbuffer->setIndexCount(ctx.Materials[m]->Indices.size()-baseVertex);
            actualVertexCount = meshbuffer->getIndexCount();
        }
        else
        {
            baseVertex = 0;
            actualVertexCount = ctx.Materials[m]->Vertices.size();
#ifndef NEW_SHADERS
			{
				auto indexbuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t)*ctx.Materials[m]->Indices.size());
				memcpy(indexbuf->getPointer(),&ctx.Materials[m]->Indices[0],indexbuf->getSize());
				desc->setIndexBuffer(std::move(indexbuf));
			}
#endif // !NEW_SHADERS

            meshbuffer->setIndexType(asset::EIT_32BIT);
            meshbuffer->setIndexCount(ctx.Materials[m]->Indices.size());
        }

#ifndef NEW_SHADERS
		{
			auto vertexbuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(actualVertexCount*sizeof(SObjVertex));
			desc->setVertexAttrBuffer(core::smart_refctd_ptr(vertexbuf),asset::EVAI_ATTR0,asset::EF_R32G32B32_SFLOAT,sizeof(SObjVertex),0);
			desc->setVertexAttrBuffer(core::smart_refctd_ptr(vertexbuf),asset::EVAI_ATTR2,asset::EF_R32G32_SFLOAT,sizeof(SObjVertex),12);
			desc->setVertexAttrBuffer(core::smart_refctd_ptr(vertexbuf),asset::EVAI_ATTR3,asset::EF_A2B10G10R10_SNORM_PACK32,sizeof(SObjVertex),20); //normal
			memcpy(vertexbuf->getPointer(),ctx.Materials[m]->Vertices.data()+baseVertex,vertexbuf->getSize());
		}
		meshbuffer->setMeshDataAndFormat(std::move(desc));
#endif
        SAssetBundle bundle({std::move(meshbuffer)});
        _override->insertAssetIntoCache(bundle, genKeyForMeshBuf(ctx, _file->getFileName().c_str(), ctx.Materials[m]->Name, ctx.Materials[m]->Group), ctx.inner, 1u);
        //transfer ownership to smart_refctd_ptr, so instead of grab() in smart_refctd_ptr and drop() here, just do nothing (thus dont_grab goes as smart ptr ctor arg)
	}

	// more cleaning up
	ctx.Materials.clear();

	if ( 0 != mesh->getMeshBufferCount() )
		mesh->recalculateBoundingBox(true);
	else
    {
		mesh->drop();
        return {};
    }

	return SAssetBundle({core::smart_refctd_ptr<IAsset>(mesh,core::dont_grab)});
}

//! Read RGB color
const char* COBJMeshFileLoader::readColor(const char* bufPtr, video::SColor& color, const char* const bufEnd)
{
	const uint32_t COLOR_BUFFER_LENGTH = 16;
	char colStr[COLOR_BUFFER_LENGTH];

	float tmp;

	color.setAlpha(255);
	bufPtr = goAndCopyNextWord(colStr, bufPtr, COLOR_BUFFER_LENGTH, bufEnd);
	sscanf(colStr,"%f",&tmp);
	color.setRed((int32_t)(tmp * 255.0f));
	bufPtr = goAndCopyNextWord(colStr,   bufPtr, COLOR_BUFFER_LENGTH, bufEnd);
	sscanf(colStr,"%f",&tmp);
	color.setGreen((int32_t)(tmp * 255.0f));
	bufPtr = goAndCopyNextWord(colStr,   bufPtr, COLOR_BUFFER_LENGTH, bufEnd);
	sscanf(colStr,"%f",&tmp);
	color.setBlue((int32_t)(tmp * 255.0f));
	return bufPtr;
}


//! Read 3d vector of floats
const char* COBJMeshFileLoader::readVec3(const char* bufPtr, core::vector3df& vec, const char* const bufEnd)
{
	const uint32_t WORD_BUFFER_LENGTH = 256;
	char wordBuffer[WORD_BUFFER_LENGTH];

	bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
	sscanf(wordBuffer,"%f",&vec.X);
	bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
	sscanf(wordBuffer,"%f",&vec.Y);
	bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
	sscanf(wordBuffer,"%f",&vec.Z);

	vec.X = -vec.X; // change handedness
	return bufPtr;
}


//! Read 2d vector of floats
const char* COBJMeshFileLoader::readUV(const char* bufPtr, core::vector2df& vec, const char* const bufEnd)
{
	const uint32_t WORD_BUFFER_LENGTH = 256;
	char wordBuffer[WORD_BUFFER_LENGTH];

	bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
	sscanf(wordBuffer,"%f",&vec.X);
	bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
	sscanf(wordBuffer,"%f",&vec.Y);

	vec.Y = 1-vec.Y; // change handedness
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

std::string COBJMeshFileLoader::genKeyForMeshBuf(const SContext & _ctx, const std::string & _baseKey, const std::string & _mtlName, const std::string & _grpName) const
{
    if (_ctx.useMaterials)
    {   
        if (_ctx.useGroups)
            return _baseKey + "?" + _grpName + "?" + _mtlName;
        return _baseKey + "?" +  _mtlName;
    }
    return ""; // if nothing's broken this will never happen
}




} // end namespace scene
} // end namespace irr

#endif // _IRR_COMPILE_WITH_OBJ_LOADER_
