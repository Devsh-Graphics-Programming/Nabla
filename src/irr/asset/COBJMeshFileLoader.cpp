// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"
#ifdef _IRR_COMPILE_WITH_OBJ_LOADER_

#include "IrrlichtDevice.h"
#include "IFileSystem.h"
#include "ISceneManager.h"
#include "COBJMeshFileLoader.h"
#include "irr/asset/IMeshManipulator.h"
#include "IVideoDriver.h"
#include "irr/video/SGPUMesh.h"
#include "SVertexManipulator.h"
#include "IReadFile.h"
#include "coreutil.h"
#include "os.h"
#include "irr/asset/IAssetManager.h"

#include "irr/core/Types.h"
#include "irr/core/math/plane3dSIMD.h"

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

//#ifdef _IRR_DEBUG
#define _IRR_DEBUG_OBJ_LOADER_
//#endif

static const uint32_t WORD_BUFFER_LENGTH = 512;


//! Constructor
COBJMeshFileLoader::COBJMeshFileLoader(IrrlichtDevice* _dev)
: Device(_dev), SceneManager(_dev->getSceneManager()), FileSystem(_dev->getFileSystem())
{
#ifdef _IRR_DEBUG
	setDebugName("COBJMeshFileLoader");
#endif
}


//! destructor
COBJMeshFileLoader::~COBJMeshFileLoader()
{
}

asset::IAsset* COBJMeshFileLoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
    SContext ctx(
        asset::IAssetLoader::SAssetLoadContext{
            _params,
            _file
        },
        _override
    );

	const long filesize = _file->getSize();
	if (!filesize)
		return nullptr;

	const uint32_t WORD_BUFFER_LENGTH = 512;

	core::vector<core::vector3df> vertexBuffer;
	core::vector<core::vector3df> normalsBuffer;
	core::vector<core::vector2df> textureCoordBuffer;

	SObjMtl * currMtl = new SObjMtl();
	ctx.Materials.push_back(currMtl);
	uint32_t smoothingGroup=0;

	const io::path fullName = _file->getFileName();
	const io::path relPath = io::IFileSystem::getFileDir(fullName)+"/";

	char* buf = new char[filesize];
	memset(buf, 0, filesize);
	_file->read((void*)buf, filesize);
	const char* const bufEnd = buf+filesize;

	// Process obj information
	const char* bufPtr = buf;
	std::string grpName, mtlName;
	bool mtlChanged=false;
    bool submeshLoadedFromCache = false;
	while(bufPtr != bufEnd)
	{
		switch(bufPtr[0])
		{
		case 'm':	// mtllib (material)
		{
			if (ctx.useMaterials)
			{
				char name[WORD_BUFFER_LENGTH];
				bufPtr = goAndCopyNextWord(name, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
#ifdef _IRR_DEBUG_OBJ_LOADER_
				os::Printer::log("Reading material _file",name);
#endif
				readMTL(ctx, name, relPath);
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
					vertexBuffer.push_back(vec);
				}
				break;

			case 'n':       // normal
				{
					core::vector3df vec;
					bufPtr = readVec3(bufPtr, vec, bufEnd);
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
			{
				char grp[WORD_BUFFER_LENGTH];
				bufPtr = goAndCopyNextWord(grp, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
#ifdef _IRR_DEBUG_OBJ_LOADER_
	os::Printer::log("Loaded group start",grp, ELL_DEBUG);
#endif
				if (ctx.useGroups)
				{
					if (0 != grp[0])
						grpName = grp;
					else
						grpName = "default";

                    asset::IAsset::E_TYPE types[] {asset::IAsset::ET_SUB_MESH, (asset::IAsset::E_TYPE)0u };
                    asset::IAsset* mb = _override->findCachedAsset(genKeyForMeshBuf(ctx, _file->getFileName().c_str(), mtlName, grpName), types, ctx.inner, 1u);
                    if (mb)
                    {
                        mb->grab();
                        SObjMtl* mtl = findMtl(ctx, mtlName, grpName);
                        ctx.preloadedSubmeshes.insert(std::make_pair(mtl, static_cast<asset::ICPUMeshBuffer*>(mb)));
                    }
                    else mtlChanged=true;

                    submeshLoadedFromCache = bool(mb);
				}
			}
			break;

		case 's': // smoothing can be a group or off (equiv. to 0)
			{
				char smooth[WORD_BUFFER_LENGTH];
				bufPtr = goAndCopyNextWord(smooth, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
#ifdef _IRR_DEBUG_OBJ_LOADER_
	os::Printer::log("Loaded smoothing group start",smooth, ELL_DEBUG);
#endif
				if (core::stringc("off")==smooth)
					smoothingGroup=0;
				else
                    sscanf(smooth,"%u",&smoothingGroup);
			}
			break;

		case 'u': // usemtl
			// get name of material
			{
				char matName[WORD_BUFFER_LENGTH];
				bufPtr = goAndCopyNextWord(matName, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
#ifdef _IRR_DEBUG_OBJ_LOADER_
	os::Printer::log("Loaded material start",matName, ELL_DEBUG);
#endif
				mtlName=matName;

                if (ctx.useMaterials && !ctx.useGroups)
                {
                    asset::IAsset::E_TYPE types[] {asset::IAsset::ET_SUB_MESH, (asset::IAsset::E_TYPE)0u };
                    asset::IAsset* mb = _override->findCachedAsset(genKeyForMeshBuf(ctx, _file->getFileName().c_str(), mtlName, grpName), types, ctx.inner, 1u);
                    if (mb)
                    {
                        mb->grab();
                        SObjMtl* mtl = findMtl(ctx, mtlName, grpName);
                        ctx.preloadedSubmeshes.insert(std::make_pair(mtl, static_cast<asset::ICPUMeshBuffer*>(mb)));
                    }
                    else mtlChanged=true;

                    submeshLoadedFromCache = bool(mb);
                }
			}
			break;

		case 'f':               // face
		{
            if (submeshLoadedFromCache)
                break;
			char vertexWord[WORD_BUFFER_LENGTH]; // for retrieving vertex data
			SObjVertex v;
			// Assign vertex color from currently active material's diffuse color
			if (mtlChanged)
			{
				// retrieve the material
				SObjMtl *useMtl = findMtl(ctx, mtlName, grpName);
				// only change material if we found it
				if (useMtl)
					currMtl = useMtl;
				mtlChanged=false;
			}

			// get all vertices data in this face (current line of obj _file)
			const core::stringc wordBuffer = copyLine(bufPtr, bufEnd);
			const char* linePtr = wordBuffer.c_str();
			const char* const endPtr = linePtr+wordBuffer.size();

			core::vector<uint32_t> faceCorners;
			faceCorners.reserve(32); // should be large enough

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
				uint32_t wlength = copyWord(vertexWord, linePtr, WORD_BUFFER_LENGTH, endPtr);
				// this function will also convert obj's 1-based index to c++'s 0-based index
				retrieveVertexIndices(vertexWord, Idx, vertexWord+wlength+1, vertexBuffer.size(), textureCoordBuffer.size(), normalsBuffer.size());
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
					v.uv[0] = 0.f;
					v.uv[1] = 0.f;
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
					currMtl->RecalculateNormals=true;
				}

				int vertLocation;
				core::map<SObjVertex, int>::iterator n = currMtl->VertMap.find(v);
				if (n!=currMtl->VertMap.end())
				{
					vertLocation = n->second;
				}
				else
				{
					currMtl->Vertices.push_back(v);
					vertLocation = currMtl->Vertices.size() -1;
					currMtl->VertMap.insert(std::pair<SObjVertex, int>(v, vertLocation));
				}

				faceCorners.push_back(vertLocation);

				// go to next vertex
				linePtr = goNextWord(linePtr, endPtr);
			}

			// triangulate the face
			for ( uint32_t i = 1; i < faceCorners.size() - 1; ++i )
			{
				// Add a triangle
				currMtl->Indices.push_back( faceCorners[i+1] );
				currMtl->Indices.push_back( faceCorners[i] );
				currMtl->Indices.push_back( faceCorners[0] );
			}
			faceCorners.resize(0); // fast clear
			faceCorners.reserve(32);
		}
		break;

		case '#': // comment
		default:
			break;
		}	// end switch(bufPtr[0])
		// eat up rest of line
		bufPtr = goNextLine(bufPtr, bufEnd);
	}	// end while(bufPtr && (bufPtr-buf<filesize))
	// Clean up the allocate obj _file contents
	delete [] buf;

	asset::SCPUMesh* mesh = new asset::SCPUMesh();

	// Combine all the groups (meshbuffers) into the mesh
	for ( uint32_t m = 0; m < ctx.Materials.size(); ++m )
	{
        {//arbitrary scope
            auto preloadedMbItr = ctx.preloadedSubmeshes.find(ctx.Materials[m]);
            if (preloadedMbItr != ctx.preloadedSubmeshes.end())
            {
                mesh->addMeshBuffer(preloadedMbItr->second);
                preloadedMbItr->second->drop(); // after grab inside addMeshBuffer()
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
        if (ctx.Materials[m]->Material.MaterialType == -1)
            os::Printer::log("Loading OBJ Models with normal maps and tangents not supported!\n",ELL_ERROR);/*
        {
            SMesh tmp;
            tmp.addMeshBuffer(ctx.Materials[m]->Meshbuffer);
            IMesh* tangentMesh = SceneManager->getMeshManipulator()->createMeshWithTangents(&tmp);
            mesh->addMeshBuffer(tangentMesh->getMeshBuffer(0));
            tangentMesh->drop();
        }
        else*/

        asset::ICPUMeshBuffer* meshbuffer = new asset::ICPUMeshBuffer();
        mesh->addMeshBuffer(meshbuffer);

        meshbuffer->getMaterial() = ctx.Materials[m]->Material;

        asset::ICPUMeshDataFormatDesc* desc = new asset::ICPUMeshDataFormatDesc();
        meshbuffer->setMeshDataAndFormat(desc);
        desc->drop();

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

        asset::ICPUBuffer* vertexbuf;
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

            asset::ICPUBuffer* indexbuf = new asset::ICPUBuffer(ctx.Materials[m]->Indices.size()*4);
            desc->setIndexBuffer(indexbuf);
            indexbuf->drop();
            memcpy(indexbuf->getPointer(),&ctx.Materials[m]->Indices[0],indexbuf->getSize());

            meshbuffer->setIndexType(asset::EIT_32BIT);
            meshbuffer->setIndexCount(ctx.Materials[m]->Indices.size());
        }

        vertexbuf = new asset::ICPUBuffer(actualVertexCount*sizeof(SObjVertex));
        desc->setVertexAttrBuffer(vertexbuf,asset::EVAI_ATTR0,asset::EF_R32G32B32_SFLOAT,sizeof(SObjVertex),0);
        desc->setVertexAttrBuffer(vertexbuf,asset::EVAI_ATTR2,asset::EF_R32G32_SFLOAT,sizeof(SObjVertex),12);
        desc->setVertexAttrBuffer(vertexbuf,asset::EVAI_ATTR3,asset::EF_A2B10G10R10_SNORM_PACK32,sizeof(SObjVertex),20); //normal
        memcpy(vertexbuf->getPointer(),ctx.Materials[m]->Vertices.data()+baseVertex,vertexbuf->getSize());
        vertexbuf->drop();

        _override->insertAssetIntoCache(meshbuffer, genKeyForMeshBuf(ctx, _file->getFileName().c_str(), ctx.Materials[m]->Name, ctx.Materials[m]->Group), ctx.inner, 1u);
        meshbuffer->drop();
	}

	// more cleaning up
	ctx.Materials.clear();

	if ( 0 != mesh->getMeshBufferCount() )
		mesh->recalculateBoundingBox(true);
	else
    {
		mesh->drop();
		mesh = NULL;
    }

	return mesh;
}


const char* COBJMeshFileLoader::readTextures(const SContext& _ctx, const char* bufPtr, const char* const bufEnd, SObjMtl* currMaterial, const io::path& relPath)
{
	E_TEXTURE_TYPE type = ETT_COLOR_MAP; // map_Kd - diffuse color texture map
	// map_Ks - specular color texture map
	// map_Ka - ambient color texture map
	// map_Ns - shininess texture map
	if ((!strncmp(bufPtr,"map_bump",8)) || (!strncmp(bufPtr,"bump",4)))
		type = ETT_NORMAL_MAP;
	else if ((!strncmp(bufPtr,"map_d",5)) || (!strncmp(bufPtr,"map_opacity",11)))
		type = ETT_OPACITY_MAP;
	else if (!strncmp(bufPtr,"map_refl",8))
		type = ETT_REFLECTION_MAP;
	// extract new material's name
	char textureNameBuf[WORD_BUFFER_LENGTH];
	bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);

	float bumpiness = 6.0f;
	bool clamp = false;
	// handle options
	while (textureNameBuf[0]=='-')
	{
		if (!strncmp(bufPtr,"-bm",3))
		{
			bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
			sscanf(textureNameBuf,"%f",&currMaterial->Material.MaterialTypeParam);
			bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
			continue;
		}
		else
		if (!strncmp(bufPtr,"-blendu",7))
			bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
		else
		if (!strncmp(bufPtr,"-blendv",7))
			bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
		else
		if (!strncmp(bufPtr,"-cc",3))
			bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
		else
		if (!strncmp(bufPtr,"-clamp",6))
			bufPtr = readBool(bufPtr, clamp, bufEnd);
		else
		if (!strncmp(bufPtr,"-texres",7))
			bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
		else
		if (!strncmp(bufPtr,"-type",5))
			bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
		else
		if (!strncmp(bufPtr,"-mm",3))
		{
			bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
			bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
		}
		else
		if (!strncmp(bufPtr,"-o",2)) // texture coord translation
		{
			bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
			// next parameters are optional, so skip rest of loop if no number is found
			bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
			if (!core::isdigit(textureNameBuf[0]))
				continue;
			bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
			if (!core::isdigit(textureNameBuf[0]))
				continue;
		}
		else
		if (!strncmp(bufPtr,"-s",2)) // texture coord scale
		{
			bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
			// next parameters are optional, so skip rest of loop if no number is found
			bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
			if (!core::isdigit(textureNameBuf[0]))
				continue;
			bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
			if (!core::isdigit(textureNameBuf[0]))
				continue;
		}
		else
		if (!strncmp(bufPtr,"-t",2))
		{
			bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
			// next parameters are optional, so skip rest of loop if no number is found
			bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
			if (!core::isdigit(textureNameBuf[0]))
				continue;
			bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
			if (!core::isdigit(textureNameBuf[0]))
				continue;
		}
		// get next word
		bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
	}

	if ((type==ETT_NORMAL_MAP) && (core::isdigit(textureNameBuf[0])))
	{
		sscanf(textureNameBuf,"%f",&currMaterial->Material.MaterialTypeParam);
		bufPtr = goAndCopyNextWord(textureNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
	}
	if (clamp)
    {
        for (size_t i=0; i<_IRR_MATERIAL_MAX_TEXTURES_; i++)
        {
            currMaterial->Material.TextureLayer[i].SamplingParams.TextureWrapU = video::ETC_CLAMP_TO_EDGE;
            currMaterial->Material.TextureLayer[i].SamplingParams.TextureWrapV = video::ETC_CLAMP_TO_EDGE;
            currMaterial->Material.TextureLayer[i].SamplingParams.TextureWrapW = video::ETC_CLAMP_TO_EDGE;
        }
    }

	io::path texname(textureNameBuf);
	handleBackslashes(&texname);

	asset::ICPUTexture* texture = nullptr;
	if (texname.size())
	{
        if (FileSystem->existFile(texname))
		{
            texture = static_cast<asset::ICPUTexture*>(
                interm_getAssetInHierarchy(Device->getAssetManager(), texname.c_str(), _ctx.inner.params, 2u, _ctx.loaderOverride)
            );
		}
		else
		{
			// try to read in the relative path, the .obj is loaded from
            texture = static_cast<asset::ICPUTexture*>(
                interm_getAssetInHierarchy(Device->getAssetManager(), (relPath + texname).c_str(), _ctx.inner.params, 2u, _ctx.loaderOverride)
            );
		}
	}
	if ( texture )
	{
		if (type==ETT_COLOR_MAP)
        {
			currMaterial->Material.setTexture(0, texture);
        }
		else if (type==ETT_NORMAL_MAP)
		{
#ifdef _IRR_DEBUG
            os::Printer::log("Loading OBJ Models with normal maps not supported!\n",ELL_ERROR);
#endif // _IRR_DEBUG
			currMaterial->Material.setTexture(1, texture);
			currMaterial->Material.MaterialType=(video::E_MATERIAL_TYPE)-1;
			currMaterial->Material.MaterialTypeParam=0.035f;
		}
		else if (type==ETT_OPACITY_MAP)
		{
			currMaterial->Material.setTexture(0, texture);
			currMaterial->Material.MaterialType=video::EMT_TRANSPARENT_ADD_COLOR;
		}
		else if (type==ETT_REFLECTION_MAP)
		{
//						currMaterial->Material.Textures[1] = texture;
//						currMaterial->Material.MaterialType=video::EMT_REFLECTION_2_LAYER;
		}
	}
	return bufPtr;
}


void COBJMeshFileLoader::readMTL(SContext& _ctx, const char* fileName, const io::path& relPath)
{
	const io::path realFile(fileName);
	io::IReadFile * mtlReader;

	if (FileSystem->existFile(realFile))
		mtlReader = FileSystem->createAndOpenFile(realFile);
	else if (FileSystem->existFile(relPath + realFile))
		mtlReader = FileSystem->createAndOpenFile(relPath + realFile);
	else if (FileSystem->existFile(io::IFileSystem::getFileBasename(realFile)))
		mtlReader = FileSystem->createAndOpenFile(io::IFileSystem::getFileBasename(realFile));
	else
		mtlReader = FileSystem->createAndOpenFile(relPath + io::IFileSystem::getFileBasename(realFile));
	if (!mtlReader)	// fail to open and read file
	{
		os::Printer::log("Could not open material file", realFile.c_str(), ELL_WARNING);
		return;
	}

	const long filesize = mtlReader->getSize();
	if (!filesize)
	{
		os::Printer::log("Skipping empty material file", realFile.c_str(), ELL_WARNING);
		mtlReader->drop();
		return;
	}

	char* buf = new char[filesize];
	mtlReader->read((void*)buf, filesize);
	const char* bufEnd = buf+filesize;

	SObjMtl* currMaterial = 0;

	const char* bufPtr = buf;
	while(bufPtr != bufEnd)
	{
		switch(*bufPtr)
		{
			case 'n': // newmtl
			{
				// if there's an existing material, store it first
				if ( currMaterial )
					_ctx.Materials.push_back( currMaterial );

				// extract new material's name
				char mtlNameBuf[WORD_BUFFER_LENGTH];
				bufPtr = goAndCopyNextWord(mtlNameBuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);

				currMaterial = new SObjMtl;
				currMaterial->Name = mtlNameBuf;
			}
			break;
			case 'i': // illum - illumination
			if ( currMaterial )
			{
				const uint32_t COLOR_BUFFER_LENGTH = 16;
				char illumStr[COLOR_BUFFER_LENGTH];

				bufPtr = goAndCopyNextWord(illumStr, bufPtr, COLOR_BUFFER_LENGTH, bufEnd);
				currMaterial->Illumination = (char)atol(illumStr);
			}
			break;
			case 'N':
			if ( currMaterial )
			{
				switch(bufPtr[1])
				{
				case 's': // Ns - shininess
					{
						const uint32_t COLOR_BUFFER_LENGTH = 16;
						char nsStr[COLOR_BUFFER_LENGTH];

						bufPtr = goAndCopyNextWord(nsStr, bufPtr, COLOR_BUFFER_LENGTH, bufEnd);
						float shininessValue;
						sscanf(nsStr,"%f",&shininessValue);

						// wavefront shininess is from [0, 1000], so scale for OpenGL
						shininessValue *= 0.128f;
						currMaterial->Material.Shininess = shininessValue;
					}
				break;
				case 'i': // Ni - refraction index
					{
						char tmpbuf[WORD_BUFFER_LENGTH];
						bufPtr = goAndCopyNextWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
					}
				break;
				}
			}
			break;
			case 'K':
			if ( currMaterial )
			{
				switch(bufPtr[1])
				{
				case 'd':		// Kd = diffuse
					{
						bufPtr = readColor(bufPtr, currMaterial->Material.DiffuseColor, bufEnd);

					}
					break;

				case 's':		// Ks = specular
					{
						bufPtr = readColor(bufPtr, currMaterial->Material.SpecularColor, bufEnd);
					}
					break;

				case 'a':		// Ka = ambience
					{
						bufPtr=readColor(bufPtr, currMaterial->Material.AmbientColor, bufEnd);
					}
					break;
				case 'e':		// Ke = emissive
					{
						bufPtr=readColor(bufPtr, currMaterial->Material.EmissiveColor, bufEnd);
					}
					break;
				}	// end switch(bufPtr[1])
			}	// end case 'K': if ( 0 != currMaterial )...
			break;
			case 'b': // bump
			case 'm': // texture maps
			if (currMaterial)
			{
				bufPtr=readTextures(_ctx, bufPtr, bufEnd, currMaterial, relPath);
			}
			break;
			case 'd': // d - transparency
			if ( currMaterial )
			{
				const uint32_t COLOR_BUFFER_LENGTH = 16;
				char dStr[COLOR_BUFFER_LENGTH];

				bufPtr = goAndCopyNextWord(dStr, bufPtr, COLOR_BUFFER_LENGTH, bufEnd);
				float dValue;
				sscanf(dStr,"%f",&dValue);

				currMaterial->Material.DiffuseColor.setAlpha( (int32_t)(dValue * 255) );
				if (dValue<1.0f)
					currMaterial->Material.MaterialType = video::EMT_TRANSPARENT_VERTEX_ALPHA;
			}
			break;
			case 'T':
			if ( currMaterial )
			{
				switch ( bufPtr[1] )
				{
				case 'f':		// Tf - Transmitivity
					const uint32_t COLOR_BUFFER_LENGTH = 16;
					char redStr[COLOR_BUFFER_LENGTH];
					char greenStr[COLOR_BUFFER_LENGTH];
					char blueStr[COLOR_BUFFER_LENGTH];

					bufPtr = goAndCopyNextWord(redStr,   bufPtr, COLOR_BUFFER_LENGTH, bufEnd);
					bufPtr = goAndCopyNextWord(greenStr, bufPtr, COLOR_BUFFER_LENGTH, bufEnd);
					bufPtr = goAndCopyNextWord(blueStr,  bufPtr, COLOR_BUFFER_LENGTH, bufEnd);

					float red,green,blue;
					sscanf(redStr,"%f",&red);
					sscanf(greenStr,"%f",&green);
					sscanf(blueStr,"%f",&blue);
					float transparency = ( red+green+blue ) / 3;

					currMaterial->Material.DiffuseColor.setAlpha( (int32_t)(transparency * 255) );
					if (transparency < 1.0f)
						currMaterial->Material.MaterialType = video::EMT_TRANSPARENT_VERTEX_ALPHA;
				}
			}
			break;
			default: // comments or not recognised
			break;
		} // end switch(bufPtr[0])
		// go to next line
		bufPtr = goNextLine(bufPtr, bufEnd);
	}	// end while (bufPtr)

	// end of file. if there's an existing material, store it
	if ( currMaterial )
		_ctx.Materials.push_back( currMaterial );

	delete [] buf;
	mtlReader->drop();
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


COBJMeshFileLoader::SObjMtl* COBJMeshFileLoader::findMtl(SContext& _ctx, const std::string& mtlName, const std::string& grpName)
{
	COBJMeshFileLoader::SObjMtl* defMaterial = 0;
	// search existing Materials for best match
	// exact match does return immediately, only name match means a new group
	for (uint32_t i = 0; i < _ctx.Materials.size(); ++i)
	{
		if (_ctx.Materials[i]->Name == mtlName )
		{
			if (_ctx.Materials[i]->Group == grpName )
				return _ctx.Materials[i];
			else
				defMaterial = _ctx.Materials[i];
		}
	}
	// we found a partial match
	if (defMaterial)
	{
        _ctx.Materials.push_back(new SObjMtl(*defMaterial));
        _ctx.Materials.back()->Group = grpName;
		return _ctx.Materials.back();
	}
	// we found a new group for a non-existant material
	else if (grpName.length())
	{
        _ctx.Materials.push_back(new SObjMtl(*_ctx.Materials[0]));
        _ctx.Materials.back()->Group = grpName;
		return _ctx.Materials.back();
	}
	return 0;
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

	uint32_t length = core::min_(i, outBufLength-1);
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
			sscanf(word,"%u",idx+idxType);
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
