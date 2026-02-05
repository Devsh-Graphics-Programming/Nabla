// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "nbl/core/declarations.h"

#include "nbl/asset/IAssetManager.h"
#include "nbl/asset/utils/CPolygonGeometryManipulator.h"

#ifdef _NBL_COMPILE_WITH_OBJ_LOADER_

#include "nbl/system/ISystem.h"
#include "nbl/system/IFile.h"

#include "COBJMeshFileLoader.h"

#include <filesystem>

namespace nbl::asset
{

static const uint32_t WORD_BUFFER_LENGTH = 512u;

struct ObjVertexKey
{
    int32_t pos;
    int32_t uv;
    int32_t normal;

    inline bool operator<(const ObjVertexKey& other) const
    {
        if (pos == other.pos)
        {
            if (uv == other.uv)
                return normal < other.normal;
            return uv < other.uv;
        }
        return pos < other.pos;
    }
};

struct Float3
{
    float x;
    float y;
    float z;
};

struct Float2
{
    float x;
    float y;
};

static_assert(sizeof(Float3) == 12);
static_assert(sizeof(Float2) == 8);

COBJMeshFileLoader::COBJMeshFileLoader(IAssetManager* _manager) : AssetManager(_manager), System(_manager->getSystem())
{
}

COBJMeshFileLoader::~COBJMeshFileLoader()
{
}

asset::SAssetBundle COBJMeshFileLoader::loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
    if (!_file)
        return {};

    const long filesize = _file->getSize();
    if (filesize <= 0)
        return {};

    std::string fileContents;
    fileContents.resize(filesize);

    system::IFile::success_t success;
    _file->read(success, fileContents.data(), 0, filesize);
    if (!success)
        return {};

    const char* const buf = fileContents.data();
    const char* const bufEnd = buf + filesize;
    const char* bufPtr = buf;

    core::vector<Float3> positions;
    core::vector<Float3> normals;
    core::vector<Float2> uvs;

    core::vector<Float3> outPositions;
    core::vector<Float3> outNormals;
    core::vector<Float2> outUVs;
    core::vector<uint32_t> indices;

    core::map<ObjVertexKey, uint32_t> vtxMap;

    bool hasNormals = false;
    bool hasUVs = false;

    char tmpbuf[WORD_BUFFER_LENGTH]{};

    while (bufPtr != bufEnd)
    {
        switch (bufPtr[0])
        {
        case 'v':
            switch (bufPtr[1])
            {
            case ' ':
                {
                    Float3 vec{};
                    bufPtr = readVec3(bufPtr, &vec.x, bufEnd);
                    positions.push_back(vec);
                }
                break;
            case 'n':
                {
                    Float3 vec{};
                    bufPtr = readVec3(bufPtr, &vec.x, bufEnd);
                    normals.push_back(vec);
                }
                break;
            case 't':
                {
                    Float2 vec{};
                    bufPtr = readUV(bufPtr, &vec.x, bufEnd);
                    uvs.push_back(vec);
                }
                break;
            default:
                break;
            }
            break;
        case 'f':
            {
                if (positions.empty())
                    return {};

                const std::string line = copyLine(bufPtr, bufEnd);
                const char* linePtr = line.c_str();
                const char* const endPtr = linePtr + line.size();

                core::vector<uint32_t> faceCorners;
                faceCorners.reserve(16ull);

                linePtr = goNextWord(linePtr, endPtr);
                while (0 != linePtr[0])
                {
                    int32_t idx[3] = { -1, -1, -1 };
                    const uint32_t wlength = copyWord(tmpbuf, linePtr, WORD_BUFFER_LENGTH, endPtr);
                    retrieveVertexIndices(tmpbuf, idx, tmpbuf + wlength + 1, positions.size(), uvs.size(), normals.size());

                    if (idx[0] < 0 || static_cast<size_t>(idx[0]) >= positions.size())
                        return {};

                    ObjVertexKey key = { idx[0], idx[1], idx[2] };
                    auto it = vtxMap.find(key);
                    uint32_t outIx = 0u;
                    if (it == vtxMap.end())
                    {
                        outIx = static_cast<uint32_t>(outPositions.size());
                        vtxMap.insert({ key, outIx });

                        outPositions.push_back(positions[idx[0]]);

                        Float2 uv = { 0.f, 0.f };
                        if (idx[1] >= 0 && static_cast<size_t>(idx[1]) < uvs.size())
                        {
                            uv = uvs[idx[1]];
                            hasUVs = true;
                        }
                        outUVs.push_back(uv);

                        Float3 normal = { 0.f, 0.f, 1.f };
                        if (idx[2] >= 0 && static_cast<size_t>(idx[2]) < normals.size())
                        {
                            normal = normals[idx[2]];
                            hasNormals = true;
                        }
                        outNormals.push_back(normal);
                    }
                    else
                    {
                        outIx = it->second;
                    }

                    faceCorners.push_back(outIx);

                    linePtr = goNextWord(linePtr, endPtr);
                }

                for (uint32_t i = 1u; i + 1u < faceCorners.size(); ++i)
                {
                    indices.push_back(faceCorners[i + 1]);
                    indices.push_back(faceCorners[i]);
                    indices.push_back(faceCorners[0]);
                }
            }
            break;
        default:
            break;
        }

        bufPtr = goNextLine(bufPtr, bufEnd);
    }

    if (outPositions.empty())
        return {};

    auto geometry = core::make_smart_refctd_ptr<ICPUPolygonGeometry>();
    geometry->setPositionView(IGeometryLoader::createView(EF_R32G32B32_SFLOAT, outPositions.size(), outPositions.data()));

    if (hasNormals)
        geometry->setNormalView(IGeometryLoader::createView(EF_R32G32B32_SFLOAT, outNormals.size(), outNormals.data()));

    if (hasUVs)
        geometry->getAuxAttributeViews()->push_back(IGeometryLoader::createView(EF_R32G32_SFLOAT, outUVs.size(), outUVs.data()));

    if (!indices.empty())
    {
        geometry->setIndexing(IPolygonGeometryBase::TriangleList());
        geometry->setIndexView(IGeometryLoader::createView(EF_R32_UINT, indices.size(), indices.data()));
    }
    else
    {
        geometry->setIndexing(IPolygonGeometryBase::PointList());
    }

    CPolygonGeometryManipulator::recomputeContentHashes(geometry.get());
    CPolygonGeometryManipulator::recomputeRanges(geometry.get());
    CPolygonGeometryManipulator::recomputeAABB(geometry.get());

    return SAssetBundle(core::smart_refctd_ptr<IAssetMetadata>(), { std::move(geometry) });
}

const char* COBJMeshFileLoader::readVec3(const char* bufPtr, float vec[3], const char* const bufEnd)
{
    const uint32_t WORD_BUFFER_LENGTH = 256;
    char wordBuffer[WORD_BUFFER_LENGTH];

    bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
    sscanf(wordBuffer, "%f", vec);
    bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
    sscanf(wordBuffer, "%f", vec + 1);
    bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
    sscanf(wordBuffer, "%f", vec + 2);

    return bufPtr;
}

const char* COBJMeshFileLoader::readUV(const char* bufPtr, float vec[2], const char* const bufEnd)
{
    const uint32_t WORD_BUFFER_LENGTH = 256;
    char wordBuffer[WORD_BUFFER_LENGTH];

    bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
    sscanf(wordBuffer, "%f", vec);
    bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
    sscanf(wordBuffer, "%f", vec + 1);

    vec[1] = 1.f - vec[1];
    return bufPtr;
}

const char* COBJMeshFileLoader::readBool(const char* bufPtr, bool& tf, const char* const bufEnd)
{
    const uint32_t BUFFER_LENGTH = 8;
    char tfStr[BUFFER_LENGTH];

    bufPtr = goAndCopyNextWord(tfStr, bufPtr, BUFFER_LENGTH, bufEnd);
    tf = strcmp(tfStr, "off") != 0;
    return bufPtr;
}

const char* COBJMeshFileLoader::goFirstWord(const char* buf, const char* const bufEnd, bool acrossNewlines)
{
    if (acrossNewlines)
        while ((buf != bufEnd) && core::isspace(*buf))
            ++buf;
    else
        while ((buf != bufEnd) && core::isspace(*buf) && (*buf != '\n'))
            ++buf;

    return buf;
}

const char* COBJMeshFileLoader::goNextWord(const char* buf, const char* const bufEnd, bool acrossNewlines)
{
    while ((buf != bufEnd) && !core::isspace(*buf))
        ++buf;

    return goFirstWord(buf, bufEnd, acrossNewlines);
}

const char* COBJMeshFileLoader::goNextLine(const char* buf, const char* const bufEnd)
{
    while (buf != bufEnd)
    {
        if (*buf == '\n' || *buf == '\r')
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
    while (inBuf[i])
    {
        if (core::isspace(inBuf[i]) || &(inBuf[i]) == bufEnd)
            break;
        ++i;
    }

    uint32_t length = core::min(i, outBufLength - 1);
    for (uint32_t j = 0; j < length; ++j)
        outBuf[j] = inBuf[j];

    outBuf[length] = 0;
    return length;
}

std::string COBJMeshFileLoader::copyLine(const char* inBuf, const char* bufEnd)
{
    if (!inBuf)
        return std::string();

    const char* ptr = inBuf;
    while (ptr < bufEnd)
    {
        if (*ptr == '\n' || *ptr == '\r')
            break;
        ++ptr;
    }
    return std::string(inBuf, (uint32_t)(ptr - inBuf + ((ptr < bufEnd) ? 1 : 0)));
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
    uint32_t idxType = 0;

    uint32_t i = 0;
    while (p != bufEnd)
    {
        if ((core::isdigit(*p)) || (*p == '-'))
        {
            word[i++] = *p;
        }
        else if (*p == '/' || *p == ' ' || *p == '\0')
        {
            word[i] = '\0';
            sscanf(word, "%d", idx + idxType);
            if (idx[idxType] < 0)
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
                idx[idxType] -= 1;

            word[0] = '\0';
            i = 0;

            if (*p == '/')
            {
                if (++idxType > 2)
                    idxType = 0;
            }
            else
            {
                while (++idxType < 3)
                    idx[idxType] = -1;
                ++p;
                break;
            }
        }

        ++p;
    }

    return true;
}

}

#endif // _NBL_COMPILE_WITH_OBJ_LOADER_
