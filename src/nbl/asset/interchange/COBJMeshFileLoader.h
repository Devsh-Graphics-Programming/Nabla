// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_ASSET_C_OBJ_MESH_FILE_LOADER_H_INCLUDED__
#define __NBL_ASSET_C_OBJ_MESH_FILE_LOADER_H_INCLUDED__

#include "nbl/core/core.h"
#include "nbl/asset/ICPUMeshBuffer.h"
#include "nbl/asset/interchange/IAssetLoader.h"
#include "nbl/asset/metadata/CMTLMetadata.h"

namespace nbl
{
namespace io
{
class IFileSystem;
}

namespace asset
{
#include "nbl/nblpack.h"
class SObjVertex
{
public:
    inline bool operator<(const SObjVertex& other) const
    {
        if(pos[0] == other.pos[0])
        {
            if(pos[1] == other.pos[1])
            {
                if(pos[2] == other.pos[2])
                {
                    if(uv[0] == other.uv[0])
                    {
                        if(uv[1] == other.uv[1])
                            return normal32bit < other.normal32bit;

                        return uv[1] < other.uv[1];
                    }
                    return uv[0] < other.uv[0];
                }
                return pos[2] < other.pos[2];
            }
            return pos[1] < other.pos[1];
        }

        return pos[0] < other.pos[0];
    }
    inline bool operator==(const SObjVertex& other) const
    {
        return pos[0] == other.pos[0] && pos[1] == other.pos[1] && pos[2] == other.pos[2] && uv[0] == other.uv[0] && uv[1] == other.uv[1] && normal32bit == other.normal32bit;
    }
    float pos[3];
    float uv[2];
    CQuantNormalCache::value_type_t<EF_A2B10G10R10_SNORM_PACK32> normal32bit;
} PACK_STRUCT;
#include "nbl/nblunpack.h"

//! Meshloader capable of loading obj meshes.
class COBJMeshFileLoader : public asset::IAssetLoader
{
    struct SContext
    {
        SContext(const IAssetLoader::SAssetLoadContext& _innerCtx, uint32_t _topHierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
            : inner(_innerCtx), topHierarchyLevel(_topHierarchyLevel), loaderOverride(_override) {}

        IAssetLoader::SAssetLoadContext inner;
        uint32_t topHierarchyLevel;
        IAssetLoader::IAssetLoaderOverride* loaderOverride;

        const bool useGroups = false;
        const bool useMaterials = true;
    };

protected:
    //! destructor
    virtual ~COBJMeshFileLoader();

public:
    //! Constructor
    COBJMeshFileLoader(IAssetManager* _manager);

    virtual bool isALoadableFileFormat(io::IReadFile* _file) const override
    {
        // OBJ doesn't really have any header but usually starts with a comment
        const size_t prevPos = _file->getPos();
        _file->seek(0u);
        char c;
        _file->read(&c, 1u);
        _file->seek(prevPos);
        return c == '#' || c == 'v';
    }

    virtual const char** getAssociatedFileExtensions() const override
    {
        static const char* ext[]{"obj", nullptr};
        return ext;
    }

    virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH; }

    virtual asset::SAssetBundle loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

private:
    // returns a pointer to the first printable character available in the buffer
    const char* goFirstWord(const char* buf, const char* const bufEnd, bool acrossNewlines = true);
    // returns a pointer to the first printable character after the first non-printable
    const char* goNextWord(const char* buf, const char* const bufEnd, bool acrossNewlines = true);
    // returns a pointer to the next printable character after the first line break
    const char* goNextLine(const char* buf, const char* const bufEnd);
    // copies the current word from the inBuf to the outBuf
    uint32_t copyWord(char* outBuf, const char* inBuf, uint32_t outBufLength, const char* const pBufEnd);
    // copies the current line from the inBuf to the outBuf
    core::stringc copyLine(const char* inBuf, const char* const bufEnd);

    // combination of goNextWord followed by copyWord
    const char* goAndCopyNextWord(char* outBuf, const char* inBuf, uint32_t outBufLength, const char* const pBufEnd);

    //! Read 3d vector of floats
    const char* readVec3(const char* bufPtr, float vec[3], const char* const pBufEnd);
    //! Read 2d vector of floats
    const char* readUV(const char* bufPtr, float vec[2], const char* const pBufEnd);
    //! Read boolean value represented as 'on' or 'off'
    const char* readBool(const char* bufPtr, bool& tf, const char* const bufEnd);

    // reads and convert to integer the vertex indices in a line of obj file's face statement
    // -1 for the index if it doesn't exist
    // indices are changed to 0-based index instead of 1-based from the obj file
    bool retrieveVertexIndices(char* vertexData, int32_t* idx, const char* bufEnd, uint32_t vbsize, uint32_t vtsize, uint32_t vnsize);

    std::string genKeyForMeshBuf(const SContext& _ctx, const std::string& _baseKey, const std::string& _mtlName, const std::string& _grpName) const;

    IAssetManager* AssetManager;
    io::IFileSystem* FileSystem;

    template<typename aType>
    static inline void performActionBasedOnOrientationSystem(aType& varToHandle, void (*performOnCertainOrientation)(aType& varToHandle))
    {
        performOnCertainOrientation(varToHandle);
    }
};

}  // end namespace asset
}  // end namespace nbl

#endif
