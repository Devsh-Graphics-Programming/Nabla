// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_ASSET_C_PLY_MESH_FILE_LOADER_H_INCLUDED__
#define __NBL_ASSET_C_PLY_MESH_FILE_LOADER_H_INCLUDED__

#include "nbl/asset/ICPUMeshBuffer.h"
#include "nbl/asset/interchange/IRenderpassIndependentPipelineLoader.h"
#include "nbl/asset/metadata/CPLYMetadata.h"

namespace nbl
{
namespace asset
{
// input buffer must be at least twice as long as the longest line in the file
#define PLY_INPUT_BUFFER_SIZE 51200  // file is loaded in 50k chunks

enum E_PLY_PROPERTY_TYPE
{
    EPLYPT_INT8 = 0,
    EPLYPT_INT16,
    EPLYPT_INT32,
    EPLYPT_FLOAT32,
    EPLYPT_FLOAT64,
    EPLYPT_LIST,
    EPLYPT_UNKNOWN
};

//! Meshloader capable of loading obj meshes.
class CPLYMeshFileLoader : public IRenderpassIndependentPipelineLoader
{
protected:
    //! Destructor
    virtual ~CPLYMeshFileLoader();

public:
    //! Constructor
    CPLYMeshFileLoader(IAssetManager* _am);

    virtual bool isALoadableFileFormat(io::IReadFile* _file) const override;

    virtual const char** getAssociatedFileExtensions() const override
    {
        static const char* ext[]{"ply", nullptr};
        return ext;
    }

    virtual uint64_t getSupportedAssetTypesBitfield() const override { return IAsset::ET_MESH; }

    //! creates/loads an animated mesh from the file.
    virtual SAssetBundle loadAsset(io::IReadFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

private:
    virtual void initialize() override;

    enum E_TYPE
    {
        ET_POS = 0,
        ET_UV = 2,
        ET_NORM = 3,
        ET_COL = 1
    };

    static const std::string getPipelineCacheKey(E_TYPE type, bool indexBufferBindingAvailable)
    {
        auto getTypeHash = [&]() -> std::string {
            bool status = true;

            switch(type)
            {
                case ET_POS:
                    return "nbl/builtin/pipeline/loader/PLY/only_position_attribute/";
                case ET_COL:
                    return "nbl/builtin/pipeline/loader/PLY/color_attribute/";
                case ET_UV:
                    return "nbl/builtin/pipeline/loader/PLY/uv_attribute/";
                case ET_NORM:
                    return "nbl/builtin/pipeline/loader/PLY/normal_attribute/";
                default: {
                    status = false;
                    assert(status);
                }
            }
            return (const char*)nullptr;
        };

        return getTypeHash() + (indexBufferBindingAvailable ? "triangle_list" : "point_list");
    }

    struct SPLYProperty
    {
        core::stringc Name;
        E_PLY_PROPERTY_TYPE Type;
#include "nbl/nblpack.h"
        union
        {
            uint8_t Int8;
            uint16_t Int16;
            uint32_t Int32;
            float Float32;
            double Double;
            struct SPLYListProperty
            {
                E_PLY_PROPERTY_TYPE CountType;
                E_PLY_PROPERTY_TYPE ItemType;
            } List PACK_STRUCT;

        } Data PACK_STRUCT;
#include "nbl/nblunpack.h"

        inline uint32_t size() const
        {
            switch(Type)
            {
                case EPLYPT_INT8:
                    return 1;
                case EPLYPT_INT16:
                    return 2;
                case EPLYPT_INT32:
                case EPLYPT_FLOAT32:
                    return 4;
                case EPLYPT_FLOAT64:
                    return 8;
                case EPLYPT_LIST:
                case EPLYPT_UNKNOWN:
                default:
                    return 0;
            }
        }

        inline bool isFloat() const
        {
            switch(Type)
            {
                case EPLYPT_FLOAT32:
                case EPLYPT_FLOAT64:
                    return true;
                case EPLYPT_INT8:
                case EPLYPT_INT16:
                case EPLYPT_INT32:
                case EPLYPT_LIST:
                case EPLYPT_UNKNOWN:
                default:
                    return false;
            }
        }
    };

    struct SPLYElement
    {
        // name of the element. We only want "vertex" and "face" elements
        // but we have to parse the others anyway.
        core::stringc Name;
        // The number of elements in the file
        uint32_t Count;
        // Properties of this element
        core::vector<SPLYProperty> Properties;
        // in binary files, true if this is a fixed size
        bool IsFixedWidth;
        // known size in bytes, 0 if unknown
        uint32_t KnownSize;
    };

    struct SContext
    {
        ~SContext()
        {
            if(Buffer)
            {
                _NBL_DELETE_ARRAY(reinterpret_cast<uint8_t*>(Buffer), PLY_INPUT_BUFFER_SIZE);
                Buffer = nullptr;
            }
            ElementList.clear();
        }

        IAssetLoader::SAssetLoadContext inner;
        uint32_t topHierarchyLevel;
        IAssetLoader::IAssetLoaderOverride* loaderOverride;

        core::vector<std::unique_ptr<SPLYElement>> ElementList;

        char* Buffer = nullptr;
        bool IsBinaryFile = false, IsWrongEndian = false, EndOfFile = false;
        int32_t LineLength = 0, WordLength = 0;
        char *StartPointer = nullptr, *EndPointer = nullptr, *LineEndPointer = nullptr;
    };

    bool allocateBuffer(SContext& _ctx);
    char* getNextLine(SContext& _ctx);
    char* getNextWord(SContext& _ctx);
    void fillBuffer(SContext& _ctx);
    E_PLY_PROPERTY_TYPE getPropertyType(const char* typeString) const;

    bool readVertex(SContext& _ctx, const SPLYElement& Element, asset::SBufferBinding<asset::ICPUBuffer> outAttributes[4], const uint32_t& currentVertexIndex, const IAssetLoader::SAssetLoadParams& _params);
    bool readFace(SContext& _ctx, const SPLYElement& Element, core::vector<uint32_t>& _outIndices);

    void skipElement(SContext& _ctx, const SPLYElement& Element);
    void skipProperty(SContext& _ctx, const SPLYProperty& Property);
    float getFloat(SContext& _ctx, E_PLY_PROPERTY_TYPE t);
    uint32_t getInt(SContext& _ctx, E_PLY_PROPERTY_TYPE t);
    void moveForward(SContext& _ctx, uint32_t bytes);

    bool genVertBuffersForMBuffer(
        ICPUMeshBuffer* _mbuf,
        const asset::SBufferBinding<asset::ICPUBuffer> attributes[4],
        SContext& context) const;

    template<typename aType>
    static inline void performActionBasedOnOrientationSystem(aType& varToHandle, void (*performOnCertainOrientation)(aType& varToHandle))
    {
        performOnCertainOrientation(varToHandle);
    }
};

}  // end namespace asset
}  // end namespace nbl

#endif
