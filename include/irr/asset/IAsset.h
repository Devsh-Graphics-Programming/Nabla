#ifndef __IRR_I_ASSET_H_INCLUDED__
#define __IRR_I_ASSET_H_INCLUDED__

#include <string>
#include "irr/core/IReferenceCounted.h"

namespace irr { namespace asset
{

class IAssetManager;

class IAsset : virtual public core::IReferenceCounted
{
public:
    enum E_TYPE : uint64_t
    {
        //! asset::ICPUBuffer
        ET_BUFFER = 1u<<0u,
        //! asset::CImageData - maybe rename to asset::CSubImageData
        ET_SUB_IMAGE = 1u<<1u,
        //! asset::ICPUTexture
        ET_IMAGE = 1u<<2u,
        //! asset::ICPUMeshBuffer
        ET_SUB_MESH = 1u<<3u,
        //! asset::ICPUMesh
        ET_MESH = 1u<<4u,
        //! asset::ICPUSkeleton - to be done by splitting CFinalBoneHierarchy
        ET_SKELETON = 1u<<5u,
        //! asset::ICPUKeyframeAnimation - from CFinalBoneHierarchy
        ET_KEYFRAME_ANIMATION = 1u<<6u,
        //! asset::ICPUShader
        ET_SHADER = 1u<<7u,
        //! asset::ICPUSpecializedShader
        ET_SPECIALIZED_SHADER = 1u<<8,
        //! asset::ICPUMeshDataFormatDesc
        ET_MESH_DATA_DESCRIPTOR = 1u<<8,
        //! reserved, to implement later
        ET_GRAPHICS_PIPELINE = 1u<<9u,
        //! reserved, to implement later
        ET_SCENE = 1u<<10u,
        //! lights, etc.
        ET_IMPLEMENTATION_SPECIFIC_METADATA = 1u<<31u
        //! Reserved special value used for things like terminating lists of this enum
    };
    constexpr static size_t ET_STANDARD_TYPES_COUNT = 12u;

    static uint32_t typeFlagToIndex(E_TYPE _type)
    {
        uint32_t type = (uint32_t)_type;
        uint32_t r = 0u;
        while (type >>= 1u) ++r;
        return r;
    }

    IAsset() : isCached{false}, isDummyObjectForCacheAliasing{false} {}

    virtual size_t conservativeSizeEstimate() const = 0;

private:
    friend class IAssetManager;

    std::string cacheKey;
    bool isCached;

    // could make a move-ctor version too
    inline void setNewCacheKey(const std::string& newKey) { cacheKey = newKey; }
    inline void setNewCacheKey(std::string&& newKey) { cacheKey = std::move(newKey); }
    inline void setCached(bool val) { isCached = val; }
    // (Criss) Why this is here if there's convertToDummyObject already
    //! Utility function to call so IAssetManager can call convertToDummyObject
    inline void IAssetManager_convertToDummyObject() { this->convertToDummyObject(); }

protected:
    bool isDummyObjectForCacheAliasing;
    //! To be implemented by base classes, dummies must retain references to other assets
    //! but cleans up all other resources which are not assets.
    virtual void convertToDummyObject() = 0;
    //! Checks if the object is either not dummy or dummy but in some cache for a purpose
    inline bool isInValidState() { return !isDummyObjectForCacheAliasing || !isCached; }
    //! Pure virtual destructor to ensure no instantiation
    virtual ~IAsset() = 0;

public:
    //! Whether this asset is in a cache and should be removed from cache to destroy
    inline bool isInAResourceCache() const { return isCached; }
    //! Only valid if IAsset:isInAResourceCache() returns true
    std::string getCacheKey() const { return cacheKey; }
    //! To be implemented by derived classes
    virtual E_TYPE getAssetType() const = 0;
    //! 
    inline bool isADummyObjectForCache() const { return isDummyObjectForCacheAliasing; }
};

}}

#endif // __IRR_I_ASSET_H_INCLUDED__
