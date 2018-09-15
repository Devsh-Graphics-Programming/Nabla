#ifndef __C_IASSET_H_INCLUDED__
#define __C_IASSET_H_INCLUDED__

#include <string>
#include "irr/core/IReferenceCounted.h"

namespace irr { namespace asset
{

    // (Criss) Rename to E_ASSET_TYPE?
enum E_TYPE
{
    //! asset::ICPUBuffer
    ET_BUFFER,
    //! asset::CImageData - maybe rename to asset::CSubImageData
    ET_SUB_IMAGE, 
    //! asset::ICPUTexture
    ET_IMAGE,
    //! asset::ICPUMeshBuffer
    ET_SUB_MESH,
    //! asset::ICPUMesh
    ET_MESH,
    //! asset::ICPUSkeleton - to be done by splitting CFinalBoneHierarchy
    ET_SKELETON, 
    //! asset::ICPUKeyframeAnimation - from CFinalBoneHierarchy
    ET_KEYFRAME_ANIMATION, 
    //! the coming asset::IProtoShader
    ET_SHADER, 
    //! reserved, to implement later
    ET_GRAPHICS_PIPELINE,
    //! reserved, to implement later
    ET_SCENE,
    //! lights, etc.
    ET_IMPLEMENTATION_SPECIFIC_METADATA = 64
};


class IAssetManager;

class IAsset : public core::IReferenceCounted
{
private:
    friend class IAssetManager;
    // (Criss) Why this is here?
    std::string cacheKey;
    // could make a move-ctor version too
    inline void setNewCacheKey(const std::string& newKey) { cacheKey = newKey; }
    bool isCached;
    inline void setNotCached() { isCached = false; }
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

#endif // __C_IASSET_H_INCLUDED__
