#ifndef __IRR_I_ASSET_H_INCLUDED__
#define __IRR_I_ASSET_H_INCLUDED__

#include <string>
#include "irr/core/core.h"

namespace irr { namespace asset
{

class IAssetManager;

class IAssetMetadata : public core::IReferenceCounted
{
protected:
    virtual ~IAssetMetadata() = default;

public:
    //! this could actually be reworked to something more usable
    virtual const char* getLoaderName() = 0;
};

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

    IAsset() : isDummyObjectForCacheAliasing{false}, m_metadata{nullptr} {}

    virtual size_t conservativeSizeEstimate() const = 0;

    IAssetMetadata* getMetadata() { return m_metadata.get(); }
    const IAssetMetadata* getMetadata() const { return m_metadata.get(); }

    friend IAssetManager;

private:
    core::smart_refctd_ptr<IAssetMetadata> m_metadata;

    void setMetadata(IAssetMetadata* _metadata) { m_metadata = _metadata; }

protected:
    bool isDummyObjectForCacheAliasing;
    //! To be implemented by base classes, dummies must retain references to other assets
    //! but cleans up all other resources which are not assets.
    virtual void convertToDummyObject() = 0;
    //! Checks if the object is either not dummy or dummy but in some cache for a purpose
    inline bool isInValidState() { return !isDummyObjectForCacheAliasing /* || !isCached TODO*/; }
    //! Pure virtual destructor to ensure no instantiation
    virtual ~IAsset() = 0;

public:
    //! To be implemented by derived classes
    virtual E_TYPE getAssetType() const = 0;
    //! 
    inline bool isADummyObjectForCache() const { return isDummyObjectForCacheAliasing; }
};

class SAssetBundle
{
    using contents_container_t = core::refctd_dynamic_array<core::smart_refctd_ptr<IAsset>>;
public:
    SAssetBundle(std::initializer_list<core::smart_refctd_ptr<IAsset>> _contents = {}) : m_contents(core::make_smart_refctd_ptr<contents_container_t>(_contents))
    {
        auto allSameTypeAndNotNull = [&_contents] {
            if (_contents.size()==0ull)
                return true;
            if (!*_contents.begin())
                return false;
            IAsset::E_TYPE t = (*_contents.begin())->getAssetType();
            for (const auto& ast : _contents)
                if (!ast || ast->getAssetType() != t)
                    return false;
            return true;
        };
        assert(allSameTypeAndNotNull());
    }

    inline IAsset::E_TYPE getAssetType() const { return m_contents->front()->getAssetType(); }

    inline std::pair<core::smart_refctd_ptr<IAsset>*, core::smart_refctd_ptr<IAsset>*> getContents()
    {
        return {m_contents->begin(), m_contents->end()};
    }
    inline std::pair<const core::smart_refctd_ptr<IAsset>*, const core::smart_refctd_ptr<IAsset>*> getContents() const
    {
        return {m_contents->begin(), m_contents->end()};
    }

    //! Whether this asset bundle is in a cache and should be removed from cache to destroy
    inline bool isInAResourceCache() const { return m_isCached; }
    //! Only valid if isInAResourceCache() returns true
    std::string getCacheKey() const { return m_cacheKey; }

    size_t getSize() const { return m_contents->size(); }
    bool isEmpty() const { return getSize()==0ull; }

    bool operator==(const SAssetBundle& _other) const
    {
        return _other.m_contents == m_contents;
    }
    bool operator!=(const SAssetBundle& _other) const
    {
        return !((*this) != _other);
    }

private:
    friend class IAssetManager;

    inline void setNewCacheKey(const std::string& newKey) { m_cacheKey = newKey; }
    inline void setNewCacheKey(std::string&& newKey) { m_cacheKey = std::move(newKey); }
    inline void setCached(bool val) { m_isCached = val; }

    std::string m_cacheKey;
    bool m_isCached = false;
    core::smart_refctd_ptr<contents_container_t> m_contents;
};

}}

#endif // __IRR_I_ASSET_H_INCLUDED__
