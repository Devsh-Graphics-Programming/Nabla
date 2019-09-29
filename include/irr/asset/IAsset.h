#ifndef __IRR_I_ASSET_H_INCLUDED__
#define __IRR_I_ASSET_H_INCLUDED__

#include <string>
#include "irr/core/core.h"

namespace irr
{
namespace asset
{

class IAssetManager;

//! A class managing Asset's metadata context
/**
	Sometimes there may be nedeed attaching some metadata by a Loader
	into Asset structure - that's why the class is defined.

	Pay attention that it hasn't been done exactly yet, engine doesn't provide
	metadata injecting.

	Metadata are extra data retrieved by the loader, which aren't ubiquitously representable by the engine.
	These could be for instance global data about the file or scene, IDs, names, default view/projection,
	complex animations and hierarchies, physics simulation data, AI data, lighting or extra material metadata.

	Flexibility has been provided, it is expected each loader has its own base metadata class implementing the 
	IAssetMetadata interface with its own type enum that other loader's metadata classes derive from the base.
*/
class IAssetMetadata : public core::IReferenceCounted
{
protected:
    virtual ~IAssetMetadata() = default;

public:
    //! This could actually be reworked to something more usable
    virtual const char* getLoaderName() = 0;
};

//! An abstract data type class representing an interface of any kind of cpu-objects with capability of being cached
/**
	An Asset is a CPU object which needs to be loaded for caching it to RAM memory,
	or needs to be filled with Asset-data if it has to be created by hand,
	and then may be converted to GPU object, but it isn't necessary in each case! For instance
	ICPUBuffer which is an Asset opens a nice way to represent a plain byte array, but doesn't need
	to be converted into GPU object!

	@see ICPUBuffer

	Actually an Asset is a class deriving from it that can be anything like cpu-side meshes scenes, texture data and material pipelines, 
	but must be serializable into/from .baw file format, unless it comes from an extension (irr::ext), 
	so forth cached. There are different asset types you can find at IAsset::E_TYPE. 
	IAsset doesn't provide direct instantiation (virtual destructor), much like IReferenceCounted.

	Asset type's naming-convention is ICPU_x, where _x is a name of an Asset, eg. ICPUBuffer, ICPUMesh e.t.c

	Every asset must be loaded by a particular class derived from IAssetLoader.
	
	@see IAssetLoader
	@see IAssetManager
	@see IAssetWriter
	@see IReferenceCounted
*/

class IAsset : virtual public core::IReferenceCounted
{
public:

	/**
		Values of E_TYPE represents an Asset type.
		
		Types are provided, so known is that the type you're casting to is right, eg.
		If there is an Asset represeting ICPUBuffer, after calling a function returning
		a type, it should return you a type associated with it, so
		
		\code{.cpp}
		IAsset* asset;
		if(asset->getType() == ET_BUFFER)
			static_cast<ICPUBuffer*>(asset) // it is safe
		\endcode

		@see IAsset

		ET_BUFFER represents asset::ICPUBuffer
		ET_SUB_IMAGE represents asset::CImageData
		ET_IMAGE represents ICPUTexture
		ET_SUB_MESH represents
		ET_MESH represents
		ET_SKELETON represents
		ET_KEYFRAME_ANIMATION represents
		ET_SHADER represents
		ET_SPECIALIZED_SHADER represents
		ET_MESH_DATA_DESCRIPTOR represents
		ET_GRAPHICS_PIPELINE represents
		ET_SCENE represents
		ET_IMPLEMENTATION_SPECIFIC_METADATA represents a special value used for things like terminating lists of this enum

		Pay attention that an Asset type represents one single bit, so there is a limit to 64 bits.

	*/
	
    enum E_TYPE : uint64_t
    {
        ET_BUFFER = 1u<<0u,								//!< asset::ICPUBuffer
        ET_SUB_IMAGE = 1u<<1u,						    //!< asset::CImageData - maybe rename to asset::CSubImageData
        ET_IMAGE = 1u<<2u,								//!< asset::ICPUTexture
        ET_SUB_MESH = 1u<<3u,							//!< asset::ICPUMeshBuffer
        ET_MESH = 1u<<4u,								//!< asset::ICPUMesh
        ET_SKELETON = 1u<<5u,							//!< asset::ICPUSkeleton - to be done by splitting CFinalBoneHierarchy
        ET_KEYFRAME_ANIMATION = 1u<<6u,					//!< asset::ICPUKeyframeAnimation - from CFinalBoneHierarchy
        ET_SHADER = 1u<<7u,								//!< asset::ICPUShader
        ET_SPECIALIZED_SHADER = 1u<<8,					//!< asset::ICPUSpecializedShader
        ET_MESH_DATA_DESCRIPTOR = 1u<<8,				//!< asset::ICPUMeshDataFormatDesc
        ET_GRAPHICS_PIPELINE = 1u<<9u,					//!< reserved, to implement later
        ET_SCENE = 1u<<10u,								//!< reserved, to implement later
        ET_IMPLEMENTATION_SPECIFIC_METADATA = 1u<<31u	//!< lights, etc.
        //! Reserved special value used for things like terminating lists of this enum
    };
    constexpr static size_t ET_STANDARD_TYPES_COUNT = 12u; //!< The variable shows valid amount of available Asset types in E_TYPE enum

	//! Returns a representaion of an Asset type in decimal system
	/**
		Each value is returned from the range 0 to (ET_STANDARD_TYPES_COUNT - 1) to provide
		easy way to handle arrays operations. For instance ET_SUB_IMAGE returns 1 and ET_MESH
		returns 4.
	*/
    static uint32_t typeFlagToIndex(E_TYPE _type)
    {
        uint32_t type = (uint32_t)_type;
        uint32_t r = 0u;
        while (type >>= 1u) ++r;
        return r;
    }

    IAsset() : isDummyObjectForCacheAliasing{false}, m_metadata{nullptr} {}

	//! Returns whole size associated with an Asset and its data
	/**
		The size is used to determine compression level while writing process 
		is performed. As you expect, the more size Asset has, the more compression
		level is.
	*/
    virtual size_t conservativeSizeEstimate() const = 0;

	//! Returns Asset's metadata. @see IAssetMetadata
    IAssetMetadata* getMetadata() { return m_metadata.get(); }

	//! Returns Asset's metadata. @see IAssetMetadata
    const IAssetMetadata* getMetadata() const { return m_metadata.get(); }

    friend IAssetManager;

private:
    core::smart_refctd_ptr<IAssetMetadata> m_metadata;

    void setMetadata(core::smart_refctd_ptr<IAssetMetadata>&& _metadata) { m_metadata = std::move(_metadata); }

protected:
    bool isDummyObjectForCacheAliasing; //!< A bool for setting whether Asset is in dummy state. @see convertToDummyObject()

    //! To be implemented by base classes, dummies must retain references to other assets
    //! but cleans up all other resources which are not assets.
	/**
		Dummy object is an object which is converted to GPU object or which is about to be converted to GPU object.
		Take into account that\b convertToDummyObject() itself doesn't perform exactly converting to GPU object\b. 

		@see IAssetManager::convertAssetToEmptyCacheHandle(IAsset* _asset, core::smart_refctd_ptr<core::IReferenceCounted>&& _gpuObject)

		If an Asset is being converted to a GPU object, its resources are no longer needed in RAM memory,
		so everything it has allocated becomes deleted, but the Asset itself remains untouched, so that is the 
		pointer for an Asset and the memory allocated for that pointer. It's because it's needed as a key by some 
		functions that find GPU objects. It involves all CPU objects (Assets).

		So an Asset signed as dummy becomes GPU object and deletes some resources in RAM memory.
	*/
    virtual void convertToDummyObject() = 0;

    //! Checks if the object is either not dummy or dummy but in some cache for a purpose
    inline bool isInValidState() { return !isDummyObjectForCacheAliasing /* || !isCached TODO*/; }

    //! Pure virtual destructor to ensure no instantiation
    virtual ~IAsset() = 0;

public:
    //! To be implemented by derived classes. Returns a type of an Asset
    virtual E_TYPE getAssetType() const = 0;

    //! Returning isDummyObjectForCacheAliasing, specifies whether Asset in dummy state
    inline bool isADummyObjectForCache() const { return isDummyObjectForCacheAliasing; }
};

//! A class storing Assets with the same type
class SAssetBundle
{
	inline bool allSameTypeAndNotNull()
	{
		if (m_contents->size() == 0ull)
			return true;
		if (!*m_contents->begin())
			return false;
		IAsset::E_TYPE t = (*m_contents->begin())->getAssetType();
		for (auto it=m_contents->cbegin(); it!=m_contents->cend(); it++)
			if (!(*it) || (*it)->getAssetType()!=t)
				return false;
		return true;
	}
public:
    using contents_container_t = core::refctd_dynamic_array<core::smart_refctd_ptr<IAsset> >;
    
	SAssetBundle() : m_contents(contents_container_t::create_dynamic_array(0u), core::dont_grab) {}
	SAssetBundle(std::initializer_list<core::smart_refctd_ptr<IAsset> > _contents) : m_contents(contents_container_t::create_dynamic_array(_contents),core::dont_grab)
	{
		assert(allSameTypeAndNotNull());
	}
	template<typename container_t, typename iterator_t = typename container_t::iterator>
	SAssetBundle(const container_t& _contents) : m_contents(contents_container_t::create_dynamic_array(_contents), core::dont_grab)
	{
		assert(allSameTypeAndNotNull());
	}
	template<typename container_t, typename iterator_t = typename container_t::iterator>
    SAssetBundle(container_t&& _contents) : m_contents(contents_container_t::create_dynamic_array(std::move(_contents)), core::dont_grab)
    {
        assert(allSameTypeAndNotNull());
    }

	//! Returning a type associated with current stored Assets
	/**
		An Asset type is specified in E_TYPE enum.
		@see E_TYPE
	*/
    inline IAsset::E_TYPE getAssetType() const { return m_contents->front()->getAssetType(); }

	//! Getting beginning and end of an Asset stored by m_contents
    inline std::pair<const core::smart_refctd_ptr<IAsset>*, const core::smart_refctd_ptr<IAsset>*> getContents() const
    {
        return {m_contents->begin(), m_contents->end()};
    }

    //! Whether this asset bundle is in a cache and should be removed from cache to destroy
    inline bool isInAResourceCache() const { return m_isCached; }

    //! Only valid if isInAResourceCache() returns true
    std::string getCacheKey() const { return m_cacheKey; }

	//! Getting size of a collection of Assets stored by m_contents
    size_t getSize() const { return m_contents->size(); }

	//! Checking if a collection of Assets stored by m_contents is empty
    bool isEmpty() const { return getSize()==0ull; }

	//! Overloaded operator checking if both collections of Assets\b are\b the same arrays in memory
    bool operator==(const SAssetBundle& _other) const
    {
        return _other.m_contents == m_contents;
    }

	//! Overloaded operator checking if both collections of Assets\b aren't\b the same arrays in memory
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

}
}

#endif // __IRR_I_ASSET_H_INCLUDED__
