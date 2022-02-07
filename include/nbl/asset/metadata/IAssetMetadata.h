// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_I_ASSET_METADATA_H_INCLUDED_
#define _NBL_ASSET_I_ASSET_METADATA_H_INCLUDED_

#include "nbl/core/core.h"

#include "nbl/asset/metadata/IImageMetadata.h"
#include "nbl/asset/metadata/IImageViewMetadata.h"
#include "nbl/asset/metadata/IRenderpassIndependentPipelineMetadata.h"
#include "nbl/asset/metadata/IMeshMetadata.h"

namespace nbl::asset
{
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
    template<class Asset>
    struct asset_metadata;

    template<>
    struct asset_metadata<ICPUImage>
    {
        using type = IImageMetadata;
    };
    template<>
    struct asset_metadata<ICPUImageView>
    {
        using type = IImageViewMetadata;
    };
    template<>
    struct asset_metadata<ICPURenderpassIndependentPipeline>
    {
        using type = IRenderpassIndependentPipelineMetadata;
    };
    template<>
    struct asset_metadata<ICPUMesh>
    {
        using type = IMeshMetadata;
    };

    template<class Asset>
    using asset_metadata_t = typename asset_metadata<Asset>::type;

    template<class Asset>
    using asset_metadata_map_t = core::map<const Asset*, const asset_metadata_t<Asset>*>;

    template<class Meta>
    using meta_container_t = core::smart_refctd_dynamic_array<Meta>;

    template<class Meta>
    static inline meta_container_t<Meta> createContainer(const uint32_t length)
    {
        return core::make_refctd_dynamic_array<meta_container_t<Meta>>(length);
    }

    std::tuple<
        asset_metadata_map_t<ICPUImage>,
        asset_metadata_map_t<ICPUImageView>,
        asset_metadata_map_t<ICPURenderpassIndependentPipeline>,
        asset_metadata_map_t<ICPUMesh>>
        m_metaMaps;

    IAssetMetadata() = default;
    virtual ~IAssetMetadata() = default;

    //!
    template<class Asset>
    inline void insertAssetSpecificMetadata(const Asset* asset, const asset_metadata_t<Asset>* meta)
    {
        std::get<asset_metadata_map_t<Asset>>(m_metaMaps).emplace(asset, meta);
    }

public:
    /*
			To implement by user. Returns a Loader name that may attach some metadata into Asset structure.

			@see IAssetMetadata

			Due to external and custom Asset Loaders static_cast cannot be protected with a type enum comparision, 
			so a string is provided.
		*/
    virtual const char* getLoaderName() const = 0;

    //!
    template<class MetaType>
    inline const MetaType* selfCast() const
    {
        if(strcmp(getLoaderName(), MetaType::LoaderName) != 0)
            return nullptr;
        return static_cast<const MetaType*>(this);
    }
    template<class MetaType>
    inline MetaType* selfCast()
    {
        if(strcmp(getLoaderName(), MetaType::LoaderName) != 0)
            return nullptr;
        return static_cast<MetaType*>(this);
    }

    //!
    template<class Asset>
    inline const asset_metadata_t<Asset>* getAssetSpecificMetadata(const Asset* asset) const
    {
        const auto& map = std::get<asset_metadata_map_t<Asset>>(m_metaMaps);
        auto found = map.find(asset);
        if(found != map.end())
            return found->second;
        return nullptr;
    }
};

}

#endif
