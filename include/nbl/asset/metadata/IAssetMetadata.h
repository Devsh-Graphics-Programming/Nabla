// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_ASSET_METADATA_H_INCLUDED__
#define __NBL_ASSET_I_ASSET_METADATA_H_INCLUDED__

#include "nbl/core/core.h"

#include "nbl/asset/IImageMetadata.h"
#include "nbl/asset/IRenderpassIndependentPipelineMetadata.h"
#include "nbl/asset/IMeshMetadata.h"

namespace nbl
{
namespace asset
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
		struct asset_metadata<ICPURenderpassIndependentPipeline>
		{
			using type = IRenderpassIndependentPipelineMetadata;
		};

		template<class Asset>
		using asset_metadata_t = typename asset_metadata<Asset>::type;

		template<class Asset>
		using asset_metadata_map_t = core::map<const Asset*,const asset_metadata_t<Asset>*>;


		std::tuple<
			asset_metadata_map_t<ICPUImage>,
			asset_metadata_map_t<ICPURenderpassIndependentPipeline>,
			asset_metadata_map_t<ICPUMesh>
		> m_metaMaps;


		virtual ~IAssetMetadata() = default;

	public:
		/*
			To implement by user. Returns a Loader name that may attach some metadata into Asset structure.

			@see IAssetMetadata

			Due to external and custom Asset Loaders static_cast cannot be protected with a type enum comparision, 
			so a string is provided.
		*/
		virtual const char* getLoaderName() const = 0;

		//!
		template<class Asset>
		inline const asset_metadata_t<Asset>* getAssetSpecificMetadata(const Asset* asset)
		{
			const auto& map = std::get<asset_metadata_map_t<Asset>>(m_metaMaps);
			auto found = map.find(asset);
			if (found != map.end())
				return found->second;
			return nullptr;
		}
};


}
}

#endif
