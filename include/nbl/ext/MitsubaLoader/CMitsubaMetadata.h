// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_C_MITSUBA_METADATA_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_C_MITSUBA_METADATA_H_INCLUDED_


#include "nbl/asset/metadata/IAssetMetadata.h"
#include "nbl/asset/ICPUImage.h"

#include "nbl/ext/MitsubaLoader/CElementIntegrator.h"
#include "nbl/ext/MitsubaLoader/CElementSensor.h"
#include "nbl/ext/MitsubaLoader/CElementShape.h"


namespace nbl::ext::MitsubaLoader
{

//! A class to derive mitsuba scene loader metadata objects from
class CMitsubaMetadata : public asset::IAssetMetadata
{
	public:
		class CID
		{
			public:
				std::string m_id;
		};
		class CGeometryCollection final : public asset::IGeometryCollectionMetadata, public CID
		{
			public:
				inline CGeometryCollection() : asset::IGeometryCollectionMetadata(), CID(), type(CElementShape::Type::INVALID) {}
				inline CGeometryCollection(CGeometryCollection&& other) : CGeometryCollection() {operator=(std::move(other));}
				inline ~CGeometryCollection() = default;

				inline CGeometryCollection& operator=(CGeometryCollection&& other)
				{
					asset::IGeometryCollectionMetadata::operator=(std::move(other));
					CID::operator=(std::move(other));
					return *this;
				}

				CElementShape::Type type;
		};

		struct SGlobal
		{
			public:
				inline SGlobal() : m_integrator("invalid") {}

				CElementIntegrator m_integrator;
				core::vector<CElementSensor> m_sensors;
		} m_global;

		inline CMitsubaMetadata() :	IAssetMetadata(), m_metaPolygonGeometryStorage() {}

		constexpr static inline const char* LoaderName = "ext::MitsubaLoader::CMitsubaLoader";
		const char* getLoaderName() const override {return LoaderName;}

        // add more overloads when more asset implementations of IGeometry<ICPUBuffer> exist
        inline const CGeometryCollection* getAssetSpecificMetadata(const asset::ICPUGeometryCollection* asset) const
        {
            const auto found = IAssetMetadata::getAssetSpecificMetadata(asset);
            return static_cast<const CGeometryCollection*>(found);
        }

	private:
		friend struct SContext;
		struct SGeometryCollectionMetaPair
		{
			core::smart_refctd_ptr<asset::ICPUGeometryCollection> collection;
			CMitsubaMetadata::CGeometryCollection meta;
		};
		template<typename Key>
		inline void setGeometryCollectionMeta(core::unordered_map<Key,SGeometryCollectionMetaPair>&& container)
		{
			const uint32_t count = container.size();
			m_metaPolygonGeometryStorage = IAssetMetadata::createContainer<CGeometryCollection>(count);
			auto outIt = m_metaPolygonGeometryStorage->begin();
			for (auto& el : container)
			{
				*outIt = std::move(el.second.meta);
				IAssetMetadata::insertAssetSpecificMetadata(el.second.collection.get(),outIt++);
			}
		}

		meta_container_t<CGeometryCollection> m_metaPolygonGeometryStorage;
};

}
#endif
