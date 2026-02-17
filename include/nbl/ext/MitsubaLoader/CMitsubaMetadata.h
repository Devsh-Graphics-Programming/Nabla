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
		class IGeometry : public CID
		{
			public:
				inline IGeometry() : CID(), type(CElementShape::Type::INVALID) {}
				inline ~IGeometry() = default;

				CElementShape::Type type;
		};
		class CPolygonGeometry final : public asset::IPolygonGeometryMetadata, public IGeometry
		{
			public:
				inline CPolygonGeometry() : asset::IPolygonGeometryMetadata(), IGeometry() {}
				inline CPolygonGeometry(CPolygonGeometry&& other) : CPolygonGeometry() {operator=(std::move(other));}
				inline ~CPolygonGeometry() = default;

				inline CPolygonGeometry& operator=(CPolygonGeometry&& other)
				{
					asset::IPolygonGeometryMetadata::operator=(std::move(other));
					IGeometry::operator=(std::move(other));
					return *this;
				}
		};
		class CGeometryCollection final : public asset::IGeometryCollectionMetadata, public CID
		{
			public:
				inline CGeometryCollection() : asset::IGeometryCollectionMetadata(), CID() {}
				inline ~CGeometryCollection() = default;
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
        inline const CPolygonGeometry* getAssetSpecificMetadata(const asset::ICPUPolygonGeometry* asset) const
        {
            const auto found = IAssetMetadata::getAssetSpecificMetadata(asset);
            return static_cast<const CPolygonGeometry*>(found);
        }

	private:
		friend struct SContext;
		struct SGeometryMetaPair
		{
			core::smart_refctd_ptr<asset::ICPUPolygonGeometry> geom;
			CMitsubaMetadata::CPolygonGeometry meta;
		};
		inline void setPolygonGeometryMeta(core::unordered_map<const CElementShape*,SGeometryMetaPair>&& container)
		{
			const uint32_t count = container.size();
			m_metaPolygonGeometryStorage = IAssetMetadata::createContainer<CPolygonGeometry>(count);
			auto outIt = m_metaPolygonGeometryStorage->begin();
			for (auto& el : container)
			{
				*outIt = std::move(el.second.meta);
				IAssetMetadata::insertAssetSpecificMetadata(el.second.geom.get(),outIt++);
			}
		}

		meta_container_t<CPolygonGeometry> m_metaPolygonGeometryStorage;
};

}
#endif
