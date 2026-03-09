// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_PLY_METADATA_H_INCLUDED_
#define _NBL_ASSET_C_PLY_METADATA_H_INCLUDED_


#include "nbl/asset/metadata/IAssetMetadata.h"
#include <string>
#include <string_view>


namespace nbl::asset
{

class CPLYMetadata final : public IAssetMetadata
{
    public:
		class CPolygonGeometry : public IPolygonGeometryMetadata
		{
			public:
				using IPolygonGeometryMetadata::IPolygonGeometryMetadata;
				inline CPolygonGeometry& operator=(CPolygonGeometry&& other)
				{
					IPolygonGeometryMetadata::operator=(std::move(other));
					std::swap(m_auxAttributeNames, other.m_auxAttributeNames);
					return *this;
				}
				inline std::string_view getAuxAttributeName(const uint32_t auxViewIx) const
				{
					return auxViewIx < m_auxAttributeNames.size() ? std::string_view(m_auxAttributeNames[auxViewIx]) : std::string_view{};
				}
				core::vector<std::string> m_auxAttributeNames;
		};
        CPLYMetadata(const uint32_t geometryCount = 0u) : IAssetMetadata(), m_geometryMetaStorage(createContainer<CPolygonGeometry>(geometryCount)) {}

        _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CPLYMeshFileLoader";
        const char* getLoaderName() const override { return LoaderName; }
	private:
		meta_container_t<CPolygonGeometry> m_geometryMetaStorage;
		friend class CPLYMeshFileLoader;
		inline void placeMeta(const uint32_t offset, const ICPUPolygonGeometry* geometry, core::vector<std::string>&& auxAttributeNames)
		{
			auto& meta = m_geometryMetaStorage->operator[](offset);
			meta = CPolygonGeometry{};
			meta.m_auxAttributeNames = std::move(auxAttributeNames);
			IAssetMetadata::insertAssetSpecificMetadata(geometry, &meta);
		}
};

}
#endif
