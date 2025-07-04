// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_C_MITSUBA_SERIALIZED_PIPELINE_METADATA_H_INCLUDED_
#define _NBL_C_MITSUBA_SERIALIZED_PIPELINE_METADATA_H_INCLUDED_


#include "nbl/asset/ICPUPolygonGeometry.h"
#include "nbl/asset/metadata/IAssetMetadata.h"


namespace nbl::ext::MitsubaLoader
{

class CMitsubaSerializedMetadata final : public asset::IAssetMetadata
{
    public:
        class CPolygonGeometry : public asset::IPolygonGeometryMetadata
        {
            public:
                inline CPolygonGeometry() : IPolygonGeometryMetadata() {}
                inline CPolygonGeometry(std::string&& _name, const uint32_t _id) : m_name(std::move(_name)), m_id(_id) {}

                std::string m_name;
                uint32_t m_id;
        };

        inline CMitsubaSerializedMetadata(const uint32_t meshBound) : m_polygonGeometryStorage(asset::IAssetMetadata::createContainer<CPolygonGeometry>(meshBound)) {}

        _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "ext::MitsubaLoader::CSerializedLoader";
        const char* getLoaderName() const override { return LoaderName; }

        //!
        inline const CPolygonGeometry* getAssetSpecificMetadata(const asset::ICPUPolygonGeometry* asset) const
        {
            const auto found = IAssetMetadata::getAssetSpecificMetadata(asset);
            return static_cast<const CPolygonGeometry*>(found);
        }

    private:
        meta_container_t<CPolygonGeometry> m_polygonGeometryStorage;

        friend class CSerializedLoader;
        inline void placeMeta(uint32_t offset, const asset::ICPUPolygonGeometry* geo, CPolygonGeometry&& geoMeta)
        {
            auto& geoMetaRef = m_polygonGeometryStorage->operator[](offset) = std::move(geoMeta);
            IAssetMetadata::insertAssetSpecificMetadata(geo,&geoMetaRef);
        }
};

}
#endif