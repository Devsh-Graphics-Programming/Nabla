// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_C_MITSUBA_METADATA_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_C_MITSUBA_METADATA_H_INCLUDED_


#include "nbl/asset/metadata/IAssetMetadata.h"
#include "nbl/asset/ICPUImage.h"

//#include "nbl/ext/MitsubaLoader/SContext.h"
//#include "nbl/ext/MitsubaLoader/CElementEmitter.h"
#include "nbl/ext/MitsubaLoader/CElementIntegrator.h"
#include "nbl/ext/MitsubaLoader/CElementSensor.h"
//#include "nbl/ext/MitsubaLoader/CElementShape.h"


namespace nbl::ext::MitsubaLoader
{

//! A class to derive mitsuba mesh loader metadata objects from
class CMitsubaMetadata : public asset::IAssetMetadata
{
	public:
		class CID
		{
			public:
				std::string m_id;
		};
#if 0
		class CMesh : public asset::IMeshMetadata, public CID
		{
			public:
				CMesh() : IMeshMetadata(), CID(), type(CElementShape::Type::INVALID) {}
				~CMesh() {}

				CElementShape::Type type;
		};
#endif
		struct SGlobal
		{
			public:
				inline SGlobal() : m_integrator("invalid") {}// TODO

				CElementIntegrator m_integrator;
				core::vector<CElementSensor> m_sensors;
		} m_global;

		inline CMitsubaMetadata() :	IAssetMetadata()/*, m_metaMeshStorage(), m_metaMeshInstanceStorage(), m_metaMeshInstanceAuxStorage(),
			m_meshStorageIt(nullptr), m_instanceStorageIt(nullptr), m_instanceAuxStorageIt(nullptr)*/
		{
		}

		constexpr static inline const char* LoaderName = "ext::MitsubaLoader::CMitsubaLoader";
		const char* getLoaderName() const override {return LoaderName;}
#if 0
        //!
        inline const CMesh* getAssetSpecificMetadata(const asset::ICPUMesh* asset) const
        {
            const auto found = IAssetMetadata::getAssetSpecificMetadata(asset);
            return static_cast<const CMesh*>(found);
        }
#endif
	private:
//		friend class CMitsubaLoader;
#if 0
		meta_container_t<CMesh> m_metaMeshStorage;
		CMesh* m_meshStorageIt;

		inline void reserveMeshStorage(uint32_t meshCount, uint32_t instanceCount)
		{
			m_metaMeshStorage = IAssetMetadata::createContainer<CMesh>(meshCount);
			m_metaMeshInstanceStorage = IAssetMetadata::createContainer<CMesh::SInstance>(instanceCount);
			m_metaMeshInstanceAuxStorage = IAssetMetadata::createContainer<CMesh::SInstanceAuxilaryData>(instanceCount);
			m_meshStorageIt = m_metaMeshStorage->begin();
			m_instanceStorageIt = m_metaMeshInstanceStorage->begin();
			m_instanceAuxStorageIt = m_metaMeshInstanceAuxStorage->begin();
		}
		template<typename InstanceIterator>
		inline uint32_t addMeshMeta(const asset::ICPUMesh* mesh, std::string&& id, const CElementShape::Type type, InstanceIterator instancesBegin, InstanceIterator instancesEnd)
		{
			auto instanceStorageBegin = m_instanceStorageIt;
			auto instanceAuxStorageBegin = m_instanceAuxStorageIt;

			auto* meta = m_meshStorageIt++;
			meta->m_id = std::move(id);
			{
				// copy instance data
				for (auto it=instancesBegin; it!=instancesEnd; ++it)
				{
					auto& inst = it->second;
					(m_instanceStorageIt++)->worldTform = inst.tform;
					*(m_instanceAuxStorageIt++) = {
						inst.emitter.front,
						inst.emitter.back,
						inst.bsdf
					};
				}
				meta->m_instances = { instanceStorageBegin,m_instanceStorageIt };
				meta->m_instanceAuxData = { instanceAuxStorageBegin,m_instanceAuxStorageIt };
			}
			meta->type = type;
			IAssetMetadata::insertAssetSpecificMetadata(mesh,meta);

			return meta->m_instances.size();
		}
#endif
};

}
#endif
