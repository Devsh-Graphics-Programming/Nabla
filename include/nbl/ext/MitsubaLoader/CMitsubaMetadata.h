// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_C_MITSUBA_METADATA_H_INCLUDED__
#define __NBL_C_MITSUBA_METADATA_H_INCLUDED__

#include "nbl/core/compile_config.h"
#include "nbl/asset/metadata/IAssetMetadata.h"

#include "nbl/ext/MitsubaLoader/SContext.h"
#include "nbl/ext/MitsubaLoader/CElementEmitter.h"
#include "nbl/ext/MitsubaLoader/CElementIntegrator.h"
#include "nbl/ext/MitsubaLoader/CElementSensor.h"
#include "nbl/ext/MitsubaLoader/CElementShape.h"

namespace nbl
{
namespace ext
{
namespace MitsubaLoader
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
		class CDerivativeMap : public asset::IImageMetadata
		{
			public:
				CDerivativeMap() : m_scale(1.f) {}
				explicit CDerivativeMap(float scale) : m_scale(scale) {}

				float m_scale;
		};
		class CRenderpassIndependentPipeline : public asset::IRenderpassIndependentPipelineMetadata
		{
			public:
				CRenderpassIndependentPipeline() : IRenderpassIndependentPipelineMetadata(), m_ds0() {}
				template<typename... Args>
				CRenderpassIndependentPipeline(core::smart_refctd_ptr<asset::ICPUDescriptorSet>&& _ds0, Args&&... args) : IRenderpassIndependentPipelineMetadata(std::forward<Args>(args)...), m_ds0(std::move(_ds0))
				{
				}

				inline CRenderpassIndependentPipeline& operator=(CRenderpassIndependentPipeline&& other)
				{
					IRenderpassIndependentPipelineMetadata::operator=(std::move(other));
					std::swap(m_ds0, other.m_ds0);
					return *this;
				}

				core::smart_refctd_ptr<asset::ICPUDescriptorSet> m_ds0;
		};
		class CMesh : public asset::IMeshMetadata, public CID
		{
			public:
				CMesh() : IMeshMetadata(), CID(), m_instanceAuxData(nullptr,nullptr), type(CElementShape::Type::INVALID) {}
				~CMesh() {}

				struct SInstanceAuxilaryData
				{
					SInstanceAuxilaryData& operator=(SInstanceAuxilaryData&& other)
					{
						frontEmitter = std::move(other.frontEmitter);
						backEmitter = std::move(other.backEmitter);
						bsdf = std::move(other.bsdf);
						return *this;
					}

					CElementEmitter frontEmitter; // type is invalid if not used
					CElementEmitter backEmitter; // type is invalid if not used
					CMitsubaMaterialCompilerFrontend::front_and_back_t bsdf;
#if defined(_NBL_DEBUG) || defined(_NBL_RELWITHDEBINFO)
					std::string bsdf_id;
#endif
				};

				core::SRange<const SInstanceAuxilaryData> m_instanceAuxData;

				CElementShape::Type type;
		};
		struct SGlobal
		{
			public:
				SGlobal() : m_integrator("invalid") {}// TODO

				inline uint32_t getVTStorageViewCount() const { return m_VT->getFloatViews().size(); }

				CElementIntegrator m_integrator;
				core::vector<CElementSensor> m_sensors;
				core::vector<CElementEmitter> m_emitters;
				core::smart_refctd_ptr<asset::ICPUVirtualTexture> m_VT;
				core::smart_refctd_ptr<asset::ICPUDescriptorSet> m_ds0;
				//has to go after #version and before required user-provided descriptors and functions
				std::string m_materialCompilerGLSL_declarations;
				//has to go after required user-provided descriptors and functions and before the rest of shader (especially entry point function)
				std::string m_materialCompilerGLSL_source;
		} m_global;

		CMitsubaMetadata() :
			IAssetMetadata(), m_metaPplnStorage(), m_semanticStorage(), m_metaPplnStorageIt(nullptr),
			m_metaMeshStorage(), m_metaMeshInstanceStorage(), m_metaMeshInstanceAuxStorage(),
			m_meshStorageIt(nullptr), m_instanceStorageIt(nullptr), m_instanceAuxStorageIt(nullptr)
		{
		}

		_NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "ext::MitsubaLoader::CMitsubaLoader";
		const char* getLoaderName() const override { return LoaderName; }

        //!
        inline const CRenderpassIndependentPipeline* getAssetSpecificMetadata(const asset::ICPURenderpassIndependentPipeline* asset) const
        {
            const auto found = IAssetMetadata::getAssetSpecificMetadata(asset);
            return static_cast<const CRenderpassIndependentPipeline*>(found);
        }
        inline const CMesh* getAssetSpecificMetadata(const asset::ICPUMesh* asset) const
        {
            const auto found = IAssetMetadata::getAssetSpecificMetadata(asset);
            return static_cast<const CMesh*>(found);
        }

	private:
		friend class CMitsubaLoader;

		meta_container_t<CRenderpassIndependentPipeline> m_metaPplnStorage;
		core::smart_refctd_dynamic_array<asset::IRenderpassIndependentPipelineMetadata::ShaderInputSemantic> m_semanticStorage;
		CRenderpassIndependentPipeline* m_metaPplnStorageIt;

		meta_container_t<CMesh> m_metaMeshStorage;
		core::smart_refctd_dynamic_array<CMesh::SInstance> m_metaMeshInstanceStorage;
		core::smart_refctd_dynamic_array<CMesh::SInstanceAuxilaryData> m_metaMeshInstanceAuxStorage;
		CMesh* m_meshStorageIt;
		CMesh::SInstance* m_instanceStorageIt;
		CMesh::SInstanceAuxilaryData* m_instanceAuxStorageIt;

		meta_container_t<CDerivativeMap> m_metaDerivMapStorage;
		CDerivativeMap* m_metaDerivMapStorageIt;

		inline void reservePplnStorage(uint32_t pplnCount, core::smart_refctd_dynamic_array<asset::IRenderpassIndependentPipelineMetadata::ShaderInputSemantic>&& _semanticStorage)
		{
			m_metaPplnStorage = IAssetMetadata::createContainer<CRenderpassIndependentPipeline>(pplnCount);
			m_semanticStorage = std::move(_semanticStorage);
			m_metaPplnStorageIt = m_metaPplnStorage->begin();
		}
		inline void reserveMeshStorage(uint32_t meshCount, uint32_t instanceCount)
		{
			m_metaMeshStorage = IAssetMetadata::createContainer<CMesh>(meshCount);
			m_metaMeshInstanceStorage = IAssetMetadata::createContainer<CMesh::SInstance>(instanceCount);
			m_metaMeshInstanceAuxStorage = IAssetMetadata::createContainer<CMesh::SInstanceAuxilaryData>(instanceCount);
			m_meshStorageIt = m_metaMeshStorage->begin();
			m_instanceStorageIt = m_metaMeshInstanceStorage->begin();
			m_instanceAuxStorageIt = m_metaMeshInstanceAuxStorage->begin();
		}
		inline void reserveDerivMapStorage(uint32_t count)
		{
			m_metaDerivMapStorage = IAssetMetadata::createContainer<CDerivativeMap>(count);
			m_metaDerivMapStorageIt = m_metaDerivMapStorage->begin();
		}
		inline void addPplnMeta(const asset::ICPURenderpassIndependentPipeline* ppln, core::smart_refctd_ptr<asset::ICPUDescriptorSet>&& _ds0)
		{
			*m_metaPplnStorageIt = CMitsubaMetadata::CRenderpassIndependentPipeline(std::move(_ds0),core::SRange<const asset::IRenderpassIndependentPipelineMetadata::ShaderInputSemantic>(m_semanticStorage->begin(),m_semanticStorage->end()));
			IAssetMetadata::insertAssetSpecificMetadata(ppln,m_metaPplnStorageIt);
			m_metaPplnStorageIt++;
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
#if defined(_NBL_DEBUG) || defined(_NBL_RELWITHDEBINFO)
						,inst.bsdf_id
#endif
					};
				}
				meta->m_instances = { instanceStorageBegin,m_instanceStorageIt };
				meta->m_instanceAuxData = { instanceAuxStorageBegin,m_instanceAuxStorageIt };
			}
			meta->type = type;
			IAssetMetadata::insertAssetSpecificMetadata(mesh,meta);

			return meta->m_instances.size();
		}
		inline void addDerivMapMeta(const asset::ICPUImage* derivmap, float scale)
		{
			auto* meta = m_metaDerivMapStorageIt++;
			meta->m_scale = scale;
			IAssetMetadata::insertAssetSpecificMetadata(derivmap, meta);
		}
};

}
}
}

#endif
