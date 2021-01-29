// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_C_MITSUBA_METADATA_H_INCLUDED__
#define __NBL_C_MITSUBA_METADATA_H_INCLUDED__

#include "nbl/asset/metadata/IAssetMetadata.h"


#include "nbl/ext/MitsubaLoader/CGlobalMitsubaMetadata.h"

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
				const std::string id;
		};
		class CRenderpassIndependentPipeline : public asset::IRenderpassIndependentPipelineMetadata
		{
			public:
				using IRenderpassIndependentPipelineMetadata::IRenderpassIndependentPipelineMetadata;

				template<typename... Args>
				CRenderpassIndependentPipeline(uint32_t _hash, Args&&... args) : IRenderpassIndependentPipelineMetadata(std::forward<Args>(args)...), m_hash(_hash)
				{
				}

				inline CRenderpassIndependentPipeline& operator=(CRenderpassIndependentPipeline&& other)
				{
					IRenderpassIndependentPipelineMetadata::operator=(std::move(other));
					std::swap(m_hash, other.m_hash);
					return *this;
				}

				uint32_t m_hash;
		};
		class CMesh : public IMeshMetadata, public CID
		{
			public:
				struct InstanceAuxilaryData
				{
					CElementEmitter emitter; // type is invalid if not used
					SContext::bsdf_type bsdf;
				};
				CElementShape::Type type;
		};
		struct SGlobal
		{
			public:
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

		//! No idea how to make it work yet
		//CMitsubaMetadata(core::smart_refctd_ptr<CGlobalMitsubaMetadata> _mitsubaMetadata) : mitsubaMetadata(std::move(_mitsubaMetadata)) {}

		_NBL_STATIC_INLINE_CONSTEXPR const char* loaderName = "ext::MitsubaLoader::CMitsubaLoader";
		const char* getLoaderName() const override { return loaderName; }

	private:
		meta_container_t<CRenderpassIndependentPipeline> m_metaPplnStorage;
		meta_container_t<CMesh> m_metaMeshStorage;
};

}
}
}

#endif
