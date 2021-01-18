// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __C_GLOBAL_MITSUBA_METADATA_H_INCLUDED__
#define __C_GLOBAL_MITSUBA_METADATA_H_INCLUDED__

#include "nbl/ext/MitsubaLoader/CElementIntegrator.h"
#include "nbl/ext/MitsubaLoader/CElementSensor.h"
#include "nbl/ext/MitsubaLoader/CElementShape.h"
#include "nbl/ext/MitsubaLoader/SContext.h"
#include <nbl/asset/ICPUVirtualTexture.h>

namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{

class CGlobalMitsubaMetadata : public core::IReferenceCounted
{
	public:
		CGlobalMitsubaMetadata() : integrator("")
		{
			integrator.type = CElementIntegrator::Type::INVALID;
		}

		CElementIntegrator integrator;
		core::vector<CElementSensor> sensors;
		core::vector<CElementEmitter> emitters;
		core::smart_refctd_ptr<asset::ICPUVirtualTexture> VT;

		//has to go after #version and before required user-provided descriptors and functions
		std::string materialCompilerGLSL_declarations;
		//has to go after required user-provided descriptors and functions and before the rest of shader (especially entry point function)
		std::string materialCompilerGLSL_source;

		uint32_t getVTStorageViewCount() const { return VT->getFloatViews().size(); }
};



//! TODO: move these to separate files


class IMitsubaMetadata : public asset::IAssetMetadata
{
	public:
		IMitsubaMetadata(core::smart_refctd_ptr<CGlobalMitsubaMetadata>&& _gmeta, std::string&& _id="") : globalMetadata(_gmeta), id(_id) {}

		_NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "Mistuba XML";
		const char* getLoaderName() const override {return LoaderName;}


		const core::smart_refctd_ptr<CGlobalMitsubaMetadata> globalMetadata;
		const std::string id;
};

// nested <shapes>
class IMeshMetadata : public IMitsubaMetadata
{
	public:
		IMeshMetadata(core::smart_refctd_ptr<CGlobalMitsubaMetadata>&& _gmeta, std::string&& _id, CElementShape* shape) :
			IMitsubaMetadata(std::move(_gmeta),std::move(_id)), type(shape->type)
		{}

		inline auto getShapeType() const {return type;}

		using Instance = SContext::SInstanceData;

		inline const auto& getInstances() const { return instances; }

	protected:
		CElementShape::Type type;
		core::vector<Instance> instances;

		friend class CMitsubaLoader;
};

// <shape>
class IMeshBufferMetadata : public IMitsubaMetadata
{
};

}
}
}

#endif