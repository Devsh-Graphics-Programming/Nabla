#ifndef __C_GLOBAL_MITSUBA_METADATA_H_INCLUDED__
#define __C_GLOBAL_MITSUBA_METADATA_H_INCLUDED__

#include "../../ext/MitsubaLoader/CElementIntegrator.h"
#include "../../ext/MitsubaLoader/CElementSensor.h"
#include "../../ext/MitsubaLoader/CElementEmitter.h"

namespace irr
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
};






class IMitsubaMetadata : public asset::IAssetMetadata
{
	public:
		IMitsubaMetadata(core::smart_refctd_ptr<CGlobalMitsubaMetadata>&& _gmeta, std::string&& _id="") : globalMetadata(_gmeta), id(_id) {}

		_IRR_STATIC_INLINE_CONSTEXPR const char* LoaderName = "Mistuba";
		const char* getLoaderName() const override {return LoaderName;}


		const core::smart_refctd_ptr<CGlobalMitsubaMetadata> globalMetadata;
		const std::string id;
};

// nested <shapes>
class IMeshMetadata : public IMitsubaMetadata
{
	public:
		using IMitsubaMetadata::IMitsubaMetadata;

		const auto& getInstances() const { return instances; }
	protected:
		core::vector<core::matrix3x4SIMD> instances;
		friend class CMitsubaLoader;
};

// <shape>
class IMeshBufferMetadata : public IMitsubaMetadata
{
};

//! not used yet
class IGraphicsPipelineMetadata : public IMitsubaMetadata
{
	public:
	protected:
};

}
}
}

#endif