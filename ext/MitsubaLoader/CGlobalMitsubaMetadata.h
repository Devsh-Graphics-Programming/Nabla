#ifndef __C_GLOBAL_MITSUBA_METADATA_H_INCLUDED__
#define __C_GLOBAL_MITSUBA_METADATA_H_INCLUDED__

#include "../../ext/MitsubaLoader/CElementIntegrator.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

class CGlobalMitsubaMetadata : public core::IReferenceCounted
{
	public:
		CGlobalMitsubaMetadata()
		{
			integrator.type = CElementIntegrator::Type::INVALID;
		}

		CElementIntegrator integrator;
		//core::vector<CElementSensor> sensor; //sensors?
		//core::vector<Emitter> emitters;
};






class IMitsubaMetadata : public asset::IAssetMetadata
{
	public:
		IMitsubaMetadata(core::smart_refctd_ptr<CGlobalMitsubaMetadata>&& _gmeta) : globalMetadata(_gmeta) {}

		const char* getLoaderName() const override {return "Mistuba";}

		const std::string id;
		const core::smart_refctd_ptr<CGlobalMitsubaMetadata> globalMetadata;
};

// nested <shapes>
class IMeshMetadata : public IMitsubaMetadata
{
	public:
		const auto& getInstances() const { return instances; }
	protected:
		core::vector<core::matrix4SIMD> instances;
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