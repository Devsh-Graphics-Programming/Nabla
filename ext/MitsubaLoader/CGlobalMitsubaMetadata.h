#ifndef __C_GLOBAL_MITSUBA_METADATA_H_INCLUDED__
#define __C_GLOBAL_MITSUBA_METADATA_H_INCLUDED__

#include "../../ext/MitsubaLoader/CElementIntegrator.h"
#include "../../ext/MitsubaLoader/CElementSensor.h"
#include "../../ext/MitsubaLoader/CElementShape.h"

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



//! TODO: move these to separate files


class IMitsubaMetadata : public asset::IAssetMetadata
{
	public:
		IMitsubaMetadata(core::smart_refctd_ptr<CGlobalMitsubaMetadata>&& _gmeta, std::string&& _id="") : globalMetadata(_gmeta), id(_id) {}

		_IRR_STATIC_INLINE_CONSTEXPR const char* LoaderName = "Mistuba XML";
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

		struct Instance
		{
			core::matrix3x4SIMD tform;
			CElementEmitter emitter; // type is invalid if not used
		};

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