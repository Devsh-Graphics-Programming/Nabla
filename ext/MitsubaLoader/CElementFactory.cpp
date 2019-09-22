#include "../../ext/MitsubaLoader/CElementFactory.h"

#include "../../ext/MitsubaLoader/ParserUtil.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{


template<>
IElement* CElementFactory::createElement<CElementIntegrator>(const char** _atts, ParserManager* _util)
{
	core::vector<CElementIntegrator>& pool = _util->objects.getPool<CElementIntegrator>();
	pool.emplace_back();
	return &pool.back();
}

const core::unordered_map<std::string, std::pair<CElementFactory::element_creation_func,bool>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> CElementFactory::createElementTable =
{
	{"integrator",{CElementFactory::createElement<CElementIntegrator>,true}}
};
/*
_IRR_STATIC_INLINE_CONSTEXPR const char* complexElements[] = {
	"alias","transform","ref","integrator","sensor","film",
	"rfilter","shape","bsdf","texture","emitter"
};
*/

#if 0
IElement* CElementFactory::createElement(const char* _el, const char** _atts)
{
	//should be removing white spaces performed before string comparison?
	IElement* result = nullptr;
	
	
	if (!std::strcmp(_el, "scene"))
	{
		return parseScene(_el, _atts);
	}
	else
	if (!std::strcmp(_el, "shape"))
	{
		return parseShape(_el, _atts);
	}
	else
	if (!std::strcmp(_el, "transform"))
	{
		CElementTransform* transform = new CElementTransform();

		if (!transform->processAttributes(_atts))
		{
			delete transform;
			_IRR_DEBUG_BREAK_IF(true);
			return nullptr;
		}

		return transform;
	}
	else
	if (!std::strcmp(_el, "sensor"))
	{
		CElementSensor* sensor = new CElementSensor();

		if (!sensor->processAttributes(_atts))
		{
			delete sensor;
			_IRR_DEBUG_BREAK_IF(true);
			return nullptr;
		}

		return sensor;
	}
	else
	if (!std::strcmp(_el, "sampler"))
	{
		CElementSampler* sampler = new CElementSampler();

		if (!sampler->processAttributes(_atts))
		{
			delete sampler;
			_IRR_DEBUG_BREAK_IF(true);
			return nullptr;
		}

		return sampler;
	}
	else
	if (!std::strcmp(_el, "film"))
	{
		CElementFilm* film = new CElementFilm();

		if (!film->processAttributes(_atts))
		{
			delete film;
			_IRR_DEBUG_BREAK_IF(true);
			return nullptr;
		}

		return film;
	}
	else
	if (!std::strcmp(_el, "emitter"))
	{
		CElementEmitter* emitter = new CElementEmitter();

		if (!emitter->processAttributes(_atts))
		{
			delete emitter;
			_IRR_DEBUG_BREAK_IF(true);
			return nullptr;
		}

		return emitter;
	}
	else
	{
		ParserLog::invalidXMLFileStructure("element " + std::string(_el) + "is unknown. \n");
		_IRR_DEBUG_BREAK_IF(true);
		return nullptr;
	}
}
IElement* CElementFactory::parseShape(const char* _el, const char** _atts)
{
	CShape* result = new CShape();

	if (!result->processAttributes(_atts))
	{
		_IRR_DEBUG_BREAK_IF(true);
		delete result;
		return nullptr;
	}

	//ParserLog::mitsubaLoaderError("There is no type attribute for shape element. \n");
	return result;
}
#endif

}
}
}