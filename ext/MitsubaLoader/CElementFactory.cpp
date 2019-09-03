#include "../../ext/MitsubaLoader/CElementFactory.h"
#include "irrlicht.h"

#include <string>

#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/IElement.h"
#include "../../ext/MitsubaLoader/CElementTransform.h"
#include "../../ext/MitsubaLoader/CElementSensor.h"
#include "../../ext/MitsubaLoader/CElementSampler.h"
#include "../../ext/MitsubaLoader/CElementFilm.h"
#include "../../ext/MitsubaLoader/CElementEmitter.h"
#include "../../ext/MitsubaLoader/CShapeCreator.h"
#include "../../ext/MitsubaLoader/Shape.h"

namespace irr { namespace ext { namespace MitsubaLoader {

//TODO: elementFactory should be an actuall class with distinct private member functions..

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

IElement* CElementFactory::parseScene(const char* _el, const char** _atts)
{
	return new CMitsubaScene();
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

}
}
}