#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/CElementFactory.h"
#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {

void ParserLog::wrongAttribute(const std::string& attribName, const std::string& elementName)
{
	std::string message = "Mitsuba loader error:\nInvalid .xml file structure: attribute '"
		+ attribName + "' is not declared for element '" + elementName + "'\n";

	//or os::Printer::log ?
	os::Printer::print(message);

	_IRR_DEBUG_BREAK_IF(true);
}

void ParserLog::wrongChildElement(const std::string& parentName, const std::string& childName)
{
	std::string message = "Mitsuba loader error:\nInvalid .xml file structure: '"
		+ parentName + "' is not a parent of '" + childName + "'\n";

	//or os::Printer::log ?
	os::Printer::print(message);

	_IRR_DEBUG_BREAK_IF(true);
}

void ParserLog::mitsubaLoaderError(const std::string& errorMessage)
{
	std::string message = "Mitsuba loader error:\nInvalid .xml file structure: '"
		+ errorMessage + '\n';

	//or os::Printer::log ?
	os::Printer::print(message);

	_IRR_DEBUG_BREAK_IF(true);
}

void elementHandlerStart(void* _data, const char* _el, const char** _atts)
{

	ParserData* data = static_cast<ParserData*>(_data);

	std::unique_ptr<IElement> element(CElementFactory::createElement(_el, _atts));

	if (!element)
	{
		XML_StopParser(data->parser, false);
		_IRR_DEBUG_BREAK_IF(true);
		return;
	}

	if (element->getType() == IElement::Type::SCENE)
	{
		if (!data->scene)
		{

			data->scene.reset(static_cast<CMitsubaScene*>(element.release()));
			data->scene->processAttributes(_atts);
			return;
		}
		else
		{
			ParserLog::wrongChildElement("scene", "scene");
			XML_StopParser(data->parser, false);
			_IRR_DEBUG_BREAK_IF(true);
			return;
		}
	}

	if (!data->scene)
	{
		ParserLog::mitsubaLoaderError("there is no scene element");
		XML_StopParser(data->parser, false);
		_IRR_DEBUG_BREAK_IF(true);
		return;
	}

	if (!element->processAttributes(_atts))
	{
		XML_StopParser(data->parser, false);
		_IRR_DEBUG_BREAK_IF(true);
		return;
	}
	
	data->elements.push_back(std::move(element));
}

void elementHandlerEnd(void* _data, const char* _el)
{

	ParserData* data = static_cast<ParserData*>(_data);

	if (data->elements.empty())
	{
		data->scene->onEndTag(data->assetManager, nullptr);
		return;
	}
	else
	{
		std::unique_ptr<IElement> element = std::move(data->elements.back());
		data->elements.pop_back();

		IElement* parent = (data->elements.empty()) ? data->scene.get() : data->elements.back().get();

		if (!element->onEndTag(data->assetManager, parent))
		{
			XML_StopParser(data->parser, false);
			_IRR_DEBUG_BREAK_IF(true);
			return;
		}
	}
}

}
}
}