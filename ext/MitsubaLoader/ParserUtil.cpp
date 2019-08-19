#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/CElementFactory.h"

#include "irrlicht.h"
#include <memory>

namespace irr { namespace ext { namespace MitsubaLoader {

void ParserLog::invalidXMLFileStructure(const std::string& errorMessage)
{
	std::string message = "Mitsuba loader error:\nInvalid .xml file structure: '"
		+ errorMessage + '\n';

	//or os::Printer::log ?
	os::Printer::print(message);

	//_IRR_DEBUG_BREAK_IF(true);
}

void elementHandlerStart(void* _data, const char* _el, const char** _atts)
{
	ParserManager* mgr = static_cast<ParserManager*>(_data);

	mgr->parseElement(_el, _atts);
}

void elementHandlerEnd(void* _data, const char* _el)
{
	ParserManager* mgr = static_cast<ParserManager*>(_data);
	
	/*
	if (data->pfc.isParsingSuspended())
	{
		data->pfc.checkForUnsuspend(_el);
		return;
	}*/

	mgr->onEnd(_el);
}

void ParserManager::parseElement(const char* _el, const char** _atts)
{
	if (!pfc.suspendParsingIfElNotSupported(_el))
	{
		ParserLog::invalidXMLFileStructure(std::string(_el) + " is not supported");
		return;
	}

	if (pfc.isParsingSuspended())
		return;

	if (!std::strcmp(_el, "scene"))
	{
		if (isSceneActive())
		{
			ParserLog::invalidXMLFileStructure("scene is already declared");
			XML_StopParser(parser, false);
			_IRR_DEBUG_BREAK_IF(true);
			return;
		}
		else if (isSceneActive() == false)
		{
			scene.reset(new CMitsubaScene());
			return;
		}
	}

	if (!isSceneActive())
	{
		ParserLog::invalidXMLFileStructure("there is no scene element.");
		XML_StopParser(parser, false);
		_IRR_DEBUG_BREAK_IF(true);
		return;
	}

	if (checkIfPropertyElement(_el))
	{
		if (!processProperty(_el, _atts))
		{
			ParserLog::invalidXMLFileStructure("invalid property element.");
			XML_StopParser(parser, false);
			_IRR_DEBUG_BREAK_IF(true);
		}

		return;
	}

	std::unique_ptr<IElement> element(CElementFactory::createElement(_el, _atts));

	if (element.get() == nullptr)
	{
		XML_StopParser(parser, false);
		_IRR_DEBUG_BREAK_IF(true);
		return;
	}

	addElementToStack(std::move(element));

}

void ParserManager::addElementToStack(std::unique_ptr<IElement>&& element)
{
	elements.push(std::move(element));
}

bool ParserManager::checkIfPropertyElement(const std::string& _el)
{
	for (int i = 0; propertyElements[i]; i++)
	{
		if (_el == propertyElements[i])
			return true;
	}

	return false;
}

bool ParserManager::processProperty(const char* _el, const char** _atts)
{
	if (elements.empty())
	{
		ParserLog::invalidXMLFileStructure("weeeeeeeeeeeeeeewwwwwww.");
		XML_StopParser(parser, false);
		_IRR_DEBUG_BREAK_IF(true);
		return false;
	}

	auto optProperty = CPropertyElementManager::createPropertyData(_el, _atts);

	if (optProperty.first == false)
		return false;

	elements.top()->addProperty(std::move(optProperty.second));

	return true;
}

void ParserManager::onEnd(const std::string& _el)
{
	if (pfc.isParsingSuspended())
	{
		pfc.checkForUnsuspend(_el);
		return;
	}

	if (checkIfPropertyElement(_el))
		return;

	if (elements.empty() == false)
	{
		std::unique_ptr<IElement> element(std::move(elements.top()));
		elements.pop();

		if (!element->onEndTag(assetManager))
		{
			ParserLog::invalidXMLFileStructure(element->getLogName());
			XML_StopParser(parser, false);
			_IRR_DEBUG_BREAK_IF(true);
			return;
		}

		if (elements.empty() == false)
		{
			if (!elements.top()->processChildData(element.get()))
			{
				XML_StopParser(parser, false);
				_IRR_DEBUG_BREAK_IF(true);
				return;
			}

			return;
		}
		else
		{
			scene->processChildData(element.get());
		}
	}
	else
	{
		scene->onEndTag(this->assetManager);
	}

}

void ParserFlowController::checkForUnsuspend(const std::string& _el)
{
	if (_el == notSupportedElement.c_str())
	{
		isParsingSuspendedFlag = false;
		notSupportedElement.clear();
	}
}

bool ParserFlowController::suspendParsingIfElNotSupported(const std::string& _el)
{
	for (int i = 0; unsElements[i]; i++)
	{
		if (_el == unsElements[i])
		{
			isParsingSuspendedFlag = true;
			notSupportedElement = _el;

			return false;
		}
	}

	return true;
}

}
}
}