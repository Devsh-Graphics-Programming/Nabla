#include "os.h"

#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/CElementFactory.h"

#include "expat/lib/expat.h"

#include <memory>

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

void ParserLog::invalidXMLFileStructure(const std::string& errorMessage)
{
	std::string message = "Mitsuba loader error - Invalid .xml file structure: \'"
		+ errorMessage + '\'';

	os::Printer::log(message.c_str(), ELL_ERROR);
	_IRR_DEBUG_BREAK_IF(true);
}

void elementHandlerStart(void* _data, const char* _el, const char** _atts)
{
	ParserManager* mgr = static_cast<ParserManager*>(_data);

	mgr->parseElement(_el, _atts);
}

void elementHandlerEnd(void* _data, const char* _el)
{
	ParserManager* mgr = static_cast<ParserManager*>(_data);

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

	if (core::strcmpi(_el, "scene")==0)
	{
		// its okay to have multiple scene declarations, because of the include file
		if (!isSceneActive())
		{
			scene.reset(new CMitsubaScene());
			return;
		}
	}

	if (!isSceneActive())
	{
		killParseWithError("there is no scene element");
		return;
	}

	if (checkIfPropertyElement(_el))
	{
		processProperty(_el, _atts);
		return;
	}

	std::unique_ptr<IElement> element(CElementFactory::createElement(_el, _atts));

	if (element.get() == nullptr)
	{
		killParseWithError(std::string("Could not create element ") + _el);
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
	for (size_t i=0u; i<sizeof(propertyElements)/sizeof(const char*); i++)
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
		killParseWithError("cannot set a property with no element on the stack.");
		return false;
	}

	auto optProperty = CPropertyElementManager::createPropertyData(_el, _atts);

	if (optProperty.first == false)
	{
		killParseWithError("could not create property data.");
		return false;
	}

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

		if (!element->onEndTag(m_override))
		{
			killParseWithError(element->getLogName()+" could not onEndTag");
			return;
		}

		if (elements.empty() == false)
		{
			if (!elements.top()->processChildData(element.get()))
				killParseWithError(element->getLogName() + " could not processChildData");

			return;
		}
		else if (!scene->processChildData(element.get()))
			killParseWithError("scene could not processChildData");
	}
	else if (scene->onEndTag(m_override))
		killParseWithError("scene could not onEndTag");
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