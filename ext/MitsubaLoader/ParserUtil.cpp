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


const core::unordered_set<std::string,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> ParserManager::propertyElements = {
	"float", "string", "boolean", "integer",
	"rgb", "srgb", "spectrum", "blackbody",
	"point", "vector",
	"matrix", "rotate", "translate", "scale", "lookat"
};

void ParserManager::parseElement(const char* _el, const char** _atts)
{
	if (!pfc.suspendParsingIfElNotSupported(_el))
	{
		ParserLog::invalidXMLFileStructure(std::string(_el) + " is not supported");
		return;
	}

	if (pfc.isParsingSuspended())
		return;

	if (core::strcmpi(_el, "scene") == 0)
	{
		auto count = 0u;
		while (_atts && _atts[count]) { count++; }
		if (count != 2u)
		{
			killParseWithError("Wrong number of attributes for scene element");
			return;
		}

		if (core::strcmpi(_atts[0], "version"))
		{
			ParserLog::invalidXMLFileStructure(std::string(_atts[0]) + " is not an attribute of scene element");
			return;
		}
		else if (core::strcmpi(_atts[1], "0.5.0"))
		{
			ParserLog::invalidXMLFileStructure("Version " + std::string(_atts[1]) + " is unsupported");
			return;
		}
		return;
	}

	if (m_sceneDeclCount==0u)
	{
		killParseWithError("there is no scene element");
		return;
	}
	
	if (core::strcmpi(_el, "include") == 0)
	{
		assert(false); // TODO
		return;
	}

	if (propertyElements.find(_el)!=propertyElements.end())
	{
		processProperty(_el, _atts);
		return;
	}

	const auto& _map = CElementFactory::createElementTable;
	auto found = _map.find(_el);
	if (found==_map.end())
	{
		killParseWithError(std::string("Could not process element ") + _el);
		return;
	}

	IElement* el = found->second.first(_atts, this);
	bool goesOnStack = found->second.second;
	if (goesOnStack)
		elements.push(el);
}

void ParserManager::processProperty(const char* _el, const char** _atts)
{
	if (elements.empty())
	{
		killParseWithError("cannot set a property with no element on the stack.");
		return;
	}
	if (!elements.top())
	{
		ParserLog::invalidXMLFileStructure("cannot set property on element that failed to be created.");
		return;
	}

	auto optProperty = CPropertyElementManager::createPropertyData(_el, _atts);

	if (optProperty.first == false)
	{
		ParserLog::invalidXMLFileStructure("could not create property data.");
		return;
	}

	elements.top()->addProperty(std::move(optProperty.second));

	return;
}

void ParserManager::onEnd(const char* _el)
{
	if (pfc.isParsingSuspended())
	{
		pfc.checkForUnsuspend(_el);
		return;
	}
	
	if (propertyElements.find(_el)!=propertyElements.end())
		return;

	if (core::strcmpi(_el, "scene") == 0)
	{
		m_sceneDeclCount--;
		return;
	}

	if (elements.empty())
		return;


	IElement* element = elements.top();
	elements.pop();

	if (!element->onEndTag(m_override, m_globalMetadata.get()))
	{
		killParseWithError(element->getLogName()+" could not onEndTag");
		return;
	}

	if (!elements.empty() == false)
	{
		if (!elements.top()->processChildData(element))
			killParseWithError(element->getLogName() + " could not processChildData");

		return;
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