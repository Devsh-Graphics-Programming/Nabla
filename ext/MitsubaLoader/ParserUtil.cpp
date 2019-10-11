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


static const core::unordered_set<std::string,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> propertyElements = {
	"float", "string", "boolean", "integer",
	"rgb", "srgb", "spectrum", "blackbody",
	"point", "vector",
	"matrix", "rotate", "translate", "scale", "lookat"
};

void ParserManager::parseElement(const char* _el, const char** _atts)
{
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
		m_sceneDeclCount++;
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
		ParserLog::invalidXMLFileStructure(std::string("Could not process element ") + _el);
		elements.push({nullptr,""});
		return;
	}

	auto el = found->second.first(_atts, this);
	bool goesOnStack = found->second.second;
	if (!goesOnStack)
		return;
	
	elements.push(el);
	if (el.first && el.first->id.size())
		handles[el.first->id] = el.first;
}

void ParserManager::processProperty(const char* _el, const char** _atts)
{
	if (elements.empty())
	{
		killParseWithError("cannot set a property with no element on the stack.");
		return;
	}
	if (!elements.top().first)
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

	elements.top().first->addProperty(std::move(optProperty.second));

	return;
}

void ParserManager::onEnd(const char* _el)
{	
	if (propertyElements.find(_el)!=propertyElements.end())
		return;

	if (core::strcmpi(_el, "scene") == 0)
	{
		m_sceneDeclCount--;
		return;
	}

	if (elements.empty())
		return;


	auto element = elements.top();
	elements.pop();

	if (element.first && !element.first->onEndTag(m_override, m_globalMetadata.get()))
	{
		killParseWithError(element.first->getLogName()+" could not onEndTag");
		return;
	}

	if (!elements.empty())
	{
		IElement* parent = elements.top().first;
		if (!parent->processChildData(element.first, element.second))
		{
			if (element.first)
				killParseWithError(element.first->getLogName() + " could not processChildData with name: " + element.second);
			else
				killParseWithError("Failed to add a nullptr child with name: " + element.second);
		}

		return;
	}
}

}
}
}