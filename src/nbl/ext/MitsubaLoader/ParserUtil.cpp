// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/MitsubaLoader/ParserUtil.h"
#include "nbl/ext/MitsubaLoader/CElementFactory.h"

#include "expat/lib/expat.h"

#include <memory>

namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{

// TODO:
void ParserLog::invalidXMLFileStructure(const std::string& errorMessage)
{
	/*std::string message = "Mitsuba loader error - Invalid .xml file structure: \'"
		+ errorMessage + '\'';

	os::Printer::log(message.c_str(), ELL_ERROR);*/
	_NBL_DEBUG_BREAK_IF(true);
}

void ParserManager::elementHandlerStart(void* _data, const char* _el, const char** _atts)
{
	auto ctx = *reinterpret_cast<Context*>(_data);

	ctx.manager->parseElement(ctx, _el, _atts);
}

void ParserManager::elementHandlerEnd(void* _data, const char* _el)
{
	auto ctx = *reinterpret_cast<Context*>(_data);

	ctx.manager->onEnd(ctx,_el);
}



bool ParserManager::parse(system::IFile* _file, const system::logger_opt_ptr& _logger)
{
	XML_Parser parser = XML_ParserCreate(nullptr);
	if (!parser)
	{
		_logger.log("Could not create XML Parser!", system::ILogger::E_LOG_LEVEL::ELL_ERROR);
		return false;
	}

	XML_SetElementHandler(parser, elementHandlerStart, elementHandlerEnd);

	//from now data (instance of ParserData struct) will be visible to expat handlers
	// TODO: test
	Context ctx = {this,parser,_file->getFileName().parent_path()/""};
	XML_SetUserData(parser, &ctx);


	char* buff = (char*)_NBL_ALIGNED_MALLOC(_file->getSize(), 4096u);

	system::future<size_t> future;
	_file->read(future, (void*)buff, 0u, _file->getSize());
	future.get();

	XML_Status parseStatus = XML_Parse(parser, buff, _file->getSize(), 0);
	_NBL_ALIGNED_FREE(buff);
	XML_ParserFree(parser);
	switch (parseStatus)
	{
		case XML_STATUS_ERROR:
			{
				_logger.log("Parse status: XML_STATUS_ERROR", system::ILogger::E_LOG_LEVEL::ELL_ERROR);
				return false;
			}
			break;
		case XML_STATUS_OK:
			#ifdef _NBL_DEBUG
			_logger.log("Parse status: XML_STATUS_OK", system::ILogger::E_LOG_LEVEL::ELL_INFO);
			#endif
			break;
		case XML_STATUS_SUSPENDED:
			{
				_logger.log("Parse status: XML_STATUS_SUSPENDED", system::ILogger::E_LOG_LEVEL::ELL_INFO);
				return false;
			}
			break;
	}

	return true;
}

static const core::unordered_set<std::string,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> propertyElements = {
	"float", "string", "boolean", "integer",
	"rgb", "srgb", "spectrum", "blackbody",
	"point", "vector",
	"matrix", "rotate", "translate", "scale", "lookat"
};

void ParserManager::parseElement(const Context& ctx, const char* _el, const char** _atts)
{
	if (core::strcmpi(_el, "scene") == 0)
	{
		auto count = 0u;
		while (_atts && _atts[count]) { count++; }
		if (count != 2u)
		{
			killParseWithError(ctx,"Wrong number of attributes for scene element");
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
		killParseWithError(ctx,"there is no scene element");
		return;
	}
	
	if (core::strcmpi(_el, "include") == 0)
	{
		//TODO: test
		system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
		bool validInput = m_system->createFile(future, ctx.currentXMLDir.string()+_atts[1], system::IFile::ECF_READ);
		if (!validInput) // try global path
			validInput = m_system->createFile(future, _atts[1], system::IFile::ECF_READ);
		if (!validInput)
		{
			ParserLog::invalidXMLFileStructure(std::string("Could not open include file: ") + _atts[1]);
			return;
		}
		auto file = future.get();
		parse(file.get(), system::logger_opt_ptr(nullptr)); // TODO: fix
		return;
	}

	if (propertyElements.find(_el)!=propertyElements.end())
	{
		processProperty(ctx, _el, _atts);
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

void ParserManager::processProperty(const Context& ctx, const char* _el, const char** _atts)
{
	if (elements.empty())
	{
		killParseWithError(ctx,"cannot set a property with no element on the stack.");
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

void ParserManager::onEnd(const Context& ctx, const char* _el)
{
	if (propertyElements.find(_el) != propertyElements.end())
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

	if (element.first && !element.first->onEndTag(m_override,m_metadata.get()))
	{
		killParseWithError(ctx,element.first->getLogName() + " could not onEndTag");
		return;
	}

	if (!elements.empty())
	{
		IElement* parent = elements.top().first;
		if (!parent->processChildData(element.first, element.second))
		{
			if (element.first)
				killParseWithError(ctx,element.first->getLogName() + " could not processChildData with name: " + element.second);
			else
				killParseWithError(ctx,"Failed to add a nullptr child with name: " + element.second);
		}

		return;
	}

	if (element.first && element.first->getType()==IElement::Type::SHAPE)
	{
		auto shape = static_cast<CElementShape*>(element.first);
		if (shape)
			shapegroups.emplace_back(shape,std::move(element.second));
	}
}


}
}
}