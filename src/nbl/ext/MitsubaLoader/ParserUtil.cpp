// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/ext/MitsubaLoader/ParserUtil.h"
#include "nbl/ext/MitsubaLoader/CElementFactory.h"

#include "expat/lib/expat.h"

#include <memory>


namespace nbl::ext::MitsubaLoader
{
using namespace nbl::system;

void ParserManager::Context::invalidXMLFileStructure(const std::string& errorMessage) const
{
	std::string message = "Mitsuba loader error - Invalid .xml file structure: \'" + errorMessage + '\'';

	logger.log(message,ILogger::E_LOG_LEVEL::ELL_ERROR);
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


bool ParserManager::parse(IFile* _file, const logger_opt_ptr& _logger)
{
	XML_Parser parser = XML_ParserCreate(nullptr);
	if (!parser)
	{
		_logger.log("Could not create XML Parser!",ILogger::E_LOG_LEVEL::ELL_ERROR);
		return false;
	}

	XML_SetElementHandler(parser,elementHandlerStart,elementHandlerEnd);

	//from now data (instance of ParserData struct) will be visible to expat handlers
	Context ctx = {_file->getFileName().parent_path()/"",this,_logger,parser};
	XML_SetUserData(parser,&ctx);

	const size_t size = _file->getSize();
	const char* buff = reinterpret_cast<const char*>(const_cast<const IFile*>(_file)->getMappedPointer());
	if (!buff)
	{
		buff = reinterpret_cast<const char*>(_NBL_ALIGNED_MALLOC(size,4096u));
		IFile::success_t success;
		_file->read(success,const_cast<char*>(buff),0u,size);
		if (!success)
		{
			_logger.log("Could read the file into XML Parser Buffer!",ILogger::E_LOG_LEVEL::ELL_ERROR);
			return false;
		}
	}
	XML_Status parseStatus = XML_Parse(parser,buff,size,0);
	if (_file->getMappedPointer()!=buff)	
		_NBL_ALIGNED_FREE(const_cast<char*>(buff));

	XML_ParserFree(parser);
	switch (parseStatus)
	{
		case XML_STATUS_ERROR:
			{
				_logger.log("Parse status: XML_STATUS_ERROR",ILogger::E_LOG_LEVEL::ELL_ERROR);
				return false;
			}
			break;
		case XML_STATUS_OK:
			_logger.log("Parse status: XML_STATUS_OK",ILogger::E_LOG_LEVEL::ELL_INFO);
			break;
		case XML_STATUS_SUSPENDED:
			{
				_logger.log("Parse status: XML_STATUS_SUSPENDED",ILogger::E_LOG_LEVEL::ELL_INFO);
				return false;
			}
			break;
	}

	return true;
}

void ParserManager::parseElement(const Context& ctx, const char* _el, const char** _atts)
{
	if (core::strcmpi(_el, "scene")==0)
	{
		auto count = 0u;
		while (_atts && _atts[count]) { count++; }
		if (count!=2u)
		{
			ctx.killParseWithError("Wrong number of attributes for scene element");
			return;
		}

		if (core::strcmpi(_atts[0],"version"))
		{
			ctx.invalidXMLFileStructure(std::string(_atts[0]) + " is not an attribute of scene element");
			return;
		}
		else if (core::strcmpi(_atts[1],"0.5.0"))
		{
			ctx.invalidXMLFileStructure("Version " + std::string(_atts[1]) + " is unsupported");
			return;
		}
		m_sceneDeclCount++;
		return;
	}

	if (m_sceneDeclCount==0u)
	{
		ctx.killParseWithError("there is no scene element");
		return;
	}
	
	if (core::strcmpi(_el,"include")==0)
	{
		core::smart_refctd_ptr<IFile> file;
		auto tryOpen = [&](const system::path& path)->bool
		{
			for (auto i=0; i<2; i++)
			{
				ISystem::future_t<core::smart_refctd_ptr<IFile>> future;
				auto flags = IFile::ECF_READ;
				if (i==0)
					flags |= IFile::ECF_MAPPABLE;
				m_system->createFile(future,ctx.currentXMLDir/_atts[1],flags);
				if (future.wait())
					future.acquire().move_into(file);
				if (file)
					return true;
			}
			return false;
		};
		// first try as relative path, then as global
		if (!tryOpen(ctx.currentXMLDir/_atts[1]))
		if (!tryOpen(_atts[1]))
		{
			ctx.invalidXMLFileStructure(std::string("Could not open include file: ")+_atts[1]);
			return;
		}
		parse(file.get(),ctx.logger);
		return;
	}
#if 0
	if (propertyElements.find(_el)!=propertyElements.end())
	{
		processProperty(ctx, _el, _atts);
		return;
	}

	const auto& _map = CElementFactory::createElementTable;
	auto found = _map.find(_el);
	if (found==_map.end())
	{
		invalidXMLFileStructure(std::string("Could not process element ") + _el);
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
#endif
}

void ParserManager::processProperty(const Context& ctx, const char* _el, const char** _atts)
{
	if (elements.empty())
	{
		ctx.killParseWithError("cannot set a property with no element on the stack.");
		return;
	}
	if (!elements.top().first)
	{
		ctx.invalidXMLFileStructure("cannot set property on element that failed to be created.");
		return;
	}

#if 0
	auto optProperty = CPropertyElementManager::createPropertyData(_el,_atts);

	if (optProperty.first == false)
	{
		invalidXMLFileStructure("could not create property data.");
		return;
	}

	elements.top().first->addProperty(std::move(optProperty.second));
#endif
}

void ParserManager::onEnd(const Context& ctx, const char* _el)
{
	if (propertyElements.find(_el)!=propertyElements.end())
		return;

	if (core::strcmpi(_el, "scene") == 0)
	{
		m_sceneDeclCount--;
		return;
	}
#if 0
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
		if (parent && !parent->processChildData(element.first, element.second))
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
#endif
}


}