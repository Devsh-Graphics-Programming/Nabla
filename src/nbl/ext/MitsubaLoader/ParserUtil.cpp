// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/ext/MitsubaLoader/ParserUtil.h"
#include "nbl/ext/MitsubaLoader/CElementIntegrator.h"
#include "nbl/ext/MitsubaLoader/CElementSensor.h"
#include "nbl/ext/MitsubaLoader/CElementFilm.h"
#include "nbl/ext/MitsubaLoader/CElementRFilter.h"
#include "nbl/ext/MitsubaLoader/CElementSampler.h"
//#include "nbl/ext/MitsubaLoader/CElementShape.h"
#include "nbl/ext/MitsubaLoader/CElementTransform.h"
//#include "nbl/ext/MitsubaLoader/CElementAnimation.h"
//#include "nbl/ext/MitsubaLoader/CElementBSDF.h"
//#include "nbl/ext/MitsubaLoader/CElementTexture.h"
//#include "nbl/ext/MitsubaLoader/CElementEmitter.h"
#include "nbl/ext/MitsubaLoader/CElementEmissionProfile.h"

#include "expat/lib/expat.h"

#include <memory>


namespace nbl::ext::MitsubaLoader
{
using namespace nbl::system;

auto ParserManager::parse(IFile* _file, const Params& _params) const -> Result
{
//	CMitsubaMetadata* obj = new CMitsubaMetadata();
	Result result = {
		.metadata = core::make_smart_refctd_ptr<CMitsubaMetadata>()
	};
	SessionContext ctx = {
		.result = &result,
		.params = &_params,
		.manager = this
	};

	if (!ctx.parse(_file))
		return {};

	return result;
}

bool ParserManager::SessionContext::parse(IFile* _file)
{
	auto logger = params->logger;

	XML_Parser parser = XML_ParserCreate(nullptr);
	if (!parser)
	{
		logger.log("Could not create XML Parser!",ILogger::E_LOG_LEVEL::ELL_ERROR);
		return false;
	}

	XML_SetElementHandler(parser,elementHandlerStart,elementHandlerEnd);

	//from now data (instance of ParserData struct) will be visible to expat handlers
	XMLContext ctx = {
		.session = this,
		.currentXMLDir = _file->getFileName().parent_path()/"",
		.parser = parser
	};
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
			logger.log("Could read the file into XML Parser Buffer!",ILogger::E_LOG_LEVEL::ELL_ERROR);
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
				logger.log("Parse status: XML_STATUS_ERROR",ILogger::E_LOG_LEVEL::ELL_ERROR);
				return false;
			}
			break;
		case XML_STATUS_OK:
			logger.log("Parse status: XML_STATUS_OK",ILogger::E_LOG_LEVEL::ELL_INFO);
			break;
		case XML_STATUS_SUSPENDED:
			{
				logger.log("Parse status: XML_STATUS_SUSPENDED",ILogger::E_LOG_LEVEL::ELL_INFO);
				return false;
			}
			break;
	}

	return true;
}

void ParserManager::elementHandlerStart(void* _data, const char* _el, const char** _atts)
{
	auto& ctx = *reinterpret_cast<XMLContext*>(_data);

	ctx.parseElement(_el,_atts);
}

void ParserManager::XMLContext::killParseWithError(const std::string& message) const
{
	session->invalidXMLFileStructure(message);
	XML_StopParser(parser,false);
}

void ParserManager::XMLContext::parseElement(const char* _el, const char** _atts)
{
	if (core::strcmpi(_el,"scene")==0)
	{
		auto count = 0u;
		while (_atts && _atts[count]) { count++; }
		if (count!=2u)
		{
			killParseWithError("Wrong number of attributes for scene element");
			return;
		}

		if (core::strcmpi(_atts[0],"version"))
		{
			session->invalidXMLFileStructure(core::string(_atts[0]) + " is not an attribute of scene element");
			return;
		}
		else if (core::strcmpi(_atts[1],"0.5.0"))
		{
			session->invalidXMLFileStructure("Version " + core::string(_atts[1]) + " is unsupported");
			return;
		}
		session->sceneDeclCount++;
		return;
	}

	if (session->sceneDeclCount==0u)
	{
		killParseWithError("there is no scene element");
		return;
	}
	
	const ParserManager* manager = session->manager;
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
				session->params->system->createFile(future,currentXMLDir/_atts[1],flags);
				if (future.wait())
					future.acquire().move_into(file);
				if (file)
					return true;
			}
			return false;
		};
		// first try as relative path, then as global
		if (!tryOpen(currentXMLDir/_atts[1]))
		if (!tryOpen(_atts[1]))
		{
			session->invalidXMLFileStructure(std::string("Could not open include file: ")+_atts[1]);
			return;
		}
		if (!session->parse(file.get()))
			killParseWithError(core::string("Could not parse include file: ")+_atts[1]);
		return;
	}

	const auto& propertyElements = manager->propertyElements;
	if (propertyElements.find(_el)!=propertyElements.end())
	{
		auto& elements = session->elements;
		if (elements.empty())
		{
			killParseWithError("cannot set a property with no element on the stack.");
			return;
		}
		if (!elements.top().element)
		{
			session->invalidXMLFileStructure("cannot set property on element that failed to be created.");
			return;
		}

		auto optProperty = manager->propertyElementManager.createPropertyData(_el,_atts,session->params->logger);
		if (!optProperty.has_value())
		{
			session->invalidXMLFileStructure("could not create property data.");
			return;
		}

		elements.top().element->addProperty(std::move(optProperty.value()),session->params->logger);
		return;
	}

	// TODO: don't have this table be a global
	const auto& _map = manager->createElementTable;
	auto found = _map.find(_el);
	if (found==_map.end())
	{
		session->invalidXMLFileStructure(std::string("Could not process element ")+_el);
		session->elements.push({nullptr,""});
		return;
	}

	auto created = found->second.create(_atts,session);
	// we still push nullptr (failed creation) onto the stack, we only stop parse on catastrphic failure
	if (!found->second.retvalGoesOnStack)
		return;
	if (created.element && created.name.size())
		session->handles[created.name] = created.element;
	session->elements.push(std::move(created));
}

void ParserManager::elementHandlerEnd(void* _data, const char* _el)
{
	auto& ctx = *reinterpret_cast<XMLContext*>(_data);

	ctx.onEnd(_el);
}

void ParserManager::XMLContext::onEnd(const char* _el)
{
	const auto& propertyElements = session->manager->propertyElements;
	if (propertyElements.find(_el)!=propertyElements.end())
		return;

	if (core::strcmpi(_el,"scene")==0)
	{
		session->sceneDeclCount--;
		return;
	}

	auto& elements = session->elements;
	if (elements.empty())
		return;

	auto element = elements.top();
	elements.pop();

	auto& result = *session->result;
	if (element.element && !element.element->onEndTag(session->params->_override,result.metadata.get()))
	{
		killParseWithError(element.element->getLogName()+" could not onEndTag");
		return;
	}

	if (!elements.empty())
	{
		IElement* parent = elements.top().element;
		if (parent && !parent->processChildData(element.element,element.name))
		{
			if (element.element)
				killParseWithError(element.element->getLogName()+" could not processChildData with name: "+element.name);
			else
				killParseWithError("Failed to add a nullptr child with name: "+element.name);
		}

		return;
	}

	if (element.element && element.element->getType()==IElement::Type::SHAPE)
	{
		auto shape = static_cast<CElementShape*>(element.element);
		if (shape)
			result.shapegroups.emplace_back(shape,std::move(element.name));
	}
}

//
ParserManager::ParserManager() : propertyElements({
	"float", "string", "boolean", "integer",
	"rgb", "srgb", "spectrum", "blackbody",
	"point", "vector",
	"matrix", "rotate", "translate", "scale", "lookat"
}), propertyElementManager(), createElementTable({
//	{"integrator",		{.create=createElement<CElementIntegrator>,.retvalGoesOnStack=true}},
//	{"sensor",			{.create=createElement<CElementSensor>,.retvalGoesOnStack=true}},
//	{"film",			{.create=createElement<CElementFilm>,.retvalGoesOnStack=true}},
//	{"rfilter",			{.create=createElement<CElementRFilter>,.retvalGoesOnStack=true}},
//	{"sampler",			{.create=createElement<CElementSampler>,.retvalGoesOnStack=true}},
//	{"shape",			{.create=createElement<CElementShape>,.retvalGoesOnStack=true}},
	{"transform",		{.create=createElement<CElementTransform>,.retvalGoesOnStack=true}},
//	{"animation",		{.create=createElement<CElementAnimation>,.retvalGoesOnStack=true}},
//	{"bsdf",			{.create=createElement<CElementBSDF>,.retvalGoesOnStack=true}},
//	{"texture",			{.create=createElement<CElementTexture>,.retvalGoesOnStack=true}},
//	{"emitter",			{.create=createElement<CElementEmitter>,.retvalGoesOnStack=true}},
	{"emissionprofile", {.create=createElement<CElementEmissionProfile>,.retvalGoesOnStack=true}},
	{"alias",			{.create=processAlias,.retvalGoesOnStack=true}},
	{"ref",				{.create=processRef,.retvalGoesOnStack=true}}
}){}

auto ParserManager::processAlias(const char** _atts, SessionContext* ctx) -> SNamedElement
{
	const char* id = nullptr;
	const char* as = nullptr;
	if (IElement::areAttributesInvalid(_atts,4u))
	{
		ctx->invalidXMLFileStructure("Invalid attributes for <alias>");
		return {};
	}

	core::string name;
	while (*_atts)
	{
		if (core::strcmpi(_atts[0], "id")==0)
			id = _atts[1];
		else if (core::strcmpi(_atts[0], "as")==0)
			as = _atts[1];
		else if (core::strcmpi(_atts[0], "name")==0)
			name = _atts[1];
		_atts += 2;
	}
	// not finding the alias doesn't kill XML parse
	if (!id || !as)
	{
		ctx->invalidXMLFileStructure("Alias ID and what we're aliasing is not found");
		return {nullptr,std::move(name)};
	}

	auto& handles = ctx->handles;
	auto* original = handles[id];
	handles[as] = original;
	return {original,std::move(name)};
}

auto ParserManager::processRef(const char** _atts, SessionContext* ctx) -> SNamedElement
{
	const char* id;
	std::string name;
	if (!IElement::getIDAndName(id,name,_atts))
	{
		ctx->invalidXMLFileStructure("Malformed `<ref>` element!");
		return {nullptr,std::move(name)};
	}

	auto* original = ctx->handles[id];
	if (!original)
		ctx->invalidXMLFileStructure(core::string("Used a `<ref name=\"")+name+"\" id=\""+id+"\">` element but referenced element not defined in preceeding XML!");
	return {original,std::move(name)};
}

}