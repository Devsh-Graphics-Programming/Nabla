// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_I_PARSER_UTIL_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_I_PARSER_UTIL_H_INCLUDED_


#include "nbl/asset/interchange/IAssetLoader.h"

#include "nbl/ext/MitsubaLoader/CMitsubaMetadata.h"
#include "nbl/ext/MitsubaLoader/CElementIntegrator.h"
#include "nbl/ext/MitsubaLoader/CElementSensor.h"
#include "nbl/ext/MitsubaLoader/CElementFilm.h"
#include "nbl/ext/MitsubaLoader/CElementRFilter.h"
#include "nbl/ext/MitsubaLoader/CElementSampler.h"
#include "nbl/ext/MitsubaLoader/CElementShape.h"
#include "nbl/ext/MitsubaLoader/CElementBSDF.h"
#include "nbl/ext/MitsubaLoader/CElementTexture.h"
#include "nbl/ext/MitsubaLoader/CElementEmitter.h"
#include "nbl/ext/MitsubaLoader/CElementEmissionProfile.h"

#include <stack>

// don't leak expat headers
struct XML_ParserStruct;
typedef struct XML_ParserStruct* XML_Parser;


namespace nbl::ext::MitsubaLoader
{
class IElement;

// TODO: replace with common Class for Material Compiler V3 Node Pool
template<typename... types>
class ElementPool // similar to : public std::tuple<core::vector<types>...>
{
		core::SimpleBlockBasedAllocator<core::LinearAddressAllocator<uint32_t>,core::aligned_allocator> poolAllocator;
	public:
		ElementPool() : poolAllocator(4096u*1024u, 256u, 256u) {} // TODO: is it correct?

		template<typename T, typename... Args>
		inline T* construct(Args&& ... args)
		{
			T* ptr = reinterpret_cast<T*>(poolAllocator.allocate(sizeof(T), alignof(T)));
			return new (ptr) T(std::forward<Args>(args)...);
		}
};

//struct, which will be passed to expat handlers as user data (first argument) see: XML_StartElementHandler or XML_EndElementHandler in expat.h
class ParserManager final
{
	public:
		//! Constructor 
		ParserManager();

		//
		static void elementHandlerStart(void* _data, const char* _el, const char** _atts);
		static void elementHandlerEnd(void* _data, const char* _el);

		struct Params
		{
			system::logger_opt_ptr logger;
			// for opening included XML files
			system::ISystem* system;
			asset::IAssetLoader::IAssetLoaderOverride* _override;
		};
		struct Result
		{
			explicit inline operator bool() const {return bool(metadata);}

			// note that its shared between per-file contexts
			core::smart_refctd_ptr<CMitsubaMetadata> metadata = nullptr;
			//
			core::vector<std::pair<CElementShape*,std::string> > shapegroups = {};
		};
		Result parse(system::IFile* _file, const Params& _params) const;
		
		// Properties are simple XML nodes which are not `IElement` and neither children of an` IElement`
		// If we match any `<name` we call the `processProperty` method instead of element creation method
		const core::unordered_set<std::string,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> propertyElements;
		const CPropertyElementManager propertyElementManager;

		using supported_elements_t = core::type_list<
			CElementIntegrator,
			CElementSensor,
			CElementFilm,
			CElementRFilter,
			CElementSampler,
			CElementShape,
			CElementTransform,
///			CElementBSDF,
///			CElementTexture,
///			CElementEmitter,
			CElementEmissionProfile
		>;

	private:
		const core::tuple_transform_t<IElement::AddPropertyMap,supported_elements_t> addPropertyMaps;

		struct SNamedElement
		{
			IElement* element = nullptr;
			core::string name = {};
		};
		// the XMLs can include each other, so this stores the stuff across files
		struct SessionContext
		{
			// prints this message:
			// Mitsuba loader error:
			// Invalid .xml file structure: message
			inline void invalidXMLFileStructure(const std::string& errorMessage) const
			{
				::nbl::ext::MitsubaLoader::invalidXMLFileStructure(params->logger,errorMessage);
			}
			// meant for parsing one file in an include chain
			bool parse(system::IFile* _file);

			Result* const result;
			const Params* const params;
			const ParserManager* const manager;
			//
			uint32_t sceneDeclCount = 0;
			// TODO: This leaks memory all over the place because destructors are not ran!
			ElementPool<> objects = {};
			// aliases and names (in Mitsbua XML you can give nodes names and `ref` them)
			core::unordered_map<core::string,IElement*,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> handles = {};
			// stack of currently processed elements, each element of index N is parent of the element of index N+1
			// the scene element is a parent of all elements of index 0
			core::stack<SNamedElement> elements = {};
		};
		// This is for a single XML File
		struct XMLContext
		{
			//
			void killParseWithError(const std::string& message) const;
			void parseElement(const char* _el, const char** _atts);
			void onEnd(const char* _el);

			SessionContext* const session;
			//
			const system::path currentXMLDir;
			//
			XML_Parser parser;
		};
		
		struct SElementCreator
		{
			// we still push nullptr (failed creation) onto the stack, we only stop parse on catastrphic failure later on if a use of the element pops up
			// this is why we don't need XMLCOntext for `killParseWithError`
			using func_t = SNamedElement(*)(const char**/*attributes*/,SessionContext*);
			func_t create;
			bool retvalGoesOnStack;
		};
		const core::unordered_map<std::string/*elementName*/,SElementCreator,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> createElementTable;
		//
		template<typename Element>
		struct CreateElement;
		//
		static SNamedElement processAlias(const char** _atts, SessionContext* ctx);
		static SNamedElement processRef(const char** _atts, SessionContext* ctx);
};

}
#endif