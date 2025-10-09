// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_I_PARSER_UTIL_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_I_PARSER_UTIL_H_INCLUDED_


#include "nbl/asset/interchange/IAssetLoader.h"

//#include "nbl/ext/MitsubaLoader/CElementFactory.h"
#include "nbl/ext/MitsubaLoader/CMitsubaMetadata.h"

#include "expat/lib/expat.h"

#include <stack>


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
class ParserManager
{
	protected:
		// TODO: need per-file/per-parse contexts and per-load (one shapegroup, one metadata, one stack, etc. - basically the members of `ParserManager` now)
		struct Context
		{
			/*prints this message:
			Mitsuba loader error:
			Invalid .xml file structure: message */
			void invalidXMLFileStructure(const std::string& errorMessage) const;

			//
			inline void killParseWithError(const std::string& message) const
			{
				invalidXMLFileStructure(message);
				XML_StopParser(parser,false);
			}

			system::path currentXMLDir;
			//
			ParserManager* manager;
			system::logger_opt_ptr logger;
			//
			XML_Parser parser;
		};

	public:
		//! Constructor 
		inline ParserManager(system::ISystem* _system, asset::IAssetLoader::IAssetLoaderOverride* _override) :
			propertyElements({
				"float", "string", "boolean", "integer",
				"rgb", "srgb", "spectrum", "blackbody",
				"point", "vector",
				"matrix", "rotate", "translate", "scale", "lookat"
			}),	m_system(_system), m_override(_override), m_metadata(core::make_smart_refctd_ptr<CMitsubaMetadata>()) {}

		//
		static void elementHandlerStart(void* _data, const char* _el, const char** _atts);
		static void elementHandlerEnd(void* _data, const char* _el);

		bool parse(system::IFile* _file, const system::logger_opt_ptr& _logger);

		void parseElement(const Context& ctx, const char* _el, const char** _atts);

		void onEnd(const Context& ctx, const char* _el);

#if 0
		//
		core::vector<std::pair<CElementShape*,std::string> > shapegroups;
#endif
		// note that its shared between per-file contexts
		core::smart_refctd_ptr<CMitsubaMetadata> m_metadata;

	private:
		//
		void processProperty(const Context& ctx, const char* _el, const char** _atts);

		const core::unordered_set<std::string,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> propertyElements;
		// TODO: re-architect this and move into context so the PArserManager can be persistent
		system::ISystem* m_system;
		asset::IAssetLoader::IAssetLoaderOverride* m_override;
		//
		uint32_t m_sceneDeclCount = 0;
		// TODO: This leaks memory all over the place because destructors are not ran!
		ElementPool</*
			CElementIntegrator,
			CElementSensor,
			CElementFilm,
			CElementRFilter,
			CElementSampler,
			CElementShape,
			CElementBSDF,
			CElementTexture,
			CElementEmitter*/
		> objects;
		// aliases and names
		core::unordered_map<std::string,IElement*,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> handles;
		/*stack of currently processed elements
		each element of index N is parent of the element of index N+1
		the scene element is a parent of all elements of index 0 */
		core::stack<std::pair<IElement*,std::string> > elements; 

		friend class CElementFactory;
};

}
#endif