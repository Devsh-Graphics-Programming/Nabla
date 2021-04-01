// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __I_PARSER_UTIL_H_INCLUDED__
#define __I_PARSER_UTIL_H_INCLUDED__

#include "nbl/core/core.h"

#include "IFileSystem.h"

#include "nbl/asset/interchange/IAssetLoader.h"

#include "nbl/ext/MitsubaLoader/CElementFactory.h"
#include "nbl/ext/MitsubaLoader/CMitsubaMetadata.h"

#include "expat/lib/expat.h"

#include <stack>


namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{

	   	  

class ParserLog
{
public:
	/*prints this message:
	Mitsuba loader error:
	Invalid .xml file structure: message */
	static void invalidXMLFileStructure(const std::string& errorMessage);

};


template<typename... types>
class ElementPool // : public std::tuple<core::vector<types>...>
{
		core::SimpleBlockBasedAllocator<core::LinearAddressAllocator<uint32_t>,core::aligned_allocator> poolAllocator;
	public:
		ElementPool() : poolAllocator(4096u*1024u, 256u) {}

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
		struct Context
		{
			ParserManager* manager;
			XML_Parser parser;
			io::path currentXMLDir;
		};
	public:
		//! Constructor 
		ParserManager(io::IFileSystem* _filesystem, asset::IAssetLoader::IAssetLoaderOverride* _override) :
								m_filesystem(_filesystem), m_override(_override), m_sceneDeclCount(0),
								m_metadata(core::make_smart_refctd_ptr<CMitsubaMetadata>())
		{
		}

		//
		static void elementHandlerStart(void* _data, const char* _el, const char** _atts);
		static void elementHandlerEnd(void* _data, const char* _el);

		//
		inline void killParseWithError(const Context& ctx, const std::string& message)
		{
			_NBL_DEBUG_BREAK_IF(true);
			ParserLog::invalidXMLFileStructure(message);
			XML_StopParser(ctx.parser, false);
		}

		bool parse(io::IReadFile* _file);

		void parseElement(const Context& ctx, const char* _el, const char** _atts);

		void onEnd(const Context& ctx, const char* _el);

		//
		core::vector<std::pair<CElementShape*,std::string> > shapegroups;
		//
		core::smart_refctd_ptr<CMitsubaMetadata> m_metadata;

	private:
		//
		void processProperty(const Context& ctx, const char* _el, const char** _atts);

		//
		io::IFileSystem* m_filesystem;
		asset::IAssetLoader::IAssetLoaderOverride* m_override;
		//
		uint32_t m_sceneDeclCount;
		//
		ElementPool<
			CElementIntegrator,
			CElementSensor,
			CElementFilm,
			CElementRFilter,
			CElementSampler,
			CElementShape,
			CElementBSDF,
			CElementTexture,
			CElementEmitter
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
}
}

#endif