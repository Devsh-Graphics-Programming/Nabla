#ifndef __I_PARSER_UTIL_H_INCLUDED__
#define __I_PARSER_UTIL_H_INCLUDED__

#include "irr/asset/IAssetLoader.h"
#include "../../ext/MitsubaLoader/CMitsubaScene.h"
#include "../../ext/MitsubaLoader/IElement.h"

#include "expat/lib/expat.h"

#include <stack>


namespace irr
{
namespace ext
{
namespace MitsubaLoader
{


//now unsupported elements (like  <sensor> (for now), for example) and its children elements will be ignored
class ParserFlowController
{
public:
	ParserFlowController()
		:isParsingSuspendedFlag(false) {};

	bool suspendParsingIfElNotSupported(const std::string& _el);
	void checkForUnsuspend(const std::string& _el);

	inline bool isParsingSuspended() const { return isParsingSuspendedFlag; }

private:
	static constexpr const char* unsElements[] = 
	{ 
		"integrator", "emitter", "ref", "bsdf",  
		"rfilter", "medium", "include", nullptr 
	};
	bool isParsingSuspendedFlag;
	std::string notSupportedElement;

};

class ParserLog
{
public:
	/*prints this message:
	Mitsuba loader error:
	Invalid .xml file structure: message */
	static void invalidXMLFileStructure(const std::string& errorMessage);

};


//struct, which will be passed to expat handlers as user data (first argument) see: XML_StartElementHandler or XML_EndElementHandler in expat.h
class ParserManager
{
	public:
		//! Constructor 
		ParserManager(asset::IAssetLoader::IAssetLoaderOverride* _override, XML_Parser _parser)
			: m_override(_override), m_parser(_parser), scene(nullptr)
		{
		}

		auto& getParserFlowController() { return pfc; }
		auto& getParserFlowController() const { return pfc; }

		inline void killParseWithError(const std::string& message)
		{
			_IRR_DEBUG_BREAK_IF(true);
			ParserLog::invalidXMLFileStructure(message);
			XML_StopParser(m_parser, false);
		}

		void parseElement(const char* _el, const char** _atts);
		//TODO: getAssetBundle();

		void onEnd(const std::string&);

		inline CMitsubaScene& getScene() { return *scene; }

	private:
		void addElementToStack(std::unique_ptr<IElement>&& element);

		inline bool isSceneActive() { return static_cast<bool>(scene.get()); }

		bool checkIfPropertyElement(const std::string& _el);

		bool processProperty(const char* _el, const char** _atts);

	private:
		asset::IAssetLoader::IAssetLoaderOverride* m_override;
		XML_Parser m_parser;

		/*root element, which will hold all loaded assets and material data
		in asset::SCPUMesh (for now) instance*/
		std::unique_ptr<CMitsubaScene> scene;

		/*stack of currently processed elements
		each element of index N is parent of the element of index N+1
		the scene element is a parent of all elements of index 0 */
		core::stack<std::unique_ptr<IElement>> elements; 

		ParserFlowController pfc;

		_IRR_STATIC_INLINE_CONSTEXPR const char* propertyElements[] = { 
			"float", "string", "boolean", "integer", 
			"rgb", "srgb", "spectrum", "blackbody",
			"point", "vector", 
			"matrix", "rotate", "translate", "scale", "lookat",
			"animation"
		};
};

void elementHandlerStart(void* _data, const char* _el, const char** _atts);
void elementHandlerEnd(void* _data, const char* _el);

}
}
}

#endif