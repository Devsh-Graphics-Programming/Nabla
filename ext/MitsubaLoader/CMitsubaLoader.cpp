#include "CMitsubaLoader.h"
#include "../3rdparty/libexpat/expat/lib/expat.h"
#include "irr/asset/SCPUMesh.h"
#include "irr/asset/IAssetManager.h"

#include "IElement.h"
#include "CMitsubaScene.h"

#include "IrrlichtDevice.h"
#include <stack>
#include <chrono>

namespace irr { namespace ext { namespace MitsubaLoader {

//TODO: 
/*
 - smart pointers
 - proper handling of incorrect .xml file structure (in some situations parsing should be stoped and nullptr should be returned
   from CMitsubaLoader::loadAsset, and in other situations only warning should be shown.)
 - handle situations when elementFactory returns nullptr
 - use log functions instead of std::cout
*/

//struct, which will be passed to expat handlers as user data (first argument) see: XML_StartElementHandler or XML_EndElementHandler in expat.h
struct ParserData
{
	//!Constructor 
	ParserData(irr::asset::IAssetManager& _assetManager, XML_Parser _parser)
		: assetManager(_assetManager),
		isSceneActive(false),
		parser(_parser)
	{

	}

	irr::asset::IAssetManager& assetManager;
	
	/*root element, which will hold all loaded assets and material data
	in irr::asset::SCPUMesh (for now) instance*/
	CMitsubaScene scene;

	/*array of currently processed elements
	each element of index N is parent of the element of index N+1
	the scene element is a parent of all elements of index 0 */
	core::vector<IElement*> elements;

	/*used to control if first element in xml file is a scene,
	if not then .xml file is considered invalid and error is generated*/
	bool isSceneActive;

	
	XML_Parser parser;
};

static void elementHandlerStart(void* _data, const char* _el, const char** _atts)
{
	ParserData* data = static_cast<ParserData*>(_data);	

	if(!std::strcmp(_el, "scene"))
	{
		//if element being parsed is a scene and there was scene element parsed before (which is incorrect)
		if (data->isSceneActive)
		{
			std::cout << "invalid .xml file structure: \"scene\" cannot be child element of \"scene\" \n";
			//stop parsing and return nullptr from CMitsubaLoader::loadAsset
		}
		else
		{
			data->isSceneActive = true;
			data->scene.processAttributes(_atts);
		}
	}
	else
	{
		//if element being parsed is NOT a scene and there was scene element parsed before
		if (data->isSceneActive)
		{
			IElement* newElement = elementFactory(_el, _atts);

			if (newElement)
			{
				data->elements.push_back(newElement);
			}
			else
			{
				std::cout << "invalid .xml file structure: " << _el <<  " \n";
				//stop parsing and return nullptr from CMitsubaLoader::loadAsset
			}
				
		}
		else
		{
			std::cout << "invalid .xml file structure: there is no scene as a root element \n";
			//stop parsing and return nullptr from CMitsubaLoader::loadAsset
		}
	}

	//XML_StopParser(data->parser, false);
}

static void elementHandlerEnd(void* _data, const char* _el)
{
	ParserData* data = static_cast<ParserData*>(_data);
	
	if (!std::strcmp(_el, "scene"))
	{
		if (!data->isSceneActive) //this scenario is not possible to happen i think (parser would return with XML_STATUS_ERROR anyway)
		{
			assert(false);
			std::cout << "invalid .xml file structure: wat? \n";
			//stop parsing and return nullptr from CMitsubaLoader::loadAsset
		}
		else
		{
			data->scene.onEndTag(data->assetManager, nullptr);
			data->isSceneActive = false;
		}
	}
	else
	{
		//here, array of elements should have at least one element
		assert(!data->elements.empty());

		IElement* element = *(data->elements.end() - 1);
		data->elements.pop_back();

		//
		IElement* parent  = (data->elements.empty()) ? &(data->scene) : *(data->elements.end() - 1);

		element->onEndTag(data->assetManager, parent);
		delete element;
	}
}

CMitsubaLoader::CMitsubaLoader(IrrlichtDevice* device)
	:m_device(device),
	m_assetManager(device->getAssetManager())
{
#ifdef _IRR_DEBUG
	setDebugName("CMitsubaLoader");
#endif
}

bool CMitsubaLoader::isALoadableFileFormat(io::IReadFile* _file) const
{
	//not implemented
	_IRR_DEBUG_BREAK_IF(true);
	return true;
}

const char** CMitsubaLoader::getAssociatedFileExtensions() const
{
	static const char* ext[]{ "xml", nullptr };
	return ext;
}

asset::IAsset* CMitsubaLoader::loadAsset(io::IReadFile* _file, const SAssetLoadParams& _params, IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	//TODO: error handling
	XML_Parser parser = XML_ParserCreate(nullptr);
	XML_SetElementHandler(parser, elementHandlerStart, elementHandlerEnd);

	ParserData data(m_assetManager, parser);
	
	//from now data (instance of ParserData struct) will be visible to expat handlers
	XML_SetUserData(parser, &data);

	const char* buff = new char[_file->getSize()];

	//will replace it with something like safeRead from CBAWMeshLoader
	_file->read((void*)buff, _file->getSize());

	XML_Status parseStatus = XML_Parse(parser, buff, _file->getSize(), 0);

	switch (parseStatus)
	{
	case XML_STATUS_ERROR:
		std::cout << "Parse status: XML_STATUS_ERROR\n";
		break;

	case XML_STATUS_OK:
		std::cout << "Parse status: XML_STATUS_OK\n";
		break;

	case XML_STATUS_SUSPENDED:
		std::cout << "Parse status: XML_STATUS_SUSPENDED\n";
		break;

	default:
		std::cout << "Parse status: XML_STATUS_SUSPENDED\n";
		break;

	}

	XML_ParserFree(parser);
	return data.scene.releaseMesh();
}

}
}
}