#include "CElementTransform.h"

#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/PropertyElement.h"

namespace irr { namespace ext { namespace MitsubaLoader {


bool CElementTransform::processAttributes(const char** _atts)
{
	//only type is an acceptable argument
	for (int i = 0; _atts[i]; i += 2)
	{
		if (!std::strcmp(_atts[i], "name"))
		{
			name = _atts[i + 1];
		}
		else
		{
			//ParserLog::wrongAttribute(_atts[i], getLogName());
			return false;
		}
	}

	return true;
}

bool CElementTransform::onEndTag(asset::IAssetManager& _assetManager)
{
	for (auto& property : properties)
	{
		if (property.type == SPropertyElementData::Type::MATRIX)
		{
			matrix = core::concatenateBFollowedByA(matrix, CPropertyElementManager::retriveMatrix(property.value));
		}
		else
		{
			ParserLog::invalidXMLFileStructure("wat is this?");
			return false;
		}
	}

	return true;
	//return _parent->processChildData(this);
}

}
}
}