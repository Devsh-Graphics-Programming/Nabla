#include "../../ext/MitsubaLoader/CElementColor.h"

#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/CElementTransform.h"
#include "../../ext/MitsubaLoader/CSimpleElement.h"

namespace irr { namespace ext { namespace MitsubaLoader {

bool CElementColor::processAttributes(const char** _atts)
{
	bool isNameSet = false;

	for (int i = 0; _atts[i]; i += 2)
	{
		if (!std::strcmp(_atts[i], "name"))
		{
			nameAttr = _atts[i + 1];
			isNameSet = true;
		}


		if (!std::strcmp(_atts[i], "value"))
		{
			bool isColorValid = false;
			std::tie(isColorValid, color) = retriveColorFromValueAttribute(_atts[i + 1]);

			if (!isColorValid)
			{
				_IRR_DEBUG_BREAK_IF(true);
				return false;
			}
		}
	}

	return isNameSet;
}

bool CElementColor::onEndTag(asset::IAssetManager& _assetManager, IElement* _parent)
{
	return _parent->processChildData(this);
}

std::pair<bool, core::vector3df_SIMD> CElementColor::retriveColorFromValueAttribute(std::string value)
{
	std::replace(value.begin(), value.end(), ',', ' ');

	float r = std::numeric_limits<float>::quiet_NaN();
	float g = std::numeric_limits<float>::quiet_NaN();
	float b = std::numeric_limits<float>::quiet_NaN();

	std::stringstream ss;
	ss << value;

	if (value[0] == '#')
	{
		//TODO: process hex rgb value
		_IRR_DEBUG_BREAK_IF(true);
		return std::make_pair(false, core::vector3df_SIMD());
	}
	else
	{
		ss >> r;
		ss >> g;
		ss >> b;

		if (!isnan(r) && isnan(g) && isnan(b))
		{
			return std::make_pair(true, core::vector3df_SIMD(r));
		}
		else if (!isnan(r) && !isnan(g) && !isnan(b))
		{
			return std::make_pair(true, core::vector3df_SIMD(r, g, b));
		}
		else if (isnan(r) && isnan(g) && isnan(b))
		{
			return std::make_pair(false, core::vector3df_SIMD());
		}
		else
		{
			return std::make_pair(false, core::vector3df_SIMD());
		}
	}
	

	
}

}
}
}