#include "CElementMatrix.h"

#include "ParserUtil.h"
#include "CSimpleElement.h"

namespace irr { namespace ext { namespace MitsubaLoader {


bool CElementMatrix::processAttributes(const char** _atts)
{

	switch (type)
	{
	case CElementMatrix::Type::ARBITRARY:
		for (int i = 0; _atts[i]; i += 2)
		{
			if (!std::strcmp(_atts[i], "value"))
			{
				std::pair<bool, core::matrix4SIMD> result = getMatrixFromString(_atts[i + 1]);
				matrix = result.second;
				return result.first;
			}
			else
			{
				ParserLog::wrongAttribute(_atts[i], getLogName());
				false;
			}
		}
	case CElementMatrix::Type::TRANSLATION:
		_IRR_DEBUG_BREAK_IF(true);
	case CElementMatrix::Type::ROTATION:
		_IRR_DEBUG_BREAK_IF(true);
	case CElementMatrix::Type::SCALE:
		_IRR_DEBUG_BREAK_IF(true);
	}
	//only type is an acceptable argument
	for (int i = 0; _atts[i]; i += 2)
	{
		if (std::strcmp(_atts[i], "type"))
		{
			ParserLog::wrongAttribute(_atts[i], getLogName());
			return false;
		}
	}

	return true;
}

bool CElementMatrix::onEndTag(asset::IAssetManager& _assetManager, IElement* _parent)
{
	return _parent->processChildData(this);
}

std::pair<bool, core::matrix4SIMD> CElementMatrix::getMatrixFromString(std::string _str) const
{
	std::replace(_str.begin(), _str.end(), ',', ' ');

	float matrixData[16];
	std::stringstream ss;
	ss << _str;

	for (int i = 0; i < 16; i++)
	{
		float f = std::numeric_limits<float>::quiet_NaN();
		ss >> f;
		std::cout << f << std::endl;

		matrixData[i] = f;

		if (isnan(f))
		{
			_IRR_DEBUG_BREAK_IF(true);
			ParserLog::mitsubaLoaderError("Invalid matrix specified.");
			return std::make_pair(false, core::matrix4SIMD());
		}
	}

	return std::make_pair(true, core::matrix4SIMD(matrixData));
}

}
}
}