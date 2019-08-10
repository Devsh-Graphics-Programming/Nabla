#include "CElementTransform.h"

#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/CSimpleElement.h"

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
			ParserLog::wrongAttribute(_atts[i], getLogName());
			return false;
		}
	}

	return true;
}

bool CElementTransform::onEndTag(asset::IAssetManager& _assetManager, IElement* _parent)
{
	while (matrices.size())
	{
		resultMatrix = core::matrix4SIMD::concatenateBFollowedByA(resultMatrix, matrices.back());
		matrices.pop_back();
	}

	return _parent->processChildData(this);
}

bool CElementTransform::processChildData(IElement* _child)
{
	switch (_child->getType())
	{
	case IElement::Type::MATRIX:
	{
		matrices.emplace_back(static_cast<CElementMatrix*>(_child)->getMatrix());
		return true;
	}
	default:
		ParserLog::wrongChildElement(getLogName(), _child->getLogName());
		return false;
	}
}

}
}
}