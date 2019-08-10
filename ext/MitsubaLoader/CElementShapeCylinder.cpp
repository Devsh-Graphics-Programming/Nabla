#include "../../ext/MitsubaLoader/CElementShapeCylinder.h"

#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/CElementTransform.h"
#include "../../ext/MitsubaLoader/CSimpleElement.h"

namespace irr { namespace ext { namespace MitsubaLoader {


bool CElementShapeCylinder::processAttributes(const char** _atts)
{
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

bool CElementShapeCylinder::onEndTag(asset::IAssetManager& _assetManager, IElement* _parent)
{
	const float cylinderHeight = (p0 - p1).getLengthAsFloat();

	mesh = _assetManager.getGeometryCreator()->createCylinderMesh(radius,cylinderHeight,32);
	if (!mesh)
		return false;

	core::vectorSIMDf vec = p1 - p0;
	vec /= vec.getLengthAsFloat();

	core::quaternion rotation = core::quaternion::rotationFromTo(core::vectorSIMDf(0.0f, 1.0f, 0.0f), vec);
	core::matrix4x3 rotationMatrix = rotation.getMatrix();
	
	core::vectorSIMDf r1;
	r1.x = rotationMatrix.getColumn(0).X;
	r1.y = rotationMatrix.getColumn(1).X;
	r1.z = rotationMatrix.getColumn(2).X;
	r1.w = rotationMatrix.getColumn(3).X;

	core::vectorSIMDf r2;
	r2.x = rotationMatrix.getColumn(0).Y;
	r2.y = rotationMatrix.getColumn(1).Y;
	r2.z = rotationMatrix.getColumn(2).Y;
	r2.w = rotationMatrix.getColumn(3).Y;

	core::vectorSIMDf r3;
	r3.x = rotationMatrix.getColumn(0).Z;
	r3.y = rotationMatrix.getColumn(1).Z;
	r3.z = rotationMatrix.getColumn(2).Z;
	r3.w = rotationMatrix.getColumn(3).Z;

	core::matrix4SIMD matrix(r1, r2, r3, core::vectorSIMDf(0.0f, 0.0f, 0.0f, 1.0f));
	matrix.setTranslation(p0);


	transform = matrix;// .getTransposed();

	if (flipNormalsFlag)
		flipNormals(_assetManager);

	return _parent->processChildData(this);
}

bool CElementShapeCylinder::processChildData(IElement* _child)
{
	switch (_child->getType())
	{
	case IElement::Type::TRANSFORM:
	{
		CElementTransform* transformElement = static_cast<CElementTransform*>(_child);

		if (transformElement->getName() == "toWorld")
			this->transform = static_cast<CElementTransform*>(_child)->getMatrix();
		else
			ParserLog::mitsubaLoaderError("Unqueried attribute '" + transformElement->getName() + "' in element 'shape'");

		return true;
	}
	case IElement::Type::BOOLEAN:
	{
		CElementBoolean* boolElement = static_cast<CElementBoolean*>(_child);
		const std::string  elementName = boolElement->getNameAttribute();

		if (elementName == "flipNormals")
		{
			flipNormalsFlag = boolElement->getValueAttribute();
		}
		else
		{
			//warning
			ParserLog::mitsubaLoaderError("Unqueried attribute " + elementName + " in element \"shape\"");
		}

		return true;
	}
	case IElement::Type::POINT:
	{
		CElementPoint* pointElement = static_cast<CElementPoint*>(_child);
		const std::string  elementName = pointElement->getNameAttribute();

		if (elementName == "p0")
		{
			p0 = pointElement->getValueAttribute();
		}
		else if (elementName == "p1")
		{
			p1 = pointElement->getValueAttribute();
		}
		else
		{
			//warning
			ParserLog::mitsubaLoaderError("Unqueried attribute " + elementName + " in element \"shape\"");
		}

		return true;
	}
	case IElement::Type::FLOAT:
	{
		CElementFloat* floatElement = static_cast<CElementFloat*>(_child);
		const std::string  elementName = floatElement->getNameAttribute();

		if (elementName == "radius")
		{
			radius = floatElement->getValueAttribute();
		}
		else
		{
			//warning
			ParserLog::mitsubaLoaderError("Unqueried attribute " + elementName + " in element \"shape\"");
		}

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