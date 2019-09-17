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

bool CElementTransform::onEndTag(asset::IAssetManager* _assetManager)
{
	for (auto& property : properties)
	{
		if (property.type == SPropertyElementData::Type::MATRIX)
		{
			matrix = core::concatenateBFollowedByA(matrix, CPropertyElementManager::retriveMatrix(property.value));
		}
		else
		if (property.type == SPropertyElementData::Type::TRANSLATE)
		{
			core::vector3df_SIMD translate = CPropertyElementManager::retriveVector(property.value);
			core::matrix4SIMD tmpMatrix;
			tmpMatrix.setTranslation(translate);

			matrix = core::concatenateBFollowedByA(tmpMatrix, matrix);
		}
		else
		if (property.type == SPropertyElementData::Type::ROTATE)
		{
			core::vectorSIMDf rotVec = CPropertyElementManager::retriveVector(property.value);
			core::quaternion rot(rotVec.x, rotVec.y, rotVec.z, rotVec.w);
			
			//not implemented yet
			_IRR_DEBUG_BREAK_IF(true);
			return false;
			//matrix = core::concatenateBFollowedByA(rot.getMatrix(), matrix);
		}
		else
		if (property.type == SPropertyElementData::Type::SCALE)
		{
			core::vector3df_SIMD scale = CPropertyElementManager::retriveVector(property.value);
			core::matrix4SIMD tmpMatrix;
			tmpMatrix.setScale(scale);

			matrix = core::concatenateBFollowedByA(tmpMatrix, matrix);
		}
		else
		if (property.type == SPropertyElementData::Type::LOOKAT)
		{
			core::matrix4SIMD tmpMatrix = CPropertyElementManager::retriveMatrix(property.value);

			matrix = core::concatenateBFollowedByA(tmpMatrix, matrix);
		}
		else
		{
			ParserLog::invalidXMLFileStructure("wat is this?");
			_IRR_DEBUG_BREAK_IF(true);
			return false;
		}
	}

	return true;
	//return _parent->processChildData(this);
}

}
}
}