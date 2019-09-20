#include "CElementTransform.h"

#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/PropertyElement.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{


bool CElementTransform::processAttributes(const char** _atts)
{
	if (IElement::areAttributesInvalid(_atts, 0u))
		return false;

	if (_atts && _atts[0])
	{
		if (!core::strcmpi(_atts[0], "name"))
			name = _atts[1];
		else
			return false;
	}

	return true;
}

bool CElementTransform::onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override)
{
#ifdef NEW_MITSUBA
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
#endif
	return true;
}

}
}
}