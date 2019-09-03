#ifndef __PROPERTY_ELEMENT_H_INCLUDED__
#define __PROPERTY_ELEMENT_H_INCLUDED__

#include "irrlicht.h"
#include <string>

namespace irr { namespace ext { namespace MitsubaLoader {

	struct SPropertyElementData
	{
		enum class Type
		{
			FLOAT,
			INTEGER,
			BOOLEAN,
			STRING,
			RGB,
			SRGB,
			SPECTRUM,
			MATRIX,
			TRANSLATE,
			ROTATE,
			SCALE,
			LOOKAT,
			POINT,
			VECTOR
		};

		SPropertyElementData::Type type;
		std::string name;
		std::string value;
	};

class CPropertyElementManager
{
public:
	static std::pair<bool, SPropertyElementData> createPropertyData(const char* _el, const char** _atts);

	static float retriveFloatValue(const std::string& _data);
	static int retriveIntValue(const std::string& _data);
	static bool retriveBooleanValue(const std::string& _data);
	static core::matrix4SIMD retriveMatrix(const std::string& _data);
	static core::vectorSIMDf retriveVector(const std::string& _data);

private:
	static std::string findStandardValue(const char** _atts, bool& _errorOccurred, const core::vector<std::string>& _acceptableAttributes);
	static std::string findAndConvertXYZAttsToSingleString(const char** _atts, bool& _errorOccurred, const core::vector<std::string>& _acceptableAttributes);

};

}
}
}

#endif