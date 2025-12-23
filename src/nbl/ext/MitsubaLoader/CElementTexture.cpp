// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/ext/MitsubaLoader/ParserUtil.h"
#include "nbl/ext/MitsubaLoader/CElementTexture.h"

#include "nbl/ext/MitsubaLoader/ElementMacros.h"

#include <functional>


namespace nbl::ext::MitsubaLoader
{
inline CElementTexture::Bitmap::WRAP_MODE getWrapMode(const SPropertyElementData& _property)
{
	using mode_e = CElementTexture::Bitmap::WRAP_MODE;
	static const core::unordered_map<std::string,mode_e,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> StringToWrap =
	{
		{"repeat",	mode_e::REPEAT},
		{"mirror",	mode_e::MIRROR},
		{"clamp",	mode_e::CLAMP},
		{"zero",	mode_e::ZERO},
		{"one",		mode_e::ONE}
	};
	assert(_property.type==SPropertyElementData::Type::STRING);
	auto found = StringToWrap.find(_property.getProperty<SPropertyElementData::Type::STRING>());
	if (found != StringToWrap.end())
		return found->second;
	return mode_e::REPEAT;
}

auto CElementTexture::compAddPropertyMap() -> AddPropertyMap<CElementTexture>
{
	using this_t = CElementTexture;
	AddPropertyMap<CElementTexture> retval;

	// bitmap
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED("filename",STRING,std::is_same,Bitmap)
		{
			setLimitedString("filename",_this->bitmap.filename,_property,logger); return true;
		}
	);
	// special
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED("wrapMode",STRING,std::is_same,Bitmap)
		{
			_this->bitmap.wrapModeV = _this->bitmap.wrapModeU = getWrapMode(_property);
			return true;
		}
	);
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED("wrapModeU",STRING,std::is_same,Bitmap)
		{
			_this->bitmap.wrapModeU = getWrapMode(_property);
			return true;
		}
	);
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED("wrapModeV",STRING,std::is_same,Bitmap)
		{
			_this->bitmap.wrapModeV = getWrapMode(_property);
			return true;
		}
	);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(gamma,FLOAT,std::is_same,Bitmap);
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED("filterType",STRING,std::is_same,Bitmap)
		{
			static const core::unordered_map<std::string,Bitmap::FILTER_TYPE,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> StringToType =
			{
				{"ewa",			Bitmap::FILTER_TYPE::EWA},
				{"trilinear",	Bitmap::FILTER_TYPE::TRILINEAR},
				{"nearest",		Bitmap::FILTER_TYPE::NEAREST}
			};
			auto found = StringToType.find(_property.getProperty<SPropertyElementData::Type::STRING>());
			if (found==StringToType.end())
				return false;
			_this->bitmap.filterType = found->second;
			return true;
		}
	);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(maxAnisotropy,FLOAT,std::is_same,Bitmap);
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED("cache",BOOLEAN,std::is_same,Bitmap)
		{
			return true; // silently drop
		}
	);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(uoffset,FLOAT,std::is_same,Bitmap);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(voffset,FLOAT,std::is_same,Bitmap);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(uscale,FLOAT,std::is_same,Bitmap);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(vscale,FLOAT,std::is_same,Bitmap);
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED("channel",STRING,std::is_same,Bitmap)
		{
			static const core::unordered_map<std::string,Bitmap::CHANNEL,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> StringToType =
			{
				{"r",	Bitmap::CHANNEL::R},
				{"g",	Bitmap::CHANNEL::G},
				{"b",	Bitmap::CHANNEL::B},
				{"a",	Bitmap::CHANNEL::A}/*,
				{"x",	Bitmap::CHANNEL::X},
				{"y",	Bitmap::CHANNEL::Y},
				{"z",	Bitmap::CHANNEL::Z}*/
			};
			auto found = StringToType.find(_property.getProperty<SPropertyElementData::Type::STRING>());
			if (found==StringToType.end())
				return false;
			_this->bitmap.channel = found->second;
			return true;
		}
	);

	// scale
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(scale,FLOAT,std::is_same,Scale);

	return retval;
}


bool CElementTexture::processChildData(IElement* _child, const std::string& name, system::logger_opt_ptr logger)
{
	if (!_child)
		return true;

	switch (_child->getType())
	{
		case IElement::Type::TEXTURE:
			{
				auto _texture = static_cast<CElementTexture*>(_child);
				switch (type)
				{
					case Type::SCALE:
						scale.texture = _texture;
						break;
					default:
						_NBL_DEBUG_BREAK_IF(true);
						logger.log("Only <texture type=\"scale\"> can have nested <texture> elements",system::ILogger::ELL_ERROR);
						return false;
				}
			}
			return true;
		default:
			break;
	}
	logger.log("<texture type=\"%d\"> does not support nested <%s> elements",system::ILogger::ELL_ERROR,type,_child->getLogName());
	return false;
}

bool CElementTexture::onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger)
{
	NBL_EXT_MITSUBA_LOADER_ELEMENT_INVALID_TYPE_CHECK(true);
	
	// TODO: Validation
	{
	}

	return true;
}

}