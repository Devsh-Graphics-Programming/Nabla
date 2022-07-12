// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MITSUBA_LOADER_I_ELEMENT_H_INCLUDED_
#define _NBL_EXT_MITSUBA_LOADER_I_ELEMENT_H_INCLUDED_

#include "nbl/asset/interchange/IAssetLoader.h"
#include "nbl/ext/MitsubaLoader/PropertyElement.h"

namespace nbl::ext::MitsubaLoader
{


class CMitsubaMetadata;

class IElement
{
	public:
		enum class Type
		{
			INTEGRATOR,
			SENSOR,
			FILM,
			RFILTER,
			SAMPLER,

			SHAPE,
			INSTANCE,
			EMITTER,

			//shapes
			BSDF,
			TEXTURE,

			// those that should really be properties
			TRANSFORM,
			ANIMATION
		};

		std::string id;

		IElement(const char* _id) : id(_id ? _id:"") {}
		virtual ~IElement() = default;
	
		virtual IElement::Type getType() const = 0;
		virtual std::string getLogName() const = 0;

		virtual bool addProperty(SNamedPropertyElement&& _property) = 0;
		virtual bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CMitsubaMetadata* globalMetadata) = 0;
		//! default implementation for elements that doesnt have any children
		virtual bool processChildData(IElement* _child, const std::string& name)
		{
			return !_child;
		}
		//
		static inline bool getTypeIDAndNameStrings(std::add_lvalue_reference<const char*>::type outType, std::add_lvalue_reference<const char*>::type outID, std::string& name, const char** _atts)
		{
			outType = nullptr;
			outID = nullptr;
			name = "";
			if (areAttributesInvalid(_atts,2u))
				return false;

			while (*_atts)
			{
				if (core::strcmpi(_atts[0], "id") == 0)
					outID = _atts[1];
				else if (core::strcmpi(_atts[0], "type") == 0)
					outType = _atts[1];
				else if (core::strcmpi(_atts[0], "name") == 0)
					name = _atts[1];
				_atts += 2;
			}
			return outType;
		}
		static inline bool getIDAndName(std::add_lvalue_reference<const char*>::type id, std::string& name, const char** _atts)
		{
			const char* thrownAwayType;
			getTypeIDAndNameStrings(thrownAwayType,id,name,_atts);
			return id;
		}
		static inline bool areAttributesInvalid(const char** _atts, uint32_t minAttrCount)
		{
			if (!_atts)
				return true;

			uint32_t i = 0u;
			while (_atts[i])
			{
				i++;
			}

			return i < minAttrCount || (i % 2u);
		}
		static inline bool invalidAttributeCount(const char** _atts, uint32_t attrCount)
		{
			if (!_atts)
				return true;

			for (uint32_t i=0u; i<attrCount; i++)
			if (!_atts[i])
				return true;
			
			return _atts[attrCount];
		}
};

}

#endif