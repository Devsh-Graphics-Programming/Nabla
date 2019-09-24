#ifndef __I_ELEMENT_H_INCLUDED__
#define __I_ELEMENT_H_INCLUDED__

#include "irr/asset/IAssetLoader.h"
#include "../../ext/MitsubaLoader/PropertyElement.h"


namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

class CGlobalMitsubaMetadata;

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
	public:
		std::string id;

		IElement(const char* _id) : id(_id) {}
		virtual ~IElement() = default;
	
		virtual IElement::Type getType() const = 0;
		virtual std::string getLogName() const = 0;

		virtual bool addProperty(SPropertyElementData&& _property) = 0;
		virtual bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CGlobalMitsubaMetadata* globalMetadata) = 0;
		//! default implementation for elements that doesnt have any children
		virtual bool processChildData(IElement* _child)
		{
			return !_child;
		}
		//

		static inline bool getTypeString(std::add_lvalue_reference<const char*>::type outType, const char** _atts)
		{
			const char* thrownAwayID;
			return getTypeAndIDStrings(outType,thrownAwayID,_atts);
		}
		static inline bool getTypeAndIDStrings(std::add_lvalue_reference<const char*>::type outType, std::add_lvalue_reference<const char*>::type outID, const char** _atts)
		{
			outType = nullptr;
			outID = nullptr;
			if (areAttributesInvalid(_atts,2u))
				return false;
			if (core::strcmpi(_atts[0],"type"))
			{
				if (core::strcmpi(_atts[2],"type"))
					return false;
				outType = _atts[3];
				if (core::strcmpi(_atts[0], "id"))
					outID = _atts[1];
			}
			else
			{
				outType = _atts[1];
				if (core::strcmpi(_atts[2],"id"))
					outID = _atts[3];
			}
			return true;
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

			return i < minAttrCount || (i % 2u == 0u);
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
}
}

#endif