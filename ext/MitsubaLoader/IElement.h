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
			SAMPLER,

			FILM,
			SENSOR,
			EMITTER,

			//shapes
			SHAPE,
			INSTANCE,

			//other
			TRANSFORM,
			TEXTURE,
			MATERIAL,
			MEDIUM,
			INVALID,
		};

	public:
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