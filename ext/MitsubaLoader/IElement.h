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

		//! default implementation for elements that doesnt have any attributes
		virtual bool processAttributes(const char** _atts)
		{
			if (_atts && _atts[0])
				return false;

			return true;
		}

		//! default implementation for elements that doesnt have any children
		virtual bool processChildData(IElement* _child)
		{
			return !_child;
		}
		//
	
		virtual bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override) = 0;
		virtual IElement::Type getType() const = 0;
		virtual std::string getLogName() const = 0;

		// TODO refactor
		virtual void addProperty(const SPropertyElementData& _property) = 0;

		virtual void addProperty(SPropertyElementData&& _property) = 0;

	protected:
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
};

}
}
}

#endif