// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/ext/MitsubaLoader/IElement.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"


// Return value is if there's no error during the setting once basic checks are done
// For when you want to do custom handling of when property with string NAME and SNamedPropertyElement::Type::PROP_TYPE is getting added to this_t
#define NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY(NAME,PROP_TYPE) retval.registerCallback(SNamedPropertyElement::Type::PROP_TYPE,NAME,{\
	.func=[](this_t* _this, SNamedPropertyElement&& _property, const system::logger_opt_ptr logger)->bool

// when you know that there's a member of `this_t` with identifier equal to NAME
#define NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(NAME,PROP_TYPE) NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY(#NAME,PROP_TYPE) \
		{\
			static_assert(SNamedPropertyElement::Type::PROP_TYPE!=SNamedPropertyElement::Type::STRING); \
			_this->NAME = _property.getProperty<SNamedPropertyElement::Type::PROP_TYPE>(); \
			return true; \
		} \
	} \
)

// Similar to `NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY` but for `this_t` which declare `variant_list_t` (list of union types)
// this adds a compile-time filter against the constraint, such that only variant types matching the constraint are visited.
// Useful when multiple variants derive from the same base struct, or have the same member.
#define NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED(NAME,PROP_TYPE,CONSTRAINT,...) retval.template registerCallback<CONSTRAINT __VA_OPT__(,) __VA_ARGS__>( \
	SNamedPropertyElement::Type::PROP_TYPE,NAME,[](this_t* _this, SNamedPropertyElement&& _property, const system::logger_opt_ptr logger)->bool

// TODO: docs
#define NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED(NAME,PROP_TYPE,CONSTRAINT,...) NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED(#NAME,PROP_TYPE,CONSTRAINT  __VA_OPT__(,) __VA_ARGS__) \
	{\
		bool success = false; \
		_this->visit([&_property,logger,&success](auto& state)->void \
			{ \
				if constexpr (CONSTRAINT<std::remove_reference_t<decltype(state)> __VA_OPT__(,) __VA_ARGS__>::value) \
				{

#define NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED_END \
				} \
			} \
		); \
		return success; \
	} \
)

// This it to `NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED` what `NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY` is to `NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY`
// So basically you know the member is the same across the constraint filtered types
#define NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(NAME,PROP_TYPE,CONSTRAINT,...) NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED(NAME,PROP_TYPE,CONSTRAINT  __VA_OPT__(,) __VA_ARGS__) \
					static_assert(SNamedPropertyElement::Type::PROP_TYPE!=SNamedPropertyElement::Type::STRING); \
					state. ## NAME = _property.getProperty<SNamedPropertyElement::Type::PROP_TYPE>(); \
					success = true; \
NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_VARIANT_PROPERTY_CONSTRAINED_END


// just to reverse `is_base_of`
namespace nbl::ext::MitsubaLoader
{
template<typename D, typename B>
struct derived_from : std::is_base_of<B,D> {};
}

#define NBL_EXT_MITSUBA_LOADER_ELEMENT_INVALID_TYPE_CHECK(NON_FATAL) if (type==Type::INVALID) \
{ \
	invalidXMLFileStructure(logger,getLogName()+": type not specified"); \
	return NON_FATAL; \
}