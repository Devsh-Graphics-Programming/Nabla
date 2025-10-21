// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/ext/MitsubaLoader/CElementTransform.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"


namespace nbl::ext::MitsubaLoader
{

auto CElementTransform::compAddPropertyMap() -> AddPropertyMap<CElementTransform>
{
	using this_t = CElementTransform;
	AddPropertyMap<CElementTransform> retval;

	auto setMatrix = [](this_t* _this, SNamedPropertyElement&& _property, const system::logger_opt_ptr logger)->bool
	{
		_this->matrix = _property.mvalue;
		return true;
	};
	for (const auto& type : {
		SNamedPropertyElement::Type::MATRIX,
		SNamedPropertyElement::Type::TRANSLATE,
		SNamedPropertyElement::Type::ROTATE,
		SNamedPropertyElement::Type::SCALE,
		SNamedPropertyElement::Type::LOOKAT
	})
		retval.registerCallback(type,"",{.func=setMatrix});

	return retval;
}

}