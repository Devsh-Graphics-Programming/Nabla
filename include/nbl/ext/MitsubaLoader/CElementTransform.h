// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_TRANSFORM_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_TRANSFORM_H_INCLUDED_


#include "nbl/ext/MitsubaLoader/IElement.h"


namespace nbl::ext::MitsubaLoader
{

class CElementTransform final : public IElement
{
	public:
		static AddPropertyMap<CElementTransform> compAddPropertyMap();

		inline CElementTransform() : IElement(""), matrix(1.f) {}
		inline  ~CElementTransform() {}

		inline bool onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger) override {return true;}

		constexpr static inline auto ElementType = IElement::Type::TRANSFORM;
		inline IElement::Type getType() const override { return ElementType; }
		inline std::string getLogName() const override { return "transform"; }
		/*
		inline CElementTransform& operator=(const CElementTransform& other)
		{
			IElement::operator=(other);
			matrix = other.matrix;
			return *this;
		}
		*/

		hlsl::float32_t4x4 matrix; // TODO: HLSL diagonal(1.f)
};

}
#endif