// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_R_FILTER_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_R_FILTER_H_INCLUDED_


#include "nbl/ext/MitsubaLoader/PropertyElement.h"
#include "nbl/ext/MitsubaLoader/IElement.h"


namespace nbl::ext::MitsubaLoader
{


class CElementRFilter final : public IElement
{
	public:
		enum Type : uint8_t
		{
			INVALID,
			BOX,
			TENT,
			GAUSSIAN,
			MITCHELL,
			CATMULLROM,
			LANCZOS
		};

		struct Gaussian
		{
			constexpr static inline Type VariantType = Type::GAUSSIAN;

			float sigma = NAN; // can't look at mitsuba source to figure out the default it uses
		};
		struct MitchellNetravali
		{
			constexpr static inline Type VariantType = Type::MITCHELL;

			float B = 1.f / 3.f;
			float C = 1.f / 3.f;
		};
		struct CatmullRom
		{
			constexpr static inline Type VariantType = Type::CATMULLROM;

			float B = 1.f / 3.f;
			float C = 1.f / 3.f;
		};
		struct LanczosSinc
		{
			constexpr static inline Type VariantType = Type::LANCZOS;

			int32_t lobes = 3;
		};

		using variant_list_t = core::type_list<
			Gaussian,
			MitchellNetravali,
			CatmullRom,
			LanczosSinc
		>;
		static inline core::unordered_map<core::string,Type,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> compStringToTypeMap()
		{
			return {
				std::make_pair("box", Type::BOX),
				std::make_pair("tent", Type::TENT),
				std::make_pair("gaussian", Type::GAUSSIAN),
				std::make_pair("mitchell", Type::MITCHELL),
				std::make_pair("catmullrom", Type::CATMULLROM),
				std::make_pair("lanczos", Type::LANCZOS)
			};
		}
		static AddPropertyMap<CElementRFilter> compAddPropertyMap();

		inline CElementRFilter(const char* id) : IElement(id), type(GAUSSIAN)
		{
			gaussian = Gaussian();
		}
		inline ~CElementRFilter() {}

		template<typename Visitor>
		inline void visit(Visitor&& visitor)
		{
			switch (type)
			{
				case Type::BOX:
					[[fallthrough]];
				case Type::TENT:
					break;
				case Type::GAUSSIAN:
					visit(gaussian);
					break;
				case Type::MITCHELL:
					visit(mitchell);
					break;
				case Type::CATMULLROM:
					visit(catmullrom);
					break;
				case Type::LANCZOS:
					visit(lanczos);
					break;
				default:
					break;
			}
		}
		template<typename Visitor>
		inline void visit(Visitor&& visitor) const
		{
			const_cast<CElementRFilter*>(this)->visit([&]<typename T>(T& var)->void
				{
					visitor(const_cast<const T&>(var));
				}
			);
		}

		bool onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger) override;

		constexpr static inline auto ElementType = IElement::Type::RFILTER;
		inline IElement::Type getType() const override { return ElementType; }
		inline std::string getLogName() const override { return "rfilter"; }

		// make these public
		Type type;
		union
		{
			Gaussian			gaussian;
			MitchellNetravali	mitchell;
			CatmullRom			catmullrom;
			LanczosSinc			lanczos;
		};
		float kappa = 0.f;
		float Emin = core::nan<float>();
};


}
#endif