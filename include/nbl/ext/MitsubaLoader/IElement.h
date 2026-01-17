// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_I_ELEMENT_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_I_ELEMENT_H_INCLUDED_


#include "nbl/core/algorithm/utility.h"
#include "nbl/asset/interchange/IAssetLoader.h"

#include "nbl/ext/MitsubaLoader/PropertyElement.h"


namespace nbl::ext::MitsubaLoader
{
class CMitsubaMetadata;

namespace impl
{
template<template<typename...> class Pred, typename... Args>
struct ToUnaryPred
{
	template<typename T>
	struct type : bool_constant<Pred<T,Args...>::value> {};
};

template<typename T, typename TypeList>
struct mpl_of_passing;
template<typename T, typename TypeList>
using mpl_of_passing_t = mpl_of_passing<T,TypeList>::type;
}

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

			EMISSION_PROFILE,

			//shapes
			BSDF,
			TEXTURE,

			// those that should really be properties
			TRANSFORM,
			ANIMATION
		};

		IElement(const char* _id) : id(_id ? _id:"") {}
		virtual ~IElement() = default;
	
		virtual IElement::Type getType() const = 0;
		virtual std::string getLogName() const = 0;

		virtual bool onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger) = 0;
		//! default implementation for elements that doesnt have any children
		virtual bool processChildData(IElement* _child, const std::string& name, system::logger_opt_ptr logger)
		{
			return !_child;
		}

		//
		static inline bool getTypeIDAndNameStrings(std::add_lvalue_reference_t<const char*> outType, std::add_lvalue_reference_t<const char*> outID, std::string& name, const char** _atts)
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
		static inline bool getIDAndName(std::add_lvalue_reference_t<const char*> id, std::string& name, const char** _atts)
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

		// if we used `variant` instead of union we could default implement this
		//template<typename Derived, typename... Variants>
		//static inline void defaultVisit(Derived* this)
		//{
		// generated switch / visit of `Variant`
		//}
		template<typename Derived>
		static inline void copyVariant(Derived* to, const Derived* from)
		{
			to->visit([from](auto& selfEl)->void
				{
					from->visit([&selfEl](const auto& otherEl)->void
						{
							if constexpr (std::is_same_v<std::decay_t<decltype(selfEl)>,std::decay_t<decltype(otherEl)>>)
								selfEl = otherEl;
						}
					);
				}
			);
		}
		
		// could move it to `nbl/builtin/hlsl/mpl`
		template<typename Type, Type... values>
		struct mpl_array
		{
			constexpr static inline Type data[] = { values... };
		};
		//
		template<typename Derived>
		struct AddPropertyCallback
		{
			using element_t = Derived;
			// TODO: list or map of supported variants (if `visit` is present)
			using func_t = bool(*)(Derived*,SNamedPropertyElement&&,const system::logger_opt_ptr);

			inline bool operator()(Derived* d, SNamedPropertyElement&& p, const system::logger_opt_ptr l) const {return func(d,std::move(p),l);}

			func_t func;
			// will usually point at 
			std::span<const typename Derived::Type> allowedVariantTypes = {};
		};
		template<typename Derived>
		using PropertyNameCallbackMap = core::unordered_map<core::string,AddPropertyCallback<Derived>,core::CaseInsensitiveHash,core::CaseInsensitiveEquals>;
		template<typename Derived>
		class AddPropertyMap
		{
				template<Derived::Type... types>
				inline void registerCallback(const SNamedPropertyElement::Type type, std::string&& propertyName, AddPropertyCallback<Derived> cb)
				{
					if constexpr (sizeof...(types))
						cb.allowedVariantTypes = mpl_array<Type,types...>::data;
					registerCallback(type,std::move(propertyName),cb);
				}

			public:
				using element_type = Derived;

				inline void registerCallback(const SNamedPropertyElement::Type type, std::string&& propertyName, const AddPropertyCallback<Derived>& cb)
				{
					auto [nameIt,inserted] = byPropertyType[type].emplace(std::move(propertyName),cb);
					assert(inserted);
				}
				template<template<typename...> class Pred, typename... Args>
				inline void registerCallback(const SNamedPropertyElement::Type type, std::string&& propertyName, AddPropertyCallback<Derived>::func_t cb)
				{
					AddPropertyCallback<Derived> callback = {.func=cb};
					using UnaryPred = impl::ToUnaryPred<Pred,Args...>;
					using passing_types = core::filter_t<typename UnaryPred::type,typename Derived::variant_list_t>;
					if constexpr (core::type_list_size_v<passing_types>)
						callback.allowedVariantTypes = impl::mpl_of_passing_t<typename Derived::Type,passing_types>::data;
					registerCallback(type,std::move(propertyName),callback);
				}

				std::array<PropertyNameCallbackMap<Derived>,SNamedPropertyElement::Type::INVALID> byPropertyType = {};
		};
		//
		template<typename Derived>
		struct ProcessChildCallback
		{
			using element_t = Derived;
			// TODO: list or map of supported variants (if `visit` is present)
			using func_t = bool(*)(Derived*,IElement* _child,const system::logger_opt_ptr);

			inline bool operator()(Derived* d, IElement* _child, const system::logger_opt_ptr l) const {return func(d,_child,l);}

			func_t func;
			// TODO: allowed IElement types
		};
		template<typename Derived>
		using ProcessChildCallbackMap = core::unordered_map<core::string,ProcessChildCallback<Derived>,core::CaseInsensitiveHash,core::CaseInsensitiveEquals>;

		// members
		std::string id;

	protected:
		static inline void setLimitedString(const std::string_view memberName, std::span<char> out, const SNamedPropertyElement& _property, const system::logger_opt_ptr logger)
		{
			auto len = strlen(_property.svalue);
			if (len>=out.size())
				logger.log(
					"String property assigned to %s is too long, max allowed length %d, is %d, property value: \"%s\"",
					system::ILogger::ELL_ERROR,memberName.data(),out.size(),len,_property.svalue
				);
			len = std::min(out.size()-1,len);
			memcpy(out.data(),_property.svalue,len);
			out[len] = 0;
		}
};

namespace impl
{
template<typename T, typename... V>
struct mpl_of_passing<T,core::type_list<V...>>
{
	using type = IElement::mpl_array<T,V::VariantType...>;
};
}

}
#endif