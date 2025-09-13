// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_CORE_RATIONAL_H_INCLUDED_
#define _NBL_CORE_RATIONAL_H_INCLUDED_

#include <type_traits>

//#include "nbl/core/math/irrMath.h"

namespace nbl::core
{


template<typename NumeratorType=int32_t, typename DenominatorType=typename std::make_unsigned<NumeratorType>::type>
class rational
{
		// TODO: detect the usage of an atomic as the type and strip it to its normal variable
		using non_atomic_numerator = NumeratorType;
		using non_atomic_denominator = DenominatorType;
		using this_type = rational<NumeratorType,DenominatorType>;
		// I'm lazy
		using comparison_promoted_type = int64_t;

		template<typename OtherNumType, typename OtherDenType>
		friend class rational;
	public:
		rational() : rational(NumeratorType(0u)) {}
		rational(non_atomic_numerator units) : rational(units,1u) {}
		rational(non_atomic_numerator _numerator, non_atomic_denominator _denominator) : numerator(_numerator), denominator(_denominator)
		{
			normalize();
		}

		template<typename OtherNumType, typename OtherDenType>
		rational(const rational<OtherNumType,OtherDenType>& other) : rational(other.numerator,other.denominator) {}

		inline bool operator!=(const this_type& other) const requires (sizeof(NumeratorType)+sizeof(DenominatorType)<=sizeof(comparison_promoted_type))
		{
			return comparison_promoted_type(numerator)*other.denominator!=comparison_promoted_type(other.numerator)*other.denominator;
		}
		inline bool operator==(const this_type& other) const requires (sizeof(NumeratorType)+sizeof(DenominatorType)<=sizeof(comparison_promoted_type))
		{
			return !operator==(other);
		}

		inline this_type  operator*(const this_type& other) const
		{
			return this_type(numerator*other.numerator,denominator*other.denominator);
		}
		inline this_type& operator*=(const this_type& other)
		{
			return operator=(operator*(other));
		}

		inline this_type  operator*(non_atomic_numerator val) const { return (*this) * this_type(val); }
		inline this_type& operator*=(non_atomic_numerator val) { return ((*this) *= this_type(val)); }


		inline non_atomic_numerator getIntegerApprox() const { return numerator/non_atomic_numerator(denominator); }
		inline non_atomic_numerator getRoundedUpInteger() const { return (numerator+non_atomic_numerator(denominator)-non_atomic_numerator(1u))/non_atomic_numerator(denominator); }
		template<typename FloatType=float>
		inline FloatType getFloatApprox() const {return FloatType(numerator)/FloatType(denominator);}


		inline NumeratorType& getNumerator() {return numerator;}
		inline const NumeratorType& getNumerator() const {return numerator;}
		inline DenominatorType& getDenominator() {return denominator;}
		inline const DenominatorType& getDenominator() const {return denominator;}
	protected:
		NumeratorType numerator;
		DenominatorType denominator;

		inline void normalize()
		{
			// nothing for now, would do gcd
		}
};

template<typename NumeratorType = int32_t, typename DenominatorType = typename std::make_unsigned<NumeratorType>::type>
rational<NumeratorType,DenominatorType> operator*(const NumeratorType& val, rational<NumeratorType, DenominatorType> const& _this) { return _this * val; }

} // end namespace nbl::core

#endif

