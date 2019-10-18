// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_RATIONAL_H_INCLUDED__
#define __IRR_RATIONAL_H_INCLUDED__

#include <type_traits>

//#include "irr/core/math/irrMath.h"

namespace irr
{
namespace core
{

template<typename NumeratorType=int32_t, typename DenominatorType=typename std::make_unsigned<NumeratorType>::type>
class rational
{
		using this_type = rational<NumeratorType,DenominatorType>;

		template<typename OtherNumType, typename OtherDenType>
		friend class rational;
	public:
		rational() : rational(NumeratorType(0u)) {}
		rational(const NumeratorType& units) : rational(units,1u) {}
		rational(const NumeratorType& _numerator, const DenominatorType& _denominator) : numerator(_numerator), denominator(_denominator)
		{
			normalize();
		}

		template<typename OtherNumType, typename OtherDenType>
		rational(const rational<OtherNumType,OtherDenType>& other) : rational(other.numerator,other.denominator) {}


		inline this_type  operator*(const this_type& other) const
		{
			return this_type(numerator*other.numerator,denominator*other.denominator);
		}
		inline this_type& operator*=(const this_type& other)
		{
			return operator=(operator*(other));
		}

		inline this_type  operator*(NumeratorType val) const { return (*this) * this_type(val); }
		inline this_type& operator*=(NumeratorType val) { return ((*this) *= this_type(val)); }


		inline NumeratorType getIntegerApprox() const { return numerator/NumeratorType(denominator); }
		inline NumeratorType getRoundedUpInteger() const { return (numerator+NumeratorType(denominator)-NumeratorType(1u))/NumeratorType(denominator); }
		template<typename FloatType=float>
		inline FloatType getFloatApprox() const {return FloatType(numerator)/FloatType(denominator);}


		inline const NumeratorType& getNumerator() const {return numerator;}
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

} // end namespace core
} // end namespace irr

#endif

