// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COMPLEX_INCLUDED_
#define _NBL_BUILTIN_HLSL_COMPLEX_INCLUDED_

#include <nbl/builtin/hlsl/binops.hlsl>

namespace nbl
{
namespace hlsl
{

template<typename Scalar>
struct complex_t
{
    Scalar m_real;
    Scalar m_imag;
            
    // ------------------------- Constructors ---------------------------------------
    static complex_t create(const Scalar real, const Scalar imag)
    {
            complex_t retVal = { real, imag };
            return retVal;
    }
            
    static complex_t create(const complex_t other)
    {
            return other;
    }
            
    // ------------------------- Member functions -------------------------------      
    Scalar real() {
        return m_real;
    }
            
    void real(const Scalar value)
    {
        m_real = value;
    }
            
    Scalar imag()
    {
        return m_imag;
    }
            
    void imag(const Scalar value)
    {
        m_imag = value;
    }
    // ------------------------- Arithmetic operators -------------------------------   
    complex_t operator+(const complex_t rhs)
    {
        complex_t result;
        result.m_real = m_real + rhs.m_real;
        result.m_imag = m_imag + rhs.m_imag;

        return result;
    }
            
    complex_t operator-(const complex_t rhs)
    {
        complex_t result;

        result.m_real = m_real - rhs.m_real;
        result.m_imag = m_imag - rhs.m_imag;

        return result;
    }

    complex_t operator*(const complex_t rhs)
    {
        complex_t result;

        result.m_real = m_real * rhs.m_real - m_imag * rhs.m_imag;
        result.m_imag = m_real * rhs.m_imag + m_imag * rhs.m_real;

        return result;
    }
            
    // multiply by scalar
    complex_t operator*(const Scalar scalar)
    {
        complex_t result;
        result.m_real = m_real * scalar;
        result.m_imag = m_imag * scalar;

        return result;
    }
            
    // Divide by scalar
    complex_t operator/(const Scalar scalar)
    {
        complex_t result;
        result.m_real = m_real / scalar;
        result.m_imag = m_imag / scalar;

        return result;
    }
            
    complex_t operator/(const complex_t rhs)
    {
        complex_t result;

        T denominator = rhs.m_real * rhs.m_real + rhs.m_imag * rhs.m_imag;
        result.m_real = (m_real * rhs.m_real + m_imag * rhs.m_imag) / denominator;
        result.m_imag = (m_imag * rhs.m_real - m_real * rhs.m_imag) / denominator;

        return result;
    }
            
    // ----------------- Relational operators -----------------------------
    bool operator==(const complex_t rhs)
    {
            return !(uint64_t(m_real) ^ uint64_t(rhs.m_real)) && !(uint64_t(m_imag) ^ uint64_t(rhs.m_imag));
    }
    bool operator!=(const complex_t rhs)
    {
            return uint64_t(m_real) ^ uint64_t(rhs.m_real) || uint64_t(m_imag) ^ uint64_t(rhs.m_imag);
    }
            
};

// ---------------------- Compound assignment functors ----------------------
template <typename Scalar>
struct ComplexCompoundHelper{              
    using add = assign_add_t<complex_t<Scalar> >;
    using subtract = assign_subtract_t<complex_t<Scalar> >;
    using mul = assign_mul_t<complex_t<Scalar> >;
    using div = assign_div_t<complex_t<Scalar> >;
};

template <typename Scalar>
using assign_add_complex = typename ComplexCompoundHelper<Scalar>::add;
template <typename Scalar>
using assign_subtract_complex = typename ComplexCompoundHelper<Scalar>::subtract;
template <typename Scalar>
using assign_mul_complex = typename ComplexCompoundHelper<Scalar>::mul;
template <typename Scalar>
using assign_div_complex = typename ComplexCompoundHelper<Scalar>::div;

// ---------------------- Non-member functions -----------------------------    
template<typename Scalar>
Scalar real(const complex_t<Scalar> c) {
    return c.m_real;
}
    
template<typename Scalar>
Scalar imag(const complex_t<Scalar> c) {
    return c.m_imag;
}
    
template<typename Scalar>
Scalar norm(const complex_t<Scalar> c) {
    return c.m_real * c.m_real + c.m_imag * c.m_imag;
}

template<typename Scalar>
Scalar abs(const complex_t<Scalar> c) {
    return sqrt(norm(c));
}

template<typename Scalar>
Scalar arg(const complex_t<Scalar> c) {
    return atan2(c.m_imag, c.m_real);
}

template<typename Scalar>
complex_t<Scalar> conj(const complex_t<Scalar> c) {
    complex_t<Scalar> retVal = {c.m_real, - c.m_imag};
    return retVal;
}

template<typename Scalar>
complex_t<Scalar> proj(const complex_t<Scalar> c) {
    Scalar den = norm(c) + Scalar(1.0);
    complex_t<Scalar> retVal = { (Scalar(2.0) * c.m_real) / den , (Scalar(2.0) * c.m_imag) / den};
    return retVal;
}

template<typename Scalar>
complex_t<Scalar> polar(const Scalar r, const Scalar theta) {
    complex_t<Scalar> retVal = {r * cos(theta), r * sin(theta)};
    return retVal;
}

}
}

#endif