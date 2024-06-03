// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COMPLEX_INCLUDED_
#define _NBL_BUILTIN_HLSL_COMPLEX_INCLUDED_

#include "nbl/builtin/hlsl/functional.hlsl"

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

        Scalar denominator = rhs.m_real * rhs.m_real + rhs.m_imag * rhs.m_imag;
        result.m_real = (m_real * rhs.m_real + m_imag * rhs.m_imag) / denominator;
        result.m_imag = (m_imag * rhs.m_real - m_real * rhs.m_imag) / denominator;

        return result;
    }
            
    // ----------------- Relational operators -----------------------------
    bool operator==(const complex_t rhs)
    {
            return m_real == rhs.m_real && m_imag == rhs.m_imag;
    }
    bool operator!=(const complex_t rhs)
    {
            return m_real != rhs.m_real || m_imag != rhs.m_imag;
    }
};

// ---------------------- Compound assign operators ------------------------
// Specializations of the structs found in functional.hlsl
// TODO: Figure out a way of making them have an identity (const static member struct doesn't compile on Godbolt)

template<typename Scalar> 
struct plus_assign< complex_t<Scalar> > {
    void operator()(NBL_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(complex_t<Scalar>) rhs) {
        lhs = lhs + rhs;                                                             
    }                                          
};

template<typename Scalar> 
struct multiplies_assign< complex_t<Scalar> > {
    void operator()(NBL_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(complex_t<Scalar>) rhs) {
        lhs = lhs * rhs;                                                             
    }

    void operator()(NBL_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(Scalar) rhs) {
        lhs = lhs * rhs;                                                             
    }                                          
};

template<typename Scalar> 
struct minus_assign< complex_t<Scalar> > {
    void operator()(NBL_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(complex_t<Scalar>) rhs) {
        lhs = lhs - rhs;                                                             
    }                                          
};

template<typename Scalar> 
struct divides_assign< complex_t<Scalar> > {
    void operator()(NBL_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(complex_t<Scalar>) rhs) {
        lhs = lhs / rhs;                                                             
    }

    void operator()(NBL_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(Scalar) rhs) {
        lhs = lhs / rhs;                                                             
    }                                          
};


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