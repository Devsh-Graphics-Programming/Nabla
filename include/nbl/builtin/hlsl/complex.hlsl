// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COMPLEX_INCLUDED_
#define _NBL_BUILTIN_HLSL_COMPLEX_INCLUDED_

#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/cpp_compat/promote.hlsl"

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

// ---------------------- STD arithmetic operators ------------------------
// Specializations of the structs found in functional.hlsl
// Could X-Macro, left as-is for readability
// These all have to be specialized because of the identity that can't be initialized inside the struct definition, 
// and the latter (mul, div) have two overloads for operator()

template<typename Scalar> 
struct plus< complex_t<Scalar> > 
{
    complex_t<Scalar> operator()(NBL_CONST_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(complex_t<Scalar>) rhs) 
    {
        return lhs + rhs;                                                             
    }

    NBL_CONSTEXPR_STATIC_INLINE complex_t<Scalar> identity;                                          
};

template<typename Scalar> 
struct minus< complex_t<Scalar> > 
{
    complex_t<Scalar> operator()(NBL_CONST_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(complex_t<Scalar>) rhs) 
    {
        return lhs - rhs;                                                             
    }

    NBL_CONSTEXPR_STATIC_INLINE complex_t<Scalar> identity;                                          
};

template<typename Scalar> 
struct multiplies< complex_t<Scalar> > 
{
    complex_t<Scalar> operator()(NBL_CONST_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(complex_t<Scalar>) rhs) 
    {
        return lhs * rhs;                                                             
    }

    complex_t<Scalar> operator()(NBL_CONST_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(Scalar) rhs) 
    {
        return lhs * rhs;                                                             
    }

    NBL_CONSTEXPR_STATIC_INLINE complex_t<Scalar> identity;                                          
};

template<typename Scalar> 
struct divides< complex_t<Scalar> > 
{
    complex_t<Scalar> operator()(NBL_CONST_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(complex_t<Scalar>) rhs) 
    {
        return lhs / rhs;                                                             
    }

    complex_t<Scalar> operator()(NBL_CONST_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(Scalar) rhs) 
    {
        return lhs / rhs;                                                             
    }

    NBL_CONSTEXPR_STATIC_INLINE complex_t<Scalar> identity;                                          
};


// Out of line generic initialization of static member data not yet supported so we X-Macro identities for Scalar types we want to support
// (left X-Macro here since it's pretty readable)

#define COMPLEX_ARITHMETIC_IDENTITIES(SCALAR) \
template<> \
const static complex_t< SCALAR > plus< complex_t< SCALAR > >::identity = { promote< SCALAR , uint32_t>(0), promote< SCALAR , uint32_t>(0)}; \
template<> \
const static complex_t< SCALAR > minus< complex_t< SCALAR > >::identity = { promote< SCALAR , uint32_t>(0),  promote< SCALAR , uint32_t>(0)}; \
template<> \
const static complex_t< SCALAR > multiplies< complex_t< SCALAR > >::identity = { promote< SCALAR , uint32_t>(1),  promote< SCALAR , uint32_t>(0)}; \
template<> \
const static complex_t< SCALAR > divides< complex_t< SCALAR > >::identity = { promote< SCALAR , uint32_t>(1),  promote< SCALAR , uint32_t>(0)};

COMPLEX_ARITHMETIC_IDENTITIES(float16_t)
COMPLEX_ARITHMETIC_IDENTITIES(float16_t2)
COMPLEX_ARITHMETIC_IDENTITIES(float16_t3)
COMPLEX_ARITHMETIC_IDENTITIES(float16_t4)  
COMPLEX_ARITHMETIC_IDENTITIES(float32_t)
COMPLEX_ARITHMETIC_IDENTITIES(float32_t2)
COMPLEX_ARITHMETIC_IDENTITIES(float32_t3)
COMPLEX_ARITHMETIC_IDENTITIES(float32_t4)  
COMPLEX_ARITHMETIC_IDENTITIES(float64_t)
COMPLEX_ARITHMETIC_IDENTITIES(float64_t2)
COMPLEX_ARITHMETIC_IDENTITIES(float64_t3)
COMPLEX_ARITHMETIC_IDENTITIES(float64_t4)

#undef COMPLEX_ARITHMETIC_IDENTITIES


// --------------------------------- Compound assignment operators ------------------------------------------
// Specializations of the structs found in functional.hlsl
// Could X-Macro these as well
// Once again the identity forces us to template specialize instead of using the generic version implied by functional.hlsl

template<typename Scalar> 
struct plus_assign< complex_t<Scalar> > 
{
    using type_t = complex_t<Scalar>;
    using base_t = plus<type_t>;
    base_t baseOp;
    void operator()(NBL_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(complex_t<Scalar>) rhs) 
    {
        lhs = baseOp(lhs, rhs);                                                             
    }

    NBL_CONSTEXPR_STATIC_INLINE complex_t<Scalar> identity;                                          
};

template<typename Scalar> 
struct minus_assign< complex_t<Scalar> > 
{
    using type_t = complex_t<Scalar>;
    using base_t = minus<type_t>;
    base_t baseOp;
    void operator()(NBL_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(complex_t<Scalar>) rhs) 
    {
        lhs = baseOp(lhs, rhs);                                                             
    }

    NBL_CONSTEXPR_STATIC_INLINE complex_t<Scalar> identity;                                          
};

template<typename Scalar> 
struct multiplies_assign< complex_t<Scalar> > 
{
    using type_t = complex_t<Scalar>;
    using base_t = multiplies<type_t>;
    base_t baseOp;
    void operator()(NBL_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(complex_t<Scalar>) rhs) 
    {
        lhs = baseOp(lhs, rhs);                                                           
    }
    
    void operator()(NBL_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(Scalar) rhs) 
    {
        lhs = baseOp(lhs, rhs);                                                           
    }

    NBL_CONSTEXPR_STATIC_INLINE complex_t<Scalar> identity;                                     
};

template<typename Scalar> 
struct divides_assign< complex_t<Scalar> > 
{
    using type_t = complex_t<Scalar>;
    using base_t = divides<type_t>;
    base_t baseOp;
    void operator()(NBL_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(complex_t<Scalar>) rhs) 
    {
        lhs = baseOp(lhs, rhs);                                                        
    }

    void operator()(NBL_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(Scalar) rhs) 
    {
        lhs = baseOp(lhs, rhs);                                                           
    }

    NBL_CONSTEXPR_STATIC_INLINE complex_t<Scalar> identity;                                        
};


// Once again have to do some ugly out of line initialization for each Scalar type we want to support

#define COMPLEX_COMPOUND_ASSIGN_IDENTITY(NAME, SCALAR) \
template<> \
const static complex_t< SCALAR > NAME##_assign< complex_t< SCALAR > >::identity = NAME < complex_t< SCALAR > >::identity;

#define COMPLEX_COMPOUND_ASSIGN_IDENTITIES(SCALAR) \
COMPLEX_COMPOUND_ASSIGN_IDENTITY(plus, SCALAR) \
COMPLEX_COMPOUND_ASSIGN_IDENTITY(minus, SCALAR) \
COMPLEX_COMPOUND_ASSIGN_IDENTITY(multiplies, SCALAR) \
COMPLEX_COMPOUND_ASSIGN_IDENTITY(divides, SCALAR)

COMPLEX_COMPOUND_ASSIGN_IDENTITIES(float16_t)
COMPLEX_COMPOUND_ASSIGN_IDENTITIES(float16_t2)
COMPLEX_COMPOUND_ASSIGN_IDENTITIES(float16_t3)
COMPLEX_COMPOUND_ASSIGN_IDENTITIES(float16_t4)
COMPLEX_COMPOUND_ASSIGN_IDENTITIES(float32_t)
COMPLEX_COMPOUND_ASSIGN_IDENTITIES(float32_t2)
COMPLEX_COMPOUND_ASSIGN_IDENTITIES(float32_t3)
COMPLEX_COMPOUND_ASSIGN_IDENTITIES(float32_t4)
COMPLEX_COMPOUND_ASSIGN_IDENTITIES(float64_t)
COMPLEX_COMPOUND_ASSIGN_IDENTITIES(float64_t2)
COMPLEX_COMPOUND_ASSIGN_IDENTITIES(float64_t3)
COMPLEX_COMPOUND_ASSIGN_IDENTITIES(float64_t4)

#undef COMPLEX_COMPOUND_ASSIGN_IDENTITIES
#undef COMPLEX_COMPOUND_ASSIGN_IDENTITY

// -------------------------------- Non-member functions --------------------------------------
    
template<typename Scalar>
Scalar real(const complex_t<Scalar> c) 
{
    return c.m_real;
}
    
template<typename Scalar>
Scalar imag(const complex_t<Scalar> c) 
{
    return c.m_imag;
}
    
template<typename Scalar>
Scalar norm(const complex_t<Scalar> c) 
{
    return c.m_real * c.m_real + c.m_imag * c.m_imag;
}

template<typename Scalar>
Scalar abs(const complex_t<Scalar> c) 
{
    return sqrt(norm(c));
}

template<typename Scalar>
Scalar arg(const complex_t<Scalar> c) 
{
    return atan2(c.m_imag, c.m_real);
}

template<typename Scalar>
complex_t<Scalar> conj(const complex_t<Scalar> c) 
{
    complex_t<Scalar> retVal = {c.m_real, - c.m_imag};
    return retVal;
}

template<typename Scalar>
complex_t<Scalar> proj(const complex_t<Scalar> c) 
{
    Scalar den = norm(c) + Scalar(1.0);
    complex_t<Scalar> retVal = { (Scalar(2.0) * c.m_real) / den , (Scalar(2.0) * c.m_imag) / den};
    return retVal;
}

template<typename Scalar>
complex_t<Scalar> polar(const Scalar r, const Scalar theta) 
{
    complex_t<Scalar> retVal = {r * cos(theta), r * sin(theta)};
    return retVal;
}


// --------------------------------------------- Some more functions that come in handy --------------------------------------
// Fast mul by i
template<typename Scalar>
complex_t<Scalar> rotateLeft(NBL_CONST_REF_ARG(complex_t<Scalar>) value)
{
    complex_t<Scalar> retVal = { -value.imag(), value.real() };
    return retVal;
}

// Fast mul by -i
template<typename Scalar>
complex_t<Scalar> rotateRight(NBL_CONST_REF_ARG(complex_t<Scalar>) value)
{
    complex_t<Scalar> retVal = { value.imag(), -value.real() };
    return retVal;
}

}
}

// due to lack of alignof and typeid in DXC, need C++03 style tricks
NBL_REGISTER_OBJ_TYPE(complex_t<float16_t>,::nbl::hlsl::alignment_of_v<float16_t>)
NBL_REGISTER_OBJ_TYPE(complex_t<float16_t2>,::nbl::hlsl::alignment_of_v<float16_t2>)
NBL_REGISTER_OBJ_TYPE(complex_t<float16_t3>,::nbl::hlsl::alignment_of_v<float16_t3>)
NBL_REGISTER_OBJ_TYPE(complex_t<float16_t4>,::nbl::hlsl::alignment_of_v<float16_t4>)
NBL_REGISTER_OBJ_TYPE(complex_t<float32_t>,::nbl::hlsl::alignment_of_v<float32_t>)
NBL_REGISTER_OBJ_TYPE(complex_t<float32_t2>,::nbl::hlsl::alignment_of_v<float32_t2>)
NBL_REGISTER_OBJ_TYPE(complex_t<float32_t3>,::nbl::hlsl::alignment_of_v<float32_t3>)
NBL_REGISTER_OBJ_TYPE(complex_t<float32_t4>,::nbl::hlsl::alignment_of_v<float32_t4>)
NBL_REGISTER_OBJ_TYPE(complex_t<float64_t>,::nbl::hlsl::alignment_of_v<float64_t>)
NBL_REGISTER_OBJ_TYPE(complex_t<float64_t2>,::nbl::hlsl::alignment_of_v<float64_t2>)
NBL_REGISTER_OBJ_TYPE(complex_t<float64_t3>,::nbl::hlsl::alignment_of_v<float64_t3>)
NBL_REGISTER_OBJ_TYPE(complex_t<float64_t4>,::nbl::hlsl::alignment_of_v<float64_t4>)

#endif
