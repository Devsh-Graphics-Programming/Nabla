// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COMPLEX_INCLUDED_
#define _NBL_BUILTIN_HLSL_COMPLEX_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/functional.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>

using namespace nbl::hlsl;

// -------------------------------------- CPP VERSION ------------------------------------
#ifndef __HLSL_VERSION

#include <complex>

namespace nbl
{
namespace hlsl
{

template<typename Scalar>
struct complex_t : public std::complex<Scalar>
{
    using base_t = std::complex<Scalar>;
    constexpr complex_t(const Scalar real = Scalar(), const Scalar imag = Scalar()) : base_t(real, imag) {}
    static constexpr complex_t create(const Scalar real, const Scalar imag)
    {
        complex_t retVal(real, imag);
        return retVal;
    }
};

}
}

// -------------------------------------- END CPP VERSION ------------------------------------

// -------------------------------------- HLSL VERSION ---------------------------------------
#else

namespace nbl
{
namespace hlsl
{

// TODO: make this BDA compatible (no unspecialized templates yet)
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
    using type_t = complex_t<Scalar>;

    complex_t<Scalar> operator()(NBL_CONST_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(complex_t<Scalar>) rhs) 
    {
        return lhs + rhs;                                                             
    }

    NBL_CONSTEXPR_STATIC_INLINE complex_t<Scalar> identity;                                          
};

template<typename Scalar> 
struct minus< complex_t<Scalar> > 
{
    using type_t = complex_t<Scalar>;

    complex_t<Scalar> operator()(NBL_CONST_REF_ARG(complex_t<Scalar>) lhs, NBL_CONST_REF_ARG(complex_t<Scalar>) rhs) 
    {
        return lhs - rhs;                                                             
    }

    NBL_CONSTEXPR_STATIC_INLINE complex_t<Scalar> identity;                                          
};

template<typename Scalar> 
struct multiplies< complex_t<Scalar> > 
{
    using type_t = complex_t<Scalar>;

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
    using type_t = complex_t<Scalar>;

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

#define COMPLEX_ARITHMETIC_IDENTITIES(SCALAR, COMPONENT) \
template<> \
const static complex_t< SCALAR > plus< complex_t< SCALAR > >::identity = { promote< SCALAR, COMPONENT>(0), promote< SCALAR, COMPONENT>(0)}; \
template<> \
const static complex_t< SCALAR > minus< complex_t< SCALAR > >::identity = { promote< SCALAR, COMPONENT>(0),  promote< SCALAR, COMPONENT>(0)}; \
template<> \
const static complex_t< SCALAR > multiplies< complex_t< SCALAR > >::identity = { promote< SCALAR, COMPONENT>(1),  promote< SCALAR, COMPONENT>(0)}; \
template<> \
const static complex_t< SCALAR > divides< complex_t< SCALAR > >::identity = { promote< SCALAR, COMPONENT>(1),  promote< SCALAR, COMPONENT>(0)};

COMPLEX_ARITHMETIC_IDENTITIES(float16_t, float16_t)
COMPLEX_ARITHMETIC_IDENTITIES(float16_t2, float16_t)
COMPLEX_ARITHMETIC_IDENTITIES(float16_t3, float16_t)
COMPLEX_ARITHMETIC_IDENTITIES(float16_t4, float16_t)  
COMPLEX_ARITHMETIC_IDENTITIES(float32_t, float32_t)
COMPLEX_ARITHMETIC_IDENTITIES(float32_t2, float32_t)
COMPLEX_ARITHMETIC_IDENTITIES(float32_t3, float32_t)
COMPLEX_ARITHMETIC_IDENTITIES(float32_t4, float32_t)  
COMPLEX_ARITHMETIC_IDENTITIES(float64_t, float64_t)
COMPLEX_ARITHMETIC_IDENTITIES(float64_t2, float64_t)
COMPLEX_ARITHMETIC_IDENTITIES(float64_t3, float64_t)
COMPLEX_ARITHMETIC_IDENTITIES(float64_t4, float64_t)

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
    complex_t<Scalar> retVal = {r * nbl::hlsl::cos<Scalar>(theta), r * nbl::hlsl::sin<Scalar>(theta)};
    return retVal;
}

// ---------------------------------------------------------------------------------- PRIMITIVE ROOTS OF UNITY ----------------------------------------------------------------------------
// These are the first primitive roots of unity `e^(2pi/N)` for different PoT values of N, used by the FFT. Precomputed this way due to lack of consteval in HLSL. "Inverse" returns the conjugate `e^(-2pi/N)`
// TODO: Can use consteval in cpp so the definition can be different there idk.

template<uint16_t logN, bool Inverse, typename Scalar>
NBL_CONSTEXPR_STATIC complex_t<Scalar> PrimitiveRootOfUnity;

// ------------------------------------------------------------------------------------------ logN = 1 ----------------------------------------------------------------------------------------

#define LOGN_1_ROOT(SCALAR)\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<1, false, SCALAR > = { SCALAR (-1), SCALAR (0)};\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<1, true, SCALAR > = { SCALAR (-1), SCALAR (0)};

LOGN_1_ROOT(float16_t)
LOGN_1_ROOT(float32_t)
LOGN_1_ROOT(float64_t)

#undef LOGN_1_ROOT

// ------------------------------------------------------------------------------------------ logN = 2 ----------------------------------------------------------------------------------------

#define LOGN_2_ROOT(SCALAR)\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<2, false, SCALAR > = {SCALAR(0), SCALAR(-1)};\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<2, true, SCALAR > = {SCALAR(0), SCALAR(1)};

LOGN_2_ROOT(float16_t)
LOGN_2_ROOT(float32_t)
LOGN_2_ROOT(float64_t)

#undef LOGN_2_ROOT

// ------------------------------------------------------------------------------------------ logN = 3 ----------------------------------------------------------------------------------------

#define LOGN_3_ROOT(SCALAR)\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<3, false, SCALAR > = {SCALAR(0.70710678118654757), SCALAR(-0.70710678118654757)};\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<3, true, SCALAR > = {SCALAR(0.70710678118654757), SCALAR(0.70710678118654757)};

LOGN_3_ROOT(float16_t)
LOGN_3_ROOT(float32_t)
LOGN_3_ROOT(float64_t)

#undef LOGN_3_ROOT

// ------------------------------------------------------------------------------------------ logN = 4 ----------------------------------------------------------------------------------------

#define LOGN_4_ROOT(SCALAR)\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<4, false, SCALAR > = {SCALAR(0.92387953251128674), SCALAR(-0.38268343236508978)};\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<4, true, SCALAR > = {SCALAR(0.92387953251128674), SCALAR(0.38268343236508978)};

LOGN_4_ROOT(float16_t)
LOGN_4_ROOT(float32_t)
LOGN_4_ROOT(float64_t)

#undef LOGN_4_ROOT

// ------------------------------------------------------------------------------------------ logN = 5 -----------------------------------------------------------------------------------------

#define LOGN_5_ROOT(SCALAR)\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<5, false, SCALAR > = {SCALAR(0.98078528040323043), SCALAR(-0.19509032201612825)};\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<5, true, SCALAR > = {SCALAR(0.98078528040323043), SCALAR(0.19509032201612825)};

LOGN_5_ROOT(float16_t)
LOGN_5_ROOT(float32_t)
LOGN_5_ROOT(float64_t)

#undef LOGN_5_ROOT

// ------------------------------------------------------------------------------------------ logN = 6 ----------------------------------------------------------------------------------------

#define LOGN_6_ROOT(SCALAR)\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<6, false, SCALAR > = {SCALAR(0.99518472667219693), SCALAR(-0.098017140329560604)};\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<6, true, SCALAR > = {SCALAR(0.99518472667219693), SCALAR(0.098017140329560604)};

LOGN_6_ROOT(float16_t)
LOGN_6_ROOT(float32_t)
LOGN_6_ROOT(float64_t)

#undef LOGN_6_ROOT

// ------------------------------------------------------------------------------------------ logN = 7 ----------------------------------------------------------------------------------------

#define LOGN_7_ROOT(SCALAR)\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<7, false, SCALAR > = {SCALAR(0.99879545620517241), SCALAR(-0.049067674327418015)};\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<7, true, SCALAR > = {SCALAR(0.99879545620517241), SCALAR(0.049067674327418015)};

LOGN_7_ROOT(float16_t)
LOGN_7_ROOT(float32_t)
LOGN_7_ROOT(float64_t)

#undef LOGN_7_ROOT

// ------------------------------------------------------------------------------------------ logN = 8 ----------------------------------------------------------------------------------------

#define LOGN_8_ROOT(SCALAR)\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<8, false, SCALAR > = {SCALAR(0.99969881869620425), SCALAR(-0.024541228522912288)};\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<8, true, SCALAR > = {SCALAR(0.99969881869620425), SCALAR(0.024541228522912288)};

LOGN_8_ROOT(float16_t)
LOGN_8_ROOT(float32_t)
LOGN_8_ROOT(float64_t)

#undef LOGN_8_ROOT

// ------------------------------------------------------------------------------------------ logN = 9 ----------------------------------------------------------------------------------------

#define LOGN_9_ROOT(SCALAR)\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<9, false, SCALAR > = {SCALAR(0.9999247018391445), SCALAR(-0.012271538285719925)};\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<9, true, SCALAR > = {SCALAR(0.9999247018391445), SCALAR(0.012271538285719925)};

LOGN_9_ROOT(float16_t)
LOGN_9_ROOT(float32_t)
LOGN_9_ROOT(float64_t)

#undef LOGN_9_ROOT

// ------------------------------------------------------------------------------------------ logN = 10 ----------------------------------------------------------------------------------------

#define LOGN_10_ROOT(SCALAR)\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<10, false, SCALAR > = {SCALAR(0.99998117528260111), SCALAR(-0.0061358846491544753)};\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<10, true, SCALAR > = {SCALAR(0.99998117528260111), SCALAR(0.0061358846491544753)};

LOGN_10_ROOT(float16_t)
LOGN_10_ROOT(float32_t)
LOGN_10_ROOT(float64_t)

#undef LOGN_10_ROOT

// ------------------------------------------------------------------------------------------ logN = 11 ----------------------------------------------------------------------------------------

#define LOGN_11_ROOT(SCALAR)\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<11, false, SCALAR > = {SCALAR(0.99999529380957619), SCALAR(-0.0030679567629659761)};\
template<>\
NBL_CONSTEXPR_STATIC complex_t< SCALAR > PrimitiveRootOfUnity<11, true, SCALAR > = {SCALAR(0.99999529380957619), SCALAR(0.0030679567629659761)};

LOGN_11_ROOT(float16_t)
LOGN_11_ROOT(float32_t)
LOGN_11_ROOT(float64_t)

#undef LOGN_11_ROOT

} //namespace hlsl
} //namespace nbl

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


// -------------------------------------- END HLSL VERSION ---------------------------------------
#endif

// ---------------------------------------- COMMON ---------------------------------------

namespace nbl
{
namespace hlsl
{

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

// Fast square - I think a good optimizer does this anyway
template<typename Scalar>
complex_t<Scalar> square(NBL_CONST_REF_ARG(complex_t<Scalar>) value)
{
    Scalar real = value.real() * value.real() - value.imag() * value.imag();
    Scalar imag = 2 * value.real() * value.imag();
    complex_t<Scalar> retVal = { real, imag };
    return retVal;
}

} //namespace hlsl
} //namespace nbl

#endif
