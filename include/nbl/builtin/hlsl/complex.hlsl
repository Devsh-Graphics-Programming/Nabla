// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COMPLEX_INCLUDED_
#define _NBL_BUILTIN_HLSL_COMPLEX_INCLUDED_



#ifndef __HLSL_VERSION


namespace nbl
{
namespace hlsl
{

    template<typename T>
    struct complex_t
    {
        T m_real;
        T m_imag;
            
        // ------------------------- Constructors ---------------------------------------
        static complex_t create(const T real, const T imag)
        {
                complex_t retVal = { real, imag };
                return retVal;
        }
            
        static complex_t create(const complex_t other)
        {
                return other;
        }
            
        // ------------------------- Member functions -------------------------------      
        T real() {
            return m_real;
        }
            
        void real(const T value)
        {
            m_real = value;
        }
            
        T imag()
        {
            return m_imag;
        }
            
        void imag(const T value)
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
        complex_t operator*(const T scalar)
        {
            complex_t result;
            result.m_real = m_real * scalar;
            result.m_imag = m_imag * scalar;

            return result;
        }
            
        // Divide by scalar
        complex_t operator/(const T scalar)
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

    // ---------------------- Non-member functions -----------------------------    
    template<typename T>
    T real(const complex_t<T> c) {
        return c.m_real;
    }
    
    template<typename T>
    T imag(const complex_t<T> c) {
        return c.m_imag;
    }
    
    template<typename T>
    T norm(const complex_t<T> c) {
        return c.m_real * c.m_real + c.m_imag * c.m_imag;
    }

    template<typename T>
    T abs(const complex_t<T> c) {
        return sqrt(norm(c));
    }

    template<typename T>
    T arg(const complex_t<T> c) {
        return atan2(c.m_imag, c.m_real);
    }

    template<typename T>
    complex_t<T> conj(const complex_t<T> c) {
        complex_t<T> retVal = {c.m_real, - c.m_imag};
        return retVal;
    }

    template<typename T>
    complex_t<T> proj(const complex_t<T> c) {
        T den = norm(c) + T(1.0);
        complex_t<T> retVal = { (T(2.0) * c.m_real) / den , (T(2.0) * c.m_imag) / den};
        return retVal;
    }

    template<typename T>
    complex_t<T> polar(const T r, const T theta) {
        complex_t<T> retVal = {r * cos(theta), r * sin(theta)};
        return retVal;
    }

}
}

#endif

#endif