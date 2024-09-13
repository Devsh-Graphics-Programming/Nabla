#ifndef _NBL_BUILTIN_HLSL_EMULATED_VECTOR_T_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_VECTOR_T_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/portable/float64_t.hlsl>

namespace nbl
{
namespace hlsl
{

template<typename EmulatedType, uint32_t N>
struct emulated_vector {};

template<typename EmulatedType>
struct emulated_vector<EmulatedType, 2>
{
    using this_t = emulated_vector<EmulatedType, 2>;

    EmulatedType x;
    EmulatedType y;

    EmulatedType calcComponentSum() NBL_CONST_MEMBER_FUNC
    {
        return x + y;
    }

    NBL_CONSTEXPR_STATIC_INLINE this_t create(EmulatedType x, EmulatedType y)
    {
        this_t output;
        output.x = x;
        output.y = y;

        return output;
    }

    this_t operator+(float rhs)
    {
        this_t output;
        EmulatedType rhsAsEF64 = EmulatedType::create(rhs);
        output.x = x + rhsAsEF64;
        output.y = y + rhsAsEF64;

        return output;
    }

    this_t operator+(EmulatedType rhs)
    {
        this_t output;
        output.x = x + rhs;
        output.y = y + rhs;

        return output;
    }

    this_t operator+(this_t rhs)
    {
        this_t output;
        output.x = x + rhs.x;
        output.y = y + rhs.y;

        return output;
    }

    this_t operator-(float rhs)
    {
        return create(x, y) + (-rhs);
    }

    this_t operator-(EmulatedType rhs)
    {
        return create(x, y) + (rhs.flipSign());
    }

    this_t operator-(this_t rhs)
    {
        rhs.x = rhs.x.flipSign();
        rhs.y = rhs.y.flipSign();
        return create(x, y) + rhs;
    }

    this_t operator*(float rhs)
    {
        this_t output;
        EmulatedType rhsAsEF64 = EmulatedType::create(rhs);
        output.x = x * rhsAsEF64;
        output.y = y * rhsAsEF64;

        return output;
    }

    this_t operator*(EmulatedType rhs)
    {
        this_t output;
        output.x = x * rhs;
        output.y = y * rhs;

        return output;
    }

    this_t operator*(this_t rhs)
    {
        this_t output;
        output.x = x * rhs.x;
        output.y = y * rhs.y;

        return output;
    }
};

template<typename EmulatedType>
struct emulated_vector<EmulatedType, 3>
{
    using this_t = emulated_vector<EmulatedType, 3>;

    EmulatedType x;
    EmulatedType y;
    EmulatedType z;

    EmulatedType calcComponentSum() NBL_CONST_MEMBER_FUNC
    {
        return x + y + z;
    }

    this_t operator*(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        this_t output;
        output.x = x * rhs.x;
        output.y = y * rhs.y;
        output.z = z * rhs.z;

        return output;
    }
};

template<typename EmulatedType>
struct emulated_vector<EmulatedType, 4>
{
    using type = emulated_vector<EmulatedType, 4>;

    EmulatedType x;
    EmulatedType y;
    EmulatedType z;
    EmulatedType w;
};

template<typename EmulatedType>
using emulated_vector_t2 = emulated_vector<EmulatedType, 2>;
template<typename EmulatedType>
using emulated_vector_t3 = emulated_vector<EmulatedType, 3>;
template<typename EmulatedType>
using emulated_vector_t4 = emulated_vector<EmulatedType, 4>;

namespace impl
{
template<typename To, typename From>
struct static_cast_helper<emulated_vector_t2<To>, vector<From, 2>, void>
{
    static inline emulated_vector_t2<To> cast(vector<From, 2> vec)
    {
        return emulated_vector_t2<To>(_static_cast<To, From>(vec.x), _static_cast<To, From>(vec.y));
    }
};

template<typename To, typename From>
struct static_cast_helper<emulated_vector_t3<To>, vector<From, 3>, void>
{
    static inline emulated_vector_t3<To> cast(vector<From, 3> vec)
    {
        return emulated_vector_t3<To>(_static_cast<To, From>(vec.x), _static_cast<To, From>(vec.y), _static_cast<To, From>(vec.z));
    }
};

template<typename To, typename From>
struct static_cast_helper<emulated_vector_t4<To>, vector<From, 4>, void>
{
    static inline emulated_vector_t4<To> cast(vector<From, 4> vec)
    {
        return emulated_vector_t4<To>(_static_cast<To, From>(vec.x), _static_cast<To, From>(vec.y), _static_cast<To, From>(vec.z), _static_cast<To, From>(vec.w));
    }
};
}

}
}
#endif