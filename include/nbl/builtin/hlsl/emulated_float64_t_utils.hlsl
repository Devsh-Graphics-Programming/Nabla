#ifndef _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_UTILS_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_UTILS_INCLUDED_

#include <nbl/builtin/hlsl/emulated_float64_t.hlsl>

namespace nbl
{
namespace hlsl
{
// should i use this namespace?
//namespace ef64_util
//{
// TODO: this is mess, refactorize it
#ifndef __HLSL_VERSION
using ef64_t2 = float64_t2;
using ef64_t3 = float64_t3;
using ef64_t4 = float64_t4;
using ef64_t3x3 = float64_t3x3;
using ef64_t2x2 = float64_t4x4;
#else
struct ef64_t2
{
    emulated_float64_t<false, true> x;
    emulated_float64_t<false, true> y;

    emulated_float64_t<false, true> calcComponentSum() NBL_CONST_MEMBER_FUNC
    {
        return x + y;
    }

    NBL_CONSTEXPR_STATIC_INLINE ef64_t2 create(emulated_float64_t<false, true> x, emulated_float64_t<false, true> y)
    {
        ef64_t2 output;
        output.x = x;
        output.y = y;

        return output;
    }

    NBL_CONSTEXPR_STATIC_INLINE ef64_t2 create(float val)
    {
        ef64_t2 output;
        output.x = emulated_float64_t<false, true>::create(val);
        output.y = emulated_float64_t<false, true>::create(val);

        return output;
    }

    NBL_CONSTEXPR_STATIC_INLINE ef64_t2 create(float32_t2 val)
    {
        ef64_t2 output;
        output.x = emulated_float64_t<false, true>::create(val.x);
        output.y = emulated_float64_t<false, true>::create(val.y);

        return output;
    }

    NBL_CONSTEXPR_STATIC_INLINE ef64_t2 create(uint32_t2 val)
    {
        ef64_t2 output;
        output.x = emulated_float64_t<false, true>::create(val.x);
        output.y = emulated_float64_t<false, true>::create(val.y);

        return output;
    }

    ef64_t2 operator+(float rhs)
    {
        ef64_t2 output;
        emulated_float64_t<false, true> rhsAsEF64 = emulated_float64_t<false, true>::create(rhs);
        output.x = x + rhsAsEF64;
        output.y = y + rhsAsEF64;

        return output;
    }

    ef64_t2 operator+(emulated_float64_t<false, true> rhs)
    {
        ef64_t2 output;
        output.x = x + rhs;
        output.y = y + rhs;

        return output;
    }

    ef64_t2 operator+(ef64_t2 rhs)
    {
        ef64_t2 output;
        output.x = x + rhs.x;
        output.y = y + rhs.y;

        return output;
    }

    ef64_t2 operator-(float rhs)
    {
        return create(x, y) + (-rhs);
    }

    ef64_t2 operator-(emulated_float64_t<false, true> rhs)
    {
        return create(x, y) + (rhs.flipSign());
    }

    ef64_t2 operator-(ef64_t2 rhs)
    {
        rhs.x = rhs.x.flipSign();
        rhs.y = rhs.y.flipSign();
        return create(x, y) + rhs;
    }

    ef64_t2 operator*(float rhs)
    {
        ef64_t2 output;
        emulated_float64_t<false, true> rhsAsEF64 = emulated_float64_t<false, true>::create(rhs);
        output.x = x * rhsAsEF64;
        output.y = y * rhsAsEF64;

        return output;
    }

    ef64_t2 operator*(emulated_float64_t<false, true> rhs)
    {
        ef64_t2 output;
        output.x = x * rhs;
        output.y = y * rhs;

        return output;
    }

    ef64_t2 operator*(ef64_t2 rhs)
    {
        ef64_t2 output;
        output.x = x * rhs.x;
        output.y = y * rhs.y;

        return output;
    }

#ifdef __HLSL_VERSION
    float2 getAsFloat2()
    {
        return float2(x.getAsFloat32(), y.getAsFloat32());
    }
#endif
};

struct ef64_t3
{
    emulated_float64_t<false, true> x;
    emulated_float64_t<false, true> y;
    emulated_float64_t<false, true> z;

    NBL_CONSTEXPR_STATIC_INLINE ef64_t3 create(NBL_REF_ARG(ef64_t3) other)
    {
        ef64_t3 output;

        output.x = other.x;
        output.y = other.y;
        output.z = other.z;

        return output;
    }

    NBL_CONSTEXPR_STATIC_INLINE ef64_t3 create(NBL_REF_ARG(ef64_t2) other, emulated_float64_t<false, true> z)
    {
        ef64_t3 output;

        output.x = other.x;
        output.y = other.y;
        output.z = z;

        return output;
    }

    NBL_CONSTEXPR_STATIC_INLINE ef64_t3 create(NBL_REF_ARG(ef64_t2) other, int z)
    {
        ef64_t3 output;

        output.x = other.x;
        output.y = other.y;
        output.z = emulated_float64_t<false, true>::create(z);

        return output;
    }

    emulated_float64_t<false, true> calcComponentSum() NBL_CONST_MEMBER_FUNC
    {
        return x + y + z;
    }

    ef64_t3 operator*(NBL_CONST_REF_ARG(ef64_t3) rhs) NBL_CONST_MEMBER_FUNC
    {
        ef64_t3 output;
        output.x = x * rhs.x;
        output.y = y * rhs.y;
        output.z = z * rhs.z;

        return output;
    }
};

struct ef64_t4
{
    emulated_float64_t<false, true> x;
    emulated_float64_t<false, true> y;
    emulated_float64_t<false, true> z;
    emulated_float64_t<false, true> w;
};

struct ef64_t3x3
{
    ef64_t3 columns[3];

    ef64_t3x3 getTransposed() NBL_CONST_MEMBER_FUNC
    {
        ef64_t3x3 output;

        output.columns[0].x = columns[0].x;
        output.columns[1].x = columns[0].y;
        output.columns[2].x = columns[0].z;

        output.columns[0].y = columns[1].x;
        output.columns[1].y = columns[1].y;
        output.columns[2].y = columns[1].z;

        output.columns[0].z = columns[2].x;
        output.columns[1].z = columns[2].y;
        output.columns[2].z = columns[2].z;

        return output;
    }

    ef64_t3x3 operator*(NBL_CONST_REF_ARG(ef64_t3x3) rhs) NBL_CONST_MEMBER_FUNC
    {
        ef64_t3x3 output;
        ef64_t3x3 lhsTransposed = getTransposed();

        output.columns[0].x = (lhsTransposed.columns[0] * rhs.columns[0]).calcComponentSum();
        output.columns[0].y = (lhsTransposed.columns[0] * rhs.columns[1]).calcComponentSum();
        output.columns[0].z = (lhsTransposed.columns[0] * rhs.columns[2]).calcComponentSum();

        output.columns[1].x = (lhsTransposed.columns[1] * rhs.columns[0]).calcComponentSum();
        output.columns[1].y = (lhsTransposed.columns[1] * rhs.columns[1]).calcComponentSum();
        output.columns[1].z = (lhsTransposed.columns[1] * rhs.columns[2]).calcComponentSum();

        output.columns[2].x = (lhsTransposed.columns[2] * rhs.columns[0]).calcComponentSum();
        output.columns[2].y = (lhsTransposed.columns[2] * rhs.columns[1]).calcComponentSum();
        output.columns[2].z = (lhsTransposed.columns[2] * rhs.columns[2]).calcComponentSum();

        // TODO: avoid transpose
        return output.getTransposed();
    }

    ef64_t3 operator*(NBL_CONST_REF_ARG(ef64_t3) rhs)
    {
        ef64_t3 output;
        ef64_t3x3 lhsTransposed = getTransposed();

        output.x = (columns[0] * rhs).calcComponentSum();
        output.y = (columns[1] * rhs).calcComponentSum();
        output.z = (columns[2] * rhs).calcComponentSum();

        return output;
    }
};

struct ef64_t2x2
{
    ef64_t2 columns[2];

    ef64_t2x2 getTransposed() NBL_CONST_MEMBER_FUNC
    {
        ef64_t2x2 output;

        output.columns[0].x = columns[0].x;
        output.columns[1].x = columns[0].y;

        output.columns[0].y = columns[1].x;
        output.columns[1].y = columns[1].y;

        return output;
    }

    ef64_t2x2 operator*(NBL_CONST_REF_ARG(ef64_t2x2) rhs) NBL_CONST_MEMBER_FUNC
    {
        ef64_t2x2 output;
        ef64_t2x2 lhsTransposed = getTransposed();

        output.columns[0].x = (lhsTransposed.columns[0] * rhs.columns[0]).calcComponentSum();
        output.columns[0].y = (lhsTransposed.columns[0] * rhs.columns[1]).calcComponentSum();

        output.columns[1].x = (lhsTransposed.columns[1] * rhs.columns[0]).calcComponentSum();
        output.columns[1].y = (lhsTransposed.columns[1] * rhs.columns[1]).calcComponentSum();

        return output.getTransposed();
    }

    ef64_t2 operator*(NBL_CONST_REF_ARG(ef64_t2) rhs)
    {
        ef64_t2 output;
        ef64_t2x2 lhsTransposed = getTransposed();

        output.x = (columns[0] * rhs).calcComponentSum();
        output.y = (columns[1] * rhs).calcComponentSum();

        return output;
    }
};

#endif

// struct VecT is solution to
// error: 'nbl::hlsl::emulated_float64_t<false, true>' cannot be used as a type parameter where a scalar is required
// using float_t2 = typename conditional<is_same<float_t, emulated_float64_t<false, true> >::value, ef64_t2, vector<float_t, 2> >::type;

// TODO: better solution

#ifndef __HLSL_VERSION
using F64_t = double;
#else
using F64_t = emulated_float64_t<false, true>;
#endif

template<typename T, uint16_t N>
struct VecT { using type = void; };

template<>
struct VecT<float, 2> { using type = vector<float, 2>; };
template<>
struct VecT<float, 3> { using type = vector<float, 3>; };
template<>
struct VecT<float, 4> { using type = vector<float, 4>; };

#ifndef __HLSL_VERSION
template<>
struct VecT<double, 2> { using type = float64_t2; };
template<>
struct VecT<double, 3> { using type = float64_t3; };
template<>
struct VecT<double, 4> { using type = float64_t4; };
#endif

template<>
struct VecT<emulated_float64_t<false, true>, 2> { using type = ef64_t2; };
template<>
struct VecT<emulated_float64_t<false, true>, 3> { using type = ef64_t3; };
template<>
struct VecT<emulated_float64_t<false, true>, 4> { using type = ef64_t4; };

template<typename T>
struct Mat2x2T { using type = void; };
template<>
struct Mat2x2T<float> { using type = float32_t2x2; };
#ifndef __HLSL_VERSION
template<>
struct Mat2x2T<double> { using type = float64_t2x2; };
#endif
template<>
struct Mat2x2T<emulated_float64_t<false, true> > { using type = ef64_t2x2; };

//}
}
}
#endif