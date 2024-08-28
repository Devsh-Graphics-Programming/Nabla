#ifndef _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_UTILS_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_UTILS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>
#include <nbl/builtin/hlsl/emulated_float64_t.hlsl>

namespace nbl
{
namespace hlsl
{
// TODO: enable
//template<typename device_capabilities = void, bool FastMath = false, bool FlushDenormToZero = true>
//using portable_float64_t = conditional_t<device_capability_traits<device_capabilities>::shaderFloat64, float64_t, typename emulated_float64_t<FastMath, FlushDenormToZero> >;

#ifndef __HLSL_VERSION
template<typename device_capabilities = void, bool FastMathIfEmulated = false, bool FlushDenormToZeroIfEmulated = true>
using portable_float64_t = typename conditional<true, float64_t, emulated_float64_t<FastMathIfEmulated, FlushDenormToZeroIfEmulated> >::type;
#else
template<typename device_capabilities = void, bool FastMathIfEmulated = false, bool FlushDenormToZeroIfEmulated = true>
using portable_float64_t = typename conditional<false, float64_t, emulated_float64_t<FastMathIfEmulated, FlushDenormToZeroIfEmulated> >::type;
#endif

template<typename EmulatedType, uint32_t N>
struct emulated_vector {};

template<typename EmulatedType>
struct emulated_vector<EmulatedType, 2>
{
    using type = emulated_vector<EmulatedType, 2>;

    EmulatedType x;
    EmulatedType y;

    EmulatedType calcComponentSum() NBL_CONST_MEMBER_FUNC
    {
        return x + y;
    }

    NBL_CONSTEXPR_STATIC_INLINE type create(EmulatedType x, EmulatedType y)
    {
        type output;
        output.x = x;
        output.y = y;

        return output;
    }

    type operator+(float rhs)
    {
        type output;
        EmulatedType rhsAsEF64 = EmulatedType::create(rhs);
        output.x = x + rhsAsEF64;
        output.y = y + rhsAsEF64;

        return output;
    }

    type operator+(EmulatedType rhs)
    {
        type output;
        output.x = x + rhs;
        output.y = y + rhs;

        return output;
    }

    type operator+(type rhs)
    {
        type output;
        output.x = x + rhs.x;
        output.y = y + rhs.y;

        return output;
    }

    type operator-(float rhs)
    {
        return create(x, y) + (-rhs);
    }

    type operator-(EmulatedType rhs)
    {
        return create(x, y) + (rhs.flipSign());
    }

    type operator-(type rhs)
    {
        rhs.x = rhs.x.flipSign();
        rhs.y = rhs.y.flipSign();
        return create(x, y) + rhs;
    }

    type operator*(float rhs)
    {
        type output;
        EmulatedType rhsAsEF64 = EmulatedType::create(rhs);
        output.x = x * rhsAsEF64;
        output.y = y * rhsAsEF64;

        return output;
    }

    type operator*(EmulatedType rhs)
    {
        type output;
        output.x = x * rhs;
        output.y = y * rhs;

        return output;
    }

    type operator*(type rhs)
    {
        type output;
        output.x = x * rhs.x;
        output.y = y * rhs.y;

        return output;
    }
};

template<typename EmulatedType>
struct emulated_vector<EmulatedType, 3>
{
    using type = emulated_vector<EmulatedType, 3>;

    EmulatedType x;
    EmulatedType y;
    EmulatedType z;

    EmulatedType calcComponentSum() NBL_CONST_MEMBER_FUNC
    {
        return x + y + z;
    }

    type operator*(NBL_CONST_REF_ARG(type) rhs) NBL_CONST_MEMBER_FUNC
    {
        type output;
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

// TODO: works only for float, fix
namespace impl
{

template<typename T, typename U, uint32_t Dim>
struct static_cast_helper<vector<T,Dim>,emulated_vector<U,Dim>,void>
{
    static inline vector<T,Dim> cast(emulated_vector<U,Dim> vec)
    {
        return vector<T,Dim>(_static_cast<T,U>(vec.x), _static_cast<T,U>(vec.y));
    }
};

}

//template<typename EmulatedType, uint32_t N, uint32_t M>
//struct emulated_matrix_base
//{
//    using vec_t = emulated_vector<EmulatedType, N>;
//    vec_t columns[M];
//};

template<typename EmulatedType, uint32_t N, uint32_t M>
struct emulated_matrix {}; // : emulated_matrix_base<EmulatedType, N, M> {};

template<typename EmulatedType>
struct emulated_matrix<EmulatedType, 2, 2>// : emulated_matrix_base<EmulatedType, 2, 2>
{
    using vec_t = emulated_vector_t2<EmulatedType>;
    using type = emulated_matrix<EmulatedType, 2, 2>;

    vec_t columns[2];

    type getTransposed() NBL_CONST_MEMBER_FUNC
    {
        type output;

        output.columns[0].x = columns[0].x;
        output.columns[1].x = columns[0].y;

        output.columns[0].y = columns[1].x;
        output.columns[1].y = columns[1].y;

        return output;
    }

    type operator*(NBL_CONST_REF_ARG(type) rhs) NBL_CONST_MEMBER_FUNC
    {
        type output;
        type lhsTransposed = getTransposed();

        output.columns[0].x = (lhsTransposed.columns[0] * rhs.columns[0]).calcComponentSum();
        output.columns[0].y = (lhsTransposed.columns[0] * rhs.columns[1]).calcComponentSum();

        output.columns[1].x = (lhsTransposed.columns[1] * rhs.columns[0]).calcComponentSum();
        output.columns[1].y = (lhsTransposed.columns[1] * rhs.columns[1]).calcComponentSum();

        return output.getTransposed();
    }

    vec_t operator*(NBL_CONST_REF_ARG(vec_t) rhs)
    {
        vec_t output;
        type lhsTransposed = getTransposed();

        output.x = (columns[0] * rhs).calcComponentSum();
        output.y = (columns[1] * rhs).calcComponentSum();

        return output;
    }
};

template<typename EmulatedType>
struct emulated_matrix<EmulatedType, 3, 3> // : emulated_matrix_base<EmulatedType, 3, 3>
{
    using vec_t = emulated_vector_t3<EmulatedType>;
    using type = emulated_matrix<EmulatedType, 3, 3>;

    vec_t columns[3];

    type getTransposed() NBL_CONST_MEMBER_FUNC
    {
        type output;

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

    type operator*(NBL_CONST_REF_ARG(type) rhs) NBL_CONST_MEMBER_FUNC
    {
        type output;
        type lhsTransposed = getTransposed();

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

    vec_t operator*(NBL_CONST_REF_ARG(vec_t) rhs)
    {
        vec_t output;
        type lhsTransposed = getTransposed();

        output.x = (columns[0] * rhs).calcComponentSum();
        output.y = (columns[1] * rhs).calcComponentSum();
        output.z = (columns[2] * rhs).calcComponentSum();

        return output;
    }

    vec_t operator[](uint32_t columnIdx)
    {
        return columns[columnIdx];
    }
};

template<typename EmulatedType>
using emulated_matrix_t2x2 = emulated_matrix<EmulatedType, 2, 2>;
template<typename EmulatedType>
using emulated_matrix_t3x3 = emulated_matrix<EmulatedType, 3, 3>;

namespace impl
{
template<typename T>
struct is_emulated
{
    NBL_CONSTEXPR_STATIC_INLINE bool value = is_same_v<T, emulated_float64_t<true, true> > ||
        is_same_v<T, emulated_float64_t<false, false> > ||
        is_same_v<T, emulated_float64_t<true, false> > ||
        is_same_v<T, emulated_float64_t<false, true> >;
};

template<typename T, uint32_t N, bool native = is_scalar<T>::value >
struct portable_vector
{
    using type = emulated_vector<T, N>;
};
// specialization for builtins
template<typename T, uint32_t N>
struct portable_vector<T, N, true>
{
    using type = vector<T, N>;
};

template<typename T, uint32_t N, uint32_t M, bool native = is_scalar<T>::value >
struct portable_matrix
{
    using type = emulated_matrix<T, N, M>;
};

template<typename T, uint32_t N, uint32_t M>
struct portable_matrix<T, N, M, true>
{
    using type = matrix<T, N, M>;
};

}

template<typename T, uint32_t N>
using portable_vector_t = typename impl::portable_vector<T, N>::type;

template<typename T>
using portable_vector_t2 = portable_vector_t<T, 2>;
template<typename T>
using portable_vector_t3 = portable_vector_t<T, 3>;
template<typename T>
using portable_vector_t4 = portable_vector_t<T, 4>;

using portable_vector64_t2 = portable_vector_t2<portable_float64_t<> >;
using portable_vector64_t3 = portable_vector_t3<portable_float64_t<> >;
using portable_vector64_t4 = portable_vector_t4<portable_float64_t<> >;

template<typename T, uint32_t N, uint32_t M>
using portable_matrix_t = typename impl::portable_matrix<T, N, M>::type;

template<typename T>
using portable_matrix_t2x2 = portable_matrix_t<T, 2, 2>;
template<typename T>
using portable_matrix_t3x3 = portable_matrix_t<T, 3, 3>;

using portable_matrix64_t2x2 = portable_matrix_t2x2<portable_float64_t<> >;
using portable_matrix64_t3x3 = portable_matrix_t3x3<portable_float64_t<> >;


// TODO: fix
template<typename T>
NBL_CONSTEXPR_INLINE_FUNC portable_float64_t<> create_portable_float64_t(T val)
{
    return _static_cast<portable_float64_t<> >(val);
}

template<typename T>
NBL_CONSTEXPR_INLINE_FUNC portable_vector64_t2 create_portable_vector64_t2(T val)
{
    portable_vector64_t2 output;
    output.x = create_portable_float64_t(val);
    output.y = create_portable_float64_t(val);

    return output;
}

template<typename X, typename Y>
NBL_CONSTEXPR_INLINE_FUNC portable_vector64_t2 create_portable_vector64_t2(X x, Y y)
{
    portable_vector64_t2 output;
    output.x = create_portable_float64_t(x);
    output.y = create_portable_float64_t(y);

    return output;
}

template<typename VecType>
NBL_CONSTEXPR_INLINE_FUNC portable_vector64_t2 create_portable_vector64_t2_from_2d_vec(VecType vec)
{
    portable_vector64_t2 output;
    output.x = create_portable_float64_t(vec.x);
    output.y = create_portable_float64_t(vec.y);

    return output;
}

template<typename T>
NBL_CONSTEXPR_INLINE_FUNC portable_vector64_t3 create_portable_vector64_t3(T val)
{
    portable_vector64_t3 output;
    output.x = create_portable_float64_t(val);
    output.y = create_portable_float64_t(val);
    output.z = create_portable_float64_t(val);

    return output;
}

template<typename X, typename Y, typename Z>
NBL_CONSTEXPR_INLINE_FUNC portable_vector64_t3 create_portable_vector64_t3(X x, Y y, Z z)
{
    portable_vector64_t3 output;
    output.x = create_portable_float64_t(x);
    output.y = create_portable_float64_t(y);
    output.z = create_portable_float64_t(z);

    return output;
}

template<typename VecType>
NBL_CONSTEXPR_INLINE_FUNC portable_vector64_t3 create_portable_vector64_t2_from_3d_vec(VecType vec)
{
    portable_vector64_t3 output;
    output.x = create_portable_float64_t(vec.x);
    output.y = create_portable_float64_t(vec.y);
    output.z = create_portable_float64_t(vec.z);

    return output;
}

namespace impl
{
    template<typename M, typename V, typename PortableFloat>
    struct PortableMul64Helper
    {
        static inline V multiply(M mat, V vec)
        {
            return mat * vec;
        }
    };

    template<typename M, typename V>
    struct PortableMul64Helper<M, V, float64_t>
    {
        static inline V multiply(M mat, V vec)
        {
            return mul(mat, vec);
        }
    };
}

template<typename M, typename V>
V portableMul64(M mat, V vec)
{
    return PortableMul64Helper<M, V, portable_float64_t<> >::multiply(mat, vec);
}


}
}
#endif