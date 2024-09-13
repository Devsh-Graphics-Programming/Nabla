#ifndef _NBL_BUILTIN_HLSL_PORTABLE_FLOAT64_T_MATH_INCLUDED_
#define _NBL_BUILTIN_HLSL_PORTABLE_FLOAT64_T_MATH_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>
#include <nbl/builtin/hlsl/emulated/emulated_float64_t.hlsl>
#include <nbl/builtin/hlsl/portable_float64_t.hlsl>

namespace nbl
{
namespace hlsl
{

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

template<typename EmulatedType, uint32_t N, uint32_t M>
struct emulated_matrix {};

template<typename EmulatedType>
struct emulated_matrix<EmulatedType, 2, 2>
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

    /*type operator*(NBL_CONST_REF_ARG(type) rhs) NBL_CONST_MEMBER_FUNC
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
    }*/
};

template<typename EmulatedType>
struct emulated_matrix<EmulatedType, 3, 3>
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

    /*type operator*(NBL_CONST_REF_ARG(type) rhs) NBL_CONST_MEMBER_FUNC
    {
        type output;

        output.columns[0].x = (columns[0] * rhs.columns[0]).calcComponentSum();
        output.columns[0].y = (columns[0] * rhs.columns[1]).calcComponentSum();
        output.columns[0].z = (columns[0] * rhs.columns[2]).calcComponentSum();

        output.columns[1].x = (columns[1] * rhs.columns[0]).calcComponentSum();
        output.columns[1].y = (columns[1] * rhs.columns[1]).calcComponentSum();
        output.columns[1].z = (columns[1] * rhs.columns[2]).calcComponentSum();

        output.columns[2].x = (columns[2] * rhs.columns[0]).calcComponentSum();
        output.columns[2].y = (columns[2] * rhs.columns[1]).calcComponentSum();
        output.columns[2].z = (columns[2] * rhs.columns[2]).calcComponentSum();

        return output.getTransposed();
    }*/

    vec_t operator[](uint32_t columnIdx)
    {
        return columns[columnIdx];
    }
};

template<typename EmulatedType>
using emulated_matrix_t2x2 = emulated_matrix<EmulatedType, 2, 2>;
template<typename EmulatedType>
using emulated_matrix_t3x3 = emulated_matrix<EmulatedType, 3, 3>;

template<typename T, uint32_t N, bool fundamental = is_fundamental<T>::value>
struct portable_vector
{
    using type = vector<T, N>;
};
#ifdef __HLSL_VERSION
template<typename T, uint32_t N>
struct portable_vector<T, N, false>
{
    using type = portable_vector<T, N>;
};
#endif

template<typename T, uint32_t N>
using portable_vector_t = typename portable_vector<T, N>::type;

template<typename T>
using portable_vector_t2 = portable_vector_t<T, 2>;
template<typename T>
using portable_vector_t3 = portable_vector_t<T, 3>;
template<typename T>
using portable_vector_t4 = portable_vector_t<T, 4>;

#ifdef __HLSL_VERSION
template<typename device_caps = void>
using portable_vector64_t2 = portable_vector_t2<portable_float64_t<device_caps> >;
template<typename device_caps = void>
using portable_vector64_t3 = portable_vector_t3<portable_float64_t<device_caps> >;
template<typename device_caps = void>
using portable_vector64_t4 = portable_vector_t4<portable_float64_t<device_caps> >;
#else
template<typename device_caps = void>
using portable_vector64_t2 = portable_vector_t2<float64_t>;
template<typename device_caps = void>
using portable_vector64_t3 = portable_vector_t3<float64_t>;
template<typename device_caps = void>
using portable_vector64_t4 = portable_vector_t4<float64_t>;
#endif

template<typename T, uint32_t N, uint32_t M, bool fundamental = is_fundamental<T>::value>
struct portable_matrix
{
    using type = matrix<T, N, M>;
};
#ifdef __HLSL_VERSION
template<typename T, uint32_t N, uint32_t M>
struct portable_matrix<T, N, M, false>
{
    using type = emulated_matrix<T, N, M>;
};
#endif

template<typename T, uint32_t N, uint32_t M>
using portable_matrix_t = typename portable_matrix<T, N, M>::type;

template<typename T>
using portable_matrix_t2x2 = portable_matrix_t<T, 2, 2>;
template<typename T>
using portable_matrix_t3x3 = portable_matrix_t<T, 3, 3>;


#ifdef __HLSL_VERSION
template<typename device_caps = void>
using portable_matrix64_t2x2 = portable_matrix_t2x2<portable_float64_t<device_caps> >;
template<typename device_caps = void>
using portable_matrix64_t3x3 = portable_matrix_t3x3<portable_float64_t<device_caps> >;
#else
template<typename device_caps = void>
using portable_matrix64_t2x2 = portable_matrix_t2x2<float64_t>;
template<typename device_caps = void>
using portable_matrix64_t3x3 = portable_matrix_t3x3<float64_t>;
#endif

namespace impl
{
    template<typename To, typename From>
    struct static_cast_helper<emulated_vector<To, 2>, vector<From, 2>, void>
    {
        static inline emulated_vector<To, 2> cast(vector<From, 2> vec)
        {
            return portable_vector_t<To, 2>(_static_cast<To, From>(vec.x), _static_cast<To, From>(vec.y));
        }
    };

    template<typename To, typename From>
    struct static_cast_helper<emulated_vector<To, 3>, vector<From, 3>, void>
    {
        static inline emulated_vector<To, 3> cast(vector<From, 3> vec)
        {
            return portable_vector_t<To, 3>(_static_cast<To, From>(vec.x), _static_cast<To, From>(vec.y), _static_cast<To, From>(vec.z));
        }
    };

    template<typename To, typename From>
    struct static_cast_helper<emulated_vector<To, 4>, vector<From, 4>, void>
    {
        static inline emulated_vector<To, 4> cast(vector<From, 4> vec)
        {
            return portable_vector_t<To, 4>(_static_cast<To, From>(vec.x), _static_cast<To, From>(vec.y), _static_cast<To, From>(vec.z), _static_cast<To, From>(vec.w));
        }
    };

    /*template<typename To, typename From>
    struct static_cast_helper<emulated_vector<To, 2>, From, void>
    {
        static inline emulated_vector<To, 2> cast(From val)
        {
            To vecComponent = To::create(val);
            return emulated_vector<To, 2>(vecComponent, vecComponent);
        }
    };

    template<typename To, typename From>
    struct static_cast_helper<To, 3>, From, void>
    {
        static inline emulated_vector<To, 2> cast(From val)
        {
            To vecComponent = To::create(val);
            return emulated_vector<To, 2>(vecComponent, vecComponent, vecComponent);
        }
    };

    template<emulated_vector<typename To, typename From>
    struct static_cast_helper<To, 4>, vector<From, 4>, void>
    {
        static inline emulated_vector<To, 4> cast(From val)
        {
            To vecComponent = To::create(val);
            return emulated_vector<To, 4>(vecComponent, vecComponent, vecComponent, vecComponent);
        }
    };*/
}

namespace impl
{
template<typename M, typename V, typename PortableFloat>
struct PortableMul64Helper
{
    static inline V multiply(M mat, V vec)
    {
        V output;
        M matTransposed = mat.getTransposed();

        output.x = (matTransposed.columns[0] * vec).calcComponentSum();
        output.y = (matTransposed.columns[1] * vec).calcComponentSum();
        output.z = (matTransposed.columns[2] * vec).calcComponentSum();

        return output;
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

#ifdef __HLSL_VERSION
template<typename M, typename V, typename device_caps = void>
V portableMul64(M mat, V vec)
{
    return impl::PortableMul64Helper<M, V, portable_float64_t<device_caps> >::multiply(mat, vec);
}
#else
template<typename M, typename V>
V portableMul64(M mat, V vec)
{
    return impl::PortableMul64Helper<M, V, float64_t>::multiply(mat, vec);
}
#endif

}
}
#endif