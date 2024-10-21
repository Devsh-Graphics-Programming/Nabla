#ifndef _NBL_BUILTIN_HLSL_PORTABLE_MATRIX_T_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_PORTABLE_MATRIX_T_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/emulated/matrix_t.hlsl>
#include <nbl/builtin/hlsl/portable/float64_t.hlsl>

namespace nbl
{
namespace hlsl
{

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
using portable_float64_t2x2 = portable_matrix_t2x2<portable_float64_t<device_caps> >;
template<typename device_caps = void>
using portable_float64_t3x3 = portable_matrix_t3x3<portable_float64_t<device_caps> >;
#else
template<typename device_caps = void>
using portable_float64_t2x2 = portable_matrix_t2x2<float64_t>;
template<typename device_caps = void>
using portable_float64_t3x3 = portable_matrix_t3x3<float64_t>;
#endif

namespace impl
{
// TODO: move to emulated/matrix.hlsl
// TODO: make one template for all dimensions
template<typename M, typename V, typename PortableFloat>
struct PortableMul64Helper
{
    static inline V multiply(M mat, V vec)
    {
        V output;

        output.x = (mat.rows[0] * vec).calcComponentSum();
        output.y = (mat.rows[1] * vec).calcComponentSum();
        output.z = (mat.rows[2] * vec).calcComponentSum();

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