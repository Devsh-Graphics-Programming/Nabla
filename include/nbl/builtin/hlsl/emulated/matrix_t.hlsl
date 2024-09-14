#ifndef _NBL_BUILTIN_HLSL_EMULATED_MATRIX_T_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_MATRIX_T_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/portable/float64_t.hlsl>

namespace nbl
{
namespace hlsl
{

template<typename EmulatedType, uint32_t N, uint32_t M>
struct emulated_matrix {};

template<typename EmulatedType>
struct emulated_matrix<EmulatedType, 2, 2>
{
    using vec_t = emulated_vector_t2<EmulatedType>;
    using this_t = emulated_matrix<EmulatedType, 2, 2>;

    vec_t rows[2];

    this_t getTransposed() NBL_CONST_MEMBER_FUNC
    {
        this_t output;

        output.rows[0].x = rows[0].x;
        output.rows[1].x = rows[0].y;

        output.rows[0].y = rows[1].x;
        output.rows[1].y = rows[1].y;

        return output;
    }
};

template<typename EmulatedType>
struct emulated_matrix<EmulatedType, 3, 3>
{
    using vec_t = emulated_vector_t3<EmulatedType>;
    using this_t = emulated_matrix<EmulatedType, 3, 3>;

    vec_t rows[3];

    this_t getTransposed() NBL_CONST_MEMBER_FUNC
    {
        this_t output;

        output.rows[0].x = rows[0].x;
        output.rows[1].x = rows[0].y;
        output.rows[2].x = rows[0].z;

        output.rows[0].y = rows[1].x;
        output.rows[1].y = rows[1].y;
        output.rows[2].y = rows[1].z;

        output.rows[0].z = rows[2].x;
        output.rows[1].z = rows[2].y;
        output.rows[2].z = rows[2].z;

        return output;
    }

    vec_t operator[](uint32_t columnIdx)
    {
        return rows[columnIdx];
    }
};

template<typename EmulatedType>
using emulated_matrix_t2x2 = emulated_matrix<EmulatedType, 2, 2>;
template<typename EmulatedType>
using emulated_matrix_t3x3 = emulated_matrix<EmulatedType, 3, 3>;

}
}
#endif