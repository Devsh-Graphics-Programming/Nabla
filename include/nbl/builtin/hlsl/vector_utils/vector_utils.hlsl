#ifndef _NBL_BUILTIN_HLSL_VECTOR_UTILS_VECTOR_UTILS_INCLUDED_
#define _NBL_BUILTIN_HLSL_VECTOR_UTILS_VECTOR_UTILS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{

// TODO: why cant I NBL_CONST_REF_ARG(vector<T, N>)
template<typename T, uint32_t N>
inline T lengthsquared(vector<T, N> vec)
{
	return hlsl::dot(vec, vec);
}

template<typename T, uint32_t N>
inline T distancesquared(vector<T, N> vecA, vector<T, N> vecB)
{
	const vector<T, N> ASubB = vecA - vecB;
	return hlsl::dot(ASubB, ASubB);
}

//! Rotates the vector by a specified number of degrees around the X axis and the specified center.
/** \param degrees: Number of degrees to rotate around the X axis.
\param center: The center of the rotation. */
template<typename T>
inline vector<T, 3> rotateYZByRAD(float radians, const vector<T, 3>& vec, const vector<T, 3>& center = vector<T, 3>(0,0,0))
{
	// TODO: implement
	//static_assert(false);
	return vector<T, 3>();
}

}
}

#endif