#ifndef _NBL_BUILTIN_HLSL_BITREVERSE_INCLUDED_
#define _NBL_BUILTIN_HLSL_BITREVERSE_INCLUDED_


#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{

template<typename T, uint16_t Bits NBL_FUNC_REQUIRES(is_unsigned_v<T>&& Bits <= sizeof(T) * 8)
/**
* @brief Takes the binary representation of `value` as a string of `Bits` bits and returns a value of the same type resulting from reversing the string
*
* @tparam T Type of the value to operate on.
* @tparam Bits The length of the string of bits used to represent `value`.
*
* @param [in] value The value to bitreverse.
*/
T bitReverseAs(T value)
{
	return bitReverse<T>(value) >> promote<T, scalar_type_t<T> >(scalar_type_t <T>(sizeof(T) * 8 - Bits));
}

template<typename T NBL_FUNC_REQUIRES(is_unsigned_v<T>)
/**
* @brief Takes the binary representation of `value` and returns a value of the same type resulting from reversing the string of bits as if it was `bits` long.
* Keep in mind `bits` cannot exceed `8 * sizeof(T)`.
*
* @tparam T type of the value to operate on.
*
* @param [in] value The value to bitreverse.
* @param [in] bits The length of the string of bits used to represent `value`.
*/
T bitReverseAs(T value, uint16_t bits)
{
	return bitReverse<T>(value) >> promote<T, scalar_type_t<T> >(scalar_type_t <T>(sizeof(T) * 8 - bits));
}


}
}



#endif