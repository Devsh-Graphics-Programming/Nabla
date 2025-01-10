#ifndef _NBL_BUILTIN_HLSL_FFT_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_FFT_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/complex.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/math/intutil.hlsl>
#include <nbl/builtin/hlsl/numbers.hlsl>

namespace nbl
{
namespace hlsl
{
namespace fft
{


template <uint16_t N NBL_FUNC_REQUIRES(N > 0 && N <= 4)
/**
* @brief Returns the size of the full FFT computed, in terms of number of complex elements.
*
* @tparam N Number of dimensions of the signal to perform FFT on.
*
* @param [in] dimensions Size of the signal.
* @param [in] realFFT Indicates whether the signal is real. False by default.
* @param [in] firstAxis Indicates which axis the FFT is performed on first. Only relevant for real-valued signals. Must be less than N. 0 by default.
*/
inline vector<uint64_t, N> padDimensions(NBL_CONST_REF_ARG(vector<uint32_t, N>) dimensions, bool realFFT = false, uint16_t firstAxis = 0u)
{
    vector<uint32_t, N> newDimensions = dimensions;
    for (uint16_t i = 0u; i < N; i++)
    {
        newDimensions[i] = hlsl::roundUpToPoT(newDimensions[i]);
    }
    if (realFFT)
        newDimensions[firstAxis] /= 2;
    return newDimensions;
}

template <uint16_t N NBL_FUNC_REQUIRES(N > 0 && N <= 4)
/**
* @brief Returns the size required by a buffer to hold the result of the FFT of a signal after a certain pass.
*
* @tparam N Number of dimensions of the signal to perform FFT on.
*
* @param [in] numChannels Number of channels of the signal.
* @param [in] inputDimensions Size of the signal.
* @param [in] passIx Which pass the size is being computed for.
* @param [in] axisPassOrder Order of the axis in which the FFT is computed in. Default is xyzw.
* @param [in] realFFT True if the signal is real. False by default.
* @param [in] halfFloats True if using half-precision floats. False by default.
*/
inline uint64_t getOutputBufferSize(
    uint32_t numChannels,
    NBL_CONST_REF_ARG(vector<uint32_t, N>) inputDimensions,
    uint16_t passIx,
    NBL_CONST_REF_ARG(vector<uint16_t, N>) axisPassOrder = _static_cast<vector<uint16_t, N> >(uint16_t4(0, 1, 2, 3)),
    bool realFFT = false,
    bool halfFloats = false
)
{
    const vector<uint32_t, N> paddedDimensions = padDimensions<N>(inputDimensions, realFFT, axisPassOrder[0]);
    vector<bool, N> axesDone = promote<vector<bool, N>, bool>(false);
    for (uint16_t i = 0; i <= passIx; i++)
        axesDone[axisPassOrder[i]] = true;
    const vector<uint32_t, N> passOutputDimension = lerp(inputDimensions, paddedDimensions, axesDone);
    uint64_t numberOfComplexElements = uint64_t(numChannels);
    for (uint16_t i = 0; i < N; i++)
        numberOfComplexElements *= uint64_t(passOutputDimension[i]);
    return numberOfComplexElements * (halfFloats ? sizeof(complex_t<float16_t>) : sizeof(complex_t<float32_t>));
}


// Computes the kth element in the group of N roots of unity
// Notice 0 <= k < N/2, rotating counterclockwise in the forward (DIF) transform and clockwise in the inverse (DIT)
template<bool inverse, typename Scalar>
complex_t<Scalar> twiddle(uint32_t k, uint32_t halfN)
{
    complex_t<Scalar> retVal;
    const Scalar kthRootAngleRadians = numbers::pi<Scalar> *Scalar(k) / Scalar(halfN);
    retVal.real(cos(kthRootAngleRadians));
    if (!inverse)
        retVal.imag(sin(-kthRootAngleRadians));
    else
        retVal.imag(sin(kthRootAngleRadians));
    return retVal;
}

template<bool inverse, typename Scalar>
struct DIX
{
    static void radix2(NBL_CONST_REF_ARG(complex_t<Scalar>) twiddle, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi)
    {
        plus_assign< complex_t<Scalar> > plusAss;
        //Decimation in time - inverse           
        if (inverse) {
            complex_t<Scalar> wHi = twiddle * hi;
            hi = lo - wHi;
            plusAss(lo, wHi);
        }
        //Decimation in frequency - forward   
        else {
            complex_t<Scalar> diff = lo - hi;
            plusAss(lo, hi);
            hi = twiddle * diff;
        }
    }
};

template<typename Scalar>
using DIT = DIX<true, Scalar>;

template<typename Scalar>
using DIF = DIX<false, Scalar>;

// ------------------------------------------------- Utils ---------------------------------------------------------
// 
// Util to unpack two values from the packed FFT X + iY - get outputs in the same input arguments, storing x to lo and y to hi
template<typename Scalar>
void unpack(NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi)
{
    complex_t<Scalar> x = (lo + conj(hi)) * Scalar(0.5);
    hi = rotateRight<Scalar>(lo - conj(hi)) * Scalar(0.5);
    lo = x;
}

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
}

#endif