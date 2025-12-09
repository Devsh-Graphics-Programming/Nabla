// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_QUANTIZED_SEQUENCE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_QUANTIZED_SEQUENCE_INCLUDED_

#include "nbl/builtin/hlsl/concepts/vector.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T, uint16_t Dim NBL_STRUCT_CONSTRAINABLE>
struct QuantizedSequence;


namespace impl
{
template<typename T, uint16_t D>
struct decode_helper;

template<typename T>
struct decode_helper<T, 1>
{
    using scalar_type = typename vector_traits<T>::scalar_type;
    using fp_type = typename float_of_size<sizeof(scalar_type)>::type;
    using return_type = vector<fp_type, 1>;

    static return_type __call(NBL_CONST_REF_ARG(QuantizedSequence<T, 1>) val, const scalar_type scrambleKey)
    {
        scalar_type seqVal = val.getX();
        seqVal ^= scrambleKey;
        return hlsl::promote<return_type>(seqVal) * bit_cast<fp_type>(0x2f800004u);
    }
};
template<typename T>
struct decode_helper<T, 2>
{
    using scalar_type = typename vector_traits<T>::scalar_type;
    using fp_type = typename float_of_size<sizeof(scalar_type)>::type;
    using uvec_type = vector<scalar_type, 2>;
    using return_type = vector<fp_type, 2>;

    static return_type __call(NBL_CONST_REF_ARG(QuantizedSequence<T, 2>) val, const uvec_type scrambleKey)
    {
        uvec_type seqVal;
        seqVal[0] = val.getX();
        seqVal[1] = val.getY();
        seqVal ^= scrambleKey;
        return return_type(seqVal) * bit_cast<fp_type>(0x2f800004u);
    }
};
template<typename T>
struct decode_helper<T, 3>
{
    using scalar_type = typename vector_traits<T>::scalar_type;
    using fp_type = typename float_of_size<sizeof(scalar_type)>::type;
    using uvec_type = vector<scalar_type, 3>;
    using return_type = vector<fp_type, 3>;

    static return_type __call(NBL_CONST_REF_ARG(QuantizedSequence<T, 3>) val, const uvec_type scrambleKey)
    {
        uvec_type seqVal;
        seqVal[0] = val.getX();
        seqVal[1] = val.getY();
        seqVal[2] = val.getZ();
        seqVal ^= scrambleKey;
        return return_type(seqVal) * bit_cast<fp_type>(0x2f800004u);
    }
};
template<typename T>
struct decode_helper<T, 4>
{
    using scalar_type = typename vector_traits<T>::scalar_type;
    using fp_type = typename float_of_size<sizeof(scalar_type)>::type;
    using uvec_type = vector<scalar_type, 4>;
    using return_type = vector<fp_type, 4>;

    static return_type __call(NBL_CONST_REF_ARG(QuantizedSequence<T, 4>) val, const uvec_type scrambleKey)
    {
        uvec_type seqVal;
        seqVal[0] = val.getX();
        seqVal[1] = val.getY();
        seqVal[2] = val.getZ();
        seqVal[3] = val.getW();
        seqVal ^= scrambleKey;
        return return_type(seqVal) * bit_cast<fp_type>(0x2f800004u);
    }
};
}

template<typename T, uint16_t D>
vector<typename float_of_size<sizeof(typename vector_traits<T>::scalar_type)>::type, D> decode(NBL_CONST_REF_ARG(QuantizedSequence<T, D>) val, const vector<typename vector_traits<T>::scalar_type, D> scrambleKey)
{
    return impl::decode_helper<T,D>::__call(val, scrambleKey);
}


#define SEQUENCE_SPECIALIZATION_CONCEPT concepts::UnsignedIntegral<typename vector_traits<T>::scalar_type> && size_of_v<typename vector_traits<T>::scalar_type> <= 4

// all Dim=1
template<typename T> NBL_PARTIAL_REQ_TOP(SEQUENCE_SPECIALIZATION_CONCEPT)
struct QuantizedSequence<T, 1 NBL_PARTIAL_REQ_BOT(SEQUENCE_SPECIALIZATION_CONCEPT) >
{
    using store_type = T;

    store_type getX() { return data; }
    void setX(const store_type value) { data = value; }

    store_type data;
};

// uint16_t, uint32_t; Dim=2,3,4
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 1 && Dim > 1 && Dim < 5)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 1 && Dim > 1 && Dim < 5) >
{
    using store_type = T;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t StoreBits = size_of_v<store_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = StoreBits / Dim;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Mask = (uint16_t(1u) << BitsPerComponent) - uint16_t(1u);

    store_type getX() { return data & Mask; }
    store_type getY() { return (data >> (BitsPerComponent * uint16_t(1u))) & Mask; }
    template<typename C=bool_constant<2 < Dim> NBL_FUNC_REQUIRES(C::value && 2 < Dim)
    store_type getZ() { return (data >> (BitsPerComponent * uint16_t(2u))) & Mask; }
    template<typename C=bool_constant<3 < Dim> NBL_FUNC_REQUIRES(C::value && 3 < Dim)
    store_type getW() { return (data >> (BitsPerComponent * uint16_t(3u))) & Mask; }

    void setX(const store_type value)
    {
        data &= ~Mask;
        data |= value & Mask;
    }
    void setY(const store_type value)
    {
        data &= ~(Mask << BitsPerComponent);
        data |= (value & Mask) << BitsPerComponent;
    }
    template<typename C=bool_constant<2 < Dim> NBL_FUNC_REQUIRES(C::value && 2 < Dim)
    void setZ(const store_type value)
    {
        const uint16_t bits = (BitsPerComponent * uint16_t(2u));
        data &= ~(Mask << bits);
        data |= (value & Mask) << bits;
    }
    template<typename C=bool_constant<3 < Dim> NBL_FUNC_REQUIRES(C::value && 3 < Dim)
    void setW(const store_type value)
    {
        const uint16_t bits = (BitsPerComponent * uint16_t(3u));
        data &= ~(Mask << bits);
        data |= (value & Mask) << bits;
    }

    store_type data;
};

// Dim 2,3,4 matches vector dim
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == Dim && Dim > 1 && Dim < 5)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == Dim && Dim > 1 && Dim < 5) >
{
    using store_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;

    scalar_type getX() { return data[0]; }
    scalar_type getY() { return data[1]; }
    template<typename C=bool_constant<2 < Dim> NBL_FUNC_REQUIRES(C::value && 2 < Dim)
    scalar_type getZ() { return data[2]; }
    template<typename C=bool_constant<3 < Dim> NBL_FUNC_REQUIRES(C::value && 3 < Dim)
    scalar_type getW() { return data[3]; }

    void setX(const scalar_type value) { data[0] = value; }
    void setY(const scalar_type value) { data[1] = value; }
    template<typename C=bool_constant<2 < Dim> NBL_FUNC_REQUIRES(C::value && 2 < Dim)
    void setZ(const scalar_type value) { data[2] = value; }
    template<typename C=bool_constant<3 < Dim> NBL_FUNC_REQUIRES(C::value && 3 < Dim)
    void setW(const scalar_type value) { data[3] = value; }

    store_type data;
};

// uint16_t2, uint32_t2; Dim=3
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 2 && Dim == 3)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 2 && Dim == 3) >
{
    using store_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t StoreBits = size_of_v<store_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = StoreBits / Dim;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Mask = (uint16_t(1u) << BitsPerComponent) - uint16_t(1u);

    scalar_type getX() { return data[0] & Mask; }
    scalar_type getY()
    {
        scalar_type y = data[0] >> BitsPerComponent;
        y |= (data[1] >> BitsPerComponent) << (StoreBits-BitsPerComponent);
        return y;
    }
    scalar_type getZ() { return data[1] & Mask; }

    void setX(const scalar_type value)
    {
        data[0] &= ~Mask;
        data[0] |= value & Mask;
    }
    void setY(const scalar_type value)
    {
        const uint16_t ybits = StoreBits-BitsPerComponent;
        const uint16_t ymask = uint16_t(1u) << ybits;
        data[0] &= Mask;
        data[1] &= Mask;
        data[0] |= (value & ymask) << BitsPerComponent;
        data[1] |= (value >> (ybits) & ymask) << BitsPerComponent;
    }
    void setZ(const scalar_type value)
    {
        data[1] &= ~Mask;
        data[1] |= value & Mask;
    }

    store_type data;
};

// uint16_t2, uint32_t2; Dim=4
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 2 && Dim == 4)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 2 && Dim == 4) >
{
    using store_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t StoreBits = size_of_v<store_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = StoreBits / Dim;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Mask = (uint16_t(1u) << BitsPerComponent) - uint16_t(1u);

    scalar_type getX() { return data[0] & Mask; }
    scalar_type getY() { return data[0] >> BitsPerComponent; }
    scalar_type getZ() { return data[1] & Mask; }
    scalar_type getW() { return data[1] >> BitsPerComponent; }

    void setX(const scalar_type value)
    {
        data[0] &= ~Mask;
        data[0] |= value & Mask;
    }
    void setY(const scalar_type value)
    {
        data[0] &= Mask;
        data[0] |= (value & Mask) << BitsPerComponent;
    }
    void setZ(const scalar_type value)
    {
        data[1] &= ~Mask;
        data[1] |= value & Mask;
    }
    void setW(const scalar_type value)
    {
        data[1] &= Mask;
        data[1] |= (value & Mask) << BitsPerComponent;
    }

    store_type data;
};

// uint16_t4, uint32_t4; Dim=2
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 4 && Dim == 2)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 4 && Dim == 2) >
{
    using store_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;
    using base_type = vector<scalar_type, 2>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t StoreBits = size_of_v<store_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = StoreBits / Dim;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Mask = (uint16_t(1u) << BitsPerComponent) - uint16_t(1u);

    base_type getX() { return data.xy; }
    base_type getY() { return data.zw; }

    void setX(const base_type value) { data.xy = value; }
    void setY(const base_type value) { data.zw = value; }

    store_type data;
};

// uint16_t4, uint32_t4; Dim=3
// uint16_t4 --> returns uint16_t2 - 21 bits per component: 16 in x, 5 in y
// uint16_t4 --> returns uint32_t2 - 42 bits per component: 32 in x, 10 in y
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 4 && Dim == 3)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 4 && Dim == 3) >
{
    using store_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;
    using base_type = vector<scalar_type, 2>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t StoreBits = size_of_v<store_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = StoreBits / Dim;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t LeftoverBitsPerComponent = BitsPerComponent - size_of_v<scalar_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Mask = (uint16_t(1u) << LeftoverBitsPerComponent) - uint16_t(1u);

    base_type getX()
    {
        base_type x;
        x[0] = data[0];
        x[1] = data[3] & Mask;
        return x;
    }
    base_type getY()
    {
        base_type y;
        y[0] = data[1];
        y[1] = (data[3] >> LeftoverBitsPerComponent) & Mask;
        return y;
    }
    base_type getZ()
    {
        base_type z;
        z[0] = data[1];
        z[1] = (data[3] >> (LeftoverBitsPerComponent * uint16_t(2u))) & Mask;
        return z;
    }

    void setX(const base_type value)
    {
        data[0] = value[0];
        data[3] &= ~Mask;
        data[3] |= value[1] & Mask;
    }
    void setY(const base_type value)
    {
        data[1] = value[0];
        data[3] &= ~Mask;
        data[3] |= (value[1] & Mask) << LeftoverBitsPerComponent;
    }
    void setZ(const base_type value)
    {
        data[2] = value[0];
        data[3] &= ~Mask;
        data[3] |= (value[1] & Mask) << (LeftoverBitsPerComponent * uint16_t(2u));
    }

    store_type data;
};

#undef SEQUENCE_SPECIALIZATION_CONCEPT

}

}
}

#endif
