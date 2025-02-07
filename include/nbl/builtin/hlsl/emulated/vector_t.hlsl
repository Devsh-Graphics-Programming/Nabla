#ifndef _NBL_BUILTIN_HLSL_EMULATED_VECTOR_T_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_VECTOR_T_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/portable/float64_t.hlsl>
#include <nbl/builtin/hlsl/functional.hlsl>
#include <nbl/builtin/hlsl/array_accessors.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>

namespace nbl
{
namespace hlsl
{

namespace emulated_vector_impl
{


template<typename T>
struct _2_component_vec
{
    T x;
    T y;

    static_assert(sizeof(T) <= 8);

    NBL_CONSTEXPR_INLINE_FUNC void setComponent(uint32_t componentIdx, T val)
    {
        if (componentIdx == 0)
            x = val;
        if (componentIdx == 1)
            y = val;
    }

    NBL_CONSTEXPR_INLINE_FUNC T getComponent(uint32_t componentIdx) NBL_CONST_MEMBER_FUNC
    {
        if (componentIdx == 0)
            return x;
        if (componentIdx == 1)
            return y;

        // TODO: avoid code duplication, make it constexpr
        using TAsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
        TAsUint invalidComponentValue = nbl::hlsl::_static_cast<TAsUint>(0xdeadbeefbadcaffeull);
        return nbl::hlsl::bit_cast<T>(invalidComponentValue);
    }

    NBL_CONSTEXPR_STATIC uint32_t Dimension = 2;
};

template<typename T>
struct _3_component_vec
{
    T x;
    T y;
    T z;


    NBL_CONSTEXPR_INLINE_FUNC void setComponent(uint32_t componentIdx, T val)
    {
        if (componentIdx == 0)
            x = val;
        if (componentIdx == 1)
            y = val;
        if (componentIdx == 2)
            z = val;
    }

    NBL_CONSTEXPR_INLINE_FUNC T getComponent(uint32_t componentIdx) NBL_CONST_MEMBER_FUNC
    {
        if (componentIdx == 0)
            return x;
        if (componentIdx == 1)
            return y;
        if (componentIdx == 2)
            return z;

        // TODO: avoid code duplication, make it constexpr
        using TAsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
        TAsUint invalidComponentValue = nbl::hlsl::_static_cast<TAsUint>(0xdeadbeefbadcaffeull >> (64 - sizeof(T) * 8));
        return nbl::hlsl::bit_cast<T>(invalidComponentValue);
    }

    NBL_CONSTEXPR_STATIC uint32_t Dimension = 3;
};

template<typename T>
struct _4_component_vec
{
    T x;
    T y;
    T z;
    T w;

    NBL_CONSTEXPR_INLINE_FUNC void setComponent(uint32_t componentIdx, T val)
    {
        if (componentIdx == 0)
            x = val;
        if (componentIdx == 1)
            y = val;
        if (componentIdx == 2)
            z = val;
        if (componentIdx == 3)
            w = val;
    }

    NBL_CONSTEXPR_INLINE_FUNC T getComponent(uint32_t componentIdx) NBL_CONST_MEMBER_FUNC
    {
        if (componentIdx == 0)
            return x;
        if (componentIdx == 1)
            return y;
        if (componentIdx == 2)
            return z;
        if (componentIdx == 3)
            return w;

        // TODO: avoid code duplication, make it constexpr
        using TAsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
        uint64_t invalidComponentValue = nbl::hlsl::_static_cast<TAsUint>(0xdeadbeefbadcaffeull >> (64 - sizeof(T) * 8));
        return nbl::hlsl::bit_cast<T>(invalidComponentValue);
    }

    NBL_CONSTEXPR_STATIC uint32_t Dimension = 4;
};

template <typename ComponentType, typename CRTP, bool IsComponentTypeFundamental = is_fundamental<ComponentType>::value>
struct emulated_vector : CRTP
{
    using this_t = emulated_vector<ComponentType, CRTP>;
    using component_t = ComponentType;

    NBL_CONSTEXPR_STATIC_INLINE this_t create(this_t other)
    {
        CRTP output;

        for (uint32_t i = 0u; i < CRTP::Dimension; ++i)
            output.setComponent(i, other.getComponent(i));
    }
    NBL_CONSTEXPR_STATIC_INLINE this_t create(vector<component_t, CRTP::Dimension> other)
    {
        this_t output;

        for (uint32_t i = 0u; i < CRTP::Dimension; ++i)
            output.setComponent(i, other[i]);

        return output;
    }

    NBL_CONSTEXPR_INLINE_FUNC this_t operator+(component_t val)
    {
        this_t output;

        for (uint32_t i = 0u; i < CRTP::Dimension; ++i)
            output.setComponent(i, this_t::getComponent(i) + val);

        return output;
    }
    NBL_CONSTEXPR_INLINE_FUNC this_t operator+(this_t other)
    {
        this_t output;

        for (uint32_t i = 0u; i < CRTP::Dimension; ++i)
            output.setComponent(i, this_t::getComponent(i) + other.getComponent(i));

        return output;
    }
    NBL_CONSTEXPR_INLINE_FUNC this_t operator+(vector<component_t, CRTP::Dimension> other)
    {
        this_t output;

        for (uint32_t i = 0u; i < CRTP::Dimension; ++i)
            output.setComponent(i, this_t::getComponent(i) + other[i]);

        return output;
    }
    
    NBL_CONSTEXPR_INLINE_FUNC this_t operator-(component_t val)
    {
        this_t output;

        for (uint32_t i = 0u; i < CRTP::Dimension; ++i)
            output.setComponent(i, CRTP::getComponent(i) - val);

        return output;
    }
    NBL_CONSTEXPR_INLINE_FUNC this_t operator-(this_t other)
    {
        this_t output;

        for (uint32_t i = 0u; i < CRTP::Dimension; ++i)
            output.setComponent(i, CRTP::getComponent(i) - other.getComponent(i));

        return output;
    }
    NBL_CONSTEXPR_INLINE_FUNC this_t operator-(vector<component_t, CRTP::Dimension> other)
    {
        this_t output;

        for (uint32_t i = 0u; i < CRTP::Dimension; ++i)
            output.setComponent(i, CRTP::getComponent(i) - other[i]);

        return output;
    }

    NBL_CONSTEXPR_INLINE_FUNC this_t operator*(component_t val)
    {
        this_t output;

        for (uint32_t i = 0u; i < CRTP::Dimension; ++i)
            output.setComponent(i, CRTP::getComponent(i) * val);

        return output;
    }
    NBL_CONSTEXPR_INLINE_FUNC this_t operator*(this_t other)
    {
        this_t output;

        for (uint32_t i = 0u; i < CRTP::Dimension; ++i)
            output.setComponent(i, CRTP::getComponent(i) * other.getComponent(i));

        return output;
    }
    NBL_CONSTEXPR_INLINE_FUNC this_t operator*(vector<component_t, CRTP::Dimension> other)
    {
        this_t output;

        for (uint32_t i = 0u; i < CRTP::Dimension; ++i)
            output.setComponent(i, CRTP::getComponent(i) * other[i]);

        return output;
    }

    NBL_CONSTEXPR_INLINE_FUNC component_t calcComponentSum()
    {
        component_t sum = 0;
        for (uint32_t i = 0u; i < CRTP::Dimension; ++i)
            sum = sum + CRTP::getComponent(i);

        return sum;
    }
};

#define DEFINE_OPERATORS_FOR_TYPE(...)\
NBL_CONSTEXPR_INLINE_FUNC this_t operator+(__VA_ARGS__ val)\
{\
    this_t output;\
    for (uint32_t i = 0u; i < CRTP::Dimension; ++i)\
        output.setComponent(i, CRTP::getComponent(i) + component_t::create(val));\
\
    return output;\
}\
\
NBL_CONSTEXPR_INLINE_FUNC this_t operator-(__VA_ARGS__ val)\
{\
    this_t output;\
    for (uint32_t i = 0u; i < CRTP::Dimension; ++i)\
        output.setComponent(i, CRTP::getComponent(i) - component_t::create(val));\
\
    return output;\
}\
\
NBL_CONSTEXPR_INLINE_FUNC this_t operator*(__VA_ARGS__ val)\
{\
    this_t output;\
    for (uint32_t i = 0u; i < CRTP::Dimension; ++i)\
        output.setComponent(i, CRTP::getComponent(i) * component_t::create(val));\
\
    return output;\
}\
\

// TODO: some of code duplication could be avoided
template <typename ComponentType, typename CRTP>
struct emulated_vector<ComponentType, CRTP, false> : CRTP
{
    using component_t = ComponentType;
    using this_t = emulated_vector<ComponentType, CRTP, false>;

    NBL_CONSTEXPR_STATIC_INLINE this_t create(this_t other)
    {
        this_t output;

        for (uint32_t i = 0u; i < CRTP::Dimension; ++i)
            output.setComponent(i, other.getComponent(i));

        return output;
    }

    template<typename T>
    NBL_CONSTEXPR_STATIC_INLINE this_t create(vector<T, CRTP::Dimension> other)
    {
        this_t output;

        for (uint32_t i = 0u; i < CRTP::Dimension; ++i)
            output.setComponent(i, ComponentType::create(other[i]));

        return output;
    }

    NBL_CONSTEXPR_INLINE_FUNC this_t operator+(this_t other)
    {
        this_t output;

        for (uint32_t i = 0u; i < CRTP::Dimension; ++i)
            output.setComponent(i, CRTP::getComponent(i) + other.getComponent(i));

        return output;
    }
    NBL_CONSTEXPR_INLINE_FUNC this_t operator-(this_t other)
    {
        this_t output;

        for (uint32_t i = 0u; i < CRTP::Dimension; ++i)
            output.setComponent(i, CRTP::getComponent(i) - other.getComponent(i));

        return output;
    }
    NBL_CONSTEXPR_INLINE_FUNC this_t operator*(this_t other)
    {
        this_t output;

        for (uint32_t i = 0u; i < CRTP::Dimension; ++i)
            output.setComponent(i, CRTP::getComponent(i) * other.getComponent(i));

        return output;
    }

    DEFINE_OPERATORS_FOR_TYPE(emulated_float64_t<true, true>)
    DEFINE_OPERATORS_FOR_TYPE(emulated_float64_t<true, false>)
    DEFINE_OPERATORS_FOR_TYPE(emulated_float64_t<false, true>)
    DEFINE_OPERATORS_FOR_TYPE(emulated_float64_t<false, false>)
    DEFINE_OPERATORS_FOR_TYPE(float32_t)
    DEFINE_OPERATORS_FOR_TYPE(float64_t)
    DEFINE_OPERATORS_FOR_TYPE(uint16_t)
    DEFINE_OPERATORS_FOR_TYPE(uint32_t)
    DEFINE_OPERATORS_FOR_TYPE(uint64_t)
    DEFINE_OPERATORS_FOR_TYPE(int16_t)
    DEFINE_OPERATORS_FOR_TYPE(int32_t)
    DEFINE_OPERATORS_FOR_TYPE(int64_t)

    NBL_CONSTEXPR_INLINE_FUNC ComponentType calcComponentSum()
    {
        ComponentType sum = ComponentType::create(0);
        for (uint32_t i = 0u; i < CRTP::Dimension; ++i)
            sum = sum + CRTP::getComponent(i);

        return sum;
    }
};

#undef DEFINE_OPERATORS_FOR_TYPE

template<typename T, uint32_t N>
struct CRTPParentStructSelector
{
    using type = void;
};
template<typename T>
struct CRTPParentStructSelector<T, 2>
{
    using type = _2_component_vec<T>;
};
template<typename T>
struct CRTPParentStructSelector<T, 3>
{
    using type = _3_component_vec<T>;
};
template<typename T>
struct CRTPParentStructSelector<T, 4>
{
    using type = _4_component_vec<T>;
};

}

template<typename T, uint32_t N>
using emulated_vector_t = emulated_vector_impl::emulated_vector<T, typename emulated_vector_impl::CRTPParentStructSelector<T, N>::type>;
template<typename T>
using emulated_vector_t2 = emulated_vector_impl::emulated_vector<T, typename emulated_vector_impl::CRTPParentStructSelector<T, 2>::type>;
template<typename T>
using emulated_vector_t3 = emulated_vector_impl::emulated_vector<T, typename emulated_vector_impl::CRTPParentStructSelector<T, 3>::type>;
template<typename T>
using emulated_vector_t4 = emulated_vector_impl::emulated_vector<T, typename emulated_vector_impl::CRTPParentStructSelector<T, 4>::type>;

// used this macro, because I can't make it work with templated array dimension
#define DEFINE_ARRAY_GET_SET_SPECIALIZATION(DIMENSION)\
template<typename ScalarType>\
struct array_get<emulated_vector_t##DIMENSION<ScalarType>, ScalarType, uint32_t>\
{\
    inline ScalarType operator()(NBL_CONST_REF_ARG(emulated_vector_t##DIMENSION<ScalarType>) vec, const uint32_t ix) NBL_CONST_MEMBER_FUNC\
    {\
        return vec.getComponent(ix);\
    }\
};\
template<typename ScalarType>\
struct array_set<emulated_vector_t##DIMENSION<ScalarType>, ScalarType, uint32_t>\
{\
    void operator()(NBL_REF_ARG(emulated_vector_t##DIMENSION<ScalarType>) vec, uint32_t index, ScalarType value) NBL_CONST_MEMBER_FUNC\
    {\
        vec.setComponent(index, value);\
    }\
};\

DEFINE_ARRAY_GET_SET_SPECIALIZATION(2)
DEFINE_ARRAY_GET_SET_SPECIALIZATION(3)
DEFINE_ARRAY_GET_SET_SPECIALIZATION(4)
#undef DEFINE_ARRAY_GET_SET_SPECIALIZATION

#define DEFINE_SCALAR_OF_SPECIALIZATION(DIMENSION)\
template<typename T>\
struct vector_traits<emulated_vector_t<T, DIMENSION> >\
{\
    using scalar_type = T;\
    NBL_CONSTEXPR_STATIC_INLINE uint32_t Dimension = DIMENSION;\
    NBL_CONSTEXPR_STATIC_INLINE bool IsVector = true;\
};\

DEFINE_SCALAR_OF_SPECIALIZATION(2)
DEFINE_SCALAR_OF_SPECIALIZATION(3)
DEFINE_SCALAR_OF_SPECIALIZATION(4)
#undef DEFINE_SCALAR_OF_SPECIALIZATION

namespace impl
{
template<typename To, typename From>
struct static_cast_helper<emulated_vector_t2<To>, vector<From, 2>, void>
{
    static inline emulated_vector_t2<To> cast(vector<From, 2> vec)
    {
        emulated_vector_t2<To> output;
        output.x = _static_cast<To, From>(vec.x);
        output.y = _static_cast<To, From>(vec.y);

        return output;
    }
};

template<typename To, typename From>
struct static_cast_helper<emulated_vector_t3<To>, vector<From, 3>, void>
{
    static inline emulated_vector_t3<To> cast(vector<From, 3> vec)
    {
        emulated_vector_t3<To> output;
        output.x = _static_cast<To, From>(vec.x);
        output.y = _static_cast<To, From>(vec.y);
        output.z = _static_cast<To, From>(vec.z);

        return output;
    }
};

template<typename To, typename From>
struct static_cast_helper<emulated_vector_t4<To>, vector<From, 4>, void>
{
    static inline emulated_vector_t4<To> cast(vector<From, 4> vec)
    {
        emulated_vector_t4<To> output;
        output.x = _static_cast<To, From>(vec.x);
        output.y = _static_cast<To, From>(vec.y);
        output.z = _static_cast<To, From>(vec.z);
        output.w = _static_cast<To, From>(vec.w);

        return output;
    }
};

template<typename ToComponentType, typename FromComponentType, uint32_t N>
struct static_cast_helper<vector<ToComponentType, N>, emulated_vector_t<FromComponentType, N>, void>
{
    using OutputVecType = vector<ToComponentType, N>;
    using InputVecType = emulated_vector_t<FromComponentType, N>;

    static inline OutputVecType cast(InputVecType vec)
    {
        array_get<InputVecType, FromComponentType> getter;
        array_set<OutputVecType, ToComponentType> setter;
        
        OutputVecType output;
        for (int i = 0; i < N; ++i)
            setter(output, i, _static_cast<ToComponentType>(getter(vec, i)));

        return output;
    }
};

}

}
}
#endif