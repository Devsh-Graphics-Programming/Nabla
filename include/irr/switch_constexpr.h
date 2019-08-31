#ifndef __IRR_SWITCH_CONSTEXPR_H_INCLUDED__
#define __IRR_SWITCH_CONSTEXPR_H_INCLUDED__

#include <type_traits>

namespace irr
{

template<size_t CASE, class CASE_TYPE>
struct switch_constexpr_case
{
    constexpr static size_t Case = CASE;
    typedef CASE_TYPE CaseType;
};

template<size_t Value, class DEFAULT_TYPE, class... SWITCH_CASES>
struct switch_constexpr;

template<size_t Value, class DEFAULT_TYPE, class SWITCH_CASE>
struct switch_constexpr<Value,DEFAULT_TYPE,SWITCH_CASE>
{
    typedef typename std::conditional<  Value==SWITCH_CASE::Case,
                                        typename SWITCH_CASE::CaseType,
                                        DEFAULT_TYPE>::type type;
};

template<size_t Value, class DEFAULT_TYPE, class FIRST_SWITCH_CASE, class... SWITCH_CASES>
struct switch_constexpr<Value,DEFAULT_TYPE,FIRST_SWITCH_CASE,SWITCH_CASES...>
{
    typedef typename std::conditional<  Value==FIRST_SWITCH_CASE::Case,
                                        typename FIRST_SWITCH_CASE::CaseType,
                                        typename switch_constexpr<Value,DEFAULT_TYPE,SWITCH_CASES...>::type >::type type;
};

}

#endif // __IRR_SWITCH_CONSTEXPR_H_INCLUDED__
