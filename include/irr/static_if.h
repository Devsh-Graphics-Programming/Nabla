//=======================================================================
// Copyright (c) 2013-2016 Baptiste Wicht.
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================
#ifndef __IRR_STATIC_IF_H_INCLUDED__
#define __IRR_STATIC_IF_H_INCLUDED__

#if __cplusplus >= 201703L

#define IRR_PSEUDO_IF_CONSTEXPR_BEGIN(...) if constexpr (__VA_ARGS__)
#define IRR_PSEUDO_ELSE_CONSTEXPR			else
#define IRR_PSEUDO_IF_CONSTEXPR_END			

#else

#define IRR_PSEUDO_IF_CONSTEXPR_BEGIN(...) irr::static_if<__VA_ARGS__>([&](auto f)
#define IRR_PSEUDO_ELSE_CONSTEXPR			).else_([&](auto f)
#define IRR_PSEUDO_IF_CONSTEXPR_END			);

namespace irr {

namespace static_if_detail {

/*!
 * \brief Identify functor
 */
struct identity {
    /*!
     * \brief Returns exactly what was passsed as argument
     */
    template <typename T>
    T operator()(T&& x) const {
        return std::forward<T>(x);
    }
};

/*!
 * \brief Helper for static if
 *
 * This base type is when the condition is true
 */
template <bool Cond>
struct statement {
    /*!
     * \brief Execute the if part of the statement
     * \param f The functor to execute
     */
    template <typename F>
    void then(const F& f) {
        f(identity());
    }

    /*!
     * \brief Execute the else part of the statement
     * \param f The functor to execute
     */
    template <typename F>
    void else_(const F& f) {
        (void)(f);
    }
};

/*!
 * \brief Helper for static if
 *
 * Specialization for condition is false
 */
template <>
struct statement<false> {
    /*!
     * \brief Execute the if part of the statement
     * \param f The functor to execute
     */
    template <typename F>
    void then(const F& f) {
        (void)(f);
    }

    /*!
     * \brief Execute the else part of the statement
     * \param f The functor to execute
     */
    template <typename F>
    void else_(const F& f) {
        f(identity());
    }
};

} //end of namespace static_if_detail

/*!
 * \brief Execute the lambda if the static condition is verified
 *
 * This should be usd to auto lambda to ensure instantiation is only made for
 * the "true" branch
 *
 * \tparam Cond The static condition
 * \param f The lambda to execute if true
 * \return a statement object to execute else_ if necessary
 */
template <bool Cond, typename F>
static_if_detail::statement<Cond> static_if(F const& f) {
    static_if_detail::statement<Cond> if_;
    if_.then(f);
    return if_;
}

} //end of namespace irr

#endif


#endif
