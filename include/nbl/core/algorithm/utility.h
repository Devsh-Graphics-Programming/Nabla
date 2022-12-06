#ifndef __NBL_CORE_ALGORITHM_UTILITY_H_INCLUDED__
#define __NBL_CORE_ALGORITHM_UTILITY_H_INCLUDED__

namespace nbl
{
namespace core
{

namespace impl
{
    template<typename T, typename F, std::size_t... Is>
    inline void for_each(T&& t, F f, std::index_sequence<Is...>)
    {
        auto l = { (f(std::get<Is>(t)), 0)... };
    }
}

template<typename... Ts, typename F>
inline void for_each_in_tuple(std::tuple<Ts...> const& t, F f)
{
    constexpr std::size_t N = std::tuple_size<std::remove_reference_t<decltype(t)>>::value;
    impl::for_each(t, f, std::make_index_sequence<N>());
}

template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

}
}

#endif //__NBL_CORE_ALGORITHM_UTILITY_H_INCLUDED__