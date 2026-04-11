#ifndef _NBL_IRANGE_HPP_
#define _NBL_IRANGE_HPP_

namespace nbl::core
{

/// @brief Minimal concepts used by camera persistence and tooling helpers.
template<typename R>
concept GeneralPurposeRange = requires
{
    typename std::ranges::range_value_t<R>;
};

template<typename R, typename T>
concept ContiguousGeneralPurposeRangeOf = GeneralPurposeRange<R> &&
std::ranges::contiguous_range<R> &&
std::same_as<std::ranges::range_value_t<R>, T>;

} // namespace nbl::core

#endif // _NBL_IRANGE_HPP_
