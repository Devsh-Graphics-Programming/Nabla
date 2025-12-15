#ifndef _NBL_SYSTEM_TO_STRING_INCLUDED_
#define _NBL_SYSTEM_TO_STRING_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/emulated/int64_t.hlsl>
#include <nbl/builtin/hlsl/morton.hlsl>

namespace nbl
{
namespace system
{
namespace impl
{

template<typename T>
struct to_string_helper
{
    static std::string __call(const T& value)
    {
        return std::to_string(value);
    }
};

template<>
struct to_string_helper<hlsl::emulated_uint64_t>
{
    static std::string __call(const hlsl::emulated_uint64_t& value)
    {
        return std::to_string(static_cast<uint64_t>(value));
    }
};

template<>
struct to_string_helper<hlsl::emulated_int64_t>
{
    static std::string __call(const hlsl::emulated_int64_t& value)
    {
        return std::to_string(static_cast<int64_t>(value));
    }
};

template<typename T, int16_t N>
struct to_string_helper<hlsl::vector<T, N>>
{
    static std::string __call(const hlsl::vector<T, N>& value)
    {
        std::stringstream output;
        output << "{ ";
        for (int i = 0; i < N; ++i)
        {
            output << to_string_helper<T>::__call(value[i]);

            if (i < N - 1)
                output << ", ";
        }
        output << " }";

        return output.str();
    }
};

template<bool Signed, uint16_t Bits, uint16_t D, typename _uint64_t>
struct to_string_helper<hlsl::morton::code<Signed, Bits, D, _uint64_t>>
{
    using value_t = hlsl::morton::code<Signed, Bits, D, _uint64_t>;
    static std::string __call(value_t value)
    {
        return to_string_helper<value_t::storage_t>::__call(value.value);
    }
};

}

template<typename T>
std::string to_string(T value)
{
    return impl::to_string_helper<T>::__call(value);
}
}
}

#endif