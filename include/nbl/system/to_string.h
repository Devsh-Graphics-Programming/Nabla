#ifndef _NBL_SYSTEM_TO_STRING_INCLUDED_
#define _NBL_SYSTEM_TO_STRING_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

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

}

template<typename T>
std::string to_string(T value)
{
    return impl::to_string_helper<T>::__call(value);
}
}
}

#endif