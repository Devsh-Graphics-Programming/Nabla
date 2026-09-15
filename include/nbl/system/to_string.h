#ifndef _NBL_SYSTEM_TO_STRING_INCLUDED_
#define _NBL_SYSTEM_TO_STRING_INCLUDED_

#include <format>
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

template<std::floating_point T>
struct to_string_helper<T>
{
    static std::string __call(const T& value)
    {
        return std::format("{}", value);
    }
};

template<>
struct to_string_helper<core::blake3_hash_t>
{
    static std::string __call(const core::blake3_hash_t& value)
    {
        // fast base64 optimized for this without leaking deps
        std::string retval(44,'=');
        constexpr const char* base = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        auto out = retval.data();
        // every 3 bytes of input create 4 bytes of output
        for (auto i=0; i<11; i++)
        {
            const uint8_t bytes[3] = {value.data[3*i+0],value.data[3*i+1],i!=10 ? value.data[3*i+2]:uint8_t(0)};
            *(out++) = base[bytes[0]>>2];
            // take bottom bits of first byte to be top of the 6 bit, and 4 top bits of next byte to be bottom 4 bits of 6 bit
            *(out++) = base[((bytes[0]&0x03u)<<4)|(bytes[1]>>4)];
            // take bottom bits of the second byte to be top 4 bits of next byte, and top 2 bits of last byte to be bottom 2
            *(out++) = base[((bytes[1]&0x0fu)<<2)|(bytes[2]>>6)];
            *(out++) = base[bytes[2]&0x3Fu];
        }
        // padding is inferred from length
        return retval;
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

template<typename T, uint16_t N, uint16_t M>
struct to_string_helper<hlsl::matrix<T,N,M>>
{
    static std::string __call(const hlsl::matrix<T, N, M>& matrix)
    {
        std::stringstream output;
        output << '\n';
        for (int i = 0; i < N; ++i)
        {
            output << "{ ";
            for (int j = 0; j < M; ++j)
                output << matrix[i][j] << ", ";
            output << "}\n";
        }
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