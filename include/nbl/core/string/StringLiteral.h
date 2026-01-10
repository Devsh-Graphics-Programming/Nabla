// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_CORE_STRING_LITERAL_H_INCLUDED_
#define _NBL_CORE_STRING_LITERAL_H_INCLUDED_

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace nbl::core
{

template<size_t N>
struct StringLiteral
{
    constexpr StringLiteral(const char (&str)[N])
    {
        std::copy_n(str, N, value);
    }

    char value[N];
};

}

// for compatibility's sake
#define NBL_CORE_UNIQUE_STRING_LITERAL_TYPE(STRING_LITERAL) nbl::core::StringLiteral<sizeof STRING_LITERAL>(STRING_LITERAL)

namespace nbl::core::detail
{

template<nbl::core::StringLiteral Key>
struct StringLiteralBufferType
{
    using type = void;
};

template<size_t Cap>
class StringLiteralBuffer
{
public:
    constexpr void append(char c)
    {
        if (!ensure_capacity(1))
            return;
        b[n++] = c;
    }
    constexpr void append(std::string_view sv)
    {
        if (!ensure_capacity(sv.size()))
            return;
        for (char c : sv)
            b[n++] = c;
    }
    constexpr void append(const char* s)
    {
        for (; *s; ++s)
            append(*s);
    }

    constexpr std::string_view view() const { return { b.data(), n }; }
    constexpr operator std::string_view() const { return view(); }
    constexpr const char* data() const { return b.data(); }
    constexpr size_t size() const { return n; }

private:
    constexpr bool ensure_capacity(size_t add)
    {
        if (n + add <= Cap)
            return true;
        if (std::is_constant_evaluated())
            throw "overflow";
        assert(false && "StringLiteralBuffer overflow");
        return false;
    }

    std::array<char, Cap> b{};
    size_t n = 0;
};

template<size_t Cap>
constexpr std::string_view to_string_view(const StringLiteralBuffer<Cap>& v)
{
    return v.view();
}

template<class Out>
constexpr void append_uint_padded(Out& o, unsigned value, int width)
{
    char buf[16];
    int len = 0;
    do
    {
        buf[len++] = static_cast<char>('0' + (value % 10u));
        value /= 10u;
    } while (value);
    while (len < width)
        buf[len++] = '0';
    for (int i = len - 1; i >= 0; --i)
        o.append(buf[i]);
}

template<class Out, class T>
constexpr void append_float_scientific(Out& o, T v)
{
    using Limits = std::numeric_limits<T>;
    constexpr int precision = Limits::max_digits10 - 1;
    if (v != v)
    {
        assert(false && "StringLiteralBuffer float format failed");
        return;
    }
    if constexpr (Limits::has_infinity)
    {
        if (v == Limits::infinity() || v == -Limits::infinity())
        {
            assert(false && "StringLiteralBuffer float format failed");
            return;
        }
    }
    if (v < T(0))
    {
        o.append('-');
        v = -v;
    }
    if (v == T(0))
    {
        o.append('0');
        o.append('.');
        for (int i = 0; i < precision; ++i)
            o.append('0');
        o.append('e');
        o.append('+');
        append_uint_padded(o, 0u, 2);
        return;
    }

    long double m = static_cast<long double>(v);
    int exp10 = 0;
    while (m >= 10.0L)
    {
        m /= 10.0L;
        ++exp10;
    }
    while (m < 1.0L)
    {
        m *= 10.0L;
        --exp10;
    }

    std::array<int, precision + 1> digits{};
    digits[0] = static_cast<int>(m);
    long double frac = m - static_cast<long double>(digits[0]);
    for (int i = 1; i <= precision; ++i)
    {
        frac *= 10.0L;
        int d = static_cast<int>(frac);
        if (d > 9)
            d = 9;
        digits[i] = d;
        frac -= static_cast<long double>(d);
    }

    frac *= 10.0L;
    int round_digit = static_cast<int>(frac);
    if (round_digit > 9)
        round_digit = 9;
    long double remainder = frac - static_cast<long double>(round_digit);
    bool round_up = false;
    if (round_digit > 5)
        round_up = true;
    else if (round_digit == 5)
    {
        if (remainder > 0.0L)
            round_up = true;
        else
            round_up = (digits[precision] % 2) != 0;
    }

    if (round_up)
    {
        int i = precision;
        for (; i >= 0; --i)
        {
            if (digits[i] < 9)
            {
                digits[i]++;
                break;
            }
            digits[i] = 0;
        }
        if (i < 0)
        {
            digits[0] = 1;
            for (int j = 1; j <= precision; ++j)
                digits[j] = 0;
            ++exp10;
        }
    }

    o.append(static_cast<char>('0' + digits[0]));
    o.append('.');
    for (int i = 1; i <= precision; ++i)
        o.append(static_cast<char>('0' + digits[i]));
    o.append('e');
    if (exp10 < 0)
    {
        o.append('-');
        exp10 = -exp10;
    }
    else
    {
        o.append('+');
    }
    const int exp_width = (exp10 >= 100) ? 3 : 2;
    append_uint_padded(o, static_cast<unsigned>(exp10), exp_width);
}

template<class Out, class T>
constexpr void put(Out& o, const T& v)
{
    using U = std::remove_cvref_t<T>;

    if constexpr (std::is_same_v<U, bool>)
    {
        o.append(v ? '1' : '0');
    }
    else if constexpr (std::is_integral_v<U>)
    {
        using UU = std::make_unsigned_t<U>;
        UU x{};

        if constexpr (std::is_signed_v<U>)
        {
            if (v < 0)
            {
                o.append('-');
                x = UU(-(v + 1)) + 1;
            }
            else
            {
                x = UU(v);
            }
        }
        else
        {
            x = UU(v);
        }

        char tmp[3 + sizeof(U) * 8];
        size_t k = 0;
        do {
            tmp[k++] = char('0' + (x % 10));
            x /= 10;
        } while (x);
        while (k)
            o.append(tmp[--k]);
    }
    else if constexpr (std::is_same_v<U, float> || std::is_same_v<U, double>)
    {
        append_float_scientific(o, v);
    }
    else if constexpr (std::is_floating_point_v<U>)
    {
        static_assert(!sizeof(U), "Unsupported %s argument type");
    }
    else if constexpr (std::is_convertible_v<U, std::string_view>)
    {
        o.append(std::string_view(v));
    }
    else if constexpr (std::is_same_v<U, const char*> || std::is_same_v<U, char*>)
    {
        o.append((const char*)v);
    }
    else
    {
        static_assert(!sizeof(U), "Unsupported %s argument type");
    }
}

template<nbl::core::StringLiteral Fmt, class Out, class... Args>
constexpr void append_printf_s(Out& out, const Args&... args)
{
    auto tup = std::forward_as_tuple(args...);
    size_t ai = 0;

    for (size_t i = 0; Fmt.value[i]; ++i)
    {
        if (Fmt.value[i] != '%')
        {
            out.append(Fmt.value[i]);
            continue;
        }

        char c = Fmt.value[++i];
        if (c == '%')
        {
            out.append('%');
            continue;
        }
        if (c == 's')
        {
            std::apply([&](auto const&... xs) {
                size_t k = 0;
                (((k++ == ai) ? (put(out, xs), 0) : 0), ...);
            }, tup);
            ++ai;
        }
    }
}

}

#endif // _NBL_CORE_STRING_LITERAL_H_INCLUDED_
