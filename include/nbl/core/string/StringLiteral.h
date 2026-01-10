// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_CORE_STRING_LITERAL_H_INCLUDED_
#define _NBL_CORE_STRING_LITERAL_H_INCLUDED_

#include <algorithm>
#include <array>
#include <cstddef>
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
struct StringLiteralBuffer
{
    std::array<char, Cap> b{};
    size_t n = 0;

    constexpr void append(char c)
    {
        if (n >= Cap)
            throw "overflow";
        b[n++] = c;
    }
    constexpr void append(std::string_view sv) { for (char c : sv) append(c); }
    constexpr void append(const char* s) { for (; *s; ++s) append(*s); }

    constexpr std::string_view view() const { return { b.data(), n }; }
    constexpr const char* data() const { return b.data(); }
    constexpr size_t size() const { return n; }
};

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
