// Internal src-only header. Do not include from public headers.
#ifndef _NBL_ASSET_IMPL_S_TEXT_PARSE_H_INCLUDED_
#define _NBL_ASSET_IMPL_S_TEXT_PARSE_H_INCLUDED_
#include "nbl/core/string/stringutil.h"
#include <charconv>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <fast_float/fast_float.h>
namespace nbl::asset::impl
{
//! Text token and numeric parsing helpers shared by interchange text formats.
struct TextParse
{
	struct LineCursor
	{
		const char* cursor = nullptr;
		const char* end = nullptr;
		inline std::optional<std::string_view> readLine()
		{
			if (!cursor || cursor >= end)
				return std::nullopt;
			const char* lineEnd = cursor;
			while (lineEnd < end && *lineEnd != '\0' && *lineEnd != '\r' && *lineEnd != '\n')
				++lineEnd;
			const std::string_view line(cursor, static_cast<size_t>(lineEnd - cursor));
			if (lineEnd < end && *lineEnd == '\r')
				++lineEnd;
			if (lineEnd < end && *lineEnd == '\n')
				++lineEnd;
			else if (lineEnd < end && *lineEnd == '\0')
				++lineEnd;
			cursor = lineEnd;
			return line;
		}
	};
	static inline bool isDigit(const char c) { return c >= '0' && c <= '9'; }
	//! Parses one arithmetic token and advances `ptr` on success.
	template<typename T>
	static inline bool parseNumber(const char*& ptr, const char* const end, T& out)
	{
		static_assert(std::is_arithmetic_v<T>);
		if constexpr (std::is_floating_point_v<T>)
		{
			const char* const start = ptr;
			if (start >= end)
				return false;
			const char* p = start;
			bool negative = false;
			if (*p == '-' || *p == '+')
			{
				negative = (*p == '-');
				++p;
				if (p >= end)
					return false;
			}
			// Fast path for the common plain-decimal subset: optional sign, digits, and an optional fractional part, but no exponent.
			// This follows the same broad idea as RapidJSON's StrtodFast: cheaply handle the dominant simple spellings before delegating
			// harder cases to the full parser. This is not a standalone general-purpose parser. Tokens with exponents or otherwise
			// non-trivial spellings still fall back to fast_float.
			if (*p != '.' && isDigit(*p))
			{
				uint64_t integerPart = 0ull;
				while (p < end && isDigit(*p))
				{
					integerPart = integerPart * 10ull + static_cast<uint64_t>(*p - '0');
					++p;
				}
				double value = static_cast<double>(integerPart);
				if (p < end && *p == '.')
				{
					const char* const dot = p;
					if ((dot + 7) <= end)
					{
						const char d0 = dot[1];
						const char d1 = dot[2];
						const char d2 = dot[3];
						const char d3 = dot[4];
						const char d4 = dot[5];
						const char d5 = dot[6];
						if (isDigit(d0) && isDigit(d1) && isDigit(d2) && isDigit(d3) && isDigit(d4) && isDigit(d5))
						{
							const bool hasNext = (dot + 7) < end;
							const char next = hasNext ? dot[7] : '\0';
							if ((!hasNext || !isDigit(next)) && (!hasNext || (next != 'e' && next != 'E')))
							{
								const uint32_t frac =
									static_cast<uint32_t>(d0 - '0') * 100000u +
									static_cast<uint32_t>(d1 - '0') * 10000u +
									static_cast<uint32_t>(d2 - '0') * 1000u +
									static_cast<uint32_t>(d3 - '0') * 100u +
									static_cast<uint32_t>(d4 - '0') * 10u +
									static_cast<uint32_t>(d5 - '0');
								value += static_cast<double>(frac) * 1e-6;
								ptr = dot + 7;
								out = static_cast<T>(negative ? -value : value);
								return true;
							}
						}
					}
					static constexpr double InvPow10[] = {
						1.0,
						1e-1, 1e-2, 1e-3, 1e-4, 1e-5,
						1e-6, 1e-7, 1e-8, 1e-9, 1e-10,
						1e-11, 1e-12, 1e-13, 1e-14, 1e-15,
						1e-16, 1e-17, 1e-18
					};
					++p;
					uint64_t fractionPart = 0ull;
					uint32_t fractionDigits = 0u;
					while (p < end && isDigit(*p))
					{
						if (fractionDigits >= (std::size(InvPow10) - 1u))
							break;
						fractionPart = fractionPart * 10ull + static_cast<uint64_t>(*p - '0');
						++fractionDigits;
						++p;
					}
					if (fractionDigits)
						value += static_cast<double>(fractionPart) * InvPow10[fractionDigits];
					if (p < end && isDigit(*p))
					{
						const auto parseResult = fast_float::from_chars(start, end, out);
						if (parseResult.ec != std::errc() || parseResult.ptr == start)
							return false;
						ptr = parseResult.ptr;
						return true;
					}
				}
				if (p < end && (*p == 'e' || *p == 'E'))
				{
					const auto parseResult = fast_float::from_chars(start, end, out);
					if (parseResult.ec != std::errc() || parseResult.ptr == start)
						return false;
					ptr = parseResult.ptr;
					return true;
				}
				ptr = p;
				out = static_cast<T>(negative ? -value : value);
				return true;
			}
			const auto parseResult = fast_float::from_chars(ptr, end, out);
			if (parseResult.ec != std::errc() || parseResult.ptr == ptr)
				return false;
			ptr = parseResult.ptr;
			return true;
		}
		else
		{
			const auto parseResult = std::from_chars(ptr, end, out);
			if (parseResult.ec != std::errc() || parseResult.ptr == ptr)
				return false;
			ptr = parseResult.ptr;
			return true;
		}
	}

	//! Parses one arithmetic token and succeeds only if the whole range was consumed.
	template<typename T>
	static inline bool parseExactNumber(const char* const begin, const char* const end, T& out) { auto ptr = begin; return parseNumber(ptr, end, out) && ptr == end; }

	//! `std::string_view` convenience wrapper over `parseExactNumber(begin,end,...)`.
	template<typename T>
	static inline bool parseExactNumber(const std::string_view token, T& out) { return parseExactNumber(token.data(), token.data() + token.size(), out); }

	//! Parses one arithmetic token and rejects zero.
	template<typename T>
	static inline bool parseNonZeroNumber(const char*& ptr, const char* const end, T& out) { return parseNumber(ptr, end, out) && out != static_cast<T>(0); }

	//! Returns true for inline whitespace accepted inside tokenized text formats.
	static inline bool isInlineWhitespace(const char c) { return c == ' ' || c == '\t' || c == '\v' || c == '\f'; }
	//! Skips spaces and tabs that stay within the current logical line.
	static inline void skipInlineWhitespace(const char*& ptr, const char* const end) { while (ptr < end && isInlineWhitespace(*ptr)) ++ptr; }
	//! Skips generic whitespace according to `core::isspace`.
	static inline void skipWhitespace(const char*& ptr, const char* const end) { while (ptr < end && core::isspace(*ptr)) ++ptr; }
	//! Trims leading and trailing whitespace from a token view.
	static inline std::string_view trimWhitespace(std::string_view token)
	{
		while (!token.empty() && core::isspace(token.front())) token.remove_prefix(1ull);
		while (!token.empty() && core::isspace(token.back())) token.remove_suffix(1ull);
		return token;
	}
	//! Reads one whitespace-delimited token and advances `cursor` past it.
	static inline std::optional<std::string_view> readToken(const char*& cursor, const char* const end)
	{
		skipWhitespace(cursor, end);
		if (cursor >= end)
			return std::nullopt;
		const auto* tokenEnd = cursor;
		while (tokenEnd < end && !core::isspace(*tokenEnd))
			++tokenEnd;
		const std::string_view token(cursor, static_cast<size_t>(tokenEnd - cursor));
		return cursor = tokenEnd, token;
	}
	//! Reads one line view from a contiguous text buffer and advances `cursor`.
	static inline std::optional<std::string_view> readLine(const char*& cursor, const char* const end)
	{
		LineCursor lineCursor = {.cursor = cursor, .end = end};
		auto line = lineCursor.readLine();
		cursor = lineCursor.cursor;
		return line;
	}
};
}
#endif
