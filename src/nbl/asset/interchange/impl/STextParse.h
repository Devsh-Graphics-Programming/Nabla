// Internal src-only header. Do not include from public headers.
#ifndef _NBL_ASSET_IMPL_S_TEXT_PARSE_H_INCLUDED_
#define _NBL_ASSET_IMPL_S_TEXT_PARSE_H_INCLUDED_
#include "nbl/core/string/stringutil.h"
#include <charconv>
#include <optional>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <fast_float/fast_float.h>
namespace nbl::asset::impl
{
struct TextParse
{
	template<typename T>
	static inline bool parseNumber(const char*& ptr, const char* const end, T& out)
	{
		static_assert(std::is_arithmetic_v<T>);
		if constexpr (std::is_floating_point_v<T>)
		{
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
	template<typename T>
	static inline bool parseExactNumber(const char* const begin, const char* const end, T& out) { auto ptr = begin; return parseNumber(ptr, end, out) && ptr == end; }
	template<typename T>
	static inline bool parseExactNumber(const std::string_view token, T& out) { return parseExactNumber(token.data(), token.data() + token.size(), out); }
	template<typename T>
	static inline bool parseNonZeroNumber(const char*& ptr, const char* const end, T& out) { return parseNumber(ptr, end, out) && out != static_cast<T>(0); }
	static inline bool isInlineWhitespace(const char c) { return c == ' ' || c == '\t' || c == '\v' || c == '\f'; }
	static inline void skipInlineWhitespace(const char*& ptr, const char* const end) { while (ptr < end && isInlineWhitespace(*ptr)) ++ptr; }
	static inline void skipWhitespace(const char*& ptr, const char* const end) { while (ptr < end && core::isspace(*ptr)) ++ptr; }
	static inline std::string_view trimWhitespace(std::string_view token)
	{
		while (!token.empty() && core::isspace(token.front())) token.remove_prefix(1ull);
		while (!token.empty() && core::isspace(token.back())) token.remove_suffix(1ull);
		return token;
	}
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
};
}
#endif
