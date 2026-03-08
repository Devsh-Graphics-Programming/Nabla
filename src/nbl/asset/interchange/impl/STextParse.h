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
//! Text token and numeric parsing helpers shared by interchange text formats.
struct TextParse
{
	//! Parses one arithmetic token and advances `ptr` on success.
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
};
}
#endif
