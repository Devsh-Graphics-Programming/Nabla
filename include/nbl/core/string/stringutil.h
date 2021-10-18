// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_STRINGUTIL_H_INCLUDED__
#define __NBL_CORE_STRINGUTIL_H_INCLUDED__

#include <string>
#include <sstream>
#include <cwchar>
#include <cctype>
#include "stddef.h"
#include "string.h"
#include "utf8.h"

#include <nbl/system/path.h>

namespace nbl
{
namespace core
{
	//! Replaces all occurences of `findStr` in `source` string with `replaceStr`.
	/** @param source String that is to be modified.
	@param findStr String to look for in source.
	@param replaceStr String replacing found occurences of `findStr`.
	*/
	template<class T>
	inline void findAndReplaceAll(T& source, T const& findStr, T const& replaceStr)
	{
		for (size_t i = 0; (i = source.find(findStr, i)) != T::npos;)
		{
			source.replace(i, findStr.length(), replaceStr);
			i += replaceStr.length();
		}
	}

	//! Replaces all occurences of `findStr` in `source` string with `replaceStr`.
	/** @param source String that is to be modified.
	@param findStr String to look for in source.
	@param replaceStr String replacing found occurences of `findStr`.
	*/
	template<class T>
	inline void findAndReplaceAll(T& source, const typename T::pointer findStr, const typename T::pointer replaceStr)
	{
		findAndReplaceAll(source, T(findStr), T(replaceStr));
	}

	template<typename T>
	inline size_t length(const T*);
	template<>
	inline size_t length<char>(const char* str)
	{
		return strlen(str);
	}
	template<>
	inline size_t length<wchar_t>(const wchar_t* str)
	{
		return wcslen(str);
	}
	//! Calculates length of given string by trying to reach 0 (of type T) starting from `str` address.
	template<typename T>
	inline size_t length(const T* str)
	{
		for (size_t i = 0;; ++i)
			if (str[i] == T(0))
				return i;
	}

	//! Compares two strings, or their substrings, ignoring case of characters.
	/** @param str1 First cstring (i.e. const char*) to compare.
	@param pos1 Position of the first to compare character in the first string.
	@param str2 Second cstring (i.e. const char*) to compare.
	@param pos2 Position of the first to compare character in the second string.
	@param len Amount of characters to compare.
	@returns Whether strings are ignore-case-equal or not. `false` is also returned when comparing either of strings would result in going out of its range.
	*/
	template<typename T>
	inline bool equalsIgnoreCaseSubStr(const T* str1, const size_t& pos1, const T* str2, const size_t& pos2, const size_t& len)
	{
		if (pos1 + len > length(str1))
			return false;
		if (pos2 + len > length(str2))
			return false;

		for (const T* c1 = str1 + pos1, *c2 = str2 + pos2; c1 != str1 + pos1 + len; ++c1, ++c2)
		{
			if (tolower(*c1) != tolower(*c2))
				return false;
		}
		return true;
	}

	//! Compares two strings, or their substrings, ignoring case of characters.
	/** @param str1 First string to compare.
	@param pos1 Position of the first to compare character in the first string.
	@param str2 Second string to compare.
	@param pos2 Position of the first to compare character in the second string.
	@param len Amount of characters to compare.
	@returns Whether strings are ignore-case-equal or not. `false` is also returned when comparing either of strings would result in going out of its range.
	*/
	template<typename T>
	inline bool equalsIgnoreCaseSubStr(const T& str1, const size_t& pos1, const T& str2, const size_t& pos2, const size_t& len)
	{
		return equalsIgnoreCaseSubStr(str1.c_str(), pos1, str2.c_str(), pos2, len);
	}

	//! Compares two strings ignoring case of characters.
	/** @param str1 First cstring to compare.
	@param str2 Second cstring to compare.
	@returns Whether strings are ignore-case-equal or not. `false` is also returned when comparing either of strings would result in going out of its range.
	*/
	template<typename T>
	inline bool equalsIgnoreCase(const T* str1, const T* str2)
	{
		const size_t size2 = length(str2);
		if (length(str1) != size2)
			return false;

		return equalsIgnoreCaseSubStr<T>(str1, 0, str2, 0, size2);
	}

	//! Compares two strings ignoring case of characters.
	/** @param str1 First string to compare.
	@param str2 Second string to compare.
	@returns Whether strings are ignore-case-equal or not. `false` is also returned when comparing either of strings would result in going out of its range.
	*/
	template<typename T>
	inline bool equalsIgnoreCase(const T& str1, const T& str2)
	{
		if (str1.size() != str2.size())
			return false;

		return equalsIgnoreCaseSubStr<T>(str1, 0, str2, 0, str2.size());
	}

	//! Compares two strings.
	/**
	@param str1 First string to compare.
	@param str2 Second string to compare.
	@returns If sizes of the two strings differ - signed difference between two sizes (i.e. (str1.size()-str2.size()) );
		Otherwise - an integral value indicating the relationship between the strings:
			<0 - the first character that does not match has a lower value in str1 t in str2
			0  - both strings are equal
			>0 - the first character that does not match has a greater value in str1 than in str2
	*/
	template<typename T>
	inline int32_t strcmpi(const T* str1, const T* str2)
	{
		const T* c1 = str1;
		const T* c2 = str2;
		for (; (*c1!=0)&&(*c2!=0); ++c1, ++c2)
		{
			int32_t val1 = tolower(*c1);
			int32_t val2 = tolower(*c2);
			if (val1 != val2)
				return val1 - val2;
		}
		if (*c1) // first is longer
			return 1; // future core::strlen(c1)
		if (*c2)
			return -1; // future core::strlen(c2)

		return 0;
	}
	template<typename T>
	inline int32_t strcmpi(const T& str1, const T& str2)
	{
		if (str1.size() != str2.size())
			return str1.size() - str2.size();

		for (typename T::const_iterator c1 = str1.begin(), c2 = str2.begin(); c1 != str1.end(); ++c1, ++c2)
		{
			int32_t val1 = tolower(*c1);
			int32_t val2 = tolower(*c2);
			if (val1 != val2)
				return val1 - val2;
		}
		return 0;
	}

	//! DOCUMENTATION TODO
	struct CaseInsensitiveHash
	{
		inline std::size_t operator()(const std::string& val) const
		{
			std::size_t seed = 0;
			for (auto it = val.begin(); it != val.end(); it++)
			{
				seed ^= ~std::size_t(std::tolower(*it)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			}
			return seed;
		}
	};
	struct CaseInsensitiveEquals
	{
		inline bool operator()(const std::string& A, const std::string& B) const
		{
			return core::strcmpi(A, B)==0;
		}
	};

	//! Gets the the last character of given string.
	/** @param str1 Given string.
	@returns Last character of the string or 0 if contains no characters.
	*/
	template<class T>
	inline typename T::value_type lastChar(const T& str1)
	{
		if (str1.size())
		{
			return *(str1.end() - 1);
		}
		return 0;
	}

	// ----------- some basic quite often used string functions -----------------

	inline int32_t isFileExtension(const std::filesystem::path& filename, const std::initializer_list<std::string_view>& extensions)
	{
		std::string filename_str = filename.string();
		std::string file_ext = filename.extension().string();
		file_ext = nbl::system::extension_wo_dot(file_ext);
		auto found = std::find(extensions.begin(), extensions.end(), file_ext);
		if (found == extensions.end()) return 0;
		return found - extensions.begin() + 1;
	}

	//! Search if a filename has a proper extension.
	/** Compares file's extension to three given extensions ignoring case.
	@param filename String being the file's name.
	@param ext Variadic list of extensions to compare with
	@returns Boolean value indicating whether file is of one of given extensions.
	*/
	template<typename... ext_string_type>
	inline bool hasFileExtension(const std::filesystem::path& filename, const ext_string_type&... ext)
	{
		return isFileExtension(filename, { ext... }) > 0;
	}

	//! Delete path from filename.
	/** Clips given file name path to just the file's name without the rest of path.
	@param filename File names string.
	@returns Reference to filename (i.e. the parameter).
	*/
	inline std::filesystem::path& deletePathFromFilename(std::filesystem::path& filename)
	{
		filename = filename.filename();
		return filename;
	}

	//! Clips given file name string to given number of least significant path-tokens.
	/** @param filename File name string.
	@param pathCount Number of path-tokens to clip the given path to.
	@returns Reference to the first parameter.
	*/
	inline std::filesystem::path& deletePathFromPath(std::filesystem::path& filename, int32_t pathCount)
	{
		// delete path from filename
		int32_t i = filename.string().size();

		// search for path separator or beginning
		while (i >= 0)
		{
			if (filename.string()[i] == '/' || filename.string()[i] == '\\') //! On Linux just delete the '\\' from path
			{
				if (--pathCount <= 0)
					break;
			}
			--i;
		}

		if (i > 0)
		{
			filename.string()[i + 1] = 0;
		}
		else
			filename = "";
		return filename;
	}

	//! Looks if `file` is in the same directory of `path`. Returns offset of directory.
	/** @param path Path to compare with.
	@param file File name string.
	@returns offset of directory. 0 means in same directory; 1 means file is direct child of path, etc.
	*/
	inline int32_t isInSameDirectory(const std::filesystem::path& path, const std::filesystem::path& file)
	{
		int32_t subA = 0;
		int32_t subB = 0;
		int32_t pos;

		if (path.string().size() && !std::equal(path.string().begin(), path.string().end(), file.string().begin()))
			return -1;

		pos = 0;
		while ((pos = path.string().find("/", pos)) >= 0)
		{
			subA += 1;
			pos += 1;
		}

		pos = 0;
		while ((pos = file.string().find('/', pos)) >= 0)
		{
			subB += 1;
			pos += 1;
		}

		return subB - subA;
	}

	//! Replaces all occurences of backslash with regular slashes.
	/** @param inout Pointer to string to modify.
	*/
	inline void handleBackslashes(std::filesystem::path* inout)
	{
		std::replace(inout->string().begin(), inout->string().end(), '\\', '/'); //! On Linux just delete them
	}


	inline std::filesystem::path toLower(const std::filesystem::path& path)
	{
		return std::transform(path.string().begin(), path.string().end(), path.string().begin(),
			[](unsigned char c) { return std::tolower(c); });
	}

	//! Splits a path into essential components.
	/** Each of output parameters can be NULL - then this component will not be returned.
	@param[in] name Input path string.
	@param[out] path Pointer to string where path component will be returned.
	@param[out] filename Pointer to string where filename component will be returned.
	@param[out] extension Pointer to string where extension component will be returned.
	@param[in] make_lower Whether to return all components as lower-case-only strings.
	*/
	inline void splitFilename(const std::filesystem::path& name, std::filesystem::path* path = 0,
		std::filesystem::path * filename = 0, std::filesystem::path * extension = 0, bool make_lower = false)
	{
		if (path != nullptr)
		{
			*path = name.parent_path();
			handleBackslashes(path);
			if (make_lower) *path = toLower(*path);
		}
		if (filename != nullptr)
		{
			*filename = name.filename().replace_extension();
			if (make_lower) *filename = toLower(*filename);
		}
		if (extension != nullptr)
		{
			*extension = name.extension();
			if(make_lower) *extension = toLower(*extension);
		}
	}

	//! some standard function ( to remove dependencies )
#undef isdigit
#undef isspace
#undef isupper
//! Returns 0 or 1 indicating whether given character is a digit.
	inline int32_t isdigit(int32_t c) { return c >= '0' && c <= '9'; }
	//! Returns 0 or 1 indicating whether given character is a whitespace character.
	inline int32_t isspace(int32_t c) { return c == ' ' || c == '\f' || c == '\n' || c == '\r' || c == '\t' || c == '\v'; }
	//! Returns 0 or 1 indicating whether given character is an upper-case letter character.
	inline int32_t isupper(int32_t c) { return c >= 'A' && c <= 'Z'; }

}
}

#endif
