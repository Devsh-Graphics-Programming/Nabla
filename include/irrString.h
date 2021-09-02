// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_STRING_H_INCLUDED__
#define __NBL_STRING_H_INCLUDED__

#include <stdio.h>
#include <string>
#include <stdlib.h>

#include "nbl/core/alloc/aligned_allocator.h"
#include "nbl/core/alloc/AlignedBase.h"

namespace nbl
{
namespace core
{

//! Very simple string class with some useful features.
/** string<char> and string<wchar_t> both accept Unicode AND ASCII/Latin-1,
so you can assign Unicode to string<char> and ASCII/Latin-1 to string<wchar_t>
(and the other way round) if you want to.

However, note that the conversation between both is not done using any encoding.
This means that char strings are treated as ASCII/Latin-1, not UTF-8, and
are simply expanded to the equivalent wchar_t, while Unicode/wchar_t
characters are truncated to 8-bit ASCII/Latin-1 characters, discarding all
other information in the wchar_t.
*/

enum eLocaleID
{
	NBL_LOCALE_ANSI = 0,
	NBL_LOCALE_GERMAN = 1
};

static eLocaleID locale_current = NBL_LOCALE_ANSI;
static inline void locale_set ( eLocaleID id )
{
	locale_current = id;
}

//! Returns a character converted to lower case
static inline uint32_t locale_lower ( uint32_t x )
{
	switch ( locale_current )
	{
		case NBL_LOCALE_GERMAN:
		case NBL_LOCALE_ANSI:
			break;
	}
	// ansi
	return x >= 'A' && x <= 'Z' ? x + 0x20 : x;
}

//! Returns a character converted to upper case
static inline uint32_t locale_upper ( uint32_t x )
{
	switch ( locale_current )
	{
		case NBL_LOCALE_GERMAN:
		case NBL_LOCALE_ANSI:
			break;
	}

	// ansi
	return x >= 'a' && x <= 'z' ? x + ( 'A' - 'a' ) : x;
}



} // end namespace core
} // end namespace nbl

#endif

