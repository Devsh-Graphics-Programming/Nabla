// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_OS_H_INCLUDED__
#define __NBL_OS_H_INCLUDED__

#include "nbl/core/declarations.h"
#include "nbl/core/math/floatutil.h"

#include "irrString.h"
#include "ILogger.h"

namespace nbl
{

namespace os
{

	class Printer // TODO: this needs to go
	{
	    Printer() = delete;
	public:
		// prints out a string to the console out stdout or debug log or whatever
		static void print(const std::string& message) {}
		static void log(const std::string& message, ELOG_LEVEL ll = ELL_INFORMATION) {}
		static void log(const std::wstring& message, ELOG_LEVEL ll = ELL_INFORMATION) {}
		static void log(const std::string& message, const std::string& hint, ELOG_LEVEL ll = ELL_INFORMATION) {}

		static ILogger* Logger;
	};


} // end namespace os
} // end namespace nbl


#endif

