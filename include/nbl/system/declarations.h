// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_SYSTEM_DECLARATIONS_H_INCLUDED_
#define _NBL_SYSTEM_DECLARATIONS_H_INCLUDED_

#include "nbl/core/declarations.h"

// basic stuff
#include "nbl/system/DynamicLibraryFunctionPointer.h"
#include "nbl/system/FuncPtrLoader.h"
#include "nbl/system/DefaultFuncPtrLoader.h"
#include "nbl/system/DynamicFunctionCaller.h"
#include "nbl/system/SReadWriteSpinLock.h"

// files
#include "nbl/system/IFile.h"

// archives
#include "nbl/system/CMountDirectoryArchive.h"

// loggers
#include "nbl/system/CStdoutLogger.h"
#include "nbl/system/CFileLogger.h"

//whole system
#if defined(_NBL_PLATFORM_WINDOWS_)
#	include "nbl/system/CColoredStdoutLoggerWin32.h"
#	include "nbl/system/CSystemWin32.h"
#elif defined(_NBL_PLATFORM_LINUX_)
#	include "nbl/system/CColoredStdoutLoggerANSI.h"
#	include "nbl/system/CSystemLinux.h"
#else
	#error "Unsupported Platform"
#endif // TODO more platforms (android)

// frameworks (ugh, doesn't work!)
//#include "nbl/system/IApplicationFramework.h"

#endif