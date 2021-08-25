// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_SYSTEM_DEFAULT_FUNC_PTR_LOADER_H_INCLUDED__
#define __NBL_SYSTEM_DEFAULT_FUNC_PTR_LOADER_H_INCLUDED__

#include "nbl/system/FuncPtrLoader.h"

#if defined(_NBL_WINDOWS_API_)
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h> 
	#include <stdio.h> 
#elif defined(_NBL_POSIX_API_)
	#include <dlfcn.h>
#endif


namespace nbl
{
namespace system
{

class DefaultFuncPtrLoader final : FuncPtrLoader
{
	protected:
		#if defined(_NBL_WINDOWS_API_)
			HMODULE lib;
		#elif defined(_NBL_POSIX_API_)
			void* lib;
		#endif
	public:
		DefaultFuncPtrLoader() : lib(NULL) {}
		DefaultFuncPtrLoader(const char* name) : DefaultFuncPtrLoader()
		{
			#if defined(_NBL_WINDOWS_API_)
				std::string libname(name);
				libname += ".dll";
				lib = LoadLibraryA(libname.c_str());
			#elif defined(_NBL_POSIX_API_)
				std::string libname("lib");
				libname += name;
				libname += ".so";
				lib = dlopen(libname.c_str(), RTLD_LAZY);
			#endif
		}
		DefaultFuncPtrLoader(DefaultFuncPtrLoader&& other) : DefaultFuncPtrLoader()
		{
			operator=(std::move(other));
		}
		~DefaultFuncPtrLoader()
		{
			if (lib != NULL)
			#if defined(_NBL_WINDOWS_API_)
				FreeLibrary(lib);
			#elif defined(_NBL_POSIX_API_)
				dlclose(lib);
			#endif
		}

		inline DefaultFuncPtrLoader& operator=(DefaultFuncPtrLoader&& other)
		{
			std::swap(lib, other.lib);
			return *this;
		}

		inline bool isLibraryLoaded() override final
		{
			return lib!=NULL;
		}

		inline void* loadFuncPtr(const char* funcname) override final
		{
			if (isLibraryLoaded())
			{
			#if defined(_NBL_WINDOWS_API_)
				return GetProcAddress(lib,funcname);
			#elif defined(_NBL_POSIX_API_)
				return dlsym(lib,funcname);
			#endif
			}
			return nullptr;
		}
};

}
}

#endif