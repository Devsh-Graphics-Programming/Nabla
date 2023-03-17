#include "nbl/system/DefaultFuncPtrLoader.h"

#if defined(_NBL_WINDOWS_API_)
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h> 
	#include <stdio.h> 
#elif defined(_NBL_POSIX_API_)
	#include <dlfcn.h>
#endif


using namespace nbl;
using namespace nbl::system;


#if defined(_NBL_WINDOWS_API_)
#define LIB reinterpret_cast<HMODULE&>(lib)
#elif defined(_NBL_POSIX_API_)
#define lib
#endif

DefaultFuncPtrLoader::DefaultFuncPtrLoader(const char* name) : DefaultFuncPtrLoader()
{
	if (!name)
		return;

	// TODO: redo with either LoadLibraryExA or SetDllDirectoryA and linux equivalents to allow loading shared libraries
	// with other shared library dependencies from regular directories without changing CWD (which is not thread safe)
	#if defined(_NBL_WINDOWS_API_)
		std::string libname(name);
		libname += ".dll";
		LIB = LoadLibraryA(libname.c_str());
		if (!lib)
			LIB = LoadLibraryA(name);
	#elif defined(_NBL_POSIX_API_)
		std::string libname("lib");
		libname += name;
		libname += ".so";
		LIB = dlopen(libname.c_str(),RTLD_LAZY);
		if (!lib)
			LIB = dlopen(name,RTLD_LAZY);
	#endif
}

DefaultFuncPtrLoader::~DefaultFuncPtrLoader()
{
	if (lib!=nullptr)
	{
	#if defined(_NBL_WINDOWS_API_)
		FreeLibrary(LIB);
	#elif defined(_NBL_POSIX_API_)
		dlclose(LIB);
	#endif
	}
}

void* DefaultFuncPtrLoader::loadFuncPtr(const char* funcname)
{
	if (isLibraryLoaded())
	{
	#if defined(_NBL_WINDOWS_API_)
		return GetProcAddress(LIB,funcname);
	#elif defined(_NBL_POSIX_API_)
		return dlsym(lib,funcname);
	#endif
	}
	return nullptr;
}