#ifndef __IRR_DEFAULT_FUNC_PTR_LOADER_H_INCLUDED__
#define __IRR_DEFAULT_FUNC_PTR_LOADER_H_INCLUDED__

#include "irr/system/FuncPtrLoader.h"

#if defined(_IRR_WINDOWS_API_)
	#include <windows.h> 
	#include <stdio.h> 
#elif defined(_IRR_POSIX_API_)
	#include <dlfcn.h>
#endif


namespace irr
{
namespace system
{

class DefaultFuncPtrLoader final : FuncPtrLoader
{
	protected:
		#if defined(_IRR_WINDOWS_API_)
			HINSTANCE lib;
		#elif defined(_IRR_POSIX_API_)
			void* lib;
		#endif
	public:
		DefaultFuncPtrLoader() : lib(NULL) {}
		DefaultFuncPtrLoader(const char* name) : DefaultFuncPtrLoader()
		{
			#if defined(_IRR_WINDOWS_API_)
				std::string libname(name);
				libname += ".dll";
				lib = LoadLibrary(libname.c_str());
			#elif defined(_IRR_POSIX_API_)
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
			#if defined(_IRR_WINDOWS_API_)
				FreeLibrary(lib);
			#elif defined(_IRR_POSIX_API_)
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
			#if defined(_IRR_WINDOWS_API_)
				return GetProcAddress(lib,funcname);
			#elif defined(_IRR_POSIX_API_)
				return dlsym(lib,funcname);
			#endif
		}
};

}
}

#endif