// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_SYSTEM_DEFAULT_FUNC_PTR_LOADER_H_INCLUDED_
#define _NBL_SYSTEM_DEFAULT_FUNC_PTR_LOADER_H_INCLUDED_

#include "nbl/system/FuncPtrLoader.h"

#if defined(_NBL_WINDOWS_API_)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#elif defined(_NBL_POSIX_API_)
#include <dlfcn.h>
#endif

namespace nbl::system
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
    DefaultFuncPtrLoader()
        : lib(NULL) {}
    DefaultFuncPtrLoader(const char* name)
        : DefaultFuncPtrLoader()
    {
        if(!name)
            return;

// TODO: redo with either LoadLibraryExA or SetDllDirectoryA and linux equivalents to allow loading shared libraries
// with other shared library dependencies from regular directories without changing CWD (which is not thread safe)
#if defined(_NBL_WINDOWS_API_)
        std::string libname(name);
        libname += ".dll";
        lib = LoadLibraryA(libname.c_str());
        if(!lib)
            lib = LoadLibraryA(name);
#elif defined(_NBL_POSIX_API_)
        std::string libname("lib");
        libname += name;
        libname += ".so";
        lib = dlopen(libname.c_str(), RTLD_LAZY);
        if(!lib)
            lib = dlopen(name, RTLD_LAZY);
#endif
    }
    DefaultFuncPtrLoader(DefaultFuncPtrLoader&& other)
        : DefaultFuncPtrLoader()
    {
        operator=(std::move(other));
    }
    ~DefaultFuncPtrLoader()
    {
        if(lib != NULL)
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
        return lib != NULL;
    }

    inline void* loadFuncPtr(const char* funcname) override final
    {
        if(isLibraryLoaded())
        {
#if defined(_NBL_WINDOWS_API_)
            return GetProcAddress(lib, funcname);
#elif defined(_NBL_POSIX_API_)
            return dlsym(lib, funcname);
#endif
        }
        return nullptr;
    }
};

}

#endif