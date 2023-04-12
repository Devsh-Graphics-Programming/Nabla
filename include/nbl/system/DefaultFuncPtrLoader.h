// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_SYSTEM_DEFAULT_FUNC_PTR_LOADER_H_INCLUDED_
#define _NBL_SYSTEM_DEFAULT_FUNC_PTR_LOADER_H_INCLUDED_


#include "nbl/system/FuncPtrLoader.h"


namespace nbl::system
{

class DefaultFuncPtrLoader final : FuncPtrLoader
{
		void* lib;

	public:
		inline DefaultFuncPtrLoader() : lib(nullptr) {}
		NBL_API2 DefaultFuncPtrLoader(const char* name);
		inline DefaultFuncPtrLoader(DefaultFuncPtrLoader&& other) : DefaultFuncPtrLoader()
		{
			operator=(std::move(other));
		}
		NBL_API2 ~DefaultFuncPtrLoader();

		inline DefaultFuncPtrLoader& operator=(DefaultFuncPtrLoader&& other)
		{
			std::swap(lib,other.lib);
			return *this;
		}

		inline bool isLibraryLoaded() override final
		{
			return lib!=nullptr;
		}

		void* loadFuncPtr(const char* funcname) override final;
};

}

#endif