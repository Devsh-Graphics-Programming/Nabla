// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_SYSTEM_FUNC_PTR_LOADER_H_INCLUDED__
#define __NBL_SYSTEM_FUNC_PTR_LOADER_H_INCLUDED__

#include "nbl/core/core.h"
#include "nbl/system/DynamicLibraryFunctionPointer.h"

namespace nbl
{
namespace system
{
class FuncPtrLoader : public core::Uncopyable
{
protected:
    FuncPtrLoader() = default;
    FuncPtrLoader(FuncPtrLoader&& other)
    {
        operator=(std::move(other));
    }
    virtual ~FuncPtrLoader() = default;

    inline FuncPtrLoader& operator=(FuncPtrLoader&& other) { return *this; }

public:
    virtual bool isLibraryLoaded() = 0;

    virtual void* loadFuncPtr(const char* funcname) = 0;

    /* When C++ gets reflection
		template<typename FuncT>
		auto loadFuncPtr()
		{
			using FuncPtrT = decltype(&FuncT);
			constexpr char FunctionName[] = std::reflection::name<FuncPtrT>::value;
			return DynamicLibraryFunctionPointer<FuncPtrT,NBL_CORE_UNIQUE_STRING_LITERAL_TYPE(FunctionName)>(this->loadFuncPtr(FunctionName));
		}
		*/
};

}
}

#endif
