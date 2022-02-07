// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_SYSTEM_DYNAMIC_FUNCTION_CALLER_H_INCLUDED__
#define __NBL_SYSTEM_DYNAMIC_FUNCTION_CALLER_H_INCLUDED__

#include "nbl/core/core.h"
#include "nbl/system/DynamicLibraryFunctionPointer.h"

namespace nbl
{
namespace system
{
template<class FuncPtrLoaderT = DefaultFuncPtrLoader>
class DynamicFunctionCallerBase : public core::Unmovable
{
protected:
    static_assert(std::is_base_of<FuncPtrLoader, FuncPtrLoaderT>::value, "Need a function pointer loader derived from `FuncPtrLoader`");
    FuncPtrLoaderT loader;

public:
    DynamicFunctionCallerBase()
        : loader() {}
    DynamicFunctionCallerBase(DynamicFunctionCallerBase&& other)
        : DynamicFunctionCallerBase()
    {
        operator=(std::move(other));
    }
    template<typename... T>
    DynamicFunctionCallerBase(T&&... args)
        : loader(std::forward<T>(args)...)
    {
    }
    virtual ~DynamicFunctionCallerBase() = default;

    DynamicFunctionCallerBase& operator=(DynamicFunctionCallerBase&& other)
    {
        std::swap<FuncPtrLoaderT>(loader, other.loader);
        return *this;
    }
};

}
}

#define NBL_SYSTEM_IMPL_INIT_DYNLIB_FUNCPTR(FUNC_NAME) , p##FUNC_NAME##(Base::loader.loadFuncPtr(#FUNC_NAME))
#define NBL_SYSTEM_IMPL_SWAP_DYNLIB_FUNCPTR(FUNC_NAME) std::swap(p##FUNC_NAME, other.p##FUNC_NAME);

#define NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(CLASS_NAME, FUNC_PTR_LOADER_TYPE, ...)     \
    class CLASS_NAME : public nbl::system::DynamicFunctionCallerBase<FUNC_PTR_LOADER_TYPE>          \
    {                                                                                               \
    public:                                                                                         \
        using Base = nbl::system::DynamicFunctionCallerBase<FUNC_PTR_LOADER_TYPE>;                  \
                                                                                                    \
        CLASS_NAME() : Base()                                                                       \
        {                                                                                           \
        }                                                                                           \
        template<typename... T>                                                                     \
        CLASS_NAME(T&&... args) : Base(std::forward<T>(args)...)                                    \
                                      NBL_FOREACH(NBL_SYSTEM_IMPL_INIT_DYNLIB_FUNCPTR, __VA_ARGS__) \
        {                                                                                           \
        }                                                                                           \
        CLASS_NAME(CLASS_NAME&& other) : CLASS_NAME()                                               \
        {                                                                                           \
            operator=(std::move(other));                                                            \
        }                                                                                           \
        ~CLASS_NAME() final = default;                                                              \
                                                                                                    \
        CLASS_NAME& operator=(CLASS_NAME&& other)                                                   \
        {                                                                                           \
            Base::operator=(std::move(other));                                                      \
            NBL_FOREACH(NBL_SYSTEM_IMPL_SWAP_DYNLIB_FUNCPTR, __VA_ARGS__);                          \
            return *this;                                                                           \
        }                                                                                           \
                                                                                                    \
        NBL_FOREACH(NBL_SYSTEM_DECLARE_DYNLIB_FUNCPTR, __VA_ARGS__);                                \
    }

#endif
