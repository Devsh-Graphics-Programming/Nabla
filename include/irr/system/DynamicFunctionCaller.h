#ifndef __IRR_DYNAMIC_FUNCTION_CALLER_H_INCLUDED__
#define __IRR_DYNAMIC_FUNCTION_CALLER_H_INCLUDED__


#include "irr/core/core.h"
#include "irr/system/DynamicLibraryFunctionPointer.h"

namespace irr
{
namespace system
{

template<class FuncPtrLoaderT=DefaultFuncPtrLoader>
class DynamicFunctionCallerBase : public core::Unmovable
{
	protected:
		static_assert(std::is_base_of<FuncPtrLoader,FuncPtrLoaderT>::value, "Need a function pointer loader derived from `FuncPtrLoader`");
		FuncPtrLoaderT loader;
	public:
		DynamicFunctionCallerBase() : loader() {}
		DynamicFunctionCallerBase(DynamicFunctionCallerBase&& other) : DynamicFunctionCallerBase()
		{
			operator(std::move(other));
		}
		template<typename... T>
		DynamicFunctionCallerBase(T&&... args) : loader(std::forward<T>(args)...)
		{
		}
		virtual ~DynamicFunctionCallerBase() = default;

		DynamicFunctionCallerBase& operator=(DynamicFunctionCallerBase&& other)
		{
			std::swap(loader, other.loader);
			return *this;
		}
};

}
}


#define IRR_SYSTEM_IMPL_INIT_DYNLIB_FUNCPTR(FUNC_NAME) ,p ## FUNC_NAME ## (Base::loader.loadFuncPtr( #FUNC_NAME ))
#define IRR_SYSTEM_IMPL_SWAP_DYNLIB_FUNCPTR(FUNC_NAME) std::swap(p ## FUNC_NAME,other.p ## FUNC_NAME);

#define IRR_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS( CLASS_NAME, FUNC_PTR_LOADER_TYPE, ... ) \
class CLASS_NAME : public irr::system::DynamicFunctionCallerBase<FUNC_PTR_LOADER_TYPE>\
{\
	public:\
		using Base = irr::system::DynamicFunctionCallerBase<FUNC_PTR_LOADER_TYPE>;\
\
		CLASS_NAME() : Base()\
		{\
		}\
		template<typename... T>\
		CLASS_NAME(T&& ... args) : Base(std::forward<T>(args)...)\
			IRR_FOREACH(IRR_SYSTEM_IMPL_INIT_DYNLIB_FUNCPTR,__VA_ARGS__)\
		{\
		}\
		CLASS_NAME(CLASS_NAME&& other) : CLASS_NAME()\
		{\
			operator=(std::move(other));\
		}\
		~CLASS_NAME() final = default;\
\
		CLASS_NAME& operator=(CLASS_NAME&& other)\
		{\
			Base::operator=(std::move(other));\
			IRR_FOREACH(IRR_SYSTEM_IMPL_SWAP_DYNLIB_FUNCPTR,__VA_ARGS__);\
			return *this;\
		}\
\
		IRR_FOREACH(IRR_SYSTEM_DECLARE_DYNLIB_FUNCPTR,__VA_ARGS__);\
\
}

#endif