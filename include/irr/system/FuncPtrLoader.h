#ifndef __IRR_FUNC_PTR_LOADER_H_INCLUDED__
#define __IRR_FUNC_PTR_LOADER_H_INCLUDED__


#include "irr/core/core.h"
#include "irr/system/DynamicLibraryFunctionPointer.h"

namespace irr
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

		inline FuncPtrLoader&& operator=(FuncPtrLoader&& other) {}
	public:
		virtual bool isLibraryLoaded() = 0;

		virtual void* loadFuncPtr(const char* funcname) = 0;

		/* When C++ gets reflection
		template<typename FuncT>
		auto loadFuncPtr()
		{
			using FuncPtrT = decltype(&FuncT);
			constexpr char FunctionName[] = std::reflection::name<FuncPtrT>::value;
			return DynamicLibraryFunctionPointer<FuncPtrT,IRR_CORE_UNIQUE_STRING_LITERAL_TYPE(FunctionName)>(this->loadFuncPtr(FunctionName));
		}
		*/
};

}
}

#endif