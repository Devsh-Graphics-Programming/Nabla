#ifndef __IRR_DYNAMIC_LIBRARY_FUNCTION_POINTER_H_INCLUDED__
#define __IRR_DYNAMIC_LIBRARY_FUNCTION_POINTER_H_INCLUDED__


#include <functional>


#include "irr/core/core.h"


namespace irr
{
namespace system
{

template<typename FuncT, class UniqueStringType>
class DynamicLibraryFunctionPointer
{
	public:
		using result_type = typename std::function<FuncT>::result_type;

		DynamicLibraryFunctionPointer() : p(nullptr) {}
		DynamicLibraryFunctionPointer(DynamicLibraryFunctionPointer&& other) : DynamicLibraryFunctionPointer()
		{
			operator=(std::move(other));
		}
		DynamicLibraryFunctionPointer(void* ptr) : p(reinterpret_cast<FuncT*>(ptr)) {}
		~DynamicLibraryFunctionPointer()
		{
			p = nullptr;
		}

		inline explicit operator bool() const { return p; }
		inline bool operator!() const { return !p; }
	

		template<typename... T>
		inline result_type operator()(std::function<result_type(const char*)> error, T&& ... args)
		{
			if (p)
				return p(std::forward<T>(args)...);
			assert(error);
			return error(name);
		}

		template<typename... T>
		inline result_type operator()(std::function<void(const char*)> error, T&& ... args)
		{
			if (p)
				return p(std::forward<T>(args)...);
			else if (error)
				error(name);
			return result_type{};
		}

		template<typename... T>
		inline result_type operator()(T&& ... args)
		{
			if (p)
				return p(std::forward<T>(args)...);
			return result_type{};
		}


		inline DynamicLibraryFunctionPointer& operator=(DynamicLibraryFunctionPointer&& other)
		{
			std::swap(p,other.p);
			return *this;
		}

	protected:
		FuncT* p;

		const char* name = UniqueStringType::value;
};

}
}

#define IRR_SYSTEM_DECLARE_DYNLIB_FUNCPTR(FUNC_NAME) irr::system::DynamicLibraryFunctionPointer<decltype(FUNC_NAME),IRR_CORE_UNIQUE_STRING_LITERAL_TYPE(#FUNC_NAME)> p ## FUNC_NAME;

#endif