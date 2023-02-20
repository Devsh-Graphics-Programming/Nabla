// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef __NBL_SYSTEM_DYNAMIC_LIBRARY_FUNCTION_POINTER_H_INCLUDED__
#define __NBL_SYSTEM_DYNAMIC_LIBRARY_FUNCTION_POINTER_H_INCLUDED__


#include "nbl/core/declarations.h"
#include "nbl/core/string/StringLiteral.h"

#include <functional>


namespace nbl::system
{

template<typename FuncT, core::StringLiteral name>
class NBL_API DynamicLibraryFunctionPointer
{
	public:
		using result_type = typename std::function<FuncT>::result_type;

		inline DynamicLibraryFunctionPointer() : p(nullptr) {}
		inline DynamicLibraryFunctionPointer(DynamicLibraryFunctionPointer&& other) : DynamicLibraryFunctionPointer()
		{
			operator=(std::move(other));
		}
		inline DynamicLibraryFunctionPointer(void* ptr) : p(reinterpret_cast<FuncT*>(ptr)) {}
		inline ~DynamicLibraryFunctionPointer()
		{
			p = nullptr;
		}


		inline explicit operator bool() const { return p; }
		inline bool operator!() const { return !p; }


		inline FuncT* operator&() const { return p; }
	

		/*template<typename... T>
		inline result_type operator()(std::function<result_type(const char*)> error, T&& ... args) const
		{
			if (p)
				return p(std::forward<T>(args)...);
			assert(error);
			return error(name.value);
		}*/

		template<typename... T>
		inline result_type operator()(std::function<void(const char*)> error, T&& ... args) const
		{
			if (p)
				return p(std::forward<T>(args)...);
			else if (error)
				error(name.value);
			if constexpr (!std::is_void_v<result_type>)
			{
				return result_type{};
			}
		}

		template<typename... T>
		inline result_type operator()(T&& ... args) const
		{
			if (p)
				return p(std::forward<T>(args)...);
			if constexpr (!std::is_void_v<result_type>)
			{
				return result_type{};
			}
		}


		inline DynamicLibraryFunctionPointer& operator=(DynamicLibraryFunctionPointer&& other)
		{
			std::swap(p,other.p);
			return *this;
		}

	protected:
		FuncT* p;
};

}

#define NBL_SYSTEM_DECLARE_DYNLIB_FUNCPTR(FUNC_NAME) nbl::system::DynamicLibraryFunctionPointer<decltype(FUNC_NAME),NBL_CORE_UNIQUE_STRING_LITERAL_TYPE(#FUNC_NAME)> p ## FUNC_NAME;

#endif