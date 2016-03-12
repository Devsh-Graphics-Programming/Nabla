// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine" and the "irrXML" project.
// For conditions of distribution and use, see copyright notice in irrlicht.h and irrXML.h

#ifndef __IRR_ALLOCATOR_H_INCLUDED__
#define __IRR_ALLOCATOR_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "irrTypes.h"
#include <new>
// necessary for older compilers
#include <memory.h>

namespace irr
{
namespace core
{

#ifdef DEBUG_CLIENTBLOCK
#undef DEBUG_CLIENTBLOCK
#define DEBUG_CLIENTBLOCK new
#endif

//! Very simple allocator implementation, containers using it can be used across dll boundaries
 #ifdef __IRR_COMPILE_WITH_AVX
template <typename T, std::size_t Alignment=32>
#else
template <typename T, std::size_t Alignment=16>
#endif
class irrAllocator
{
public:

	//! Destructor
	virtual ~irrAllocator() {}

	//! Allocate memory for an array of objects
	T* allocate(size_t cnt)
	{
		return (T*)internal_new(cnt* sizeof(T));
	}

	//! Deallocate memory for an array of objects
	void deallocate(T* ptr)
	{
		internal_delete(ptr);
	}

	//! Construct an element
	void construct(T* ptr, const T&e)
	{
		new ((void*)ptr) T(e);
	}

	//! Destruct an element
	void destruct(T* ptr)
	{
		ptr->~T();
	}

protected:

#ifdef __IRR_COMPILE_WITH_X86_SIMD_
	virtual void* internal_new(size_t cnt)
	{
		void *memoryallocatedaligned = 0;
#ifdef _IRR_WINDOWS_
		memoryallocatedaligned = _aligned_malloc(cnt,Alignment);
#else
		int dummy = posix_memalign((void**)&memoryallocatedaligned,Alignment,cnt);
#endif
		return memoryallocatedaligned;
	}

	virtual void internal_delete(void* ptr)
	{
#ifdef _IRR_WINDOWS_
        _aligned_free(ptr);
#else
        free(ptr);
#endif
	}
#else
	virtual void* internal_new(size_t cnt)
	{
		return operator new(cnt);
	}

	virtual void internal_delete(void* ptr)
	{
		operator delete(ptr);
	}
#endif
};


//! Fast allocator, only to be used in containers inside the same memory heap.
/** Containers using it are NOT able to be used it across dll boundaries. Use this
when using in an internal class or function or when compiled into a static lib */
 #ifdef __IRR_COMPILE_WITH_AVX
template <typename T, std::size_t Alignment=32>
#else
template <typename T, std::size_t Alignment=16>
#endif
class irrAllocatorFast
{
public:

#ifdef __IRR_COMPILE_WITH_X86_SIMD_
	//! Allocate memory for an array of objects
	T* allocate(size_t cnt)
	{
		cnt *= sizeof(T);
		T *memoryallocatedaligned = 0;
#ifdef _IRR_WINDOWS_
		memoryallocatedaligned = (T*)_aligned_malloc(cnt,Alignment);
#else
		posix_memalign((void**)&memoryallocatedaligned,Alignment,cnt);
#endif
		return memoryallocatedaligned;
	}

	//! Deallocate memory for an array of objects
	void deallocate(T* ptr)
	{
#ifdef _IRR_WINDOWS_
        _aligned_free(ptr);
#else
        free(ptr);
#endif
	}
#else
	//! Allocate memory for an array of objects
	T* allocate(size_t cnt)
	{
		return (T*)operator new(cnt* sizeof(T));
	}

	//! Deallocate memory for an array of objects
	void deallocate(T* ptr)
	{
		operator delete(ptr);
	}
#endif // __IRR_COMPILE_WITH_X86_SIMD_
	//! Construct an element
	void construct(T* ptr, const T&e)
	{
		new ((void*)ptr) T(e);
	}

	//! Destruct an element
	void destruct(T* ptr)
	{
		ptr->~T();
	}
};



#ifdef DEBUG_CLIENTBLOCK
#undef DEBUG_CLIENTBLOCK
#define DEBUG_CLIENTBLOCK new( _CLIENT_BLOCK, __FILE__, __LINE__)
#endif

//! defines an allocation strategy
enum eAllocStrategy
{
	ALLOC_STRATEGY_SAFE    = 0,
	ALLOC_STRATEGY_DOUBLE  = 1,
	ALLOC_STRATEGY_SQRT    = 2
};


} // end namespace core
} // end namespace irr

#endif

