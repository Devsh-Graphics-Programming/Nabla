// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifndef _NBL_CORE_I_IREFERENCE_COUNTED_H_INCLUDED_
#define _NBL_CORE_I_IREFERENCE_COUNTED_H_INCLUDED_

#include "nbl/core/decl/Types.h"
#include "nbl/core/decl/BaseClasses.h"
#include "nbl/core/alloc/AlignedBase.h"

#include <atomic>

namespace nbl::core
{

//! Base class of most objects of the Nabla Engine.
/** This class provides reference counting through the methods grab() and drop().
It also is able to store a debug string for every instance of an object.
Most objects of the Irrlicht
Engine are derived from IReferenceCounted, and so they are reference counted.

When you create an object in the Irrlicht engine, calling a method
which starts with 'create', an object is created, and you get a pointer
to the new object. If you no longer need the object, you have
to call drop(). This will destroy the object, if grab() was not called
in another part of you program, because this part still needs the object.
Note, that you only need to call drop() to the object, if you created it,
and the method had a 'create' in it.
*/
class NBL_FORCE_EBO IReferenceCounted : public Interface, public AllocationOverrideDefault
{
	public:
		//! Grabs the object. Increments the reference counter by one.
		/** Someone who calls grab() to an object, should later also
		call drop() to it. If an object never gets as much drop() as
		grab() calls, it will never be destroyed. The
		IReferenceCounted class provides a basic reference counting
		mechanism with its methods grab() and drop(). Most objects of
		the Irrlicht Engine are derived from IReferenceCounted, and so
		they are reference counted.

		When you create an object in the Irrlicht engine, calling a
		method which starts with 'create', an object is created, and
		you get a pointer to the new object. If you no longer need the
		object, you have to call drop(). This will destroy the object,
		if grab() was not called in another part of you program,
		because this part still needs the object. Note, that you only
		need to call drop() to the object, if you created it, and the
		method had a 'create' in it.

		A simple example:

		If you want to create a texture, you may want to call an
		imaginable method IDriver::createTexture. You call
		ITexture* texture = driver->createTexture(dimension2d<uint32_t>(128, 128));
		If you no longer need the texture, call texture->drop().
		If you want to load a texture, you may want to call imaginable
		method IDriver::loadTexture. You do this like
		ITexture* texture = driver->loadTexture("example.jpg");
		You will not have to drop the pointer to the loaded texture,
		because the name of the method does not start with 'create'.
		The texture is stored somewhere by the driver. */
		inline uint32_t grab() const { return ReferenceCounter++; }

		//! Drops the object. Decrements the reference counter by one.
		/** The IReferenceCounted class provides a basic reference
		counting mechanism with its methods grab() and drop(). Most
		objects of the Irrlicht Engine are derived from
		IReferenceCounted, and so they are reference counted.

		When you create an object in the Irrlicht engine, calling a
		method which starts with 'create', an object is created, and
		you get a pointer to the new object. If you no longer need the
		object, you have to call drop(). This will destroy the object,
		if grab() was not called in another part of you program,
		because this part still needs the object. Note, that you only
		need to call drop() to the object, if you created it, and the
		method had a 'create' in it.

		A simple example:

		If you want to create a texture, you may want to call an
		imaginable method IDriver::createTexture. You call
		ITexture* texture = driver->createTexture(dimension2d<uint32_t>(128, 128));
		If you no longer need the texture, call texture->drop().
		If you want to load a texture, you may want to call imaginable
		method IDriver::loadTexture. You do this like
		ITexture* texture = driver->loadTexture("example.jpg");
		You will not have to drop the pointer to the loaded texture,
		because the name of the method does not start with 'create'.
		The texture is stored somewhere by the driver.
		\return True, if the object was deleted. */
		inline bool drop() const
		{
			auto ctrVal = ReferenceCounter--;
			// someone is doing bad reference counting.
			_NBL_DEBUG_BREAK_IF(ctrVal == 0)
			if (ctrVal==1)
			{
			    // https://eli.thegreenplace.net/2015/c-deleting-destructors-and-virtual-operator-delete/
				delete this; // aligned overrides of delete should do the job :D due to C++ standard trickery
				return true;
			}

			return false;
		}

		//! Get the reference count, due to threading it might be slightly outdated.
		/** \return Recent value of the reference counter. */
		inline int32_t getReferenceCount() const
		{
			return ReferenceCounter.load();
		}

		//! Returns the debug name of the object.
		/** The Debugname may only be set and changed by the object
		itself. This method should only be used in Debug mode.
		\return Returns a string, previously set by setDebugName(); */
		inline const char* getDebugName() const
		{
			return DebugName;
		}

	protected:
		//! Constructor.
		IReferenceCounted()
			: DebugName(0), ReferenceCounter(1)
		{
			_NBL_DEBUG_BREAK_IF(!ReferenceCounter.is_lock_free()) //incompatibile platform
			static_assert(decltype(ReferenceCounter)::is_always_lock_free,"Unsupported Platform, Lock-less Atomic Reference Couting is Impossible!");
		}

		// Old destructor, but needed virtual for abstractness!
		// _NBL_INTERFACE_CHILD_DEFAULT(IReferenceCounted);
		//! Destructor, no need to define really, but make it pure virtual to truly prevent instantiation.
		NBL_API2 virtual ~IReferenceCounted() = 0;

		//! Sets the debug name of the object.
		/** The Debugname may only be set and changed by the object
		itself. This method should only be used in Debug mode.
		\param newName: New debug name to set. */
		inline void setDebugName(const char* newName)
		{
			DebugName = newName;
		}

	private:
		//! The debug name.
		const char* DebugName;
		static_assert(alignof(const char*) <= _NBL_SIMD_ALIGNMENT/2u, "Pointer type is overaligned");
		static_assert(sizeof(const char*) <= _NBL_SIMD_ALIGNMENT/2u, "Pointer type is overaligned");

		//! The reference counter. Mutable to do reference counting on const objects.
		mutable std::atomic<uint32_t> ReferenceCounter;
		static_assert(alignof(std::atomic<uint32_t>) <= _NBL_SIMD_ALIGNMENT/2u, "This compiler has a problem with its atomic int decl!");
		static_assert(sizeof(std::atomic<uint32_t>) <= _NBL_SIMD_ALIGNMENT/2u, "This compiler has a problem with its atomic int decl!");
};
static_assert(alignof(IReferenceCounted) == _NBL_SIMD_ALIGNMENT, "This compiler has a problem respecting alignment!");

} // end namespace nbl::core

#endif

