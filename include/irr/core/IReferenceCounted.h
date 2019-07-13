// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_IREFERENCE_COUNTED_H_INCLUDED__
#define __I_IREFERENCE_COUNTED_H_INCLUDED__

#include "irr/core/Types.h"
#include "irr/core/BaseClasses.h"
#include "irr/core/alloc/AlignedBase.h"

#include <atomic>

namespace irr
{
namespace core
{

	//! Base class of most objects of the Irrlicht Engine.
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

	A simple example:

	If you want to create a texture, you may want to call an imaginable method
	IDriver::createTexture. You call
	ITexture* texture = driver->createTexture(dimension2d<uint32_t>(128, 128));
	If you no longer need the texture, call texture->drop().

	If you want to load a texture, you may want to call imaginable method
	IDriver::loadTexture. You do this like
	ITexture* texture = driver->loadTexture("example.jpg");
	You will not have to drop the pointer to the loaded texture, because
	the name of the method does not start with 'create'. The texture
	is stored somewhere by the driver.
	*/
	class IRR_FORCE_EBO IReferenceCounted : public Interface, public AllocationOverrideDefault
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
		inline void grab() const { ReferenceCounter++; }

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
			_IRR_DEBUG_BREAK_IF(ctrVal == 0)
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
			_IRR_DEBUG_BREAK_IF(!ReferenceCounter.is_lock_free()) //incompatibile platform
#if __cplusplus >= 201703L
			static_assert(decltype(ReferenceCounter)::is_always_lock_free,"Unsupported Platform, Lock-less Atomic Reference Couting is Impossible!");
#endif //
		}

		// Old destructor, but needed virtual for abstractness!
		// _IRR_INTERFACE_CHILD_DEFAULT(IReferenceCounted);
		//! Destructor, no need to define really, but make it pure virtual to truly prevent instantiation.
		virtual ~IReferenceCounted() = 0;

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

		//! The reference counter. Mutable to do reference counting on const objects.
		mutable std::atomic<uint32_t> ReferenceCounter;
	};

	// Parameter types for special overloaded constructors
	struct dont_grab_t {};
	constexpr dont_grab_t dont_grab{};
	struct dont_drop_t {};
	constexpr dont_drop_t dont_drop{};

	// A RAII-like class to help you safeguard against memory leaks.
	// Will automagically drop reference counts when it goes out of scope
	template<class I_REFERENCE_COUNTED>
	class smart_refctd_ptr
	{
			static_assert(std::is_base_of<IReferenceCounted, I_REFERENCE_COUNTED>::value,"Wrong Base Class!");
			
			mutable I_REFERENCE_COUNTED* ptr; // since IReferenceCounted declares the refcount mutable atomic
			template<class U> friend class smart_refctd_ptr;
		public:
			constexpr smart_refctd_ptr() noexcept : ptr(nullptr) {}
			constexpr smart_refctd_ptr(std::nullptr_t) noexcept : ptr(nullptr) {}
			template<class U>
			explicit smart_refctd_ptr(U* _pointer) noexcept : ptr(_pointer)
			{
				if (_pointer)
					_pointer->grab();
			}
			template<class U>
			explicit smart_refctd_ptr(U* _pointer, dont_grab_t t) noexcept : ptr(_pointer) {}
			template<class U>
			smart_refctd_ptr(const smart_refctd_ptr<U>& other) noexcept : smart_refctd_ptr(other.ptr) {}
			template<class U>
			smart_refctd_ptr(smart_refctd_ptr<U>&& other) noexcept : smart_refctd_ptr()
			{
				if (ptr) // should only happen if constexpr (is convertible)
					ptr->drop();
				ptr = other.ptr;
				other.ptr = nullptr; // should only happen if constexpr (is convertible)
			}
			~smart_refctd_ptr() noexcept
			{
				if (ptr)
					ptr->drop();
			}

			template<class U>
			inline smart_refctd_ptr& operator=(U* _pointer) noexcept
			{
				if (_pointer)
					_pointer->grab();
				if (ptr)
					ptr->drop();
				ptr = _pointer;
				return *this;
			}
			template<class U>
			inline smart_refctd_ptr& operator=(const smart_refctd_ptr<U>& other) noexcept
			{
				return operator=(other.ptr);
			}
			template<class U>
			inline smart_refctd_ptr& operator=(smart_refctd_ptr<U>&& other) noexcept
			{
				if (ptr) // should only happen if constexpr (is convertible)
					ptr->drop();
				ptr = other.ptr;
				other.ptr = nullptr; // should only happen if constexpr (is convertible)
				return *this;
			}

			inline I_REFERENCE_COUNTED* get() { return ptr; }
			inline const I_REFERENCE_COUNTED* get() const { return ptr; }

			inline I_REFERENCE_COUNTED* operator->() { return ptr; }
			inline const I_REFERENCE_COUNTED* operator->() const { return ptr; }

			inline I_REFERENCE_COUNTED& operator*() { return *ptr; }
			inline const I_REFERENCE_COUNTED& operator*() const { return *ptr; }

			inline I_REFERENCE_COUNTED& operator[](size_t idx) { return ptr[idx]; }
			inline const I_REFERENCE_COUNTED& operator[](size_t idx) const { return ptr[idx]; }


			inline explicit operator bool() const { return ptr; }
			inline bool operator!() const { return !ptr; }

			template<class U>
			inline bool operator==(const smart_refctd_ptr<U> &other) const { return ptr == other.ptr; }
			template<class U>
			inline bool operator!=(const smart_refctd_ptr<U> &other) const { return ptr != other.ptr; }

			template<class U>
			inline bool operator<(const smart_refctd_ptr<U> &other) const { return ptr < other.ptr; }
			template<class U>
			inline bool operator>(const smart_refctd_ptr<U>& other) const { return ptr > other.ptr; }
	};

    template< class T, class... Args >
    inline smart_refctd_ptr<T> make_smart_refctd_ptr(Args&& ... args)
    {
        T* obj = new T(std::forward<Args>(args)...);
        smart_refctd_ptr<T> smart(obj, dont_grab);
        return smart;
    }

}
} // end namespace irr

#endif

