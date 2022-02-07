// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_CORE_I_IREFERENCE_COUNTED_H_INCLUDED__
#define __NBL_CORE_I_IREFERENCE_COUNTED_H_INCLUDED__

#include "nbl/core/Types.h"
#include "nbl/core/BaseClasses.h"
#include "nbl/core/alloc/AlignedBase.h"

#include <atomic>

namespace nbl
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
        _NBL_DEBUG_BREAK_IF(ctrVal == 0)
        if(ctrVal == 1)
        {
            // https://eli.thegreenplace.net/2015/c-deleting-destructors-and-virtual-operator-delete/
            delete this;  // aligned overrides of delete should do the job :D due to C++ standard trickery
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
        _NBL_DEBUG_BREAK_IF(!ReferenceCounter.is_lock_free())  //incompatibile platform
        static_assert(decltype(ReferenceCounter)::is_always_lock_free, "Unsupported Platform, Lock-less Atomic Reference Couting is Impossible!");
    }

    // Old destructor, but needed virtual for abstractness!
    // _NBL_INTERFACE_CHILD_DEFAULT(IReferenceCounted);
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
    static_assert(alignof(const char*) <= _NBL_SIMD_ALIGNMENT / 2u, "Pointer type is overaligned");
    static_assert(sizeof(const char*) <= _NBL_SIMD_ALIGNMENT / 2u, "Pointer type is overaligned");

    //! The reference counter. Mutable to do reference counting on const objects.
    mutable std::atomic<uint32_t> ReferenceCounter;
    static_assert(alignof(std::atomic<uint32_t>) <= _NBL_SIMD_ALIGNMENT / 2u, "This compiler has a problem with its atomic int decl!");
    static_assert(sizeof(std::atomic<uint32_t>) <= _NBL_SIMD_ALIGNMENT / 2u, "This compiler has a problem with its atomic int decl!");
};
static_assert(alignof(IReferenceCounted) == _NBL_SIMD_ALIGNMENT, "This compiler has a problem respecting alignment!");

// Parameter types for special overloaded constructors
struct dont_grab_t
{
};
constexpr dont_grab_t dont_grab{};
struct dont_drop_t
{
};
constexpr dont_drop_t dont_drop{};

// A RAII-like class to help you safeguard against memory leaks.
// Will automagically drop reference counts when it goes out of scope
template<class I_REFERENCE_COUNTED>
class smart_refctd_ptr
{
    static_assert(std::is_base_of<IReferenceCounted, I_REFERENCE_COUNTED>::value || std::is_same<IReferenceCounted, I_REFERENCE_COUNTED>::value, "Wrong Base Class!");

    mutable I_REFERENCE_COUNTED* ptr;  // since IReferenceCounted declares the refcount mutable atomic

    template<class U>
    friend class smart_refctd_ptr;
    template<class U, class T>
    friend smart_refctd_ptr<U> smart_refctd_ptr_static_cast(smart_refctd_ptr<T>&&);
    template<class U, class T>
    friend smart_refctd_ptr<U> smart_refctd_ptr_dynamic_cast(smart_refctd_ptr<T>&&);

    template<class U>
    void copy(const smart_refctd_ptr<U>& other) noexcept
    {
        if(other.ptr)
            other.ptr->grab();
        ptr = other.ptr;
    }
    template<class U>
    void move(smart_refctd_ptr<U>&& other) noexcept
    {
        ptr = other.ptr;
        other.ptr = nullptr;
    }

public:
    using pointee = I_REFERENCE_COUNTED;
    using value_type = I_REFERENCE_COUNTED*;

    struct hash
    {
        inline size_t operator()(const core::smart_refctd_ptr<I_REFERENCE_COUNTED>& ptr) const
        {
            return std::hash<void*>{}(ptr.get());
        }
    };

    constexpr smart_refctd_ptr() noexcept
        : ptr(nullptr) {}
    constexpr smart_refctd_ptr(std::nullptr_t) noexcept
        : ptr(nullptr) {}
    template<class U>
    explicit smart_refctd_ptr(U* _pointer) noexcept
        : ptr(_pointer)
    {
        if(_pointer)
            _pointer->grab();
    }
    template<class U>
    explicit smart_refctd_ptr(U* _pointer, dont_grab_t t) noexcept
        : ptr(_pointer) {}

    template<class U, std::enable_if_t<!std::is_same<U, I_REFERENCE_COUNTED>::value, int> = 0>
    smart_refctd_ptr(const smart_refctd_ptr<U>& other) noexcept
    {
        this->copy(other);
    }
    smart_refctd_ptr(const smart_refctd_ptr<I_REFERENCE_COUNTED>& other) noexcept
    {
        this->copy(other);
    }

    template<class U, std::enable_if_t<!std::is_same<U, I_REFERENCE_COUNTED>::value, int> = 0>
    smart_refctd_ptr(smart_refctd_ptr<U>&& other) noexcept
    {
        this->move(std::move(other));
    }
    smart_refctd_ptr(smart_refctd_ptr<I_REFERENCE_COUNTED>&& other) noexcept
    {
        this->move(std::move(other));
    }

    ~smart_refctd_ptr() noexcept
    {
        if(ptr)
            ptr->drop();
    }

    inline smart_refctd_ptr& operator=(const smart_refctd_ptr<I_REFERENCE_COUNTED>& other) noexcept
    {
        if(other.ptr)
            other.ptr->grab();
        if(ptr)
            ptr->drop();
        ptr = other.ptr;
        return *this;
    }
    template<class U, std::enable_if_t<!std::is_same<U, I_REFERENCE_COUNTED>::value, int> = 0>
    inline smart_refctd_ptr& operator=(const smart_refctd_ptr<U>& other) noexcept
    {
        if(other.ptr)
            other.ptr->grab();
        if(ptr)
            ptr->drop();
        ptr = other.ptr;
        return *this;
    }

    inline smart_refctd_ptr& operator=(smart_refctd_ptr<I_REFERENCE_COUNTED>&& other) noexcept
    {
        if(ptr)  // should only happen if constexpr (is convertible)
            ptr->drop();
        ptr = other.ptr;
        other.ptr = nullptr;  // should only happen if constexpr (is convertible)
        return *this;
    }
    //those std::enable_if_t's most likely not needed, but just to be sure (i put them to trigger SFINAE to be sure call to non-templated ctor is always generated in case of same type)
    template<class U, std::enable_if_t<!std::is_same<U, I_REFERENCE_COUNTED>::value, int> = 0>
    inline smart_refctd_ptr& operator=(smart_refctd_ptr<U>&& other) noexcept
    {
        if(ptr)  // should only happen if constexpr (is convertible)
            ptr->drop();
        ptr = other.ptr;
        other.ptr = nullptr;  // should only happen if constexpr (is convertible)
        return *this;
    }

    // so that you don't mix refcounting methods
    void grab() = delete;
    void grab() const = delete;
    bool drop() = delete;
    bool drop() const = delete;

    inline I_REFERENCE_COUNTED* const& get() const { return ptr; }

    inline I_REFERENCE_COUNTED* operator->() const { return ptr; }

    inline I_REFERENCE_COUNTED& operator*() const { return *ptr; }

    inline I_REFERENCE_COUNTED& operator[](size_t idx) { return ptr[idx]; }
    inline const I_REFERENCE_COUNTED& operator[](size_t idx) const { return ptr[idx]; }

    inline explicit operator bool() const { return ptr; }
    inline bool operator!() const { return !ptr; }

    template<class U>
    inline bool operator==(const smart_refctd_ptr<U>& other) const { return ptr == other.ptr; }
    template<class U>
    inline bool operator!=(const smart_refctd_ptr<U>& other) const { return ptr != other.ptr; }

    template<class U>
    inline bool operator<(const smart_refctd_ptr<U>& other) const { return ptr < other.ptr; }
    template<class U>
    inline bool operator>(const smart_refctd_ptr<U>& other) const { return ptr > other.ptr; }
};
static_assert(sizeof(smart_refctd_ptr<IReferenceCounted>) == sizeof(IReferenceCounted*), "smart_refctd_ptr has a memory overhead!");

template<class T, class... Args>
inline smart_refctd_ptr<T> make_smart_refctd_ptr(Args&&... args)
{
    T* obj = new T(std::forward<Args>(args)...);
    smart_refctd_ptr<T> smart(obj, dont_grab);
    return smart;
}

template<class U, class T>
inline smart_refctd_ptr<U> smart_refctd_ptr_static_cast(const smart_refctd_ptr<T>& smart_ptr)
{
    return smart_refctd_ptr<U>(static_cast<U*>(smart_ptr.get()));
}
template<class U, class T>
inline smart_refctd_ptr<U> smart_refctd_ptr_static_cast(smart_refctd_ptr<T>&& smart_ptr)
{
    T* ptr = nullptr;
    std::swap(ptr, smart_ptr.ptr);
    return smart_refctd_ptr<U>(static_cast<U*>(ptr), dont_grab);
}

template<class U, class T>
inline smart_refctd_ptr<U> move_and_static_cast(smart_refctd_ptr<T>& smart_ptr)
{
    return smart_refctd_ptr_static_cast<U, T>(std::move(smart_ptr));
}

template<class U, class T>
inline smart_refctd_ptr<U> smart_refctd_ptr_dynamic_cast(const smart_refctd_ptr<T>& smart_ptr)
{
    return smart_refctd_ptr<U>(dynamic_cast<U*>(smart_ptr.get()));
}
template<class U, class T>
inline smart_refctd_ptr<U> smart_refctd_ptr_dynamic_cast(smart_refctd_ptr<T>&& smart_ptr)
{
    T* ptr = nullptr;
    std::swap(ptr, smart_ptr.ptr);
    return smart_refctd_ptr<U>(dynamic_cast<U*>(ptr), dont_grab);
}

template<class U, class T>
inline smart_refctd_ptr<U> move_and_dynamic_cast(smart_refctd_ptr<T>& smart_ptr)
{
    return smart_refctd_ptr_dynamic_cast<U, T>(std::move(smart_ptr));
}

}
}  // end namespace nbl

namespace std
{
template<typename T>
struct hash<nbl::core::smart_refctd_ptr<T>>
{
    std::size_t operator()(const nbl::core::smart_refctd_ptr<T>& k) const
    {
        return reinterpret_cast<std::size_t>(k.get());
    }
};

}

#endif
