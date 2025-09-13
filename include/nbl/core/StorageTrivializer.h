#ifndef _NBL_CORE_STORAGE_TRIVIALIZER_H_INCLUDED_
#define _NBL_CORE_STORAGE_TRIVIALIZER_H_INCLUDED_

namespace nbl::core
{

// This construct makes it so that we don't trigger T's constructors and destructors.
template<typename T>
struct StorageTrivializer;

template<>
struct NBL_FORCE_EBO StorageTrivializer<void>
{
    void* getStorage() {return nullptr;}
    const void* getStorage() const {return nullptr;}

    void construct() {}
    void destruct() {}
};

template<typename T>
struct alignas(T) StorageTrivializer
{
    T* getStorage() {return reinterpret_cast<T*>(storage); }
    const T* getStorage() const {return reinterpret_cast<const T*>(storage);}
    
    template<typename... Args>
    void construct(Args&&... args)
    {
        new (getStorage()) T(std::forward<Args>(args)...);
    }
    void destruct()
    {
        getStorage()->~T();
    }

    uint8_t storage[sizeof(T)];
};

}

#endif