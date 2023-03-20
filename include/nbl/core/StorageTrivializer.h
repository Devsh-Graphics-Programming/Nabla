#ifndef __NBL_CORE_STORAGE_TRIVIALIZER_H_INCLUDED__
#define __NBL_CORE_STORAGE_TRIVIALIZER_H_INCLUDED__

namespace nbl::core
{

// This construct makes it so that we don't trigger T's constructors and destructors.
template <typename T>
struct alignas(T) StorageTrivializer
{
    uint8_t storage[sizeof(T)];
};

}

#endif