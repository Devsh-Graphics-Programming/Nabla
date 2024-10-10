#include "nbl/core/hash/blake.h"

using namespace nbl;

core::blake3_hasher::blake3_hasher()
{
    ::blake3_hasher_init(&m_state);
}

core::blake3_hasher& core::blake3_hasher::update(const void* data, const size_t bytes)
{
    ::blake3_hasher_update(&m_state,data,bytes);
    return *this;
}

template<typename T>
core::blake3_hasher& core::blake3_hasher::operator<<(const T& input)
{
    update_impl<T>::__call(*this,input);
    return *this;
}

core::blake3_hasher::operator core::blake3_hash_t() const
{
    core::blake3_hash_t retval;
    // the blake3 docs say that the hasher can be finalized multiple times
    ::blake3_hasher_finalize(&m_state,retval.data,sizeof(retval));
    return retval;
}