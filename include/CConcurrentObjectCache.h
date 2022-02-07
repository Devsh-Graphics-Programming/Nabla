// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_C_CONCURRENT_OBJECT_CACHE_H_INCLUDED__
#define __NBL_C_CONCURRENT_OBJECT_CACHE_H_INCLUDED__

#include "CObjectCache.h"
#include "FW_Mutex.h"

namespace nbl
{
namespace core
{
namespace impl
{
struct NBL_FORCE_EBO CConcurrentObjectCacheBase
{
    CConcurrentObjectCacheBase() = default;
    // explicitely making concurrent caches non-copy-and-move-constructible and non-copy-and-move-assignable
    CConcurrentObjectCacheBase(const CConcurrentObjectCacheBase&) = delete;
    CConcurrentObjectCacheBase(CConcurrentObjectCacheBase&&) = delete;
    CConcurrentObjectCacheBase& operator=(const CConcurrentObjectCacheBase&) = delete;
    CConcurrentObjectCacheBase& operator=(CConcurrentObjectCacheBase&&) = delete;

    struct
    {
        void lockRead() const { FW_AtomicCounterIncr(ctr); }
        void unlockRead() const { FW_AtomicCounterDecr(ctr); }
        void lockWrite() const { FW_AtomicCounterBlock(ctr); }
        void unlockWrite() const { FW_AtomicCounterUnBlock(ctr); }

    private:
        mutable FW_AtomicCounter ctr = 0;
    } m_lock;
};

template<typename CacheT>
class CMakeCacheConcurrent : private impl::CConcurrentObjectCacheBase, private CacheT
{
    using BaseCache = CacheT;
    using K = typename BaseCache::KeyType_impl;
    using T = typename BaseCache::CachedType;

public:
    using IteratorType = typename BaseCache::IteratorType;
    using ConstIteratorType = typename BaseCache::ConstIteratorType;
    using RevIteratorType = typename BaseCache::RevIteratorType;
    using ConstRevIteratorType = typename BaseCache::ConstRevIteratorType;
    using RangeType = typename BaseCache::RangeType;
    using ConstRangeType = typename BaseCache::ConstRangeType;
    using PairType = typename BaseCache::PairType;
    using MutablePairType = typename BaseCache::MutablePairType;
    using CachedType = T;
    using KeyType = typename BaseCache::KeyType;

    using BaseCache::BaseCache;

    inline bool insert(const typename BaseCache::KeyType_impl& _key, const typename BaseCache::ValueType_impl& _val)
    {
        this->m_lock.lockWrite();
        const bool r = BaseCache::insert(_key, _val);
        this->m_lock.unlockWrite();
        return r;
    }

    inline bool contains(typename BaseCache::ImmutableValueType_impl& _object) const
    {
        this->m_lock.lockRead();
        const bool r = BaseCache::contains(_object);
        this->m_lock.unlockRead();
        return r;
    }

    inline size_t getSize() const
    {
        this->m_lock.lockRead();
        const size_t r = BaseCache::getSize();
        this->m_lock.unlockRead();
        return r;
    }

    inline void clear()
    {
        this->m_lock.lockWrite();
        BaseCache::clear();
        this->m_lock.unlockWrite();
    }

    //! Returns true if had to insert
    bool swapObjectValue(const typename BaseCache::KeyType_impl& _key, const typename BaseCache::ImmutableValueType_impl& _obj, const typename BaseCache::ValueType_impl& _val)
    {
        this->m_lock.lockWrite();
        bool r = BaseCache::swapObjectValue(_key, _obj, _val);
        this->m_lock.unlockWrite();
        return r;
    }

    bool getAndStoreKeyRangeOrReserve(const typename BaseCache::KeyType_impl& _key, size_t& _inOutStorageSize, typename BaseCache::ValueType_impl* _out, bool* _gotAll)
    {
        this->m_lock.lockWrite();
        const bool r = BaseCache::getAndStoreKeyRangeOrReserve(_key, _inOutStorageSize, _out, _gotAll);
        this->m_lock.unlockWrite();
        return r;
    }

    inline bool removeObject(const typename BaseCache::ValueType_impl& _obj, const typename BaseCache::KeyType_impl& _key)
    {
        this->m_lock.lockWrite();
        const bool r = BaseCache::removeObject(_obj, _key);
        this->m_lock.unlockWrite();
        return r;
    }

    inline bool findAndStoreRange(const typename BaseCache::KeyType_impl& _key, size_t& _inOutStorageSize, typename BaseCache::MutablePairType* _out)
    {
        m_lock.lockRead();
        const bool r = BaseCache::findAndStoreRange(_key, _inOutStorageSize, _out);
        m_lock.unlockRead();
        return r;
    }

    inline bool findAndStoreRange(const typename BaseCache::KeyType_impl& _key, size_t& _inOutStorageSize, typename BaseCache::MutablePairType* _out) const
    {
        m_lock.lockRead();
        const bool r = BaseCache::findAndStoreRange(_key, _inOutStorageSize, _out);
        m_lock.unlockRead();
        return r;
    }

    inline bool findAndStoreRange(const typename BaseCache::KeyType_impl& _key, size_t& _inOutStorageSize, typename BaseCache::ValueType_impl* _out)
    {
        m_lock.lockRead();
        const bool r = BaseCache::findAndStoreRange(_key, _inOutStorageSize, _out);
        m_lock.unlockRead();
        return r;
    }

    inline bool findAndStoreRange(const typename BaseCache::KeyType_impl& _key, size_t& _inOutStorageSize, typename BaseCache::ValueType_impl* _out) const
    {
        m_lock.lockRead();
        const bool r = BaseCache::findAndStoreRange(_key, _inOutStorageSize, _out);
        m_lock.unlockRead();
        return r;
    }

    inline bool outputAll(size_t& _inOutStorageSize, MutablePairType* _out) const
    {
        m_lock.lockRead();
        const bool r = BaseCache::outputAll(_inOutStorageSize, _out);
        m_lock.unlockRead();
        return r;
    }

    inline bool changeObjectKey(const typename BaseCache::ValueType_impl& _obj, const typename BaseCache::KeyType_impl& _key, const typename BaseCache::KeyType_impl& _newKey)
    {
        m_lock.lockWrite();
        const bool r = BaseCache::changeObjectKey(_obj, _key, _newKey);
        m_lock.unlockWrite();
        return r;
    }
};
}

template<
    typename K,
    typename T,
    template<typename...> class ContainerT_T = std::vector,
    typename Alloc = core::allocator<typename impl::key_val_pair_type_for<ContainerT_T, K, T>::type> >
using CConcurrentObjectCache =
    impl::CMakeCacheConcurrent<
        CObjectCache<K, T, ContainerT_T, Alloc> >;

template<
    typename K,
    typename T,
    template<typename...> class ContainerT_T = std::vector,
    typename Alloc = core::allocator<typename impl::key_val_pair_type_for<ContainerT_T, K, T>::type> >
using CConcurrentMultiObjectCache =
    impl::CMakeCacheConcurrent<
        CMultiObjectCache<K, T, ContainerT_T, Alloc> >;

}
}

#endif
