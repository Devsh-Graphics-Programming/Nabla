// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_C_CONCURRENT_OBJECT_CACHE_H_INCLUDED__
#define __NBL_C_CONCURRENT_OBJECT_CACHE_H_INCLUDED__

#include "CObjectCache.h"
#include "nbl/system/SReadWriteSpinLock.h"

namespace nbl { namespace core
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

        mutable system::SReadWriteSpinLock m_lock;

    protected:
        auto lock_read() const { return system::read_lock_guard<>(m_lock); }
        auto lock_write() const { return system::write_lock_guard<>(m_lock); }
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
            auto lk = lock_write();
            const bool r = BaseCache::insert(_key, _val);
            return r;
        }

        inline bool contains(typename BaseCache::ImmutableValueType_impl& _object) const
        {
            auto lk = lock_read();
            const bool r = BaseCache::contains(_object);
            return r;
        }

        inline size_t getSize() const
        {
            auto lk = lock_read();
            const size_t r = BaseCache::getSize();
            return r;
        }

        inline void clear()
        {
            auto lk = lock_write();
            BaseCache::clear();
        }

        //! Returns true if had to insert
        bool swapObjectValue(const typename BaseCache::KeyType_impl& _key, const typename BaseCache::ImmutableValueType_impl& _obj, const typename BaseCache::ValueType_impl& _val)
        {
            auto lk = lock_write();
            bool r = BaseCache::swapObjectValue(_key, _obj, _val);
            return r;
        }

        bool getAndStoreKeyRangeOrReserve(const typename BaseCache::KeyType_impl& _key, size_t& _inOutStorageSize, typename BaseCache::ValueType_impl* _out, bool* _gotAll)
        {
            auto lk = lock_write();
            const bool r = BaseCache::getAndStoreKeyRangeOrReserve(_key, _inOutStorageSize, _out, _gotAll);
            return r;
        }

        inline bool removeObject(const typename BaseCache::ValueType_impl& _obj, const typename BaseCache::KeyType_impl& _key)
        {
            auto lk = lock_write();
            const bool r = BaseCache::removeObject(_obj, _key);
            return r;
        }

        inline bool findAndStoreRange(const typename BaseCache::KeyType_impl& _key, size_t& _inOutStorageSize, typename BaseCache::MutablePairType* _out)
        {
            auto lk = lock_read();
            const bool r = BaseCache::findAndStoreRange(_key, _inOutStorageSize, _out);
            return r;
        }

        inline bool findAndStoreRange(const typename BaseCache::KeyType_impl& _key, size_t& _inOutStorageSize, typename BaseCache::MutablePairType* _out) const
        {
            auto lk = lock_read();
            const bool r = BaseCache::findAndStoreRange(_key, _inOutStorageSize, _out);
            return r;
        }

        inline bool findAndStoreRange(const typename BaseCache::KeyType_impl& _key, size_t& _inOutStorageSize, typename BaseCache::ValueType_impl* _out)
        {
            auto lk = lock_read();
            const bool r = BaseCache::findAndStoreRange(_key, _inOutStorageSize, _out);
            return r;
        }

        inline bool findAndStoreRange(const typename BaseCache::KeyType_impl& _key, size_t& _inOutStorageSize, typename BaseCache::ValueType_impl* _out) const
        {
            auto lk = lock_read();
            const bool r = BaseCache::findAndStoreRange(_key, _inOutStorageSize, _out);
            return r;
        }

        inline bool outputAll(size_t& _inOutStorageSize, MutablePairType* _out) const
        {
            auto lk = lock_read();
            const bool r = BaseCache::outputAll(_inOutStorageSize, _out);
            return true;
        }

        inline bool changeObjectKey(const typename BaseCache::ValueType_impl& _obj, const typename BaseCache::KeyType_impl& _key, const typename BaseCache::KeyType_impl& _newKey)
        {
            auto lk = lock_write();
            const bool r = BaseCache::changeObjectKey(_obj, _key, _newKey);
            return r;
        }
    };
}

template<
    typename K,
    typename T,
    template<typename...> class ContainerT_T = std::vector,
    typename Alloc = core::allocator<typename impl::key_val_pair_type_for<ContainerT_T, K, T>::type>
>
using CConcurrentObjectCache =
    impl::CMakeCacheConcurrent<
        CObjectCache<K, T, ContainerT_T, Alloc>
    >;

template<
    typename K,
    typename T,
    template<typename...> class ContainerT_T = std::vector,
    typename Alloc = core::allocator<typename impl::key_val_pair_type_for<ContainerT_T, K, T>::type>
>
using CConcurrentMultiObjectCache =
    impl::CMakeCacheConcurrent<
        CMultiObjectCache<K, T, ContainerT_T, Alloc>
    >;

}}

#endif
