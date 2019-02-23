#ifndef __C_CONCURRENT_OBJECT_CACHE_H_INCLUDED__
#define __C_CONCURRENT_OBJECT_CACHE_H_INCLUDED__

#include "CObjectCache.h"
#include "../source/Irrlicht/FW_Mutex.h"

namespace irr { namespace core
{

namespace impl
{
    struct IRR_FORCE_EBO CConcurrentObjectCacheBase
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

        template<typename RngT>
        static bool isNonZeroRange(const RngT& _rng) { return BaseCache::isNonZeroRange(_rng); }

        inline bool insert(const K& _key, T* _val)
        {
            this->m_lock.lockWrite();
            const bool r = BaseCache::insert(_key, _val);
            this->m_lock.unlockWrite();
            return r;
        }

        inline bool contains(const T* _object) const
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
        bool swapObjectValue(const K& _key, const T* _obj, T* _val)
        {
            this->m_lock.lockWrite();
            bool r = BaseCache::swapObjectValue(_key, _obj, _val);
            this->m_lock.unlockWrite();
            return r;
        }

        bool getAndStoreKeyRangeOrReserve(const K& _key, size_t& _inOutStorageSize, MutablePairType* _out, bool* _gotAll)
        {
            this->m_lock.lockWrite();
            const bool r = BaseCache::getAndStoreKeyRangeOrReserve(_key, _inOutStorageSize, _out, _gotAll);
            this->m_lock.unlockWrite();
            return r;
        }

        inline bool removeObject(T* _object, const K& _key)
        {
            this->m_lock.lockWrite();
            const bool r = BaseCache::removeObject(_object, _key);
            this->m_lock.unlockWrite();
            return r;
        }

        inline bool findAndStoreRange(const K& _key, size_t& _inOutStorageSize, MutablePairType* _out)
        {
            m_lock.lockRead();
            const bool r = BaseCache::findAndStoreRange(_key, _inOutStorageSize, _out);
            m_lock.unlockRead();
            return r;
        }

        inline bool findAndStoreRange(const K& _key, size_t& _inOutStorageSize, MutablePairType* _out) const
        {
            m_lock.lockRead();
            const bool r = BaseCache::findAndStoreRange(_key, _inOutStorageSize, _out);
            m_lock.unlockRead();
            return r;
        }

        inline bool findAndStoreRange(const K& _key, size_t& _inOutStorageSize, CachedType** _out)
        {
            m_lock.lockRead();
            const bool r = BaseCache::findAndStoreRange(_key, _inOutStorageSize, _out);
            m_lock.unlockRead();
            return r;
        }

        inline bool findAndStoreRange(const K& _key, size_t& _inOutStorageSize, CachedType** _out) const
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

        inline bool changeObjectKey(T* _obj, const K& _key, const K& _newKey)
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
