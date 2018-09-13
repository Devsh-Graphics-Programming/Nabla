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
        using RangeType = std::pair<typename BaseCache::IteratorType, typename BaseCache::IteratorType>;

        using BaseCache::BaseCache;

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

        //! Returns true if had to insert
        bool swapObjectValue(const K& _key, const const T* _obj, T* _val)
        {
            this->m_lock.lockWrite();
            bool r = BaseCache::swapObjectValue(_key, _obj, _val);
            this->m_lock.unlockWrite();
            return r;
        }

        bool getKeyRangeOrReserve(typename BaseCache::RangeType* _outrange, const K& _key)
        {
            this->m_lock.lockWrite();
            bool r = BaseCache::getKeyRangeOrReserve(_outrange, _key);
            this->m_lock.unlockWrite();
            return r;
        }

        inline typename BaseCache::RangeType findRange(const K& _key)
        {
            this->m_lock.lockRead();
            typename BaseCache::RangeType r = BaseCache::findRange(_key);
            this->m_lock.unlockRead();
            return r;
        }

        inline const typename BaseCache::RangeType findRange(const K& _key) const
        {
            this->m_lock.lockRead();
            const typename BaseCache::RangeType r = BaseCache::findRange(_key);
            this->m_lock.unlockRead();
            return r;
        }

        inline void removeObject(T* _object, const K& _key)
        {
            this->m_lock.lockWrite();
            BaseCache::removeObject(_object, _key);
            this->m_lock.unlockWrite();
        }
    };
}

template<
    typename K,
    typename T,
    template<typename...> class ContainerT_T = std::vector
>
using CConcurrentObjectCache =
    impl::CMakeCacheConcurrent<
        CObjectCache<K, T, ContainerT_T>
    >;

template<
    typename K,
    typename T,
    template<typename...> class ContainerT_T = std::vector
>
using CConcurrentMultiObjectCache =
    impl::CMakeCacheConcurrent<
        CMultiObjectCache<K, T, ContainerT_T>
    >;

}}

#endif
