#ifndef __C_CONCURRENT_OBJECT_CACHE_H_INCLUDED__
#define __C_CONCURRENT_OBJECT_CACHE_H_INCLUDED__

#include "CObjectCache.h"
#include "../source/Irrlicht/FW_Mutex.h"

namespace irr { namespace core
{

namespace impl
{
    struct CConcurrentObjectCacheBase
    {
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
    class CMakeCacheConcurrent_common : protected impl::CConcurrentObjectCacheBase, protected CacheT
    {
        using CacheBase = CacheT;
        using K = typename CacheBase::KeyType_impl;
        using T = typename CacheBase::CachedType;

    public:
        inline explicit CMakeCacheConcurrent_common(const std::function<void(T*)>& _greeting, const std::function<void(T*)>& _disposal) : CacheBase(_greeting, _disposal) {}
        inline explicit CMakeCacheConcurrent_common(std::function<void(T*)>&& _greeting = nullptr, std::function<void(T*)>&& _disposal = nullptr) : CacheBase(std::move(_greeting), std::move(_disposal)) {}

	    inline bool insert(const K& _key, T* _val)
        {
            m_lock.lockWrite();
            const bool r = CacheBase::insert(_key, _val);
            m_lock.unlockWrite();
            return r;
        }

        inline bool contains(const T* _object) const
        {
            m_lock.lockRead();
            const bool r = CacheBase::contains(_object);
            m_lock.unlockRead();
            return r;
        }

        inline size_t getSize() const
        {
            m_lock.lockRead();
            const size_t r = CacheBase::getSize();
            m_lock.unlockRead();
            return r;
        }
    };

    template<typename CacheT>
    class CMakeCacheConcurrent : public CMakeCacheConcurrent_common<CacheT>
    {
        using CacheBase = CacheT;
        using K = typename CacheBase::KeyType;
        using T = typename CacheBase::CachedType;

    public:
        inline explicit CMakeCacheConcurrent(const typename CacheBase::GreetFuncType& _greeting, const typename CacheBase::DisposalFuncType& _disposal) : CMakeCacheConcurrent_common<CacheT>(_greeting, _disposal) {}
        inline explicit CMakeCacheConcurrent(typename CacheBase::GreetFuncType&& _greeting = nullptr, typename CacheBase::DisposalFuncType&& _disposal = nullptr) : CMakeCacheConcurrent_common<CacheT>(std::move(_greeting), std::move(_disposal)) {}

        inline T* getByKey(const K& _key)
        {
            m_lock.lockRead();
            T* r = CacheBase::getByKey(_key);
            m_lock.unlockRead();
            return r;
        }

        inline const T* getByKey(const K& _key) const
        {
            m_lock.lockRead();
            const T* const r = CacheBase::getByKey(_key);
            m_lock.unlockRead();
            return r;
        }

        inline void removeByKey(const K& _key)
        {
            m_lock.lockWrite();
            CacheBase::removeByKey(_key);
            m_lock.unlockWrite();
        }
    };

    template<typename CacheT>
    class CMakeMultiCacheConcurrent : public CMakeCacheConcurrent_common<CacheT>
    {
        using CacheBase = CacheT;
        using K = typename CacheBase::KeyType_impl;
        using T = typename CacheBase::CachedType;
        using RangeType = std::pair<typename CacheBase::IteratorType, typename CacheBase::IteratorType>;

    public:
        inline explicit CMakeMultiCacheConcurrent(const typename CacheBase::GreetFuncType& _greeting, const typename CacheBase::DisposalFuncType& _disposal) : CMakeCacheConcurrent_common<CacheT>(_greeting, _disposal) {}
        inline explicit CMakeMultiCacheConcurrent(typename CacheBase::GreetFuncType&& _greeting = nullptr, typename CacheBase::DisposalFuncType&& _disposal = nullptr) : CMakeCacheConcurrent_common<CacheT>(std::move(_greeting), std::move(_disposal)) {}

        inline RangeType findRange(const K& _key)
        {
            m_lock.lockRead();
            RangeType r = CacheBase::findRange(_key);
            m_lock.unlockRead();
            return r;
        }

        inline const RangeType findRange(const K& _key) const
        {
            m_lock.lockRead();
            const RangeType r = CacheBase::findRange(_key);
            m_lock.unlockRead();
            return r;
        }

        inline void removeObject(T* _object, const K& _key)
        {
            m_lock.lockWrite();
            CacheBase::removeObject(_object, _key);
            m_lock.unlockWrite();
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
    impl::CMakeMultiCacheConcurrent<
        CMultiObjectCache<K, T, ContainerT_T>
    >;

}}

#endif