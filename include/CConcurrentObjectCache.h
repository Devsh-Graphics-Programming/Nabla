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
}

template<
    typename K,
    typename T,
    template<typename...> class ContainerT = std::vector
>
class CConcurentObjectCache : private impl::CConcurrentObjectCacheBase, private CObjectCache<K, T, ContainerT>
{
    using Base = CObjectCache<K, T, ContainerT>;

public:
    explicit CConcurentObjectCache(const std::function<void(T*)>& _disposal = nullptr) : CObjectCache<K, T, ContainerT>(_disposal) {}

    bool insert(const K& _key, T* _val)
    {
        m_lock.lockWrite();
        const bool r = Base::insert(_key, _val);
        m_lock.unlockWrite();
        return r;
    }

    T* getByKey(const K& _key)
    {
        m_lock.lockRead();
        T* const r = Base::getByKey(_key);
        m_lock.unlockRead();
        return r;
    }
    const T* getByKey(const K& _key) const
    {
        return const_cast<typename std::remove_const<decltype(*this)>::type*>(this)->getByKey(_key);
    }

    void removeByKey(const K& _key)
    {
        m_lock.lockWrite();
        Base::removeByKey(_key);
        m_lock.unlockWrite();
    }

    bool contains(const T* _object) const
    {
        m_lock.lockRead();
        const bool r = Base::contains(_object);
        m_lock.unlockRead();
        return r;
    }

    size_t getSize() const
    {
        m_lock.lockRead();
        const size_t r = Base::getSize();
        m_lock.unlockRead();
        return r;
    }
};

}}

#endif