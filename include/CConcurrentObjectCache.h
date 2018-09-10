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
    template<typename...> class ContainerT_T = core::vector
>
class CConcurrentObjectCache : private impl::CConcurrentObjectCacheBase, private CObjectCache<K, T, ContainerT_T>
{
    using Base = CObjectCache<K, T, ContainerT_T>;

public:
    inline explicit CConcurrentObjectCache(const std::function<void(T*)>& _greeting, const std::function<void(T*)>& _disposal) : CObjectCache<K, T, ContainerT_T>(_greeting, _disposal) {}
    inline explicit CConcurrentObjectCache(std::function<void(T*)>&& _greeting = nullptr, std::function<void(T*)>&& _disposal = nullptr) : CObjectCache<K, T, ContainerT_T>(std::move(_greeting), std::move(_disposal)) {}

	inline bool insert(const K& _key, T* _val)
    {
        m_lock.lockWrite();
        const bool r = Base::insert(_key, _val);
        m_lock.unlockWrite();
        return r;
    }

    //! Returns true if had to insert
    inline bool swapValue(const K& _key, T* _val)
    {
        m_lock.lockWrite();
        bool r = Base::swapValue(_key,_val);
        m_lock.unlockWrite();
        return r;
    }

    inline bool getByKeyOrReserve(T** _outval, const K& _key)
    {
        m_lock.lockWrite();
        bool r = Base::getByKeyOrReserve(_outval,_key);
        m_lock.unlockWrite();
        return r;
    }

	inline bool getByKey(T** _outval, const K& _key)
    {
        m_lock.lockRead();
        bool r = Base::getByKey(_outval,_key);
        m_lock.unlockRead();
        return r;
    }

	inline bool getByKey(const T** _outval, const K& _key) const
    {
		m_lock.lockRead();
		bool r = Base::getByKey(_outval,_key);
		m_lock.unlockRead();
		return r;
    }

	inline void removeByKey(const K& _key)
    {
        m_lock.lockWrite();
        Base::removeByKey(_key);
        m_lock.unlockWrite();
    }

	inline bool contains(const T* _object) const
    {
        m_lock.lockRead();
        const bool r = Base::contains(_object);
        m_lock.unlockRead();
        return r;
    }

    inline size_t getSize() const
    {
        m_lock.lockRead();
        const size_t r = Base::getSize();
        m_lock.unlockRead();
        return r;
    }
};

}}

#endif
