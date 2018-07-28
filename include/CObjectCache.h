#ifndef __C_OBJECT_CACHE_H_INCLUDED__
#define __C_OBJECT_CACHE_H_INCLUDED__

#include <type_traits>
#include <vector>
#include <utility>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <map>
#include <unordered_map>
#include "IReferenceCounted.h"

namespace irr { namespace core
{

namespace impl
{
    template<template<typename...> class, template<typename...> class>
    struct is_same_templ : std::false_type {};

    template<template<typename...> class T>
    struct is_same_templ<T, T> : std::true_type {};

    template<
        template<typename...> class ContainerT,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    struct CObjectCacheBase
    {
    protected:
        ContainerT<K..., T> m_container;

        using ValueType = typename ContainerT<K..., T>::value_type::second_type; // container's value_type is always instantiation of std::pair
        using NoPtrValueType = typename std::remove_pointer<ValueType>::type;
        using PtrToConstVal_t = const NoPtrValueType*; // ValueType is always pointer to derivative of irr::IReferenceCounted type

        static_assert(std::is_base_of<IReferenceCounted, NoPtrValueType>::value, "CObjectCache<K, T, ContainerT>: T must be derivative of irr::IReferenceCounted");

        std::function<void(ValueType)> m_disposalFunc;

    protected:
        void dispose(ValueType _object) const
        {
            if (m_disposalFunc == nullptr)
                _object->drop();
            else
                m_disposalFunc(_object);
        }

    public:
        explicit CObjectCacheBase(const std::function<void(ValueType)>& _disposal) : m_disposalFunc(_disposal) {}

        bool contains(PtrToConstVal_t _object) const
        {
            for (const auto& e : m_container)
                if (e.second == _object)
                    return true;
            return false;
        }

        size_t getSize() const { return m_container.size(); }
    };
}

template<
    typename K,
    typename T,
    template<typename...> class ContainerT = std::vector,
    bool = impl::is_same_templ<ContainerT, std::vector>::value
>
class CObjectCache;

template<
    typename K,
    typename T,
    template<typename...> class ContainerT
>
class CObjectCache<K, T, ContainerT, true> : public impl::CObjectCacheBase<ContainerT, std::pair<K, T*>>
{
    using ValueType = std::pair<K, T*>;

    static bool compare(const ValueType& _a, const ValueType& _b)
    {
        return _a.first < _b.first;
    }
    static int bs_compare(const void* _a, const void* _b)
    {
        const ValueType& a = *reinterpret_cast<const ValueType*>(_a);
        const ValueType& b = *reinterpret_cast<const ValueType*>(_b);

        if (a.first < b.first)
            return -1;
        else if (b.first < a.first)
            return 1;
        return 0;
    }

public:
    explicit CObjectCache(const std::function<void(T*)>& _disposal = nullptr) : impl::CObjectCacheBase<ContainerT, std::pair<K, T*>>(_disposal) {}

    bool insert(const K& _key, T* _val)
    {
        if (_val)
            _val->grab();
        else
            return false;
        const ValueType newVal{_key, _val};
        auto it = std::lower_bound(std::begin(m_container), std::end(m_container), newVal, &compare);
        m_container.insert(it, newVal);
        return true;
    }

    T* getByKey(const K& _key)
    {
        const ValueType lookingFor{_key, nullptr};
        ValueType* found = reinterpret_cast<ValueType*>(
            std::bsearch(&lookingFor, m_container.data(), getSize(), sizeof(ValueType), &bs_compare)
        );
        if (found)
            return found->second;
        return nullptr;
    }
    const T* getByKey(const K& _key) const
    {
        return const_cast<typename std::remove_const<decltype(*this)>::type*>(this)->getByKey(_key);
    }

    void removeByKey(const K& _key)
    {
        const auto it = std::remove_if(std::begin(m_container), std::end(m_container), [&_key] (const ValueType& _a) { return _a.first == _key; });
        auto itr = it;
        while (itr != std::end(m_container))
        {
            dispose(itr->second);
            std::advance(itr, 1);
        }
        m_container.erase(it, std::end(m_container));
        std::sort(std::begin(m_container), std::end(m_container), &compare);
    }
};


template<
    typename K,
    typename T,
    template<typename...> class ContainerT
>
class CObjectCache<K, T, ContainerT, false> : public impl::CObjectCacheBase<ContainerT, T*, K>
{
    static_assert(impl::is_same_templ<ContainerT, std::map>::value || impl::is_same_templ<ContainerT, std::unordered_map>::value, "ContainerT must be one of: std::vector, std::map, std::unordered_map");

public:
    explicit CObjectCache(const std::function<void(T*)>& _disposal = nullptr) : impl::CObjectCacheBase<ContainerT, T*, K>(_disposal) {}

    bool insert(const K& _key, T* _val)
    {
        if (!_val)
            return false;
        _val->grab();
        return m_container.insert({_key, _val}).second;
    }

    T* getByKey(const K& _key)
    {
        auto it = m_container.find(_key);
        if (it == std::end(m_container))
            return nullptr;
        return it->second;
    }
    const T* getByKey(const K& _key) const
    {
        return const_cast<typename std::remove_const<decltype(*this)>::type*>(this)->getByKey(_key);
    }

    void removeByKey(const K& _key)
    {
        auto it = m_container.find(_key);
        if (it != std::end(m_container))
        {
            dispose(it->second);
            m_container.erase(it);
        }
    }
};

}}

#endif