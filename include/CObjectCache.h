#ifndef __C_OBJECT_CACHE_H_INCLUDED__
#define __C_OBJECT_CACHE_H_INCLUDED__

#include <type_traits>
#include <vector>
#include <utility>
#include <algorithm>
#include <cstdlib>

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
        typename ...Ts //optionally key type for std::map/std::unordered_map
    >
    struct CObjectCacheBase
    {
    protected:
        ContainerT<Ts..., T> m_container;

        using PtrToConstVal_t = typename std::remove_pointer<typename ContainerT<Ts..., T>::value_type::second_type>::type const*; // container's value_type is always instantiation of std::pair

    public:
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
    using value_type = std::pair<K, T*>;

    static bool compare(const value_type& _a, const value_type& _b)
    {
        return _a.first < _b.first;
    }
    static int bs_compare(const void* _a, const void* _b)
    {
        const value_type& a = *reinterpret_cast<const value_type*>(_a);
        const value_type& b = *reinterpret_cast<const value_type*>(_b);

        if (a.first < b.first)
            return -1;
        else if (b.first < a.first)
            return 1;
        return 0;
    }

public:
    bool insert(const K& _key, T* _val)
    {
        if (_val)
            _val->grab();
        else
            return false;
        const value_type newVal{_key, _val};
        auto it = std::lower_bound(std::begin(m_container), std::end(m_container), newVal, &compare);
        m_container.insert(it, newVal);
        return true;
    }

    T* getByKey(const K& _key)
    {
        const value_type lookingFor{_key, nullptr};
        value_type* found = reinterpret_cast<value_type*>(
            std::bsearch(&lookingFor, m_container.data(), getSize(), sizeof(value_type), &bs_compare)
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
        const auto it = std::remove_if(std::begin(m_container), std::end(m_container), [&_key] (const value_type& _a) { return _a.first == _key; });
        auto itr = it;
        while (itr != std::end(m_container))
        {
            itr->second->drop();
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
public:
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
            it->second->drop();
            m_container.erase(it);
        }
    }
};

}}

#endif