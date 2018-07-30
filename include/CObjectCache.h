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
        template<typename...> class ContainerT_T,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    struct CObjectCacheBase
    {
    protected:
        using ContainerT = ContainerT_T<K..., T>;
        ContainerT m_container;

        using ValueType = typename ContainerT_T<K..., T>::value_type::second_type; // container's value_type is always instantiation of std::pair
        using NoPtrValueType = typename std::remove_pointer<ValueType>::type;
        using PtrToConstVal_t = const NoPtrValueType*; // ValueType is always pointer to derivative of irr::IReferenceCounted type

        static_assert(std::is_base_of<IReferenceCounted, NoPtrValueType>::value, "CObjectCache<K, T, ContainerT_T>: T must be derivative of irr::IReferenceCounted");

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
        explicit CObjectCacheBase(std::function<void(ValueType)>&& _disposal) : m_disposalFunc(std::move(_disposal)) {}

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
    template<typename...> class ContainerT_T = std::vector,
    bool = impl::is_same_templ<ContainerT_T, std::vector>::value
>
class CObjectCache;

template<
    typename K,
    typename T,
    template<typename...> class ContainerT_T
>
class CObjectCache<K, T, ContainerT_T, true> : public impl::CObjectCacheBase<ContainerT_T, std::pair<K, T*>>
{
    using ValueType = std::pair<K, T*>;

    static bool compare(const ValueType& _a, const ValueType& _b)
    {
        return _a.first < _b.first;
    }

public:
    explicit CObjectCache(const std::function<void(T*)>& _disposal) : impl::CObjectCacheBase<ContainerT_T, std::pair<K, T*>>(_disposal) {}
    explicit CObjectCache(std::function<void(T*)>&& _disposal = nullptr) : impl::CObjectCacheBase<ContainerT_T, std::pair<K, T*>>(std::move(_disposal)) {}

    bool insert(const K& _key, T* _val)
    {
        if (_val)
            _val->grab();
        else
            return false;
        const ValueType newVal{_key, _val};
        auto it = std::lower_bound(std::begin(m_container), std::end(m_container), newVal, &compare);
        if (it != std::end(m_container) && !(_key < it->first)) // used `<` instead of `==` operator here to keep consistency with std::map (so key type doesn't need to define operator==)
            return false;
        m_container.insert(it, newVal);
        return true;
    }

    T* getByKey(const K& _key)
    {
        auto it = this->find(_key);
        if (it != std::end(m_container))
            return it->second;
        return nullptr;
    }
    const T* getByKey(const K& _key) const
    {
        using MeT = typename std::remove_reference<decltype(*this)>::type; // decltype(*this) gives reference type
        return const_cast<typename std::remove_const<MeT>::type*>(this)->getByKey(_key);
    }

    void removeByKey(const K& _key)
    {
        auto it = this->find(_key);
        if (it != std::end(m_container))
        {
            dispose(it->second);
            m_container.erase(it);
        }
        std::sort(std::begin(m_container), std::end(m_container), &compare);
    }

private:
    typename ContainerT::iterator find(const K& _key)
    {
        auto it = std::lower_bound(std::begin(m_container), std::end(m_container), ValueType{_key, nullptr}, &compare);
        if (it == std::end(m_container) || it->first < _key || _key < it->first)
            return std::end(m_container);
        return it;
    }
};


template<
    typename K,
    typename T,
    template<typename...> class ContainerT_T
>
class CObjectCache<K, T, ContainerT_T, false> : public impl::CObjectCacheBase<ContainerT_T, T*, K>
{
    static_assert(impl::is_same_templ<ContainerT_T, std::map>::value || impl::is_same_templ<ContainerT_T, std::unordered_map>::value, "ContainerT_T must be one of: std::vector, std::map, std::unordered_map");

public:
    explicit CObjectCache(const std::function<void(T*)>& _disposal) : impl::CObjectCacheBase<ContainerT_T, T*, K>(_disposal) {}
    explicit CObjectCache(std::function<void(T*)>&& _disposal = nullptr) : impl::CObjectCacheBase<ContainerT_T, T*, K>(std::move(_disposal)) {}

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