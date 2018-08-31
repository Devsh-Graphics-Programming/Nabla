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
        using PtrToConstVal_t = const NoPtrValueType*; // ValueType is always pointer type

        std::function<void(ValueType)> m_greetingFunc, m_disposalFunc;
    protected:
		inline virtual ~CObjectCacheBase()
		{
			for (auto it=m_container.begin(); it!=m_container.end(); it++)
				dispose(it->second);
		}

        void dispose(ValueType _object) const        {
            if (m_disposalFunc)
                m_disposalFunc(_object);
        }

        void greet(ValueType _object) const
        {
            if (m_greetingFunc)
                m_greetingFunc(_object);
        }

    public:
        inline explicit CObjectCacheBase(const std::function<void(ValueType)>& _greeting, const std::function<void(ValueType)>& _disposal) : m_greetingFunc(_greeting), m_disposalFunc(_disposal) {}
        inline explicit CObjectCacheBase(std::function<void(ValueType)>&& _greeting, std::function<void(ValueType)>&& _disposal) : m_greetingFunc(std::move(_greeting)), m_disposalFunc(std::move(_disposal)) {}

        inline bool contains(PtrToConstVal_t _object) const
        {
            for (const auto& e : m_container)
                if (e.second == _object)
                    return true;
            return false;
        }

		inline size_t getSize() const { return m_container.size(); }
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

public:
    inline explicit CObjectCache(const std::function<void(T*)>& _greeting, const std::function<void(T*)>& _disposal) : impl::CObjectCacheBase<ContainerT_T, std::pair<K, T*>>(_greeting, _disposal) {}
    inline explicit CObjectCache(std::function<void(T*)>&& _greeting = nullptr, std::function<void(T*)>&& _disposal = nullptr) : impl::CObjectCacheBase<ContainerT_T, std::pair<K, T*>>(std::move(_greeting), std::move(_disposal)) {}

    inline bool insert(const K& _key, T* _val)
    {
        const ValueType newVal{_key, _val};
        auto it = std::lower_bound(std::begin(m_container), std::end(m_container), newVal, [](const ValueType& _a, const ValueType& _b) -> bool {return _a.first < _b.first; });
        if (it != std::end(m_container) && !(_key < it->first)) // used `<` instead of `==` operator here to keep consistency with std::map (so key type doesn't need to define operator==)
            return false;

        greet(newVal.second);
        m_container.insert(it, newVal);
        return true;
    }

	inline bool getByKey(T** _outval, const K& _key)
    {
        const T** _outval2 = _outval;
        return const_cast<std::remove_reference<decltype(*this)>::type const&>(*this).getByKey(_outval2,_key);
    }

	inline bool getByKey(const T** _outval, const K& _key) const
    {
		auto it = find(_key);
		if (it != std::end(m_container))
        {
            *_outval = it->second;
            return true;
        }
		return false;
    }

	inline void removeByKey(const K& _key)
    {
        auto it = find(_key);
        if (it != std::end(m_container))
        {
            dispose(it->second);
            m_container.erase(it);
        }
    }

private:
    inline typename ContainerT::const_iterator find(const K& _key) const
    {
        const auto it = std::lower_bound(std::begin(m_container), std::end(m_container), ValueType{_key, nullptr}, [](const ValueType& _a, const ValueType& _b) -> bool {return _a.first < _b.first; });
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
    inline explicit CObjectCache(const std::function<void(T*)>& _greeting, const std::function<void(T*)>& _disposal) : impl::CObjectCacheBase<ContainerT_T, T*, K>(_greeting, _disposal) {}
    inline explicit CObjectCache(std::function<void(T*)>&& _greeting = nullptr, std::function<void(T*)>&& _disposal = nullptr) : impl::CObjectCacheBase<ContainerT_T, T*, K>(std::move(_greeting), std::move(_disposal)) {}

	inline bool insert(const K& _key, T* _val)
    {
        auto retval = m_container.insert({_key, _val});
        if (retval.second)
            greet(newVal.second);
        return retval.second;
    }

	inline bool getByKey(T** _outval, const K& _key)
    {
        const T** _outval2 = _outval;
		return const_cast<std::remove_reference<decltype(*this)>::type const&>(*this).getByKey(_outval2,_key);
    }
	inline bool getByKey(const T** _outval, const K& _key) const
    {
		auto it = m_container.find(_key);
		if (it == std::end(m_container))
			return false;

		*_outval = it->second;
		return true;
    }

    inline void removeByKey(const K& _key)
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
