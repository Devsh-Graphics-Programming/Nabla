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

namespace irr { namespace core
{

namespace impl
{
    struct CMultiCache_tag {};

    template<template<typename...> class, template<typename...> class>
    struct is_same_templ : std::false_type {};

    template<template<typename...> class T>
    struct is_same_templ<T, T> : std::true_type {};

    template<template<typename...> class T>
    struct is_multi_container : std::false_type {};
    template<>
    struct is_multi_container<std::multimap> : std::true_type {};
    template<>
    struct is_multi_container<std::unordered_multimap> : std::true_type {};

    template<template<typename...> class T>
    struct is_assoc_container : std::false_type {};
    template<>
    struct is_assoc_container<std::map> : std::true_type {};
    template<>
    struct is_assoc_container<std::unordered_map> : std::true_type {};

    struct Dummy {};

#define DEF_CACHE_CTORS(_Name, _Base) \
    inline explicit _Name(const typename _Base::GreetFuncType& _greeting, const typename _Base::DisposalFuncType& _disposal) : _Base(_greeting, _disposal) {}\
    inline explicit _Name(typename _Base::GreetFuncType&& _greeting = nullptr, typename _Base::DisposalFuncType&& _disposal = nullptr) : _Base(std::move(_greeting), std::move(_disposal)) {}
    
    template<typename K, typename...>
    struct PropagKeyTypeTypedef_ { using KeyType = K; };
    template<typename ...K>
    struct PropagKeyTypeTypedef : PropagKeyTypeTypedef_<K...> {};

    template<typename T, typename ...K>
    struct PropagTypedefs : PropagKeyTypeTypedef<K...> { using CachedType = T; };

    template<
        template<typename...> class ContainerT_T,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    struct CObjectCacheBase
    {
        using UnderlyingContainerType = ContainerT_T<K..., T>;
        using IteratorType = typename UnderlyingContainerType::iterator;
        using ConstIteratorType = typename UnderlyingContainerType::const_iterator;
        using RevIteratorType = typename UnderlyingContainerType::reverse_iterator;
        using ConstRevIteratorType = typename UnderlyingContainerType::const_reverse_iterator;

    protected:
        using ContainerT = UnderlyingContainerType;
        ContainerT m_container;

        // typedefs for implementation only
        using PairType = typename ContainerT::value_type;
        using ValueType = typename PairType::second_type; // container's value_type is always instantiation of std::pair
        using KeyType_impl = typename PairType::first_type;
        using NoPtrValueType = typename std::remove_pointer<ValueType>::type;
        using PtrToConstVal_t = const NoPtrValueType*; // ValueType is always pointer type

    public:
        using RangeType = std::pair<IteratorType, IteratorType>;

        using GreetFuncType = std::function<void(ValueType)>;
        using DisposalFuncType = std::function<void(ValueType)>;

    protected:
        GreetFuncType m_greetingFunc;
        DisposalFuncType m_disposalFunc;
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
        inline explicit CObjectCacheBase(const GreetFuncType& _greeting, const DisposalFuncType& _disposal) : m_greetingFunc(_greeting), m_disposalFunc(_disposal) {}
        inline explicit CObjectCacheBase(GreetFuncType&& _greeting, DisposalFuncType&& _disposal) : m_greetingFunc(std::move(_greeting)), m_disposalFunc(std::move(_disposal)) {}

        inline bool contains(PtrToConstVal_t _object) const
        {
            for (const auto& e : m_container)
                if (e.second == _object)
                    return true;
            return false;
        }

		inline size_t getSize() const { return m_container.size(); }
    };


    template<template<typename...> class ContainerT_T, typename ContainerT, bool ForMultiCache, bool IsAssocContainer = impl::is_assoc_container<ContainerT_T>::value>
    struct CPreInsertionVerifier;
    template<template<typename...> class ContainerT_T, typename ContainerT, bool IsAssocContainer>
    struct CPreInsertionVerifier<ContainerT_T, ContainerT, true, IsAssocContainer>
    {
        template<typename ...Ts>
        static bool verify(Ts...) { return true; }
    };
    template<template<typename...> class ContainerT_T, typename ContainerT>
    struct CPreInsertionVerifier<ContainerT_T, ContainerT, false, false>
    {
        static bool verify(const ContainerT& _container, const typename ContainerT::iterator& _itr, const typename ContainerT::value_type::first_type& _key)
        {
            using ValueType = typename ContainerT::value_type;

            if (_itr != std::cend(_container) && !(_key < _itr->first)) // used `<` instead of `==` operator here to keep consistency with std::map (so key type doesn't need to define operator==)
                return false;

            return true;
        }
    };
    template<template<typename...> class ContainerT_T, typename ContainerT>
    struct CPreInsertionVerifier<ContainerT_T, ContainerT, false, true>
    {
        static bool verify(const std::pair<typename ContainerT::iterator, bool>& _insertionRes)
        {
            return _insertionRes.second;
        }
    };

    //! Use in non-static member functions
#define MY_TYPE typename std::remove_reference<decltype(*this)>::type
#define INSERT_IMPL_VEC \
    const typename Base::PairType newVal{ _key, _val };\
    auto it = std::lower_bound(std::begin(this->m_container), std::end(this->m_container), newVal, [](const typename Base::PairType& _a, const typename Base::PairType& _b) -> bool {return _a.first < _b.first; });\
    if (\
    !impl::CPreInsertionVerifier<ContainerT_T, typename Base::ContainerT, std::is_base_of<impl::CMultiCache_tag, MY_TYPE>::value>::verify(this->m_container, it, _key)\
    )\
        return false;\
    \
    greet(newVal.second);\
    this->m_container.insert(it, newVal);\
    return true;
#define INSERT_IMPL_ASSOC \
    auto res = this->m_container.insert({ _key, _val });\
    const bool verif = impl::CPreInsertionVerifier<ContainerT_T, typename Base::ContainerT, std::is_base_of<impl::CMultiCache_tag, MY_TYPE>::value>::verify(res);\
    if (verif)\
        greet(_val);\
    return verif;

    template<
        bool isMultiContainer,
        template<typename...> class ContainerT_T,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    struct CMultiObjectCacheBase;

    template<
        template<typename...> class ContainerT_T,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    struct CMultiObjectCacheBase<true, ContainerT_T, T, K...> : public CObjectCacheBase<ContainerT_T, T, K...>, public CMultiCache_tag
    {
    private:
        using Base = CObjectCacheBase<ContainerT_T, T, K...>;

    protected:
        using GreetFuncType = typename Base::GreetFuncType;
        using DisposalFuncType = typename Base::DisposalFuncType;

    public:
        inline explicit CMultiObjectCacheBase(const GreetFuncType& _greeting, const DisposalFuncType& _disposal) : Base(_greeting, _disposal) {}
        inline explicit CMultiObjectCacheBase(GreetFuncType&& _greeting, DisposalFuncType&& _disposal) : Base(_greeting, _disposal) {}

        inline std::pair<typename Base::IteratorType, typename Base::IteratorType> findRange(const typename Base::KeyType_impl& _key)
        {
            return Base::m_container.equal_range(_key);
        }
        inline std::pair<typename Base::ConstIteratorType, typename Base::ConstIteratorType> findRange(const typename Base::KeyType_impl& _key) const
        {
            return Base::m_container.equal_range(_key);
        }

        inline void removeObject(const typename Base::ValueType _obj, const typename Base::KeyType_impl& _key)
        {
            std::pair<typename Base::IteratorType, typename Base::IteratorType> range = findRange(_key);
            for (auto it = range.first; it != range.second; ++it)
            {
                if (it->second == _obj)
                {
                    Base::m_container.erase(it);
                    break;
                }
            }
        }
    };
    template<
        template<typename...> class ContainerT_T,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    struct CMultiObjectCacheBase<false, ContainerT_T, T, K...> : public CObjectCacheBase<ContainerT_T, T, K...>, public CMultiCache_tag
    {
    private:
        using Base = CObjectCacheBase<ContainerT_T, T, K...>;

    protected:
        using GreetFuncType = typename Base::GreetFuncType;
        using DisposalFuncType = typename Base::DisposalFuncType;

    public:
        inline explicit CMultiObjectCacheBase(const GreetFuncType& _greeting, const DisposalFuncType& _disposal) : Base(_greeting, _disposal) {}
        inline explicit CMultiObjectCacheBase(GreetFuncType&& _greeting, DisposalFuncType&& _disposal) : Base(_greeting, _disposal) {}

    private:
        template<typename ItrType>
        inline std::pair<ItrType, ItrType> findRange_(const typename Base::KeyType_impl& _key)
        {
            auto cmpf = [](const typename Base::PairType& _a, const typename Base::PairType& _b) -> bool {return _a.first < _b.first; };
            typename Base::PairType lookingFor{_key, nullptr};

            std::pair<ItrType, ItrType> range;
            range.first = std::lower_bound(std::begin(Base::m_container), std::end(Base::m_container), lookingFor, cmpf);
            if (range.first == std::end(Base::m_container) || _key < range.first->first)
            {
                range.second = range.first;
                return range;
            }
            range.second = std::upper_bound(range.first, std::end(Base::m_container), lookingFor, cmpf);
            return range;
        }

    public:
        inline std::pair<typename Base::IteratorType, typename Base::IteratorType> findRange(const typename Base::KeyType_impl& _key)
        {
            return findRange_<typename Base::IteratorType>(_key);
        }
        inline std::pair<typename Base::ConstIteratorType, typename Base::ConstIteratorType> findRange(const typename Base::KeyType_impl& _key) const
        {
            return findRange_<typename Base::ConstIteratorType>(_key);
        }

        // todo THIS DUPLICATES FUNCTION FROM ABOVE SPECIALIZATION
        inline void removeObject(const typename Base::ValueType _obj, const typename Base::KeyType_impl& _key)
        {
            std::pair<typename Base::IteratorType, typename Base::IteratorType> range = findRange(_key);
            for (auto it = range.first; it != range.second; ++it)
            {
                if (it->second == _obj)
                {
                    Base::m_container.erase(it);
                    break;
                }
            }
        }
    };

    template<
        bool IsMultiContainer,
        template<typename...> class ContainerT_T,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    struct CMultiObjectCacheBaseExt : public CMultiObjectCacheBase<IsMultiContainer, ContainerT_T, T, K...>
    {
    private:
        using Base = CMultiObjectCacheBase<IsMultiContainer, ContainerT_T, T, K...>;

    public:
        DEF_CACHE_CTORS(CMultiObjectCacheBaseExt, Base)

        //! Returns true if had to insert
        bool swapObjectValue(const typename Base::KeyType_impl& _key, const typename Base::PtrToConstVal_t _obj, typename Base::ValueType _val)
        {
            greet(_val); // grab before drop

            auto range = findRange(_key);
            typename Base::IteratorType found = find(range, _key, _obj);

            if (found != range.second)
            {
                dispose(found->second);
                found->second = _val;
                return false;
            }
            this->m_container.insert(range.second, typename Base::PairType{_key, _val});
            return true;
        }

        bool getKeyRangeOrReserve(typename Base::RangeType* _outrange, const typename Base::KeyType_impl& _key)
        {
            *_outrange = findRange(_key);
            if (std::distance(_outrange->first, _outrange->second)==0)
            {
                this->m_container.insert(_outrange->second, { _key, nullptr });
                greet(nullptr);
                return false;
            }
            return true;
        }

    private:
        typename Base::IteratorType find(const typename Base::RangeType& _range, const typename Base::KeyType_impl& _key, const typename Base::PtrToConstVal_t _obj) const
        {
            typename Base::IteratorType found = _range.second;
            for (auto it = _range.first; it != _range.second; ++it)
            {
                if (it->second == _obj)
                {
                    found = it;
                    break;
                }
            }
            return found;
        }
    };
}

template<
    typename K,
    typename T,
    template<typename...> class ContainerT_T = std::vector,
    bool = impl::is_same_templ<ContainerT_T, std::vector>::value
>
class CMultiObjectCache;

template<
    typename K,
    typename T,
    template<typename...> class ContainerT_T
>
class CMultiObjectCache<K, T, ContainerT_T, true> : 
    public impl::CMultiObjectCacheBaseExt<impl::is_multi_container<ContainerT_T>::value, ContainerT_T, std::pair<K, T*>>,
    public impl::PropagTypedefs<T, K>
{
private:
    using Base = impl::CMultiObjectCacheBaseExt<impl::is_multi_container<ContainerT_T>::value, ContainerT_T, std::pair<K, T*>>;

protected:
    using GreetFuncType = typename Base::GreetFuncType;
    using DisposalFuncType = typename Base::DisposalFuncType;

public:
    inline explicit CMultiObjectCache(const GreetFuncType& _greeting, const DisposalFuncType& _disposal) : Base(_greeting, _disposal) {}
    inline explicit CMultiObjectCache(GreetFuncType&& _greeting = nullptr, DisposalFuncType&& _disposal = nullptr) : Base(_greeting, _disposal) {}

    inline bool insert(const K& _key, T* _val)
    {
        INSERT_IMPL_VEC
    }
};
template<
    typename K,
    typename T,
    template<typename...> class ContainerT_T
>
class CMultiObjectCache<K, T, ContainerT_T, false> : 
    public impl::CMultiObjectCacheBaseExt<impl::is_multi_container<ContainerT_T>::value, ContainerT_T, T*, K>,
    public impl::PropagTypedefs<T, K>
{
    static_assert(impl::is_same_templ<ContainerT_T, std::multimap>::value || impl::is_same_templ<ContainerT_T, std::unordered_multimap>::value, "ContainerT_T must be one of: std::vector, std::multimap, std::unordered_multimap");

private:
    using Base = impl::CMultiObjectCacheBaseExt<impl::is_multi_container<ContainerT_T>::value, ContainerT_T, T*, K>;

protected:
    using GreetFuncType = typename Base::GreetFuncType;
    using DisposalFuncType = typename Base::DisposalFuncType;

public:
    inline explicit CMultiObjectCache(const GreetFuncType& _greeting, const DisposalFuncType& _disposal) : Base(_greeting, _disposal) {}
    inline explicit CMultiObjectCache(GreetFuncType&& _greeting = nullptr, DisposalFuncType&& _disposal = nullptr) : Base(_greeting, _disposal) {}

    inline bool insert(const K& _key, T* _val)
    {
        INSERT_IMPL_ASSOC
    }
};

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
class CObjectCache<K, T, ContainerT_T, true> : 
    public impl::CObjectCacheBase<ContainerT_T, std::pair<K, T*>>,
    public impl::PropagTypedefs<T, K>
{
    using ValueType = std::pair<K, T*>;
    using Base = impl::CObjectCacheBase<ContainerT_T, std::pair<K, T*>>;

public:
    inline explicit CObjectCache(const typename Base::GreetFuncType& _greeting, const typename Base::DisposalFuncType& _disposal) : Base(_greeting, _disposal) {}
    inline explicit CObjectCache(typename Base::GreetFuncType&& _greeting = nullptr, typename Base::DisposalFuncType&& _disposal = nullptr) : Base(std::move(_greeting), std::move(_disposal)) {}

    inline bool insert(const K& _key, T* _val)
    {
        INSERT_IMPL_VEC
    }

    //! Returns true if had to insert
    inline bool swapValue(const K& _key, T* _val)
    {
        greet(_val); //grab before drop

        auto it = find(_key);
        if (it != std::end(this->m_container))
        {
            dispose(it->second);
            it->second = _val;
            return false;
        }

        this->m_container.insert(it, { _key, _val });
        return true;
    }

    inline bool getByKey(T** _outval, const K& _key)
    {
        const T** _outval2 = const_cast<const T**>(_outval);
        return const_cast<typename std::remove_reference<decltype(*this)>::type const&>(*this).getByKey(_outval2, _key);
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
        auto it = this->find(_key);
        if (it != std::end(this->m_container))
        {
            dispose(it->second);
            this->m_container.erase(it);
        }
    }

private:
    inline typename Base::ConstIteratorType find(const typename Base::KeyType_impl& _key) const
    {
        using ValueType = typename Base::PairType;

        const auto it = std::lower_bound(std::begin(this->m_container), std::end(this->m_container), ValueType{ _key, nullptr }, [](const ValueType& _a, const ValueType& _b) -> bool {return _a.first < _b.first; });
        if (it == std::end(this->m_container) || it->first < _key || _key < it->first)
            return std::end(this->m_container);
        return it;
    }
    inline typename Base::IteratorType find(const typename Base::KeyType_impl& _key)
    {
        using ValueType = typename Base::PairType;

        const auto it = std::lower_bound(std::begin(this->m_container), std::end(this->m_container), ValueType{ _key, nullptr }, [](const ValueType& _a, const ValueType& _b) -> bool {return _a.first < _b.first; });
        if (it == std::end(this->m_container) || it->first < _key || _key < it->first)
            return std::end(this->m_container);
        return it;
    }
};


template<
    typename K,
    typename T,
    template<typename...> class ContainerT_T
>
class CObjectCache<K, T, ContainerT_T, false> : 
    public impl::CObjectCacheBase<ContainerT_T, T*, K>,
    public impl::PropagTypedefs<T, K>
{
    static_assert(impl::is_same_templ<ContainerT_T, std::map>::value || impl::is_same_templ<ContainerT_T, std::unordered_map>::value, "ContainerT_T must be one of: std::vector, std::map, std::unordered_map");
    using Base = impl::CObjectCacheBase<ContainerT_T, T*, K>;

public:
    inline explicit CObjectCache(const typename Base::GreetFuncType& _greeting, const typename Base::DisposalFuncType& _disposal) : Base(_greeting, _disposal) {}
    inline explicit CObjectCache(typename Base::GreetFuncType&& _greeting = nullptr, typename Base::DisposalFuncType&& _disposal = nullptr) : Base(std::move(_greeting), std::move(_disposal)) {}

	inline bool insert(const K& _key, T* _val)
    {
        INSERT_IMPL_ASSOC
    }

    //! Returns true if had to insert
    inline bool swapValue(const K& _key, T* _val)
    {
        greet(_val); //grab before drop

        auto it = this->m_container.find(_key);
        if (it != std::end(this->m_container))
        {
            dispose(it->second);
            it->second = _val;
            return false;
        }

        this->m_container.insert(it, { _key, _val });
        return true;
    }

    inline bool getByKey(T** _outval, const K& _key)
    {
        const T** _outval2 = const_cast<const T**>(_outval);
        return const_cast<std::remove_reference<decltype(*this)>::type const&>(*this).getByKey(_outval2, _key);
    }
    inline bool getByKey(const T** _outval, const K& _key) const
    {
        auto it = this->m_container.find(_key);
        if (it == std::end(this->m_container))
            return false;

        *_outval = it->second;
        return true;
    }

    inline void removeByKey(const K& _key)
    {
        auto it = this->m_container.find(_key);
        if (it != std::end(this->m_container))
        {
            dispose(it->second);
            this->m_container.erase(it);
        }
    }
};

}}

#endif //__C_OBJECT_CACHE_H_INCLUDED__