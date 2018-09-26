#ifndef __C_OBJECT_CACHE_H_INCLUDED__
#define __C_OBJECT_CACHE_H_INCLUDED__

#include <type_traits>
#include <utility>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <string>

#include "irr/macros.h"
#include "irr/core/Types.h"

namespace irr { namespace core
{

namespace impl
{
    struct IRR_FORCE_EBO CMultiCache_tag {};

    template<template<typename...> class, template<typename...> class>
    struct is_same_templ : std::false_type {};

    template<template<typename...> class T>
    struct is_same_templ<T, T> : std::true_type {};

    template <typename T>
    struct is_string : std::false_type {};
    template <typename C, typename T, typename A>
    struct is_string<std::basic_string<C, T, A>> : std::true_type {};

    template<template<typename...> class T>
    struct is_multi_container : std::false_type {};
    template<>
    struct is_multi_container<std::multimap> : std::true_type {};
    template<>
    struct is_multi_container<std::unordered_multimap> : std::true_type {};

    template<template<typename...> class>
    struct is_assoc_container : std::false_type {};
    template<>
    struct is_assoc_container<std::map> : std::true_type {};
    template<>
    struct is_assoc_container<std::unordered_map> : std::true_type {};
    template<>
    struct is_assoc_container<std::multimap> : std::true_type {};
    template<>
    struct is_assoc_container<std::unordered_multimap> : std::true_type {};
    
    template<typename K, typename...>
    struct PropagKeyTypeTypedef_ { using KeyType = K; };
    template<typename ...K>
    struct PropagKeyTypeTypedef : PropagKeyTypeTypedef_<K...> {};

    template<typename T, typename ...K>
    struct PropagTypedefs : PropagKeyTypeTypedef<K...> { using CachedType = T; };


    template<
        template<typename...> class ContainerT_T,
        typename Alloc,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    struct CObjectCacheBase
    {
    private:
        template<bool isAssoc, template<typename...> class C>
        struct help;

        template<template<typename...> class C>
        struct help<true, C>
        {
            template<typename K, typename T, typename Alloc>
            using container_t = C<K, T, std::less<K>, Alloc>;
        };
        template<template<typename...> class C>
        struct help<false, C>
        {
            template<typename T, typename Alloc>
            using container_t = C<T, Alloc>;
        };

    public:
        using AllocatorType = Alloc;

        using UnderlyingContainerType = typename help<is_assoc_container<ContainerT_T>::value, ContainerT_T>::template container_t<K..., T, Alloc>;
        using IteratorType = typename UnderlyingContainerType::iterator;
        using ConstIteratorType = typename UnderlyingContainerType::const_iterator;
        using RevIteratorType = typename UnderlyingContainerType::reverse_iterator;
        using ConstRevIteratorType = typename UnderlyingContainerType::const_reverse_iterator;

    protected:
        using ContainerT = UnderlyingContainerType;
        ContainerT m_container;

    public:
        using PairType = typename ContainerT::value_type;
        using MutablePairType = std::pair<typename std::remove_const<typename PairType::first_type>::type, typename PairType::second_type>;

    protected:
        // typedefs for implementation only
        //! Always pointer type
        using ValueType_impl = typename PairType::second_type; // container's value_type is always instantiation of std::pair
        static_assert(std::is_pointer<ValueType_impl>::value, "ValueType_impl must be pointer type!");
        using KeyType_impl = typename PairType::first_type;
        using NoPtrValueType_impl = typename std::remove_pointer<ValueType_impl>::type;
        using ValueType_PtrToConst_impl = const NoPtrValueType_impl*; // ValueType_impl is always pointer type

    public:
        using RangeType = std::pair<IteratorType, IteratorType>;
        using ConstRangeType = std::pair<ConstIteratorType, ConstIteratorType>;

        using GreetFuncType = std::function<void(ValueType_impl)>;
        using DisposalFuncType = std::function<void(ValueType_impl)>;

        template<typename RangeT>
        static bool isNonZeroRange(const RangeT& _range) { return _range.first != _range.second; }

    protected:
        GreetFuncType m_greetingFunc;
        DisposalFuncType m_disposalFunc;
    protected:
		inline virtual ~CObjectCacheBase()
		{
			for (auto it=m_container.begin(); it!=m_container.end(); it++)
				this->dispose(it->second);
		}

        void dispose(ValueType_impl _object) const {
            if (m_disposalFunc)
                m_disposalFunc(_object);
        }

        void greet(ValueType_impl _object) const
        {
            if (m_greetingFunc)
                m_greetingFunc(_object);
        }

    private:
        template<typename StorageT>
        void outputThis(const ConstIteratorType& _itr, size_t _ix, StorageT* _storage) const;
        template<>
        void outputThis<MutablePairType>(const ConstIteratorType& _itr, size_t _ix, MutablePairType* _storage) const
        {
            _storage[_ix] = *_itr;
        }
        template<>
        void outputThis<ValueType_impl>(const ConstIteratorType& _itr, size_t _ix, ValueType_impl* _storage) const
        {
            _storage[_ix] = _itr->second;
        }

        //! Only in non-concurrent cache
        template<typename StorageT>
        inline bool outputRange_(const ConstRangeType& _rng, size_t& _inOutStorageSize, StorageT* _out) const
        {
            if (_out)
            {
                _inOutStorageSize = std::distance(_rng.first, _rng.second);
                return false;
            }
            size_t i = 0u;
            for (auto it = _rng.first; it != _rng.second && i < _inOutStorageSize; ++it)
                outputThis(it, i, _out);
            const size_t reqSize = std::distance(_rng.first, _rng.second);
            bool res = _inOutStorageSize <= reqSize;
            _inOutStorageSize = i;
            return res;
        }

    public:
        inline bool outputRange(const ConstRangeType& _rng, size_t& _inOutStorageSize, MutablePairType* _out) const
        {
            return outputRange_(_rng, _inOutStorageSize, _out);
        }
        inline bool outputRange(const ConstRangeType& _rng, size_t& _inOutStorageSize, ValueType_impl* _out) const
        {
            return outputRange_(_rng, _inOutStorageSize, _out);
        }

        inline bool outputAll(size_t& _inOutStorageSize, MutablePairType* _out) const
        {
            return outputRange({cbegin(), cend()}, _inOutStorageSize, _out);
        }
        inline bool outputAll(size_t& _inOutStorageSize, ValueType_impl* _out) const
        {
            return outputRange({cbegin(), cend()}, _inOutStorageSize, _out);
        }

        CObjectCacheBase() = default;
        inline explicit CObjectCacheBase(const GreetFuncType& _greeting, const DisposalFuncType& _disposal) : m_greetingFunc(_greeting), m_disposalFunc(_disposal) {}
        inline explicit CObjectCacheBase(GreetFuncType&& _greeting, DisposalFuncType&& _disposal) : m_greetingFunc(std::move(_greeting)), m_disposalFunc(std::move(_disposal)) {}

        inline bool contains(ValueType_PtrToConst_impl _object) const
        {
            for (const auto& e : m_container)
                if (e.second == _object)
                    return true;
            return false;
        }

        inline void clear()
        {
            for (PairType& e : m_container)
                dispose(e.second);
            m_container.clear();
        }

		inline size_t getSize() const { return m_container.size(); }

        // Concurrent cache has only const-iterator getters
        inline ConstIteratorType begin() const { return cbegin(); }
        inline ConstIteratorType end() const { return cend(m_container); }
        inline ConstIteratorType cbegin() const { return std::cbegin(m_container); }
        inline ConstIteratorType cend() const { return std::cend(m_container); }
        inline ConstRevIteratorType crbegin() const { std::crbegin(m_container); }
        inline ConstRevIteratorType crend() const { std::crend(m_container); }
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
#define INSERT_IMPL_VEC \
    const typename Base::PairType newVal{ _key, _val };\
    auto it = std::lower_bound(std::begin(this->m_container), std::end(this->m_container), newVal, [](const typename Base::PairType& _a, const typename Base::PairType& _b) -> bool {return _a.first < _b.first; });\
    if (\
    !impl::CPreInsertionVerifier<ContainerT_T, typename Base::ContainerT, std::is_base_of<impl::CMultiCache_tag, typename std::decay<decltype(*this)>::type>::value>::verify(this->m_container, it, _key)\
    )\
        return false;\
    \
    this->greet(newVal.second);\
    this->m_container.insert(it, newVal);\
    return true;
#define INSERT_IMPL_ASSOC \
    auto res = this->m_container.insert({ _key, _val });\
    const bool verif = impl::CPreInsertionVerifier<ContainerT_T, typename Base::ContainerT, std::is_base_of<impl::CMultiCache_tag, typename std::decay<decltype(*this)>::type>::value>::verify(res);\
    if (verif)\
        this->greet(_val);\
    return verif;

    template<
        bool isMultiContainer, // is container a multimap or unordered_multimap (all allowed containers are those two and vector)
        template<typename...> class ContainerT_T,
        typename Alloc,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    struct CMultiObjectCacheBase;

    template<
        template<typename...> class ContainerT_T,
        typename Alloc,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    struct CMultiObjectCacheBase<true, ContainerT_T, Alloc, T, K...> : public CObjectCacheBase<ContainerT_T, Alloc, T, K...>, public CMultiCache_tag
    {
    private:
        using Base = CObjectCacheBase<ContainerT_T, Alloc, T, K...>;

    public:
        using Base::Base;

        inline bool insert(const typename Base::KeyType_impl& _key, typename Base::ValueType_impl _val)
        {
            INSERT_IMPL_ASSOC
        }

    public:
        inline typename Base::RangeType findRange(const typename Base::KeyType_impl& _key)
        {
            return Base::m_container.equal_range(_key);
        }
        inline typename Base::ConstRangeType findRange(const typename Base::KeyType_impl& _key) const
        {
            return Base::m_container.equal_range(_key);
        }
    };
    template<
        template<typename...> class ContainerT_T,
        typename Alloc,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    struct CMultiObjectCacheBase<false, ContainerT_T, Alloc, T, K...> : public CObjectCacheBase<ContainerT_T, Alloc, T, K...>, public CMultiCache_tag
    {
    private:
        using Base = CObjectCacheBase<ContainerT_T, Alloc, T, K...>;

    public:
        using Base::Base;

        inline bool insert(const typename Base::KeyType_impl& _key, typename Base::ValueType_impl _val)
        {
            INSERT_IMPL_VEC
        }

    private:
        // container not passed by **const** reference so i can take non-const iterators
        template<typename RngType>
        static inline RngType findRange_internal(typename Base::ContainerT& _container, const typename Base::KeyType_impl& _key)
        {
            auto cmpf = [](const typename Base::PairType& _a, const typename Base::PairType& _b) -> bool { return _a.first < _b.first; };
            typename Base::PairType lookingFor{_key, nullptr};

            RngType range;
            range.first = std::lower_bound(std::begin(_container), std::end(_container), lookingFor, cmpf);
            if (range.first == std::end(_container) || _key < range.first->first)
            {
                range.second = range.first;
                return range;
            }
            range.second = std::upper_bound(range.first, std::end(_container), lookingFor, cmpf);
            return range;
        }

    public:
        inline typename Base::RangeType findRange(const typename Base::KeyType_impl& _key)
        {
            return findRange_internal<typename Base::RangeType>(this->m_container, _key);
        }
        inline typename Base::ConstRangeType findRange(const typename Base::KeyType_impl& _key) const
        {
            return findRange_internal<typename Base::ConstRangeType>(this->m_container, _key);
        }
    };

    template<
        bool IsVectorContainer, // is container a vector
        template<typename...> class ContainerT_T,
        typename Alloc,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    struct CMultiObjectCacheBaseExt : public CMultiObjectCacheBase<!IsVectorContainer, ContainerT_T, Alloc, T, K...>
    {
    private:
        using Base = CMultiObjectCacheBase<!IsVectorContainer, ContainerT_T, Alloc, T, K...>;

    public:
        using Base::Base;

        //! Returns true if had to insert
        bool swapObjectValue(const typename Base::KeyType_impl& _key, const typename Base::ValueType_PtrToConst_impl _obj, typename Base::ValueType_impl _val)
        {
            this->greet(_val); // grab before drop

            auto range = this->findRange(_key);
            typename Base::IteratorType found = find(range, _key, _obj);

            if (found != range.second)
            {
                this->dispose(found->second);
                found->second = _val;
                return false;
            }
            this->m_container.insert(range.second, typename Base::PairType{_key, _val});
            return true;
        }

        //! @returns true if object was removed (i.e. was present in cache)
        inline bool removeObject(const typename Base::ValueType_impl _obj, const typename Base::KeyType_impl& _key)
        {
            typename Base::RangeType range = this->findRange(_key);
            for (auto it = range.first; it != range.second; ++it)
            {
                if (it->second == _obj)
                {
                    this->dispose(it->second);
                    Base::m_container.erase(it);
                    return true;
                }
            }
            return false;
        }

    private:
        typename Base::IteratorType find(const typename Base::RangeType& _range, const typename Base::KeyType_impl& _key, const typename Base::ValueType_PtrToConst_impl _obj) const
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

    template<
        bool isVectorContainer,
        template<typename...> class ContainerT_T,
        typename Alloc,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    struct CUniqObjectCacheBase;

    template<
        template<typename...> class ContainerT_T,
        typename Alloc,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    struct CUniqObjectCacheBase<true, ContainerT_T, Alloc, T, K...> : public CObjectCacheBase<ContainerT_T, Alloc, T, K...>
    {
    private:
        using Base = CObjectCacheBase<ContainerT_T, Alloc, T, K...>;

    public:
        using Base::Base;

        inline bool insert(const typename Base::KeyType_impl& _key, typename Base::ValueType_impl _val)
        {
            INSERT_IMPL_VEC
        }

        inline typename Base::RangeType findRange(const typename Base::KeyType_impl& _key)
        {
            auto it = std::lower_bound(std::begin(this->m_container), std::end(this->m_container), typename Base::PairType{ _key, nullptr });
            if (it == std::end(this->m_container) || it->first > _key)
                return { it, it };
            return { it, std::next(it) };
        }
        inline typename Base::ConstRangeType findRange(const typename Base::KeyType_impl& _key) const
        {
            typename Base::RangeType range =
                const_cast<typename std::decay<decltype(*this)>::type&>(*this).findRange(_key);
            return typename Base::ConstRangeType(range.first, range.second);
        }
    };
    template<
        template<typename...> class ContainerT_T,
        typename Alloc,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    struct CUniqObjectCacheBase<false, ContainerT_T, Alloc, T, K...> : public CObjectCacheBase<ContainerT_T, Alloc, T, K...>
    {
    private:
        using Base = CObjectCacheBase<ContainerT_T, Alloc, T, K...>;

    public:
        using Base::Base;

        inline bool insert(const typename Base::KeyType_impl& _key, typename Base::ValueType_impl _val)
        {
            INSERT_IMPL_ASSOC
        }

        inline typename Base::RangeType findRange(const typename Base::KeyType_impl& _key)
        {
            auto it = this->m_container.lower_bound(_key);
            if (it == std::end(this->m_container) || it->first > _key)
                return { it, it };
            return { it, std::next(it) };
        }
        inline typename Base::ConstRangeType findRange(const typename Base::KeyType_impl& _key) const
        {
            typename Base::RangeType range =
                const_cast<typename std::decay<decltype(*this)>::type&>(*this).findRange(_key);
            return typename Base::ConstRangeType(range.first, range.second);
        }
    };

    template<
        bool isVectorContainer,
        template<typename...> class ContainerT_T,
        typename Alloc,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    struct CUniqObjectCacheBaseExt : public CUniqObjectCacheBase<isVectorContainer, ContainerT_T, Alloc, T, K...>
    {
    private:
        using Base = CUniqObjectCacheBase<isVectorContainer, ContainerT_T, Alloc, T, K...>;

    public:
        using Base::Base;

        //! @returns true if object was removed (i.e. was present in cache)
        inline bool removeObject(const typename Base::ValueType_impl _obj, const typename Base::KeyType_impl& _key)
        {
            typename Base::RangeType range = this->findRange(_key);
            auto it = range.first;
            if (Base::isNonZeroRange(range) && it->second == _obj)
            {
                this->dispose(it->second);
                this->m_container.erase(it);
                return true;
            }
            return false;
        }

        //! Returns true if had to insert
        bool swapObjectValue(const typename Base::KeyType_impl& _key, const typename Base::ValueType_PtrToConst_impl _obj, typename Base::ValueType_impl _val)
        {
            this->greet(_val); // grab before drop

            typename Base::RangeType range = this->findRange(_key);
            auto it = range.first;

            if (Base::isNonZeroRange(range) && it->second == _obj)
            {
                this->dispose(it->second);
                it->second = _val;
                return false;
            }
            this->m_container.insert(it, { _key, _val });
            return true;
        }
    };

    template <
        bool forMultiCache,
        bool isVectorContainer,
        template<typename...> class ContainerT_T,
        typename Alloc,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    struct CDirectCacheBase : 
        public std::conditional<forMultiCache, CMultiObjectCacheBaseExt<isVectorContainer, ContainerT_T, Alloc, T, K...>, CUniqObjectCacheBaseExt<isVectorContainer, ContainerT_T, Alloc, T, K...>>::type
    {
    private:
        using Base = typename std::conditional<forMultiCache, CMultiObjectCacheBaseExt<isVectorContainer, ContainerT_T, Alloc, T, K...>, CUniqObjectCacheBaseExt<isVectorContainer, ContainerT_T, Alloc, T, K...>>::type;

    public:
        using Base::Base;

        inline bool changeObjectKey(typename Base::ValueType_impl _obj, const typename Base::KeyType_impl& _key, const typename Base::KeyType_impl& _newKey)
        {
            this->greet(_obj);
            if (this->removeObject(_obj, _key))
            {
                this->insert(_newKey, _obj);
                this->dispose(_obj);
                return true;
            }
            this->dispose(_obj);
            return false;
        }

        inline bool findAndStoreRange(const typename Base::KeyType_impl& _key, size_t& _inOutStorageSize, typename Base::MutablePairType* _out) const
        {
            return this->outputRange(this->findRange(_key), _inOutStorageSize, _out);
        }

        inline bool findAndStoreRange(const typename Base::KeyType_impl& _key, size_t& _inOutStorageSize, typename Base::ValueType_impl* _out) const
        {
            return this->outputRange(this->findRange(_key), _inOutStorageSize, _out);
        }

        bool getKeyRangeOrReserve(typename Base::RangeType* _outrange, const typename Base::KeyType_impl& _key)
        {
            *_outrange = this->findRange(_key);
            if (!Base::isNonZeroRange(*_outrange))
            {
                _outrange->first = this->m_container.insert(_outrange->second, { _key, nullptr });
                _outrange->second = std::next(_outrange->first);
                this->greet(nullptr);
                return false;
            }
            return true;
        }

        bool getAndStoreKeyRangeOrReserve(const typename Base::KeyType_impl& _key, size_t& _inOutStorageSize, typename Base::ValueType_impl* _out, bool* _gotAll)
        {
            bool dummy;
            if (!_gotAll)
                _gotAll = &dummy;
            auto rng = this->findRange(_key);
            bool res = true;
            if (!Base::isNonZeroRange(rng))
            {
                rng.first = this->m_container.insert(rng.second, { _key, nullptr });
                rng.second = std::next(rng.first);
                this->greet(nullptr);
                res = false;
            }
            *_gotAll = this->outputRange(rng, _inOutStorageSize, _out);
            return res;
        }
    };

    template <
        bool isVectorContainer,
        template<typename...> class ContainerT_T,
        typename Alloc,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    using CDirectMultiCacheBase = CDirectCacheBase<true, isVectorContainer, ContainerT_T, Alloc, T, K...>;
    template <
        bool isVectorContainer,
        template<typename...> class ContainerT_T,
        typename Alloc,
        typename T, //value type for container
        typename ...K //optionally key type for std::map/std::unordered_map
    >
    using CDirectUniqCacheBase = CDirectCacheBase<false, isVectorContainer, ContainerT_T, Alloc, T, K...>;
}

template<
    typename K,
    typename T,
    template<typename...> class ContainerT_T = std::vector,
    typename Alloc = core::allocator<std::pair<const K, T*>>,
    bool = impl::is_same_templ<ContainerT_T, std::vector>::value
>
class CMultiObjectCache;

template<
    typename K,
    typename T,
    template<typename...> class ContainerT_T,
    typename Alloc
>
class CMultiObjectCache<K, T, ContainerT_T, Alloc, true> : 
    public impl::CDirectMultiCacheBase<true, ContainerT_T, Alloc, std::pair<K, T*>>,
    public impl::PropagTypedefs<T, K>
{
private:
    using Base = impl::CDirectMultiCacheBase<true, ContainerT_T, Alloc, std::pair<K, T*>>;

public:
    using Base::Base;
};
template<
    typename K,
    typename T,
    template<typename...> class ContainerT_T,
    typename Alloc
>
class CMultiObjectCache<K, T, ContainerT_T, Alloc, false> : 
    public impl::CDirectMultiCacheBase<false, ContainerT_T, Alloc, T*, const K>,
    public impl::PropagTypedefs<T, const K>
{
    static_assert(impl::is_same_templ<ContainerT_T, std::multimap>::value || impl::is_same_templ<ContainerT_T, std::unordered_multimap>::value, "ContainerT_T must be one of: std::vector, std::multimap, std::unordered_multimap");

private:
    using Base = impl::CDirectMultiCacheBase<false, ContainerT_T, Alloc, T*, const K>;

public:
    using Base::Base;
};


template<
    typename K,
    typename T,
    template<typename...> class ContainerT_T = std::vector,
    typename Alloc = core::allocator<std::pair<const K, T*>>,
    bool = impl::is_same_templ<ContainerT_T, std::vector>::value
>
class CObjectCache;

template<
    typename K,
    typename T,
    template<typename...> class ContainerT_T,
    typename Alloc
>
class CObjectCache<K, T, ContainerT_T, Alloc, true> : 
    public impl::CDirectUniqCacheBase<true, ContainerT_T, Alloc, std::pair<K, T*>>,
    public impl::PropagTypedefs<T, K>
{
    using Base = impl::CDirectUniqCacheBase<true, ContainerT_T, Alloc, std::pair<K, T*>>;

public:
    using Base::Base;
};


template<
    typename K,
    typename T,
    template<typename...> class ContainerT_T,
    typename Alloc
>
class CObjectCache<K, T, ContainerT_T, Alloc, false> : 
    public impl::CDirectUniqCacheBase<false, ContainerT_T, Alloc, T*, const K>,
    public impl::PropagTypedefs<T, const K>
{
    static_assert(impl::is_same_templ<ContainerT_T, std::map>::value || impl::is_same_templ<ContainerT_T, std::unordered_map>::value, "ContainerT_T must be one of: std::vector, std::map, std::unordered_map");
    using Base = impl::CDirectUniqCacheBase<false, ContainerT_T, Alloc, T*, const K>;

public:
    using Base::Base;
};

}}

#undef INSERT_IMPL_VEC
#undef INSERT_IMPL_ASSOC
#endif //__C_OBJECT_CACHE_H_INCLUDED__
