// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_LRU_CACHE_H_INCLUDED__
#define __NBL_CORE_LRU_CACHE_H_INCLUDED__

#include "nbl/core/containers/FixedCapacityDoublyLinkedList.h"
#include <iostream>

namespace nbl
{
namespace core
{

namespace impl
{
	template<typename Key, typename Value, typename MapHash, typename MapEquals>
	class NBL_API LRUCacheBase
	{
		public:
			using list_value_t = std::pair<Key,Value>;
			using list_t = FixedCapacityDoublyLinkedList<list_value_t>;
			_NBL_STATIC_INLINE_CONSTEXPR uint32_t invalid_iterator = list_t::invalid_iterator;

			using disposal_func_t = typename list_t::disposal_func_t;

		protected:
			list_t m_list;

		private:
			MapHash m_hash;
			MapEquals m_equals;

		protected:
			const mutable Key* searchedKey;

			inline LRUCacheBase(const uint32_t capacity, MapHash&& _hash, MapEquals&& _equals, disposal_func_t&& df) : m_list(capacity, std::move(df)), m_hash(std::move(_hash)), m_equals(std::move(_equals)), searchedKey(nullptr)
			{
			}

		public:
			inline const Key& getReference(const uint32_t nodeAddr) const
			{
				if (nodeAddr!=invalid_iterator)
					return m_list.get(nodeAddr)->data.first;
				else
					return *searchedKey;
			}

			inline const MapHash& getHash() const { return m_hash; }

			inline const MapEquals& getEquals() const { return m_equals; }
	};
}

// Key-Value Least Recently Used cache
// Stores fixed size amount of elements. 
// When the cache is full inserting will remove the least used entry
template<typename Key, typename Value, typename MapHash=std::hash<Key>, typename MapEquals=std::equal_to<Key> >
class NBL_API LRUCache : private impl::LRUCacheBase<Key,Value,MapHash,MapEquals>
{
		// typedefs
		typedef impl::LRUCacheBase<Key,Value,MapHash,MapEquals> base_t;
		typedef LRUCache<Key,Value,MapHash,MapEquals> this_t;

		_NBL_STATIC_INLINE_CONSTEXPR uint32_t invalid_iterator = base_t::invalid_iterator;

		// wrappers
		struct WrapHash
		{
			const base_t* cache;

			inline std::size_t operator()(const uint32_t nodeAddr) const
			{
				return cache->getHash()(cache->getReference(nodeAddr));
			}
		};
		struct WrapEquals
		{
			const base_t* cache;

			inline bool operator()(const uint32_t lhs, const uint32_t rhs) const
			{
				return cache->getEquals()(cache->getReference(lhs),cache->getReference(rhs));
			}
		};

		// members
		unordered_set<uint32_t,WrapHash,WrapEquals> m_shortcut_map;
		using shortcut_iterator_t = typename unordered_set<uint32_t,WrapHash,WrapEquals>::const_iterator;
		inline shortcut_iterator_t common_find(const Key& key) const
		{
			base_t::searchedKey = &key;
			return m_shortcut_map.find(invalid_iterator);
		}
		inline shortcut_iterator_t common_find(const Key& key, bool& success) const
		{
			auto retval = common_find(key);
			success = retval!=m_shortcut_map.end();
			return retval;
		}

		//get iterator associated with a key, or invalid_iterator if key is not within the cache
		inline uint32_t common_peek(const Key& key) const
		{
			bool success;
			shortcut_iterator_t iterator = common_find(key,success);
			return success ? (*iterator):invalid_iterator;
		}

		template<typename K,typename V>
		inline void common_insert(K&& k, V&& v)
		{
			bool success;
			shortcut_iterator_t iterator = common_find(k,success);
			if (success)
			{
				const auto nodeAddr = *iterator;
				base_t::m_list.get(nodeAddr)->data.second = std::forward<V>(v);
				base_t::m_list.moveToFront(nodeAddr);
			}
			else
			{
				const bool overflow = m_shortcut_map.size()>=base_t::m_list.getCapacity();
				if (overflow)
				{
					m_shortcut_map.erase(base_t::m_list.getLastAddress());
					base_t::m_list.popBack();
				}
				base_t::m_list.pushFront(std::make_pair(std::forward<K>(k),std::forward<V>(v)));
				m_shortcut_map.insert(base_t::m_list.getFirstAddress());
			}
		}

	public:
		using disposal_func_t = typename base_t::disposal_func_t;
		using assoc_t = typename base_t::list_value_t;

		//Constructor
		inline LRUCache(const uint32_t capacity, disposal_func_t&& _df = disposal_func_t(), MapHash&& _hash=MapHash(), MapEquals&& _equals=MapEquals()) :
			base_t(capacity,std::move(_hash),std::move(_equals),std::move(_df)),
			m_shortcut_map(capacity>>2,WrapHash{this},WrapEquals{this}) // 4x less buckets than capacity seems reasonable
		{
			assert(capacity > 1);
			m_shortcut_map.reserve(capacity);
		}

		inline void print(std::ostream& ostream)
		{
			auto node = base_t::m_list.getBegin();
			while (true)
			{
				ostream <<"k:" << node->data.first << "    v:" << node->data.second << std::endl;
				if (node->next == invalid_iterator)
					break;
				node = base_t::m_list.get(node->next);
			}
		}

		//insert an element into the cache, or update an existing one with the same key
		inline void insert(Key&& k, Value&& v) { common_insert(std::move(k), std::move(v)); }
		inline void insert(Key&& k, const Value& v) { common_insert(std::move(k), v); }
		inline void insert(const Key& k, Value&& v) { common_insert(k, std::move(v)); }
		inline void insert(const Key& k, const Value& v) { common_insert(k, v); }

		//get the value from cache at an associated Key, or nullptr if Key is not contained within cache. Marks the returned value as most recently used
		inline Value* get(const Key& key)
		{
			auto i = common_peek(key);
			if (i!=invalid_iterator)
			{
				base_t::m_list.moveToFront(i);
				return &(base_t::m_list.get(i)->data.second);
			}
			else
				return nullptr;
		}

		//get the value from cache at an associated Key, or nullptr if Key is not contained within cache. Does not alter the value use order
		inline const Value* peek(const Key& key) const
		{
			uint32_t i = common_peek(key);
			if (i!=invalid_iterator)
				return &(base_t::m_list.get(i)->data.second);
			else
				return nullptr;
		}
		inline Value* peek(const Key& key)
		{
			uint32_t i = common_peek(key);
			if (i != invalid_iterator)
				return &(base_t::m_list.get(i)->data.second);
			else
				return nullptr;
		}

		//remove element at key if present
		inline void erase(const Key& key)
		{
			bool success;
			shortcut_iterator_t iterator = common_find(key,success);
			if (success)
			{
				base_t::m_list.erase(*iterator);
				m_shortcut_map.erase(iterator);
			}
		}
};


}	//namespace core
}		//namespace nbl
#endif 