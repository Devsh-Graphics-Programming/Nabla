// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_LRU_CACHE_H_INCLUDED__
#define __NBL_CORE_LRU_CACHE_H_INCLUDED__

// Legacy include
#include "nbl/core/containers/FixedCapacityDoublyLinkedList.h"
#include "nbl/core/containers/DoublyLinkedList.h"
#include <iostream>
#include "nbl/system/ILogger.h"

namespace nbl
{
namespace core
{

namespace impl
{

template<typename Key, typename Value, typename MapHash, typename MapEquals, typename List>
class LRUCacheBase
{
	public:
		using list_t = List;
		using list_value_t = typename list_t::value_t;
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t invalid_iterator = list_t::invalid_iterator;

		using disposal_func_t = typename list_t::disposal_func_t;

	protected:
		list_t m_list;
		MapHash m_hash;
		MapEquals m_equals;
		const mutable Key* searchedKey;

		LRUCacheBase(const uint32_t capacity, MapHash&& _hash, MapEquals&& _equals, disposal_func_t&& df) : m_list(capacity, std::move(df)), m_hash(std::move(_hash)), m_equals(std::move(_equals)), searchedKey(nullptr)
		{ }

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

} //namespace impl


// Key-Value Least Recently Used cache
// Stores fixed size amount of elements. 
// When the cache is full inserting will remove the least used entry
template<typename Key, typename Value, typename MapHash=std::hash<Key>, typename MapEquals=std::equal_to<Key> >
class [[deprecated]] LRUCache : protected impl::LRUCacheBase<Key,Value,MapHash,MapEquals,FixedCapacityDoublyLinkedList<std::pair<Key, Value> > >, public core::Unmovable, public core::Uncopyable
{
	// typedefs
	using list_t = FixedCapacityDoublyLinkedList<std::pair<Key, Value> >;
	using base_t = impl::LRUCacheBase<Key,Value,MapHash,MapEquals,list_t>;
	using this_t = LRUCache<Key,Value,MapHash,MapEquals>;

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

public:
	using disposal_func_t = typename base_t::disposal_func_t;
	using assoc_t = typename base_t::list_value_t;

	//Constructor
	LRUCache(const uint32_t capacity, disposal_func_t&& _df = disposal_func_t(), MapHash&& _hash=MapHash(), MapEquals&& _equals=MapEquals()) :
		base_t(capacity,std::move(_hash),std::move(_equals),std::move(_df)),
		m_shortcut_map(capacity>>2,WrapHash{this},WrapEquals{this}) // 4x less buckets than capacity seems reasonable
	{
		assert(capacity > 1);
		m_shortcut_map.reserve(capacity);
	}
	LRUCache() = delete;

	inline void print(core::smart_refctd_ptr<system::ILogger> logger)
	{
		logger->log("Printing LRU cache contents");
		auto nodeAddr = base_t::m_list.getLastAddress();
		while (nodeAddr != invalid_iterator)
		{
			auto node = base_t::m_list.get(nodeAddr);
			std::ostringstream stringStream;
			stringStream << "k: '" << node->data.first << "', v: '" << node->data.second << "'\t prev: " << node->prev << " | curr: " << nodeAddr << " | next: " << node->next;
			logger->log(stringStream.str());
			nodeAddr = node->prev;
			node = base_t::m_list.get(node->prev);
		}
	}
		
	template<typename K, typename V, std::invocable<const Value&> EvictionCallback> requires std::is_constructible_v<Value,V> // && (std::is_same_v<Value,V> || std::is_assignable_v<Value,V>) // is_assignable_v<int, int&> returns false :(
	inline Value* insert(K&& k, V&& v, EvictionCallback&& evictCallback)
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
				evictCallback(base_t::m_list.getBack()->data.second);
				m_shortcut_map.erase(base_t::m_list.getLastAddress());
				base_t::m_list.popBack();
			}
			if constexpr (std::is_same_v<Value, V>)
				base_t::m_list.emplaceFront(std::forward<K>(k), std::forward<V>(v));
			else
				base_t::m_list.emplaceFront(std::forward<K>(k), Value(std::forward<V>(v)) );
			m_shortcut_map.insert(base_t::m_list.getFirstAddress());
		}
		return &base_t::m_list.getBegin()->data.second;
	}

	template<typename K, typename V>
	inline Value* insert(K&& k, V&& v)
	{
		return insert(std::forward<K>(k),std::forward<V>(v),[](const Value& ejected)->void{});
	}

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

protected:
	unordered_set<uint32_t,WrapHash,WrapEquals> m_shortcut_map;
};

// Key-Value Least Recently Used cache
// Capacity can be increased at user's will
// When the cache is full inserting will remove the least used entry
template<typename Key, typename Value, typename MapHash = std::hash<Key>, typename MapEquals = std::equal_to<Key> >
class ResizableLRUCache : protected impl::LRUCacheBase<Key, Value, MapHash, MapEquals, DoublyLinkedList<std::pair<Key, Value> > >, public core::Unmovable, public core::Uncopyable
{
	// typedefs
	using list_t = DoublyLinkedList<std::pair<Key, Value> >;
	using base_t = impl::LRUCacheBase<Key, Value, MapHash, MapEquals, list_t>;
	using this_t = ResizableLRUCache<Key, Value, MapHash, MapEquals>;

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
			return cache->getEquals()(cache->getReference(lhs), cache->getReference(rhs));
		}
	};

	// members
	using shortcut_iterator_t = typename unordered_set<uint32_t, WrapHash, WrapEquals>::const_iterator;
	inline shortcut_iterator_t common_find(const Key& key) const
	{
		base_t::searchedKey = &key;
		return m_shortcut_map.find(invalid_iterator);
	}
	inline shortcut_iterator_t common_find(const Key& key, bool& success) const
	{
		auto retval = common_find(key);
		success = retval != m_shortcut_map.end();
		return retval;
	}

	//get iterator associated with a key, or invalid_iterator if key is not within the cache
	inline uint32_t common_peek(const Key& key) const
	{
		bool success;
		shortcut_iterator_t iterator = common_find(key, success);
		return success ? (*iterator) : invalid_iterator;
	}

public:
	using disposal_func_t = typename base_t::disposal_func_t;
	using assoc_t = typename base_t::list_value_t;

	//Constructor
	ResizableLRUCache(uint32_t capacity, disposal_func_t&& _df = disposal_func_t(), MapHash&& _hash = MapHash(), MapEquals&& _equals = MapEquals()) :
		base_t(capacity, std::move(_hash), std::move(_equals), std::move(_df)), m_capacity(capacity),
		m_shortcut_map(capacity >> 2, WrapHash{ this }, WrapEquals{ this }) // 4x less buckets than capacity seems reasonable
	{
		m_shortcut_map.reserve(capacity);
	}
	ResizableLRUCache() = delete;

	inline void print(core::smart_refctd_ptr<system::ILogger> logger)
	{
		logger->log("Printing LRU cache contents");
		auto nodeAddr = base_t::m_list.getLastAddress();
		while (nodeAddr != invalid_iterator)
		{
			auto node = base_t::m_list.get(nodeAddr);
			std::ostringstream stringStream;
			stringStream << "k: '" << node->data.first << "', v: '" << node->data.second << "'\t prev: " << node->prev << " | curr: " << nodeAddr << " | next: " << node->next;
			logger->log(stringStream.str());
			nodeAddr = node->prev;
		}
	}

	/**
	* @brief Returns a string representing the elements currently in the cache in LRU order
	*
	* @param [in] newCapacity New number of elements to hold. MUST be greater than current list capacity.
	*/
	inline std::string getState()
	{
		std::ostringstream stringStream;
		auto nodeAddr = base_t::m_list.getLastAddress();
		while (nodeAddr != invalid_iterator)
		{
			auto node = base_t::m_list.get(nodeAddr);
			stringStream << "{" << node->data.first << ", " << node->data.second << "}";
			nodeAddr = node->prev;
			if (nodeAddr != invalid_iterator)
				stringStream << ", ";
		}
		return stringStream.str();
	}

	template<typename K, typename V, std::invocable<const Value&> EvictionCallback> requires std::is_constructible_v<Value, V> // && (std::is_same_v<Value,V> || std::is_assignable_v<Value,V>) // is_assignable_v<int, int&> returns false :(
	inline Value* insert(K&& k, V&& v, EvictionCallback&& evictCallback)
	{
		bool success;
		shortcut_iterator_t iterator = common_find(k, success);
		if (success)
		{
			const auto nodeAddr = *iterator;
			base_t::m_list.get(nodeAddr)->data.second = std::forward<V>(v);
			base_t::m_list.moveToFront(nodeAddr);
		}
		else
		{
			const bool overflow = m_shortcut_map.size() >= base_t::m_list.getCapacity();
			if (overflow)
			{
				evictCallback(base_t::m_list.getBack()->data.second);
				m_shortcut_map.erase(base_t::m_list.getLastAddress());
				base_t::m_list.popBack();
			}
			if constexpr (std::is_same_v<Value, V>)
				base_t::m_list.emplaceFront(std::forward<K>(k), std::forward<V>(v));
			else
				base_t::m_list.emplaceFront(std::forward<K>(k), Value(std::forward<V>(v)));
			m_shortcut_map.insert(base_t::m_list.getFirstAddress());
		}
		return &base_t::m_list.getBegin()->data.second;
	}

	template<typename K, typename V>
	inline Value* insert(K&& k, V&& v)
	{
		return insert(std::forward<K>(k), std::forward<V>(v), [](const Value& ejected)->void {});
	}

	//get the value from cache at an associated Key, or nullptr if Key is not contained within cache. Marks the returned value as most recently used
	inline Value* get(const Key& key)
	{
		auto i = common_peek(key);
		if (i != invalid_iterator)
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
		if (i != invalid_iterator)
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
		shortcut_iterator_t iterator = common_find(key, success);
		if (success)
		{
			base_t::m_list.erase(*iterator);
			m_shortcut_map.erase(iterator);
		}
	}
	
	//returns key for least recently used
	inline const Key* get_least_recently_used() const
	{
		if (m_shortcut_map.size() > 0)
			return &base_t::m_list.getBack()->data.first;
		else
			return nullptr;
	}
	
	inline size_t size() const { return m_shortcut_map.size(); }

	inline bool empty() const { return size() <= 0ull;  }

	/**
	* @brief Resizes the cache by extending its capacity so it can hold more elements. Returns a bool indicating if capacity was indeed increased.
	*
	* @param [in] newCapacity New number of elements to hold. MUST be greater than current cache capacity.
	*/
	inline bool grow(uint32_t newCapacity)
	{
		if (newCapacity <= m_capacity)
			return false;
		m_shortcut_map.reserve(newCapacity);
		return base_t::m_list.grow(newCapacity);
	}

	/**
	* @brief Empties cache and resets its state
	*/
	inline void clear()
	{
		base_t::searchedKey = nullptr;
		base_t::m_list.clear();

		auto mapBegin = m_shortcut_map.begin();
		auto mapEnd = m_shortcut_map.end();
		m_shortcut_map.erase(mapBegin, mapEnd);
	}

protected:
	unordered_set<uint32_t, WrapHash, WrapEquals> m_shortcut_map;
	uint32_t m_capacity;
};

}	//namespace core
}	//namespace nbl
#endif 