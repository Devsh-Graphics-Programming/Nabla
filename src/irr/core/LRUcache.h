#ifndef __LRU_CACHE_H_INCLUDED__
#define __LRU_CACHE_H_INCLUDED__

#include "irr/core/CFixedCapacityDoublyLinkedList.h"

namespace irr {
	namespace core {

//Key-Value Least Recently Used cache
//Stores fixed size amount of elements. 
//When the cache is full inserting will remove the least used entry
template<typename Key, typename Value, typename MapHash = std::hash<Key>, typename MapEquals = std::equal_to<Key>>
class LRUcache
{
	typedef LRUcache<Key, Value, MapHash, MapEquals> base_t;
	typedef std::pair<Key*, Value> list_template_t;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t invalid_iterator = Snode<list_template_t>::invalid_iterator;

	DoublyLinkedList<list_template_t> m_list;
	unordered_map<Key, uint32_t, MapHash, MapEquals> m_shortcut_map;
	uint32_t m_capacity;

	//get iterator associated with a key, or invalid_iterator if key is not within the cache
	inline uint32_t common_peek(Key& key)
	{
		auto i = m_shortcut_map.find(key);
		if (i != m_shortcut_map.end())
			return i->second;
		return invalid_iterator;
	}
	inline const uint32_t common_peek(const Key& key) const 
	{
		return const_cast<base_t*>(this)->common_peek(const_cast<Key&>(key));
	}

	inline void common_erase(uint32_t iterator, Key& key)
	{
		m_shortcut_map.erase(key);
		m_list.erase(iterator);
	}

	template<typename K,typename V>
	inline void common_insert(K&& k, V&& v)
	{
		auto iterator = m_shortcut_map.find(k);
		if (iterator == m_shortcut_map.end())
		{
			if (m_shortcut_map.size() >= m_capacity)
			{
				m_shortcut_map.erase(*(m_list.getBack()->data.first));
				m_list.popBack();
			}
			uint32_t newElementAddress = m_list.reserveAddress();
			auto newIterator = m_shortcut_map.insert(std::make_pair(k,newElementAddress));		//newIterator type is pair<iterator,bool>
			assert(newIterator.second);															//check whether insertion was successful
			m_list.insertAt(
				newElementAddress,
				std::make_pair< Key* ,Value>(
					const_cast<Key*>(&(newIterator.first->first)  ), /* referencing the key's copy inside m_shortcut_map */
					std::forward<V>(v)));
		}
		else
		{
			m_list.get(iterator->second)->data.second = std::forward<V>(v);
			m_list.moveToFront(iterator->second);
		}
	}
public:
	//Constructor
	inline LRUcache(const uint32_t capacity) : m_capacity(capacity), m_shortcut_map(), m_list(capacity)
	{
		assert(capacity > 1);
		m_shortcut_map.reserve(capacity);
	}

#ifdef _IRR_DEBUG
	inline void print()
	{
		auto node = m_list.getBegin();
		while (true)
		{
			std::cout <<"k:" << *(node->data.first) << "    v:" << node->data.second << std::endl;
			if (node->next == invalid_iterator)
				break;
			node = m_list.get(node->next);
		}
	}
#endif // _IRR_DEBUG

	//insert an element into the cache, or update an existing one with the same key
	inline void insert(Key&& k, Value&& v) { common_insert(std::move(k), std::move(v)); }
	inline void insert(Key&& k, const Value& v) { common_insert(std::move(k), v); }
	inline void insert(const Key& k, Value&& v) { common_insert(k, std::move(v)); }
	inline void insert(const Key& k, const Value& v) { common_insert(k, v); }

	//get the value from cache at an associated Key, or nullptr if Key is not contained within cache. Marks the returned value as most recently used
	inline Value* get(Key& key)
	{
		auto i = common_peek(key);
		if (i == invalid_iterator) return nullptr;
		else {
			m_list.moveToFront(i);
			return &(m_list.get(i)->data.second);
		}
	}
	inline const Value* get(const Key& key) const
	{
		return const_cast<base_t*>(this)->get(const_cast<Key&>(key));
	}

	//get the value from cache at an associated Key, or nullptr if Key is not contained within cache. Does not alter the value use order
	inline const Value* peek(Key& key) const
	{
		uint32_t i = common_peek(key);
		if (i == invalid_iterator) return nullptr;
		else return &(m_list.get(i)->data.second);
	}
	inline const Value* peek(const Key& key) const
	{
		return const_cast<base_t*>(this)->peek(const_cast<Key&>(key));
	}
	//remove element at key if present
	inline void erase(Key& key)
	{
		uint32_t i = common_peek(key);
		if (i != invalid_iterator)
			common_erase(i, key);
	}
	inline void erase(const Key& key) const
	{
		const_cast<base_t*>(this)->erase(const_cast<Key&>(key));
	}
};


	}	//namespace core
}		//namespace irr
#endif // !__LRU_CACHE_H_INCLUDED__