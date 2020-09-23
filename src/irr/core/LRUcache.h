#ifndef __LRU_CACHE_H_INCLUDED__
#define __LRU_CACHE_H_INCLUDED__

#include "irr/core/CFixedCapacityDoublyLinkedList.h"

namespace irr {
	namespace core {

//Key-Value LRU cache
//Stores fixed size amount of elements. 
//When inserting new, prevents overflow by removing the least recently used element.
template<typename Key, typename Value, typename MapHash = std::hash<Key>, typename MapEquals = std::equal_to<Key>>
class LRUcache
{
	typedef std::pair<Key, Value> list_template_t;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t invalid_iterator = Snode<list_template_t>::invalid_iterator;

	DoublyLinkedList<list_template_t> m_list;
	unordered_map<Key, uint32_t, MapHash, MapEquals> m_shortcut_map;
	uint32_t m_capacity;

	//get iterator associated with a key, or invalid_iterator if key is not within the cache
	inline uint32_t common_peek(Key& key)
	{
		if (m_shortcut_map.find(key) != m_shortcut_map.end())
		{
			return m_shortcut_map[key];
		}
		return invalid_iterator;
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
			std::cout << node->data.first << ' ' << node->data.second << std::endl;
			if (node->next == invalid_iterator)
				break;
			node = m_list.get(node->next);
		}
	}
#endif // _IRR_DEBUG

	//add element to the cache, or move to the front if key exists. In case of the latter, it doesnt update the value
	inline void insert(Key& k, const Value& v)
	{
		auto iterator = m_shortcut_map.find(k);
		if (iterator == m_shortcut_map.end())
		{
			if (m_shortcut_map.size() >= m_capacity)
			{
				m_shortcut_map.erase(m_list.getBack()->data.first);
				m_list.popBack();
			}
			m_list.pushFront(std::pair<Key, Value>(k, v));
			m_shortcut_map[k] = m_list.getFirstAddress();
		}
		else
		{
			m_list.get(iterator->second)->data = std::pair<Key, Value>(k, v);
			m_list.moveToFront(iterator->second);
		}

	}
	inline void insert(Key& k, Value&& v)
	{
		auto iterator = m_shortcut_map.find(k);
		if (iterator == m_shortcut_map.end())
		{
			if (m_shortcut_map.size() >= m_capacity)
			{
				m_shortcut_map.erase(m_list.getBack()->data.first);
				m_list.popBack();
			}
			m_list.pushFront(std::pair<Key, Value>(k, std::move(v)));
			m_shortcut_map[k] = m_list.getFirstAddress();

		}
		else
		{
			m_list.get(iterator->second)->data = std::pair<Key, Value>(k, std::move(v));
			m_list.moveToFront(iterator->second);
		}
	}
	//get the value from the cache, or null if key doesnt exist in cache or has been removed. Moves the element to the front of the cache.
	inline Value get(Key& key)
	{
		auto i = common_peek(key);
		if (i != invalid_iterator)
		{
			m_list.moveToFront(i);
			return m_list.get(i)->data.second;
		}
		return nullptr;
	}

	//get value without reordering the list
	inline Value peek(Key& key)
	{
		uint32_t i = common_peek(key);
		if (i == invalid_iterator) return nullptr;
		else return  m_list.get(i)->data.second;
	}

	//remove element at key if present
	inline void erase(Key& key)
	{
		uint32_t i = common_peek(key);
		if (i != invalid_iterator)
		{
			m_shortcut_map.erase(key);
			m_list.erase(i);
		}
	}
};


	}	//namespace core
}		//namespace irr
#endif // !__LRU_CACHE_H_INCLUDED__