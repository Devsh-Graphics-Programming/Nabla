#ifndef __LRU_CACHE_H_INCLUDED__
#define __LRU_CACHE_H_INCLUDED__

#include "irr/core/CFixedCapacityDoublyLinkedList.h"

namespace irr {
	namespace core {

template<typename Key, typename Value, typename MapHash = std::hash<Key>, typename MapEquals = std::equal_to<Key>>
class LRUcache
{
	typedef typename Snode<std::pair<Key, Value>>* iterator_t;

	DoublyLinkedList<std::pair<Key, Value>> m_list;
	unordered_map<Key, iterator_t, MapHash, MapEquals> m_shortcut_map;
	uint32_t m_capacity;

	inline iterator_t common_peek(Key& key)
	{
		if (m_shortcut_map.find(key) != m_shortcut_map.end())
		{
			return m_shortcut_map[key];
		}
		return nullptr;
	}

public:

	//LRUcache();
	inline LRUcache(const uint32_t& capacity) : m_capacity(capacity), m_shortcut_map(), m_list(capacity)
	{
		assert(capacity > 1);
		m_shortcut_map.reserve(capacity);
	}
	//inline void print()
	//{
	//	iterator_t i = m_list.getFirst();
	//	while (i != nullptr)
	//	{
	//		std::cout << i->data.first << ' ' << i->data.second << std::endl;
	//		i = i->next;
	//	}
	//}
	inline void insert(Key& k, Value& v)
	{
		if (m_shortcut_map.find(k) == m_shortcut_map.end() && m_shortcut_map.size() == m_capacity)
		{
			m_shortcut_map.erase(m_list.getLast()->data.first);
		}
		else
		{
			m_shortcut_map.erase(k);
		}
		m_list.pushFront(std::pair<Key, Value>(k, v));
		m_shortcut_map[k] = m_list.getFirst();

	}
	//Try getting the value from the cache, or null if key doesnt exist in cache or has been removed
	inline Value get(Key& key)
	{
		auto iterator = common_peek(key);
		if (pair != nullptr)
		{
			m_list.moveToFront(iterator);
			return iterator->data.second;
		}
		return nullptr;
	}
	inline Value peek(Key& key)
	{
		iterator_t i = common_peek(key);
		if (i == nullptr) return nullptr;
		else return i->data.second;
	}
	inline void erase(Key& key)
	{
		auto i = common_peek(key);
		if (i != nullptr)
		{
			m_list.erase(i);
		}
	}
};


	}	//namespace core
}		//namespace irr
#endif // !__LRU_CACHE_H_INCLUDED__