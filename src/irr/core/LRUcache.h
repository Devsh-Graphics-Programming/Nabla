#ifndef __LRU_CACHE_H_INCLUDED__
#define __LRU_CACHE_H_INCLUDED__

#include "irr/core/CFixedCapacityDoublyLinkedList.h"

namespace irr {
	namespace core {

template<typename Key, typename Value, typename MapHash = std::hash<Key>, typename MapEquals = std::equal_to<Key>>
class LRUcache
{
#define invalid_address 0xdeadbeefu
	typedef std::pair<Key, Value> list_template_t;
	DoublyLinkedList<list_template_t> m_list;
	unordered_map<Key, uint32_t, MapHash, MapEquals> m_shortcut_map;
	uint32_t m_capacity;

	inline uint32_t common_peek(Key& key)
	{
		if (m_shortcut_map.find(key) != m_shortcut_map.end())
		{
			return m_shortcut_map[key];
		}
		return invalid_address;
	}

public:

	//LRUcache();
	inline LRUcache(const uint32_t& capacity) : m_capacity(capacity), m_shortcut_map(), m_list(capacity)
	{
		assert(capacity > 1);
		m_shortcut_map.reserve(capacity);
	}
	inline void print()
	{
		uint32_t i = m_list.getBegin();
		while (i != invalid_address)
		{
			auto node = reinterpret_cast<Snode<list_template_t>*>(i);
			std::cout << node->data.first << ' ' << node->data.second << std::endl;
			i = node->next;
		}
	}
	inline void insert(Key& k, Value& v)
	{
		if (m_shortcut_map.find(k) == m_shortcut_map.end())
		{
			if (m_shortcut_map.size() >= m_capacity)
			{
				m_shortcut_map.erase(reinterpret_cast<Snode<list_template_t>*>(m_list.getBack())->data.first);
				m_list.popBack();
				m_list.pushFront(std::pair<Key, Value>(k, v));
				m_shortcut_map[k] = m_list.getBegin();
			}
		}
		else
		{
			m_list.moveToFront(m_shortcut_map[k]);
		}

	}
	//Try getting the value from the cache, or null if key doesnt exist in cache or has been removed
	inline Value get(Key& key)
	{
		auto i = common_peek(key);
		if (i != invalid_address)
		{
			m_list.moveToFront(i);
			return reinterpret_cast<Snode<list_template_t>*>(i)->data.second;
		}
		return nullptr;
	}
	inline Value peek(Key& key)
	{
		uint32_t i = common_peek(key);
		if (i == invalid_address) return nullptr;
		else return  reinterpret_cast<Snode<list_template_t>*>(i)->data.second;
	}
	inline void erase(Key& key)
	{
		uint32_t i = common_peek(key);
		if (i != invalid_address)
			m_list.erase(i);
	}
};


	}	//namespace core
}		//namespace irr
#endif // !__LRU_CACHE_H_INCLUDED__