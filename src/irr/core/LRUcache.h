#include "irr/core/Types.h"
#include "irr/core/alloc/PoolAddressAllocator.h"

namespace irr {
	namespace core {

		template<typename Key, typename Value, typename MapHash, typename MapEquals>
		class LRUcache
		{
			typedef std::list<std::pair<Key, Value>, irr::core::PoolAddressAllocator<uint32_t>> list_t;
			//apparently, list_t::iterator is not a type
			//workaround would be 
			typedef std::iterator<
									std::bidirectional_iterator_tag,				// iterator_category
									std::pair<Key, Value>,							// value_type
									std::pair<Key, Value>,							// difference_type
									const std::pair<Key, Value>*,					// pointer
									std::pair<Key, Value>							// reference
								 > iterator_t;			
			
			list_t linkedlist;
			unordered_map<Key, iterator_t, MapHash, MapEquals> shortcut_map;
			uint32_t cap;

			inline void common_erase(iterator_t& i)
			{
				shortcut_map.erase((*i).first);
				linkedlist.erase(i);

			}

			//this return type also throws an error
			inline iterator_t common_peek(Key& key)
			{
				if (shortcut_map.find(k) != shortcut_map.end())
				{
					return shortcut_map[k];
				}
				return nullptr;
			}

		public:
			inline LRUcache() = default;
			inline LRUcache(uint32_t& capacity)
			{
				assert(capacity > 1);
				cap = capacity;
				shortcut_map.reserve(capacity);
				//should list also use reserve() ?
			}

			inline void insert(Key& k, Value& v)
			{
				if (shortcut_map.find(k) == shortcut_map.end() && linkedlist.size() == cap) {
					auto lastpair = linkedlist.back();
					shortcut_map.erase(lastpair.first);
					linkedlist.pop_back();
				}
				else
					linkedlist.remove(k);
				linkedlist.push_front(k);
				shortcut_map[k] = v;

			}
			//Try getting the value from the cache, or null if key doesnt exist in cache or has been removed
			inline Value get(Key& key)
			{
				auto iterator = common_peek(key);
				if (pair != nullptr)
				{
					auto temp = *iterator;
					linkedlist.erase(iterator);
					linkedlist.push_front(temp);
					return temp.second;
				}
				return nullptr;
			}
			inline Value peek(Key& key)
			{
				auto i = common_peek(key);
				if (i != nullptr)
				{
					return (*i).second;
				}
				return nullptr;
			}
			inline void erase(Key& key)
			{
				auto i = common_peek(key);
				if (i != nullptr)
				{
					common_erase(i);
				}
			}
		};


	}	//namespace core
}		//namespace irr