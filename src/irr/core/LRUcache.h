#include <list>
#include "irr/core/Types.h"
#include "COpenGLDriver.h"

namespace irr {
	namespace core {

		template<typename Key, typename Value,typename MapHash,typename MapEquals>
		class LRUcache {
			list<Key> stored_keys;
			unordered_map<Key, Value, MapHash,MapEquals> map;
			uint32_t cap;
			std::atomic_uint32_t timestamp;
		public:
			inline LRUcache() = default;
			inline LRUcache(uint32_t& capacity)
			{
				cap = capacity;
			}

			inline void insert(Key &k, Value &v)
			{
				if (map.find(k) == map.end() && stored_keys.size() == cap) {
					auto last = stored_keys.back();
					stored_keys.pop_back();
					map.erase(last);
				}
				else
					stored_keys.remove(k);
				stored_keys.push_front(k);
				map[k] = v;

			}
			//Try getting the value from the cache, or null if key doesnt exist in cache or has been removed
			inline Value get(Key &key)
			{
				if (map.find(key) == map.end())
					return NULL;
				timestamp++;
				return map[key];
			}
			//Try to get value without incrementing the timestamp
			inline Value peek(Key &key)
			{
				if (map.find(key) == ma.end())
					return NULL;
				return map[key];
			}

			std::uint32_t getTimestamp()
			{
				return timestamp;
			}
		};


	}	//namespace core
}		//namespace irr