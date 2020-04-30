#include <type_traits>
#include <utility>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <string>

#include "irr/static_if.h"
#include "irr/macros.h"
#include "irr/core/Types.h"
#include "COpenGLTimestampQuery.h"
#include "COpenGLDriver.h"

namespace irr {
	namespace core {

		template<typename Key, typename Value,typename MapHash,typename MapEquals>
		class LRUcache {
			list<Key> stored_keys;
			unordered_map<Key, Value, MapHash,MapEquals> map;
			uint32_t cap;
			std::atomic_uint32_t timestamp = 0U;
		public:
			inline LRUcache() = default;
			inline LRUcache(uint32_t& capacity)
			{
				cap = capacity;
			}
			inline ~LRUcache();

			inline void insert(const Key &&k, Value &&v)
			{
				if (map.find(k) == map.end() && stored_keys.size() == cap) {
					auto last = stored_keys.back();
					stored_keys.pop_back();
					map.erase(last);
				}
				else
					stored_keys.erase(k);
				stored_keys.push_front(k);
				map[k] = v;

			}
			inline Value get(Key k)
			{
				if (map.find(k) == ma.end())
					return NULL;
				timestamp++;
				return map[k];
			}
			inline Value peek()
			{

			}
		};


	}	//namespace core
}		//namespace irr