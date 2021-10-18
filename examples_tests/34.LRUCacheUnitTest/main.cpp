#define _IRR_STATIC_LIB_
#include <nabla.h>
#include "nbl/core/containers/LRUcache.h"

using namespace nbl;
using namespace nbl::core;

int main()
{
	LRUCache<int, char> hugeCache(50000000u);

	LRUCache<int, char> cache(5u);

	//const, const
	cache.insert(10, 'c');
	cache.insert(11, 'c');
	cache.insert(12, 'c');
	cache.insert(13, 'c');

	char returned = *(cache.get(11));
	assert(returned == 'c');

#ifdef _NBL_DEBUG
	cache.print();
#endif
	//non const, const
	int i = 0;
	cache.insert(++i, '1');
	cache.insert(++i, '2');
	cache.insert(++i, '3');

	returned = *(cache.get(1));
	assert(returned == '1');

	//const, non const
	char ch = 'T';
	cache.insert(4, ch);
	cache.insert(5, ch);

	returned = *(cache.get(4));
	assert(returned == 'T');

	//non const, non const
	i = 6;
	ch = 'Y';
	cache.insert(i, ch);

	returned = *(cache.get(6));
	assert(returned == 'Y');

	returned = *(cache.get(i));
	assert(returned == ch);

	cache.erase(520);
	cache.erase(5);

	auto returnedNullptr = cache.get(5);
	assert(returnedNullptr == nullptr);
	auto peekedNullptr = cache.peek(5);
	assert(peekedNullptr == nullptr);



	core::LRUCache<int, std::string> cache2(5u);

	cache2.insert(500, "five hundred");			//inserts at addr = 0
	cache2.insert(510, "five hundred and ten");	//inserts at addr = 472
	cache2.insert(52, "fifty two");
	i = 20;
	cache2.insert(++i, "key is 21");
	cache2.insert(++i, "key is 22");
	cache2.insert(++i, "key is 23");
	i = 111;
	//cache2.print();


	return 0;
}
