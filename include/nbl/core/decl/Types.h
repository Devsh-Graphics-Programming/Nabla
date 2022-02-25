// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_TYPES_H_INCLUDED__
#define __NBL_CORE_TYPES_H_INCLUDED__

#include "nbl/core/decl/compile_config.h"

#include "stdint.h"
#include <wchar.h>

#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility>
#include <iterator>
#include <parallel-hashmap/parallel_hashmap/phmap.h>

#include "nbl/core/memory/new_delete.h"

#include "nbl/core/alloc/aligned_allocator.h"
#include "nbl/core/alloc/aligned_allocator_adaptor.h"

#include <mutex>

namespace nbl::core
{

template<typename Compared, typename T>
using add_const_if_const_t = std::conditional_t<std::is_const_v<Compared>,std::add_const_t<T>,T>;


template<typename T>
using allocator = _NBL_DEFAULT_ALLOCATOR_METATYPE<T>;



template<typename T>
using deque = std::deque<T,allocator<T> >;
template<typename T>
using forward_list = std::forward_list<T,allocator<T> >;
template<typename T>
using list = std::list<T,allocator<T> >;


template<typename K, typename T, class Compare=std::less<K>, class Allocator=allocator<std::pair<const K,T> > >
using map = std::map<K,T,Compare,Allocator>;
template<typename K, typename T, class Compare=std::less<K>, class Allocator=allocator<std::pair<const K,T> > >
using multimap = std::multimap<K,T,Compare,Allocator>;

template<typename K, class Compare=std::less<K>, class Allocator=allocator<K> >
using multiset = std::multiset<K,Compare,Allocator>;
template<typename K, class Compare=std::less<K>, class Allocator=allocator<K> >
using set = std::set<K,Compare,Allocator>;

template<typename K,typename T, class Hash=std::hash<K>, class KeyEqual=std::equal_to<K>, class Allocator=allocator<std::pair<const K,T> > >
using unordered_map = phmap::flat_hash_map<K,T,Hash,KeyEqual,Allocator>;

template<typename K,typename T, class Hash=std::hash<K>, class KeyEqual=std::equal_to<K>, class Allocator=allocator<std::pair<const K,T> > >
using unordered_multimap = std::unordered_multimap<K,T,Hash,KeyEqual,Allocator>;

template<typename K, class Hash=std::hash<K>, class KeyEqual=std::equal_to<K>, class Allocator=allocator<K> >
using unordered_multiset = std::unordered_multiset<K,Hash,KeyEqual,Allocator>;
template<typename K, class Hash=std::hash<K>, class KeyEqual=std::equal_to<K>, class Allocator=allocator<K> >
using unordered_set = phmap::flat_hash_set<K,Hash,KeyEqual,Allocator>;


template<typename T, class Allocator=allocator<T> >
using vector = std::vector<T,Allocator>;


template<typename T, class Container=vector<T>, class Compare=std::less<typename Container::value_type> >
using priority_queue = std::priority_queue<T,Container,Compare>;
template<typename T, class Container=deque<T> >
using queue = std::queue<T,Container>;
template<typename T, class Container=deque<T> >
using stack = std::stack<T,Container>;


typedef std::mutex  mutex;
// change to some derivation of FW_FastLock later
typedef std::mutex  fast_mutex;
}


// memory debugging
#if defined(_NBL_DEBUG) && defined(NABLA_EXPORTS) && defined(_MSC_VER) && \
	(_MSC_VER > 1299) && !defined(_NBL_DONT_DO_MEMORY_DEBUGGING_HERE) && !defined(_WIN32_WCE)

	#define CRTDBG_MAP_ALLOC
	#define _CRTDBG_MAP_ALLOC
	#define DEBUG_CLIENTBLOCK new( _CLIENT_BLOCK, __FILE__, __LINE__)
	#include <stdlib.h>
	#include <crtdbg.h>
	#define new DEBUG_CLIENTBLOCK
#endif

#endif
