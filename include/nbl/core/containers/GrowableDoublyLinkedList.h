// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_GROWABLE_DOUBLY_LINKED_LIST_H_INCLUDED__
#define __NBL_CORE_GROWABLE_DOUBLY_LINKED_LIST_H_INCLUDED__


#include "nbl/core/containers/lists/DoublyLinkedListBase.h"

namespace nbl
{
namespace core
{

template<typename Value>
class GrowableDoublyLinkedList : public DoublyLinkedListBase<Value>
{
public:
	using base_t = DoublyLinkedListBase<Value>;
	using value_t = Value;
	using node_t = typename base_t::node_t;
	using address_allocator_t = typename base_t::address_allocator_t;
	using disposal_func_t = typename base_t::disposal_func_t;

	//Constructor, capacity determines the amount of allocated space
	GrowableDoublyLinkedList(const uint32_t capacity, disposal_func_t&& dispose_f = disposal_func_t()) : base_t(capacity, std::move(dispose_f))
	{}

	GrowableDoublyLinkedList() = default;

	GrowableDoublyLinkedList(const GrowableDoublyLinkedList& other) = delete;

	GrowableDoublyLinkedList& operator=(const GrowableDoublyLinkedList& other) = delete;

	GrowableDoublyLinkedList& operator=(GrowableDoublyLinkedList&& other)
	{
		base_t::operator=(other);
	}

	~GrowableDoublyLinkedList() = default;

	inline void grow(uint32_t newCapacity)
	{
		const auto firstPart = core::alignUp(address_allocator_t::reserved_size(1u, newCapacity, 1u), alignof(node_t));
		void* newReservedSpace = _NBL_ALIGNED_MALLOC(firstPart + newCapacity * sizeof(node_t), alignof(node_t));
		node_t* newArray = reinterpret_cast<node_t*>(reinterpret_cast<uint8_t*>(newReservedSpace) + firstPart);

		memcpy(reinterpret_cast<void*>(newArray), reinterpret_cast<void*>(this->m_array), m_cap * sizeof(node_t));

		//this->m_addressAllocator = std::unique_ptr<address_allocator_t>(new address_allocator_t(newCapacity, std::move(this->m_addressAllocator)));
		this->m_cap = capacity;
		this->m_back = invalid_iterator;
		this->m_begin = invalid_iterator;
	}
};


}
}


#endif
