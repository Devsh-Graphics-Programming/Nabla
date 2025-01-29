// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_FIXED_CAPACITY_DOUBLY_LINKED_LIST_H_INCLUDED__
#define __NBL_CORE_FIXED_CAPACITY_DOUBLY_LINKED_LIST_H_INCLUDED__


#include "nbl/core/containers/lists/DoublyLinkedListBase.h"

namespace nbl
{
namespace core
{

template<typename Value>
class FixedCapacityDoublyLinkedList : public DoublyLinkedListBase<Value>
{
public:
	using base_t = DoublyLinkedListBase<Value>;
	using disposal_func_t = typename base_t::disposal_func_t;

	//Constructor, capacity determines the amount of allocated space
	FixedCapacityDoublyLinkedList(const uint32_t capacity, disposal_func_t&& dispose_f = disposal_func_t()) : base_t(capacity, std::move(dispose_f))
	{}

	FixedCapacityDoublyLinkedList() = default;

	FixedCapacityDoublyLinkedList(const FixedCapacityDoublyLinkedList& other) = delete;

	FixedCapacityDoublyLinkedList& operator=(const FixedCapacityDoublyLinkedList& other) = delete;

	FixedCapacityDoublyLinkedList& operator=(FixedCapacityDoublyLinkedList&& other)
	{
		base_t::operator=(other);
	}

	~FixedCapacityDoublyLinkedList() = default;
};


}
}


#endif
