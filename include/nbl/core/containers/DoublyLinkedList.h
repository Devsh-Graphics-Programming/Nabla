// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_CONTAINERS_DOUBLY_LINKED_LIST_H_INCLUDED__
#define __NBL_CORE_CONTAINERS_DOUBLY_LINKED_LIST_H_INCLUDED__


#include "nbl/core/alloc/PoolAddressAllocator.h"
#include "nbl/core/decl/Types.h"

#include <functional>

namespace nbl
{
namespace core
{

//Struct for use in a doubly linked list. Stores data and pointers to next and previous elements the list, or invalid iterator if it is first/last
template<typename Value>
struct alignas(void*) SDoublyLinkedNode
{
	using this_t = SDoublyLinkedNode<Value>;
	using value_t = Value;
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t invalid_iterator = PoolAddressAllocator<uint32_t>::invalid_address;

	Value data;
	uint32_t prev;
	uint32_t next;

	SDoublyLinkedNode() {}
	SDoublyLinkedNode(const Value& val) : data(val)
	{
		prev = invalid_iterator;
		next = invalid_iterator;
	}
	template <typename... Args>
	SDoublyLinkedNode(Args&&... args) : data(std::forward<Args>(args)...)
	{
		prev = invalid_iterator;
		next = invalid_iterator;
	}
	SDoublyLinkedNode(SDoublyLinkedNode<Value>&& other) : data(std::move(other.data)), prev(std::move(other.prev)), next(std::move(other.next))
	{
	}

	~SDoublyLinkedNode()
	{
	}

	SDoublyLinkedNode<Value>& operator=(const SDoublyLinkedNode<Value>& other)
	{
		data = other.data;
		this->prev = other.prev;
		this->next = other.next;
		return *this;
	}

	SDoublyLinkedNode<Value>& operator=(SDoublyLinkedNode<Value>&& other)
	{
		this->data = std::move(other.data);
		this->prev = std::move(other.prev);
		this->next = std::move(other.next);
		return *this;
	}
};

template<typename Value, class allocator = core::allocator<SDoublyLinkedNode<Value>> >
class DoublyLinkedList
{
public:
	template <bool Mutable>
	class Iterator;
	template <bool Mutable>
	friend class Iterator;

	using iterator = Iterator<true>;
	using const_iterator = Iterator<false>;

	using allocator_t = allocator;
	using allocator_traits_t = std::allocator_traits<allocator_t>;
	using address_allocator_t = PoolAddressAllocator<uint32_t>;
	using node_t = SDoublyLinkedNode<Value>;
	using value_t = Value;
	using disposal_func_t = std::function<void(value_t&)>;

	_NBL_STATIC_INLINE_CONSTEXPR uint32_t invalid_iterator = node_t::invalid_iterator;

	// get the fixed capacity
	inline uint32_t getCapacity() const { return m_cap; }

	//get node at iterator
	inline node_t* get(const uint32_t address)
	{
		return (m_array + address);
	}
	inline const node_t* get(const uint32_t address) const
	{
		return (m_array + address);
	}

	//get node ptr of the first item in the list
	inline node_t* getBegin() { return m_array + m_begin; }
	inline const node_t* getBegin() const { return m_array + m_begin; }

	//get node ptr of the last item in the list
	inline node_t* getBack() { return m_array + m_back; }
	inline const node_t* getBack() const { return m_array + m_back; }

	//get index/iterator of the first element
	inline uint32_t getFirstAddress() const { return m_begin; }

	//get index/iterator of the last element
	inline uint32_t getLastAddress() const { return m_back; }

	//add new item to the list. This function does not make space to store the new node. in case the list is full, popBack() needs to be called beforehand
	inline void pushFront(value_t&& val)
	{
		insertAt(reserveAddress(), std::move(val));
	}

	template <typename... Args>
	inline void emplaceFront(Args&&... args)
	{
		insertAt(reserveAddress(), std::forward<Args>(args)...);
	}

	/**
	* @brief Empties list: calls disposal function on and destroys every node in list, then resets state
	*/
	inline void clear()
	{
		destroyAll();

		m_addressAllocator = address_allocator_t(m_reservedSpace, 0u, 0u, 1u, m_cap, 1u);
		m_back = invalid_iterator;
		m_begin = invalid_iterator;

	}

	//remove the last element in the list
	inline void popBack()
	{
		if (m_back == invalid_iterator)
			return;

		auto backNode = getBack();
		if (backNode->prev != invalid_iterator)
			get(backNode->prev)->next = invalid_iterator;
		uint32_t temp = m_back;
		m_back = backNode->prev;
		if (m_back == invalid_iterator)
			m_begin = invalid_iterator;
		common_delete(temp);
	}

	//remove a node at nodeAddr from the list
	inline void erase(const uint32_t nodeAddr)
	{
		assert(nodeAddr != invalid_iterator);
		assert(nodeAddr < m_cap);
		node_t* node = get(nodeAddr);

		if (m_back == nodeAddr)
			m_back = node->prev;
		if (m_begin == nodeAddr)
			m_begin = node->next;

		common_detach(node);
		common_delete(nodeAddr);
	}

	//move a node at nodeAddr to the front of the list
	inline void moveToFront(const uint32_t nodeAddr)
	{
		if (m_begin == nodeAddr || nodeAddr == invalid_iterator)
			return;

		getBegin()->prev = nodeAddr;

		auto node = get(nodeAddr);

		if (m_back == nodeAddr)
			m_back = node->prev;

		common_detach(node);
		node->next = m_begin;
		node->prev = invalid_iterator;
		m_begin = nodeAddr;
	}

	/**
	* @brief Resizes the list by extending its capacity so it can hold more elements. Returns a bool indicating if capacity was indeed increased.
	*
	* @param [in] newCapacity New number of elements to hold. MUST be greater than current list capacity.
	*/
	inline bool grow(uint32_t newCapacity)
	{
		// Must at least make list grow
		if (newCapacity <= m_cap)
			return false;
		// Have to consider allocating enough space for list AND state of the address allocator
		// Allocator can only allocate in terms of nodes, so we do `addressAllocatorStorageNodes = ceil(reserved_size / sizeof(node_t))`.
		// This means that the storage for the address allocator fits in `addressAllocatorStorageNodes * sizeof(node_t)` bytes of memory
		// All `Size`s given in terms of nodes
		const size_t addressAllocatorStorageSize = (address_allocator_t::reserved_size(1u, newCapacity, 1u) + sizeof(node_t) - 1) / sizeof(node_t);
		const size_t newAllocationSize = addressAllocatorStorageSize + newCapacity;
		void* newReservedSpace = reinterpret_cast<void*>(allocator_traits_t::allocate(m_allocator, newAllocationSize));

		// Allocation failed, not possible to grow
		if (!newReservedSpace)
			return false;

		// Offset the array start by the storage used by the address allocator
		node_t* newArray = reinterpret_cast<node_t*>(reinterpret_cast<uint8_t*>(newReservedSpace) + addressAllocatorStorageSize * sizeof(node_t));
		// Copy memory over to new buffer
		memcpy(newArray, m_array, m_cap * sizeof(node_t));
		// Create new address allocator from old one
		m_addressAllocator = address_allocator_t(newCapacity, std::move(m_addressAllocator), newReservedSpace);
		// After address allocator creation we can free the old buffer
		allocator_traits_t::deallocate(m_allocator, reinterpret_cast<node_t*>(m_reservedSpace), m_currentAllocationSize);
		m_cap = newCapacity;
		m_array = newArray;
		m_reservedSpace = newReservedSpace;
		m_currentAllocationSize = newAllocationSize;
		
		return true;
	}

	//Constructor, capacity determines the amount of allocated space
	DoublyLinkedList(const uint32_t capacity, disposal_func_t&& dispose_f = disposal_func_t(), const allocator_t& _allocator = allocator_t()) 
		: m_dispose_f(std::move(dispose_f)), m_allocator(_allocator)
	{
		// Have to consider allocating enough space for list AND state of the address allocator
		// Allocator can only allocate in terms of nodes, so we do `addressAllocatorStorageNodes = ceil(reserved_size / sizeof(node_t))`.
		// This means that the storage for the address allocator fits in `addressAllocatorStorageNodes * sizeof(node_t)` bytes of memory
		// All `Size`s given in terms of nodes
		const size_t addressAllocatorStorageSize = (address_allocator_t::reserved_size(1u, capacity, 1u) + sizeof(node_t) - 1) / sizeof(node_t);
		m_currentAllocationSize = addressAllocatorStorageSize + capacity;
		m_reservedSpace = reinterpret_cast<void*>(allocator_traits_t::allocate(m_allocator, m_currentAllocationSize));
		// Offset the array start by the storage used by the address allocator
		m_array = reinterpret_cast<node_t*>(reinterpret_cast<uint8_t*>(m_reservedSpace) + addressAllocatorStorageSize * sizeof(node_t));

		// If allocation failed, create list with no capacity to indicate creation failed
		m_cap = m_reservedSpace ? capacity : 0;
		m_addressAllocator = address_allocator_t(m_reservedSpace, 0u, 0u, 1u, m_cap, 1u);
	}

	DoublyLinkedList() = default;

	// Copy Constructor
	explicit DoublyLinkedList(const DoublyLinkedList& other) : m_dispose_f(other.m_dispose_f), m_allocator(other.m_allocator)
	{
		const size_t addressAllocatorStorageSize = (address_allocator_t::reserved_size(1u, other.m_cap, 1u) + sizeof(node_t) - 1) / sizeof(node_t);
		m_currentAllocationSize = addressAllocatorStorageSize + other.m_cap;
		m_reservedSpace = reinterpret_cast<void*>(allocator_traits_t::allocate(m_allocator, m_currentAllocationSize));
		// If allocation failed, create a list with no capacity
		m_cap = m_reservedSpace ? other.m_cap : 0;
		if (!m_cap) return; // Allocation failed
		// Offset the array start by the storage used by the address allocator
		m_array = reinterpret_cast<node_t*>(reinterpret_cast<uint8_t*>(m_reservedSpace) + addressAllocatorStorageSize * sizeof(node_t));

		if constexpr (std::is_trivially_copyable_v<Value>)
		{
			// Create new address allocator by copying state
			m_addressAllocator = address_allocator_t(m_cap, other.m_addressAllocator, m_reservedSpace);
			// Copy memory over
			memcpy(m_array, other.m_array, m_cap * sizeof(node_t));
			m_back = other.m_back;
			m_begin = other.m_begin;
		}
		else
		{
			m_addressAllocator = address_allocator_t(m_reservedSpace, 0u, 0u, 1u, m_cap, 1u);
			// Reverse iteration since we push from the front
			for (auto it = other.crbegin(); it != other.crend(); it++)
				pushFront(value_t(*it));

		}
	}

	DoublyLinkedList& operator=(const DoublyLinkedList& other) = delete;

	DoublyLinkedList& operator=(DoublyLinkedList&& other)
	{
		m_allocator = std::move(other.m_allocator);
		m_addressAllocator = std::move(other.m_addressAllocator);
		m_reservedSpace = std::move(other.m_reservedSpace);
		m_array = std::move(other.m_array);
		m_dispose_f = std::move(std::move(other.m_dispose_f));
		m_cap = std::move(other.m_cap);
		
		m_begin = other.m_begin;
		m_back = other.m_back;
		other.m_begin = invalid_iterator;
		other.m_back = invalid_iterator;

		return *this;
	}

	~DoublyLinkedList()
	{
		destroyAll();
		// Could be null if list was moved out of
		if (m_reservedSpace)
		{
			allocator_traits_t::deallocate(m_allocator, reinterpret_cast<node_t*>(m_reservedSpace), m_currentAllocationSize);
		}
	}

	// Iterator stuff
	iterator begin();
	iterator end();
	const_iterator cbegin() const;
	const_iterator cend() const;
	std::reverse_iterator<iterator> rbegin();
	std::reverse_iterator<iterator> rend();
	std::reverse_iterator<const_iterator> crbegin() const;
	std::reverse_iterator<const_iterator> crend() const;

private:
	//allocate and get the address of the next free node
	inline uint32_t reserveAddress()
	{
		uint32_t addr = m_addressAllocator.alloc_addr(1u, 1u);
		return addr;
	}

	//create a new node which stores data at already allocated address
	template <typename... Args>
	inline void insertAt(uint32_t addr, Args&&... args)
	{
		assert(addr < m_cap);
		assert(addr != invalid_iterator);
		node_t* n = m_array + addr;
		allocator_traits_t::construct(m_allocator, n, std::forward<Args>(args)...);
		n->prev = invalid_iterator;
		n->next = m_begin;

		if (m_begin != invalid_iterator)
			getBegin()->prev = addr;
		if (m_back == invalid_iterator)
			m_back = addr;
		m_begin = addr;
	}

	/**
	* @brief Calls disposal function then destroys all elements in the list.
	*/
	inline void destroyAll()
	{
		uint32_t currentAddress = m_begin;
		while (currentAddress != invalid_iterator)
		{
			node_t* currentNode = get(currentAddress);
			uint32_t nextAddress = currentNode->next;
			if (m_dispose_f) m_dispose_f(currentNode->data);
			allocator_traits_t::destroy(m_allocator, currentNode);
			currentAddress = nextAddress;
		}
	}

	inline void common_delete(uint32_t address)
	{
		if (m_dispose_f)
			m_dispose_f(get(address)->data);
		allocator_traits_t::destroy(m_allocator, get(address));
		m_addressAllocator.free_addr(address, 1u);
	}

	inline void common_detach(node_t* node)
	{
		if (node->next != invalid_iterator)
			get(node->next)->prev = node->prev;
		if (node->prev != invalid_iterator)
			get(node->prev)->next = node->next;
	}

	allocator_t m_allocator;
	address_allocator_t m_addressAllocator;
	void* m_reservedSpace;
	// In term of nodes
	size_t m_currentAllocationSize;
	node_t* m_array;

	uint32_t m_cap;
	uint32_t m_back = invalid_iterator;
	uint32_t m_begin = invalid_iterator;
	disposal_func_t m_dispose_f;
};

// ---------------------------------------------------- ITERATOR -----------------------------------------------------------

// Satifies std::bidirectional_iterator
template<typename Value, class allocator>
template<bool Mutable>
class DoublyLinkedList<Value, allocator>::Iterator
{
	using base_iterable_t = DoublyLinkedList<Value, allocator>;
	using iterable_t = std::conditional_t<Mutable, base_iterable_t, const base_iterable_t>;
	friend class base_iterable_t;
public:
	using value_type = std::conditional_t<Mutable, Value, const Value>;
	using pointer = value_type*;
	using reference = value_type&;
	using difference_type = int32_t;

	Iterator() = default;

	// Prefix 
	Iterator& operator++()
	{
		m_current = m_iterable->get(m_current)->next;
		return *this;
	}

	Iterator& operator--()
	{
		m_current = m_current != invalid_iterator ? m_iterable->get(m_current)->prev : m_iterable->m_back;
		return *this;
	}

	// Postfix
	Iterator operator++(int)
	{
		Iterator beforeIncrement = *this;
		operator++();
		return beforeIncrement;
	}

	Iterator operator--(int)
	{
		Iterator beforeDecrement = *this;
		operator--();
		return beforeDecrement;
	}

	// Comparison
	bool operator==(const Iterator& rhs) const
	{
		return m_iterable == rhs.m_iterable && m_current == rhs.m_current;
	}

	//Deref
	reference operator*() const
	{
		return m_iterable->get(m_current)->data;
	}

	pointer operator->() const
	{
		return & operator*();
	}
private:
	Iterator(iterable_t* const iterable, uint32_t idx) : m_iterable(iterable), m_current(idx) {}

	iterable_t* m_iterable;
	uint32_t m_current;
};

template<typename Value, class allocator>
DoublyLinkedList<Value, allocator>::iterator DoublyLinkedList<Value, allocator>::begin()
{
	return iterator(this, m_begin);
}

template<typename Value, class allocator>
DoublyLinkedList<Value, allocator>::const_iterator DoublyLinkedList<Value, allocator>::cbegin() const
{
	return const_iterator(this, m_begin);
}

template<typename Value, class allocator>
DoublyLinkedList<Value, allocator>::iterator DoublyLinkedList<Value, allocator>::end()
{
	return iterator(this, invalid_iterator);
}

template<typename Value, class allocator>
DoublyLinkedList<Value, allocator>::const_iterator DoublyLinkedList<Value, allocator>::cend() const
{
	return const_iterator(this, invalid_iterator);
}

template<typename Value, class allocator>
std::reverse_iterator<typename DoublyLinkedList<Value, allocator>::iterator> DoublyLinkedList<Value, allocator>::rbegin()
{
	return std::reverse_iterator<iterator>(iterator(this, invalid_iterator));
}

template<typename Value, class allocator>
std::reverse_iterator<typename DoublyLinkedList<Value, allocator>::const_iterator> DoublyLinkedList<Value, allocator>::crbegin() const
{
	return std::reverse_iterator<const_iterator>(const_iterator(this, invalid_iterator));
}

template<typename Value, class allocator>
std::reverse_iterator<typename DoublyLinkedList<Value, allocator>::iterator> DoublyLinkedList<Value, allocator>::rend()
{
	return std::reverse_iterator<iterator>(iterator(this, m_begin));
}

template<typename Value, class allocator>
std::reverse_iterator<typename DoublyLinkedList<Value, allocator>::const_iterator> DoublyLinkedList<Value, allocator>::crend() const
{
	return std::reverse_iterator<const_iterator>(const_iterator(this, m_begin));
}

} //namespace core
} //namespace nbl


#endif
