// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_FIXED_CAPACITY_DOUBLY_LINKED_LIST_H_INCLUDED__
#define __NBL_CORE_FIXED_CAPACITY_DOUBLY_LINKED_LIST_H_INCLUDED__


#include "nbl/core/alloc/PoolAddressAllocator.h"
#include "nbl/core/decl/Types.h"

#include <functional>

namespace nbl
{
namespace core
{

namespace impl
{
	class FixedCapacityDoublyLinkedListBase
	{
		public:
			_NBL_STATIC_INLINE_CONSTEXPR uint32_t invalid_iterator = PoolAddressAllocator<uint32_t>::invalid_address;
		protected:

			FixedCapacityDoublyLinkedListBase() = default;

			template<typename T>
			FixedCapacityDoublyLinkedListBase(const uint32_t capacity, void*& _reservedSpace, T*& _array)
			{
				const auto firstPart = core::alignUp(PoolAddressAllocator<uint32_t>::reserved_size(1u,capacity,1u),alignof(T));
				_reservedSpace = _NBL_ALIGNED_MALLOC(firstPart+capacity*sizeof(T),alignof(T));
				_array = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(_reservedSpace)+firstPart);
			}
	};
}

//Struct for use in a doubly linked list. Stores data and pointers to next and previous elements the list, or invalid iterator if it is first/last
template<typename Value>
struct alignas(void*) SDoublyLinkedNode
{
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t invalid_iterator = impl::FixedCapacityDoublyLinkedListBase::invalid_iterator;

	Value data;
	uint32_t prev;
	uint32_t next;

	SDoublyLinkedNode() {}
	SDoublyLinkedNode(const Value& val) : data(val)
	{
		prev = invalid_iterator;
		next = invalid_iterator;
	}
	SDoublyLinkedNode(Value&& val) : data(std::move(val))
	{
		prev = invalid_iterator;
		next = invalid_iterator;
	}
	SDoublyLinkedNode(SDoublyLinkedNode<Value>&& other) : data(std::move(other.data)), prev(std::move(other.prev)), next(std::move(other.next))
	{}
	
	~SDoublyLinkedNode()
	{}

	SDoublyLinkedNode<Value>& operator=(const SDoublyLinkedNode<Value>& other)
	{
		this->data = other.data;
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

template<typename Value>
class FixedCapacityDoublyLinkedList : private impl::FixedCapacityDoublyLinkedListBase
{
	public:
		using AddressAllocator = PoolAddressAllocator<uint32_t>;
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t invalid_iterator = impl::FixedCapacityDoublyLinkedListBase::invalid_iterator;

		using disposal_func_t = std::function<void(Value&)>;

		using node_t = SDoublyLinkedNode<Value>;

		// get the fixed capacity
		inline uint32_t getCapacity() const { return cap; }

		//get node at iterator
		inline node_t* get(const uint32_t address)
		{
			return (m_array + address);
		}
		inline const node_t* get(const uint32_t address) const
		{
			return (m_array + address);
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
			common_delete(temp);
		}

		//add new item to the list. This function does not make space to store the new node. in case the list is full, popBack() needs to be called beforehand
		inline void pushFront(Value&& val)
		{
			insertAt(reserveAddress(), std::move(val));
		}

		template <typename... Args>
		inline void emplaceFront(Args&&... args)
		{
			insertAt(reserveAddress(), Value(std::forward<Args>(args)...));
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

		//remove a node at nodeAddr from the list
		inline void erase(const uint32_t nodeAddr)
		{
			assert(nodeAddr != invalid_iterator);
			assert(nodeAddr < cap);
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

		//Constructor, capacity determines the amount of allocated space
		FixedCapacityDoublyLinkedList(const uint32_t capacity, disposal_func_t&& dispose_f = disposal_func_t()) :
			FixedCapacityDoublyLinkedListBase(capacity,m_reservedSpace,m_array),
			m_dispose_f(std::move(dispose_f))
		{
			addressAllocator = std::unique_ptr<AddressAllocator>(new AddressAllocator(m_reservedSpace, 0u, 0u, 1u, capacity, 1u));
			cap = capacity;
			m_back = invalid_iterator;
			m_begin = invalid_iterator;
		}
		
		FixedCapacityDoublyLinkedList() = default;

		FixedCapacityDoublyLinkedList(const FixedCapacityDoublyLinkedList& other) = delete;

		FixedCapacityDoublyLinkedList& operator=(const FixedCapacityDoublyLinkedList& other) = delete;
		
		FixedCapacityDoublyLinkedList& operator=(FixedCapacityDoublyLinkedList&& other)
		{
			addressAllocator = std::move(other.addressAllocator);
			m_reservedSpace = std::move(other.m_reservedSpace);
			m_array = std::move(other.m_array);
			m_dispose_f = std::move(other.m_dispose_f);
			cap = other.cap;
			m_back = other.m_back;
			m_begin = other.m_begin;

			// Nullify other
			other.addressAllocator = nullptr;
			other.m_reservedSpace = nullptr;
			other.m_array = nullptr;
			other.cap = 0u;
			other.m_back = 0u;
			other.m_begin = 0u;
			return *this;
		}

		~FixedCapacityDoublyLinkedList()
		{
			if (m_dispose_f && m_begin != invalid_iterator)
			{
				auto* begin = getBegin();
				auto* back = getBack();
				do 
				{
					m_dispose_f(begin->data);
					begin = get(begin->next);
				} while (begin != back);
			}
			_NBL_ALIGNED_FREE(m_reservedSpace);
		}
		

	private:
		std::unique_ptr<AddressAllocator> addressAllocator;
		void* m_reservedSpace;
		node_t* m_array;

		uint32_t cap;
		uint32_t m_back;
		uint32_t m_begin;

		disposal_func_t m_dispose_f;
		
		//allocate and get the address of the next free node
		inline uint32_t reserveAddress()
		{
			uint32_t addr = addressAllocator->alloc_addr(1u, 1u);
			return addr;
		}

		//create a new node which stores data at already allocated address, 
		inline void insertAt(uint32_t addr, Value&& val)
		{
			assert(addr < cap);
			assert(addr != invalid_iterator);
			SDoublyLinkedNode<Value>* n = new(m_array + addr) SDoublyLinkedNode<Value>(std::move(val));
			n->prev = invalid_iterator;
			n->next = m_begin;

			if (m_begin != invalid_iterator)
				getBegin()->prev = addr;
			if (m_back == invalid_iterator)
				m_back = addr;
			m_begin = addr;
		}

		inline void common_delete(uint32_t address)
		{
			if (m_dispose_f)
				m_dispose_f(get(address)->data);
			get(address)->~node_t();
			addressAllocator->free_addr(address, 1u);
		}

		inline void common_detach(node_t* node)
		{
			if (node->next != invalid_iterator)
				get(node->next)->prev = node->prev;
			if (node->prev != invalid_iterator)
				get(node->prev)->next = node->next;
		}
};


}
}


#endif
