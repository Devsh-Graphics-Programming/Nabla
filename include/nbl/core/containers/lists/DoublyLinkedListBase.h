#ifndef __NBL_CORE_CONTAINERS_LISTS_DOUBLY_LINKED_LIST_BASE_H_INCLUDED__
#define __NBL_CORE_CONTAINERS_LISTS_DOUBLY_LINKED_LIST_BASE_H_INCLUDED__

#include "nbl/core/containers/lists/common.h"

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
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t invalid_iterator = impl::ContiguousMemoryLinkedListBase<this_t>::invalid_iterator;

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
	{
	}

	~SDoublyLinkedNode()
	{
	}

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

template <typename Value>
class DoublyLinkedListBase : public impl::ContiguousMemoryLinkedListBase<SDoublyLinkedNode<Value> >
{
public:
	using base_t = impl::ContiguousMemoryLinkedListBase<SDoublyLinkedNode<Value> >;
	using node_t = typename base_t::node_t;
	using value_t = Value;
	using disposal_func_t = typename base_t::disposal_func_t;
	using address_allocator_t = typename base_t::address_allocator_t;

	_NBL_STATIC_INLINE_CONSTEXPR uint32_t invalid_iterator = base_t::invalid_iterator;

	//remove the last element in the list
	inline virtual void popBack() final
	{
		if (this->m_back == invalid_iterator)
			return;

		auto backNode = this->getBack();
		if (backNode->prev != invalid_iterator)
			this->get(backNode->prev)->next = invalid_iterator;
		uint32_t temp = this->m_back;
		this->m_back = backNode->prev;
		this->common_delete(temp);
	}

	//remove a node at nodeAddr from the list
	inline virtual void erase(const uint32_t nodeAddr) final
	{
		assert(nodeAddr != invalid_iterator);
		assert(nodeAddr < this->m_cap);
		node_t* node = this->get(nodeAddr);

		if (this->m_back == nodeAddr)
			this->m_back = node->prev;
		if (this->m_begin == nodeAddr)
			this->m_begin = node->next;

		common_detach(node);
		this->common_delete(nodeAddr);
	}

	//move a node at nodeAddr to the front of the list
	inline virtual void moveToFront(const uint32_t nodeAddr) final
	{
		if (this->m_begin == nodeAddr || nodeAddr == invalid_iterator)
			return;

		this->getBegin()->prev = nodeAddr;

		auto node = this->get(nodeAddr);

		if (this->m_back == nodeAddr)
			this->m_back = node->prev;

		common_detach(node);
		node->next = this->m_begin;
		node->prev = invalid_iterator;
		this->m_begin = nodeAddr;
	}

	//Constructor, capacity determines the amount of allocated space
	DoublyLinkedListBase(const uint32_t capacity, disposal_func_t&& dispose_f = disposal_func_t()) : base_t(capacity, std::move(dispose_f))
	{
	}

	DoublyLinkedListBase() = default;

	DoublyLinkedListBase(const DoublyLinkedListBase& other) = delete;

	DoublyLinkedListBase& operator=(const DoublyLinkedListBase& other) = delete;

	DoublyLinkedListBase& operator=(DoublyLinkedListBase&& other)
	{
		base_t::operator=(other);
	}

	~DoublyLinkedListBase() = default;

private:
	//create a new node which stores data at already allocated address, 
	inline virtual void insertAt(uint32_t addr, value_t&& val) final
	{
		assert(addr < this->m_cap);
		assert(addr != invalid_iterator);
		SDoublyLinkedNode<Value>* n = new(this->m_array + addr) SDoublyLinkedNode<Value>(std::move(val));
		n->prev = invalid_iterator;
		n->next = this->m_begin;

		if (this->m_begin != invalid_iterator)
			this->getBegin()->prev = addr;
		if (this->m_back == invalid_iterator)
			this->m_back = addr;
		this->m_begin = addr;
	}

	inline virtual void common_detach(node_t* node) final
	{
		if (node->next != invalid_iterator)
			this->get(node->next)->prev = node->prev;
		if (node->prev != invalid_iterator)
			this->get(node->prev)->next = node->next;
	}
};

}
}




#endif