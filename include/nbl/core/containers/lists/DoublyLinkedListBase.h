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
	inline virtual void popBack() override final
	{
		if (base_t::m_back == invalid_iterator)
			return;

		auto backNode = base_t::getBack();
		if (backNode->prev != invalid_iterator)
			base_t::get(backNode->prev)->next = invalid_iterator;
		uint32_t temp = base_t::m_back;
		base_t::m_back = backNode->prev;
		base_t::common_delete(temp);
	}

	//remove a node at nodeAddr from the list
	inline virtual void erase(const uint32_t nodeAddr) override final
	{
		assert(nodeAddr != invalid_iterator);
		assert(nodeAddr < base_t::m_cap);
		node_t* node = base_t::get(nodeAddr);

		if (base_t::m_back == nodeAddr)
			base_t::m_back = node->prev;
		if (base_t::m_begin == nodeAddr)
			base_t::m_begin = node->next;

		common_detach(node);
		base_t::common_delete(nodeAddr);
	}

	//move a node at nodeAddr to the front of the list
	inline virtual void moveToFront(const uint32_t nodeAddr) override final
	{
		if (base_t::m_begin == nodeAddr || nodeAddr == invalid_iterator)
			return;

		base_t::getBegin()->prev = nodeAddr;

		auto node = base_t::get(nodeAddr);

		if (base_t::m_back == nodeAddr)
			base_t::m_back = node->prev;

		common_detach(node);
		node->next = base_t::m_begin;
		node->prev = invalid_iterator;
		base_t::m_begin = nodeAddr;
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
	inline virtual void insertAt(uint32_t addr, value_t&& val) override final
	{
		assert(addr < base_t::m_cap);
		assert(addr != invalid_iterator);
		SDoublyLinkedNode<Value>* n = new(base_t::m_array + addr) SDoublyLinkedNode<Value>(std::move(val));
		n->prev = invalid_iterator;
		n->next = base_t::m_begin;

		if (base_t::m_begin != invalid_iterator)
			base_t::getBegin()->prev = addr;
		if (base_t::m_back == invalid_iterator)
			base_t::m_back = addr;
		base_t::m_begin = addr;
	}

	inline virtual void common_detach(node_t* node) override final
	{
		if (node->next != invalid_iterator)
			base_t::get(node->next)->prev = node->prev;
		if (node->prev != invalid_iterator)
			base_t::get(node->prev)->next = node->next;
	}
};

}
}




#endif