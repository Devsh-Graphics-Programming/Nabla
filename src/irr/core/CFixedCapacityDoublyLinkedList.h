#ifndef __DOUBLY_LINKED_LIST_H_INCLUDED__
#define __DOUBLY_LINKED_LIST_H_INCLUDED__
#include "irr/core/alloc/PoolAddressAllocator.h"
#include "irr/core/Types.h"

namespace irr {
	namespace core {

		template<typename Value>
		struct Snode
		{
			Value data;
			uint32_t prev;
			uint32_t next;

			Snode(const Value& val) : data(val)
			{ }
			Snode(Snode<Value>&& other) : data(std::move(other.data)), prev(std::move(other.prev)), next(std::move(other.next))
			{ }

			Snode<Value>& operator=(const Snode<Value>& other)
			{
				this.data = other.data;
				this.prev = other.prev;
				this.next = other.next;
				return *this;
			}

			Snode<Value>& operator=(Snode<Value>&& other)
			{
				this.data = std::move(other.data);
				this.prev = std::move(other.prev);
				this.next = std::move(other.next);
				return *this;
			}
		};

		template<typename Value>
		class DoublyLinkedList
		{
			typedef uint32_t node_address;
			
		private:
			void* reservedSpace;
			PoolAddressAllocator<uint32_t> alloc;
			node_address m_back;
			node_address m_begin;
			Snode<Value>* m_array;
			uint32_t cap;

			#define invalid_iterator alloc.invalid_address	//0xdeadbeef

		public:

			inline void popBack()
			{
				if(m_back->prev != invalid_iterator)
					m_back->prev->next = invalid_iterator;
				uint32_t temp = m_end;
				m_end = m_back->prev;
				alloc.free_addr(temp, 1u);
			}

			inline void pushFront(Value &val) 
			{
				uint32_t addr = alloc.alloc_addr(1u, 1u);
				Snode<Value>* n = (reinterpret_cast<Snode<Value>*>(reservedSpace) + addr);
				n->prev = invalid_iterator;
				n->data = val;
				n->next = m_begin;

				m_array[addr] = *n;
				if (m_begin != invalid_iterator)
					m_begin->prev = n;
				m_begin = n;
			}

			inline node_address getBegin() { return m_begin; }

			inline node_address getBack() { return m_back; }

			inline void erase(uint32_t& nodeAddr)
			{
				if(nodeAddr->prev != invalid_iterator)
				nodeAddr->prev->next = nodeAddr->next;
				if (nodeAddr->next != invalid_iterator)
				nodeAddr->next->prev = nodeAddr->prev;
				nodeAddr->~Snode<Value>();
				alloc.free_addr(nodeAddr, 1u);
			}

			inline void moveToFront(uint32_t& node)
			{
				if (m_begin == node) return;
				m_begin->prev = node;
				if (node->next != invalid_iterator)
					node->next->prev = node->prev;
				if (node->prev != invalid_iterator)
					node->next->prev = node->prev;
				node->next = m_begin;
				node->prev = invalid_iterator;
				m_begin = node;
			}

			DoublyLinkedList(const uint32_t& capacity) :
				alloc(reservedSpace, 0u, 0u, 1u, capacity, 1u),
				reservedSpace(_IRR_ALIGNED_MALLOC(PoolAddressAllocator<uint32_t>::reserved_size(1u, capacity, 1u), alignof(void*))),
				cap(capacity)
			{
				m_back = invalid_iterator;
				m_begin = invalid_iterator;
				m_array = new (reservedSpace) Snode<Value>;
			}
			~DoublyLinkedList()
			{
				_IRR_ALIGNED_FREE(reservedSpace);
			}
			#undef invalid_iterator
		};


	}


}


#endif // !__DOUBLY_LINKED_LIST_H_INCLUDED__
