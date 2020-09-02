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
			Snode* prev;
			Snode* next;

			Snode(const Value& val) : data(val)
			{ }
			Snode(Snode&& other) : data(std::move(other.data)), prev(std::move(other.prev)), next(std::move(other.next))
			{ }

			Snode& operator=(const Snode& other)
			{
				this.data = other.data;
				this.prev = other.prev;
				this.next = other.next;
				return *this;
			}

			Snode& operator=(Snode&& other)
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
			typedef Snode<Value>* iteratorptr_t;

		private:
			PoolAddressAllocator<uint32_t>* alloc;
			iteratorptr_t p_begin;
			iteratorptr_t p_end;
			uint32_t cap;
			void* reservedSpace;

			inline void popBack()
			{
				if(p_end->prev != nullptr)
					p_end->prev->next = nullptr;
				iteratorptr_t temp = p_end;
				p_end = p_end->prev;
				alloc.free_addr(reinterpret_cast<uint32_t>(temp), 1u);
			}

		public:

			inline void pushFront(Value &val) 
			{
				uint32_t addr = alloc.alloc_addr(1u, 1u);
				iteratorptr_t n = new(addr) Snode(val);
				n->prev = nullptr;
				
				n->next = p_begin;

				if (p_begin != nullptr)
					p_begin->prev = n;
				if (p_end == nullptr)
					p_end = n;
				p_begin = n;
			}

			inline iteratorptr_t begin() { return p_begin; }

			inline iteratorptr_t end() { return p_end; }

			inline void erase(iteratorptr_t node)
			{
				if(node->prev != nullptr)
				node->prev->next = node->next;
				if (node->next != nullptr)
				node->next->prev = node->prev;
				delete node;
				alloc.free_addr(reinterpret_cast<uint32_t>(node), 1u);
			}

			inline void moveToFront(iteratorptr_t node)
			{
				if (p_begin == node) return;
				p_begin->prev = node;
				node->next = p_begin;
				node->prev = nullptr;
				p_begin = node;
			}

			inline DoublyLinkedList(const uint32_t& capacity)
			{
				reservedSpace = malloc(sizeof(Snode<Value>) * capacity);
				alloc = new PoolAddressAllocator<uint32_t>(reservedSpace, 0u, 0u, (capacity - 1u), 1u, 1u);
				p_begin = nullptr;
				p_end = nullptr;
			}
		};


	}


}


#endif // !__DOUBLY_LINKED_LIST_H_INCLUDED__
