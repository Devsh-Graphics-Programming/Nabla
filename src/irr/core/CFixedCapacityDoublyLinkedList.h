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

			inline Snode(const Value& v) : data(v)
			{
			}
		};

		template<typename Value>
		class DoublyLinkedList
		{
		public:


		private:
			PoolAddressAllocator<uint32_t> alloc;
			Snode* p_begin;
			Snode* p_end;
			uint32_t cap;

			inline void popBack()
			{
				if(p_end->prev != nullptr)
					p_end->prev->next = nullptr;
				Snode* temp = p_end;
				p_end = p_end->prev;
				alloc.free_addr(reinterpret_cast<uint32_t>(temp), sizeof(Snode));
			}

		public:

			inline void pushFront(Value &val) 
			{
				uint32_t addr = alloc.alloc_addr(1u, 1u);
				Snode* n = new(addr) Snode(val);
				n->prev = nullptr;
				
				n->next = p_begin;

				if (p_begin != nullptr)
					p_begin->prev = n;
				if (p_end == nullptr)
					p_end = n;
				p_begin = n;
			}

			inline Snode* begin() { return p_begin; }

			inline Snode* end() { return p_end; }

			inline void erase(Snode* node)
			{
				if(node->prev != nullptr)
				node->prev->next = node->next;
				if (node->next != nullptr)
				node->next->prev = node->prev;
				alloc.free_addr(reinterpret_cast<uint32_t>(node), sizeof(Snode));
			}

			inline void moveToFront(Snode* node)
			{
				if (p_begin == node) return;
				p_begin->prev = node;
				node->next = p_begin;
				node->prev = nullptr;
				p_begin = node;
			}

			inline DoublyLinkedList(uint32_t& capacity) : alloc(malloc(sizeof(Snode)* capacity), 0u, 0u, (capacity - 1u), 1u, 1u)
			{
				p_begin = nullptr;
				p_end = nullptr;
			}
		};


	}


}


#endif // !__DOUBLY_LINKED_LIST_H_INCLUDED__
