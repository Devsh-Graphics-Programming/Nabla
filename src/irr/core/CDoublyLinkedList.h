#ifndef __DOUBLY_LINKED_LIST_H_INCLUDED__
#define __DOUBLY_LINKED_LIST_H_INCLUDED__
#include "irr/core/alloc/PoolAddressAllocator.h"
#include "irr/core/Types.h"

namespace irr {
	namespace core {

		template<typename Value>
		class DoublyLinkedList
		{
			struct Snode
			{
				struct Snode* prev;
				Value data;
				struct Snode* next;

				Snode(Value v) { data = v; }
			};

			uint32_t cap;
			Snode* begin;
			Snode* end;
			PoolAddressAllocator<uint32_t> alloc;

			inline void popBack()
			{
				if(end->prev != nullptr)
					end->prev->next = nullptr;
				Snode* temp = end;
				end = end->prev;
				alloc.free_addr(reinterpret_cast<uint32_t>(temp), sizeof(Snode));
			}

		public:

			inline void pushFront(Value &val) 
			{
				auto p = alloc.alloc_addr(sizeof(Snode),0u)
				Snode n = new(reinterpret_cast<void*>(p)) Snode(val);
				n.prev = nullptr;
				
				n.next = begin;

				if (begin != nullptr)
					begin->prev = &n;
				if (end == nullptr)
					end = &n;
				begin = &n;
			}
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
				if (begin == node) return;
				begin->prev = node;
				node->next = begin;
				node->prev = nullptr;
				begin = node;
			}
			DoublyLinkedList(uint32_t& capacity) 
			{
				alloc = new PoolAddressAllocator<uint32_t>(malloc(sizeof(Snode) * capacity), 1u, 1u, 
					sizeof(Snode) * (capacity - 1u),	//max allocatable alligment - should it be without `sizeof(Snode) `
					sizeof(Snode), 1u);

			}
		};


	}


}


#endif // !__DOUBLY_LINKED_LIST_H_INCLUDED__
