#ifndef __DOUBLY_LINKED_LIST_H_INCLUDED__
#define __DOUBLY_LINKED_LIST_H_INCLUDED__
#include "irr/core/alloc/PoolAddressAllocator.h"
#include "irr/core/Types.h"

namespace irr {
	namespace core {

		//Struct for use in a doubly linked list. Stores data and pointers to next and previous elements the list, or invalid iterator if it is first/last
		template<typename Value>
		struct Snode
		{
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t invalid_iterator = DoublyLinkedList<Value>::invalid_iterator;

			Value data;
			uint32_t prev;
			uint32_t next;

			Snode() {}
			Snode(const Value& val) : data(val)
			{
				prev = invalid_iterator;
				next = invalid_iterator;
			}
			Snode(Value&& val) : data(std::move(val))
			{
				prev = invalid_iterator;
				next = invalid_iterator;
			}
			Snode(Snode<Value>&& other) : data(std::move(other.data)), prev(std::move(other.prev)), next(std::move(other.next))
			{
			}

			Snode<Value>& operator=(const Snode<Value>& other)
			{
				this->data = other.data;
				this->prev = other.prev;
				this->next = other.next;
				return *this;
			}

			Snode<Value>& operator=(Snode<Value>&& other)
			{
				this->data = std::move(other.data);
				this->prev = std::move(other.prev);
				this->next = std::move(other.next);
				return *this;
			}
			~Snode()
			{

			}
		};


		template<typename Value>
		class DoublyLinkedList
		{

			void* reservedSpace;
			PoolAddressAllocator<uint32_t> alloc;
			uint32_t m_back;
			uint32_t m_begin;
			Snode<Value>* m_array;
			uint32_t cap;
			inline void common_delete(uint32_t address)
			{
				get(address)->~Snode<Value>();
				alloc.free_addr(address, 1u);
			}

			inline void common_detach(Snode<Value>* node)
			{
				if (node->next != invalid_iterator)
					get(node->next)->prev = node->prev;
				if (node->prev != invalid_iterator)
					get(node->prev)->next = node->next;
			}

		public:
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t invalid_iterator = PoolAddressAllocator<uint32_t>::invalid_address;

			//get node at iterator
			inline Snode<Value>* get(uint32_t address)
			{
				return (m_array + address);
			}
			inline const Snode<Value>* get(const uint32_t address) const
			{
				return (m_array + address);;
			}

			//remove the last element in the list
			inline void popBack()
			{
				if (m_back == invalid_iterator)	return;
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
				uint32_t addr = alloc.alloc_addr(1u, 1u);
				assert(addr < cap);
				Snode<Value>* n =  new(m_array+addr) Snode<Value>(std::move(val));
				m_array[addr] = *n;
				n->prev = invalid_iterator;
				n->next = m_begin;

				if (m_begin != invalid_iterator)
					getBegin()->prev = addr;
				if (m_back == invalid_iterator)
					m_back = addr;
				m_begin = addr;
			}

			//get node ptr of the first item in the list
			inline Snode<Value>* getBegin() { return m_array+m_begin; }

			//get node ptr of the last item in the list
			inline Snode<Value>* getBack() { return m_array+m_back; }

			//get index/iterator of the first element
			inline uint32_t getFirstAddress() const { return m_begin; } 

			//remove a node at nodeAddr from the list
			inline void erase(uint32_t nodeAddr)
			{
				assert(nodeAddr != invalid_iterator);
				assert(nodeAddr < cap);
				auto node = get(nodeAddr);
				common_detach(node);
				common_delete(nodeAddr);
			}

			//move a node at nodeAddr to the front of the list
			inline void moveToFront(uint32_t nodeAddr)
			{
				if (m_begin == nodeAddr || nodeAddr == invalid_iterator ) return;
				getBegin()->prev = nodeAddr;

				auto node = get(nodeAddr);
				common_detach(node);
				node->next = m_begin;
				node->prev = invalid_iterator;
				m_begin = nodeAddr;
			}
			//Constructor, capacity determines the amount of allocated space
			DoublyLinkedList(const uint32_t capacity) :
				cap(capacity),
				reservedSpace(_IRR_ALIGNED_MALLOC(PoolAddressAllocator<uint32_t>::reserved_size(1u, capacity, 1u), alignof(void*))),
				alloc(reservedSpace, 0u, 0u, 1u, capacity, 1u)
			{
				m_back = invalid_iterator;
				m_begin = invalid_iterator;
				m_array = new Snode<Value>[capacity];
			}
			~DoublyLinkedList()
			{
				_IRR_ALIGNED_FREE(reservedSpace);
				delete[] m_array;

			}
		};


	}


}


#endif // !__DOUBLY_LINKED_LIST_H_INCLUDED__
