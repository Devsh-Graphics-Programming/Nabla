#ifndef __NBL_CORE_CONTAINERS_LISTS_COMMON_H_INCLUDED__
#define __NBL_CORE_CONTAINERS_LISTS_COMMON_H_INCLUDED__


#include "nbl/core/alloc/PoolAddressAllocator.h"
#include "nbl/core/decl/Types.h"

#include <functional>

// All lists use PoolAddressAllocator

namespace nbl
{
namespace core
{

namespace impl
{

template<typename NodeType>
class ContiguousMemoryLinkedListBase
{
public:
	using address_allocator_t = PoolAddressAllocator<uint32_t>;
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t invalid_iterator = address_allocator_t::invalid_address;

	using node_t = NodeType;
	using value_t = typename node_t::value_t;
	using disposal_func_t = std::function<void(value_t&)>;

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
	inline node_t* getBack() { return m_array + this->m_back; }
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
		insertAt(reserveAddress(), value_t(std::forward<Args>(args)...));
	}

	//remove the last element in the list
	virtual void popBack() = 0;

	//remove a node at nodeAddr from the list
	virtual void erase(const uint32_t nodeAddr) = 0;

	//move a node at nodeAddr to the front of the list
	virtual void moveToFront(const uint32_t nodeAddr) = 0;

	//Constructor, capacity determines the amount of allocated space
	ContiguousMemoryLinkedListBase(const uint32_t capacity, disposal_func_t&& dispose_f = disposal_func_t()) : m_dispose_f(std::move(dispose_f))
	{
		const auto firstPart = core::alignUp(address_allocator_t::reserved_size(1u, capacity, 1u), alignof(node_t));
		m_reservedSpace = _NBL_ALIGNED_MALLOC(firstPart + capacity * sizeof(node_t), alignof(node_t));
		m_array = reinterpret_cast<node_t*>(reinterpret_cast<uint8_t*>(m_reservedSpace) + firstPart);

		m_addressAllocator = std::unique_ptr<address_allocator_t>(new address_allocator_t(m_reservedSpace, 0u, 0u, 1u, capacity, 1u));
		m_cap = capacity;
		m_back = invalid_iterator;
		m_begin = invalid_iterator;
	}

	ContiguousMemoryLinkedListBase() = default;

	ContiguousMemoryLinkedListBase(const ContiguousMemoryLinkedListBase& other) = delete;

	ContiguousMemoryLinkedListBase& operator=(const ContiguousMemoryLinkedListBase& other) = delete;

	ContiguousMemoryLinkedListBase& operator=(ContiguousMemoryLinkedListBase&& other)
	{
		m_addressAllocator = std::move(other.m_addressAllocator);
		m_reservedSpace = other.m_reservedSpace;
		m_array = other.m_array;
		m_dispose_f = std::move(other.m_dispose_f);
		m_cap = other.m_cap;
		m_back = other.m_back;
		m_begin = other.m_begin;

		// Nullify other
		other.m_addressAllocator = nullptr;
		other.m_reservedSpace = nullptr;
		other.m_array = nullptr;
		other.m_cap = 0u;
		other.m_back = 0u;
		other.m_begin = 0u;
		return *this;
	}

	~ContiguousMemoryLinkedListBase()
	{
		if (m_dispose_f && m_begin != invalid_iterator)
		{
			auto* begin = getBegin();
			auto* back = getBack();
			while(begin != back)
			{
				m_dispose_f(begin->data);
				begin = get(begin->next);
			}
			m_dispose_f(back->data);
		}
		_NBL_ALIGNED_FREE(m_reservedSpace);
	}

protected:
	inline void common_delete(uint32_t address)
	{
		if (m_dispose_f)
			m_dispose_f(get(address)->data);
		get(address)->~node_t();
		m_addressAllocator->free_addr(address, 1u);
	}

private:
	//allocate and get the address of the next free node
	inline uint32_t reserveAddress()
	{
		uint32_t addr = m_addressAllocator->alloc_addr(1u, 1u);
		return addr;
	}

	//create a new node which stores data at already allocated address, 
	virtual void insertAt(uint32_t addr, value_t&& val) = 0;

	virtual void common_detach(node_t* node) = 0;

protected:
	std::unique_ptr<address_allocator_t> m_addressAllocator;
	void* m_reservedSpace;
	node_t* m_array;

	uint32_t m_cap;
	uint32_t m_back;
	uint32_t m_begin;
private:
	disposal_func_t m_dispose_f;
};

} //namespace impl

} //namespace core
} //namespace nbl






















#endif