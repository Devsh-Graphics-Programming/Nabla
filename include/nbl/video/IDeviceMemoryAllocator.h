#ifndef _NBL_VIDEO_I_DEVICE_MEMORY_ALLOCATOR_H_INCLUDED_
#define _NBL_VIDEO_I_DEVICE_MEMORY_ALLOCATOR_H_INCLUDED_

#include "IDeviceMemoryAllocation.h"
#include "IDeviceMemoryBacked.h"
#include "nbl/core/definitions.h" // findLSB

namespace nbl::video
{

class IDeviceMemoryAllocator
{
public:
	static constexpr size_t InvalidMemoryOffset = 0xdeadbeefBadC0ffeull;

	struct SMemoryOffset
	{
		core::smart_refctd_ptr<IDeviceMemoryAllocation> memory = nullptr;
		size_t offset = InvalidMemoryOffset;

		bool isValid() const
		{
			return memory && (offset!=InvalidMemoryOffset);
		}
	};

	struct SAllocateInfo
	{
		size_t size : 54 = 0ull;
		size_t flags : 5 = 0u; // IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS
		size_t memoryTypeIndex : 5 = 0u;
		IDeviceMemoryBacked* dedication = nullptr; // if you make the info have a `dedication` the memory will be bound right away, also it will use VK_KHR_dedicated_allocation on vulkan
		// size_t opaqueCaptureAddress = 0u; Note that this mechanism is intended only to support capture/replay tools, and is not recommended for use in other applications.
	};

	//! IMemoryTypeIterator extracts memoryType indices from memoryTypeBits in arbitrary order
	//! which is used to give priority to memoryTypes in try-allocate usages where allocations may fail with some memoryTypes
	//! IMemoryTypeIterator will construct SAllocateInfo from object's memory requirements, allocateFlags and dedication using operator()
	class IMemoryTypeIterator
	{
	public:
		IMemoryTypeIterator(const IDeviceMemoryBacked::SDriverMemoryRequirements& reqs, core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags)
			: m_allocateFlags(static_cast<uint32_t>(allocateFlags.value))
			, m_reqs(reqs)
		{}

		static inline uint32_t end() {return 32u;}

		IMemoryTypeIterator& operator++()
		{
			advance();
			return *this;
		}

		inline SAllocateInfo operator()(IDeviceMemoryBacked* dedication)
		{
			SAllocateInfo ret;
			ret.size = m_reqs.size;
			ret.flags = m_allocateFlags;
			ret.memoryTypeIndex = dereference();
			ret.dedication = dedication;
			return ret;
		}
		
		bool operator==(uint32_t rhs) const {return dereference() == rhs;}
		bool operator!=(uint32_t rhs) const {return dereference() != rhs;}

	protected:
		virtual uint32_t dereference() const = 0;
		virtual void advance() = 0;
		
		IDeviceMemoryBacked::SDriverMemoryRequirements m_reqs;
		uint32_t m_allocateFlags;
	};

	//! DefaultMemoryTypeIterator will iterate through set bits of memoryTypeBits from LSB to MSB
	class DefaultMemoryTypeIterator : public IMemoryTypeIterator
	{
	public:
		DefaultMemoryTypeIterator(const IDeviceMemoryBacked::SDriverMemoryRequirements& reqs, core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags)
			: IMemoryTypeIterator(reqs, allocateFlags)
		{
			currentIndex = core::findLSB(m_reqs.memoryTypeBits);
		}

	protected:
		uint32_t dereference() const override
		{
			return currentIndex;
		}

		void advance() override
		{
			uint32_t leftBits = m_reqs.memoryTypeBits & ~((1u << (currentIndex + 1u)) - 1u); // set lower bits to 0
			if(leftBits > 0u)
				currentIndex = core::findLSB(leftBits);
			else
				currentIndex = IMemoryTypeIterator::end();
		}

		uint32_t currentIndex = 0u;
	};

	virtual SMemoryOffset allocate(const SAllocateInfo& info) = 0;

	template<class memory_type_iterator_t=DefaultMemoryTypeIterator>
	SMemoryOffset allocate(
		const IDeviceMemoryBacked::SDriverMemoryRequirements& reqs,
		IDeviceMemoryBacked* dedication = nullptr,
		const core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags = IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE)
	{
		for(memory_type_iterator_t memTypeIt(reqs, allocateFlags); memTypeIt != IMemoryTypeIterator::end(); ++memTypeIt)
		{
			SAllocateInfo allocateInfo = memTypeIt.operator()(dedication);
			SMemoryOffset allocation = allocate(allocateInfo);
			if (allocation.memory && allocation.offset != InvalidMemoryOffset)
				return allocation;
		}
		return {nullptr, InvalidMemoryOffset};
	}
};

}

#endif