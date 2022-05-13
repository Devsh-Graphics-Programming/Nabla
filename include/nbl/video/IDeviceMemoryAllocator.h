#ifndef _NBL_VIDEO_I_DEVICE_MEMORY_ALLOCATOR_H_INCLUDED_
#define _NBL_VIDEO_I_DEVICE_MEMORY_ALLOCATOR_H_INCLUDED_

#include "IDriverMemoryAllocation.h"
#include "IDriverMemoryBacked.h"
#include "nbl/core/math/glslFunctions.h" // findLSB

namespace nbl::video
{

class IDeviceMemoryAllocator
{
public:
	static constexpr size_t InvalidMemoryOffset = 0xdeadbeefBadC0ffeull;

	struct SMemoryOffset
	{
		core::smart_refctd_ptr<IDriverMemoryAllocation> memory = nullptr;
		size_t offset = InvalidMemoryOffset;
	};

	struct SAllocateInfo
	{
		size_t size : 54 = 0ull;
		size_t flags : 5 = 0u; // IDriverMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS
		size_t memoryTypeIndex : 5 = 0u;
		IDriverMemoryBacked* dedication = nullptr; // if you make the info have a `dedication` the memory will be bound right away, also it will use VK_KHR_dedicated_allocation on vulkan
		// size_t opaqueCaptureAddress = 0u; Note that this mechanism is intended only to support capture/replay tools, and is not recommended for use in other applications.
	};

	//! IMemoryTypeIterator extracts memoryType indices from memoryTypeBits in arbitrary order
	//! which is used to give priority to memoryTypes in try-allocate usages where allocations may fail with some memoryTypes
	//! IMemoryTypeIterator will construct SAllocateInfo from object's memory requirements, allocateFlags and dedication using operator()
	class IMemoryTypeIterator
	{
	public:
		IMemoryTypeIterator(const IDriverMemoryBacked::SDriverMemoryRequirements2& reqs, core::bitflag<IDriverMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags)
			: m_allocateFlags(static_cast<uint32_t>(allocateFlags.value))
			, m_reqs(reqs)
		{}

		static inline uint32_t end() {return 32u;}

		IMemoryTypeIterator& operator++()
		{
			advance();
			return *this;
		}

		inline SAllocateInfo operator()(IDriverMemoryBacked* dedication)
		{
			SAllocateInfo ret;
			ret.size = m_reqs.size;
			ret.flags = m_allocateFlags;
			ret.memoryTypeIndex = dereference();
			ret.dedication = dedication;
			return ret;
		}

	protected:
		virtual uint32_t dereference() const = 0;
		virtual void advance() = 0;
		virtual bool operator==(uint32_t rhs) const = 0;
		virtual bool operator!=(uint32_t rhs) const = 0;
		
		IDriverMemoryBacked::SDriverMemoryRequirements2 m_reqs;
		uint32_t m_allocateFlags;
	};

	//! DefaultMemoryTypeIterator will iterate through set bits of memoryTypeBits from LSB to MSB
	class DefaultMemoryTypeIterator : public IMemoryTypeIterator
	{
	public:
		DefaultMemoryTypeIterator(const IDriverMemoryBacked::SDriverMemoryRequirements2& reqs, core::bitflag<IDriverMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags)
			: IMemoryTypeIterator(reqs, allocateFlags)
		{
			currentIndex = core::findLSB(m_reqs.memoryTypeBits);
		}

		bool operator==(uint32_t rhs) const override { return currentIndex == rhs; }
		bool operator!=(uint32_t rhs) const override { return currentIndex != rhs; }

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

	// TODO: use template for Iterator
	// template<class memory_type_iterator_t=DefaultMemoryTypeIterator>
	SMemoryOffset allocate(
		const IDriverMemoryBacked::SDriverMemoryRequirements2& reqs,
		IDriverMemoryBacked* dedication = nullptr,
		const core::bitflag<IDriverMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags = IDriverMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE)
	{
		for(DefaultMemoryTypeIterator memTypeIt(reqs, allocateFlags); memTypeIt != IMemoryTypeIterator::end(); ++memTypeIt)
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