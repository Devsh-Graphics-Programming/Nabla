#ifndef _NBL_VIDEO_I_DEVICE_MEMORY_ALLOCATOR_H_INCLUDED_
#define _NBL_VIDEO_I_DEVICE_MEMORY_ALLOCATOR_H_INCLUDED_

#include "nbl/core/definitions.h" // findLSB

#include "IDeviceMemoryAllocation.h"
#include "IDeviceMemoryBacked.h"

namespace nbl::video
{

class IDeviceMemoryAllocator
{
	public:
		struct SAllocateInfo
		{
			size_t size : 54 = 0ull;
			size_t flags : 5 = 0u; // IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS
			size_t memoryTypeIndex : 5 = 0u;
			IDeviceMemoryBacked* dedication = nullptr; // if you make the info have a `dedication` the memory will be bound right away, also it will use VK_KHR_dedicated_allocation on vulkan
			// size_t opaqueCaptureAddress = 0u; Note that this mechanism is intended only to support capture/replay tools, and is not recommended for use in other applications.

			// Handle Type for external resources
			IDeviceMemoryAllocation::E_EXTERNAL_HANDLE_TYPE externalHandleType = IDeviceMemoryAllocation::EHT_NONE;
			//! Imports the given handle  if externalHandle != nullptr && externalHandleType != EHT_NONE
			//! Creates exportable memory if externalHandle == nullptr && externalHandleType != EHT_NONE
			void* externalHandle = nullptr;
		};

		//! IMemoryTypeIterator extracts memoryType indices from memoryTypeBits in arbitrary order
		//! which is used to give priority to memoryTypes in try-allocate usages where allocations may fail with some memoryTypes
		//! IMemoryTypeIterator will construct SAllocateInfo from object's memory requirements, allocateFlags and dedication using operator()
		class IMemoryTypeIterator
		{
			public:
				IMemoryTypeIterator(const IDeviceMemoryBacked::SDeviceMemoryRequirements& reqs,
					core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags,
					IDeviceMemoryAllocation::E_EXTERNAL_HANDLE_TYPE handleType,
					void* handle)
					: m_allocateFlags(static_cast<uint32_t>(allocateFlags.value))
					, m_reqs(reqs)
					, m_handleType(handleType)
					, m_handle(handle)
				{}

				static inline uint32_t end() {return 32u;}

				IMemoryTypeIterator& operator++()
				{
					advance();
					return *this;
				}

				inline SAllocateInfo operator()(IDeviceMemoryBacked* dedication)
				{
					SAllocateInfo ret = {};
					ret.size = m_reqs.size;
					ret.flags = m_allocateFlags;
					ret.memoryTypeIndex = dereference();
					ret.dedication = dedication;
					ret.externalHandleType = m_handleType;
					ret.externalHandle = m_handle;
					return ret;
				}
		
				bool operator==(uint32_t rhs) const {return dereference() == rhs;}
				bool operator!=(uint32_t rhs) const {return dereference() != rhs;}

			protected:
				virtual uint32_t dereference() const = 0;
				virtual void advance() = 0;
		
				IDeviceMemoryBacked::SDeviceMemoryRequirements m_reqs;
				uint32_t m_allocateFlags;
				IDeviceMemoryAllocation::E_EXTERNAL_HANDLE_TYPE m_handleType;
				void* m_handle;
		};

		//! DefaultMemoryTypeIterator will iterate through set bits of memoryTypeBits from LSB to MSB
		class DefaultMemoryTypeIterator : public IMemoryTypeIterator
		{
			public:
				DefaultMemoryTypeIterator(const IDeviceMemoryBacked::SDeviceMemoryRequirements& reqs,
					core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags,
					IDeviceMemoryAllocation::E_EXTERNAL_HANDLE_TYPE handleType,
					void* handle)
					: IMemoryTypeIterator(reqs, allocateFlags, handleType, handle)
				{
					currentIndex = hlsl::findLSB(m_reqs.memoryTypeBits);
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
						currentIndex = hlsl::findLSB(leftBits);
					else
						currentIndex = IMemoryTypeIterator::end();
				}

				uint32_t currentIndex = 0u;
		};
		

		struct SAllocation
		{
			static constexpr size_t InvalidMemoryOffset = 0xdeadbeefBadC0ffeull;
			bool isValid() const
			{
				return memory && (offset!=InvalidMemoryOffset);
			}

			core::smart_refctd_ptr<IDeviceMemoryAllocation> memory = nullptr;
			size_t offset = InvalidMemoryOffset;
		};
		virtual SAllocation allocate(const SAllocateInfo& info) = 0;

		template<class memory_type_iterator_t = DefaultMemoryTypeIterator>
		SAllocation allocate(
			const IDeviceMemoryBacked::SDeviceMemoryRequirements& reqs,
			IDeviceMemoryBacked* dedication = nullptr,
			const core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags = IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE,
			IDeviceMemoryAllocation::E_EXTERNAL_HANDLE_TYPE handleType = IDeviceMemoryAllocation::EHT_NONE,
			void* handle = nullptr,
			std::unique_ptr<struct ICleanup>&& postDestroyCleanup = nullptr)
		{
			for (memory_type_iterator_t memTypeIt(reqs, allocateFlags, handleType, handle); memTypeIt != IMemoryTypeIterator::end(); ++memTypeIt)
			{
				SAllocateInfo allocateInfo = memTypeIt.operator()(dedication);
				SAllocation allocation = allocate(allocateInfo);
				if (allocation.isValid())
				{
					allocation.memory->setPostDestroyCleanup(std::move(postDestroyCleanup));
					return allocation;
				}
			}
			return { };
		}
};

}

#endif