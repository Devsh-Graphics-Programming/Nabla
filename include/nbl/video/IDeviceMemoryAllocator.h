#ifndef _NBL_VIDEO_I_DEVICE_MEMORY_ALLOCATOR_H_INCLUDED_
#define _NBL_VIDEO_I_DEVICE_MEMORY_ALLOCATOR_H_INCLUDED_

#include "nbl/core/definitions.h" // findLSB

#include "IDeviceMemoryAllocation.h"
#include "IDeviceMemoryBacked.h"

namespace nbl::video
{

class NBL_API2 IDeviceMemoryAllocator
{
	public:
		// right now we only support this interface handing out memory for one device or group
		virtual ILogicalDevice* getDeviceForAllocations() const = 0;

		struct SAllocateInfo : IDeviceMemoryAllocation::SInfo
		{
			IDeviceMemoryBacked* dedication = nullptr; // if you make the info have a `dedication` the memory will be bound right away, also it will use VK_KHR_dedicated_allocation on vulkan
			// size_t opaqueCaptureAddress = 0u; Note that this mechanism is intended only to support capture/replay tools, and is not recommended for use in other applications.
			uint8_t memoryTypeIndex = 0u;
		};

		struct SAllocateParams {
			IDeviceMemoryBacked* dedication = nullptr;
			const core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags = IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE;
			IDeviceMemoryAllocation::E_EXTERNAL_HANDLE_TYPE externalHandleType = IDeviceMemoryAllocation::EHT_NONE;
			system::external_handle_t externalHandle = system::ExternalHandleNull;
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


		//! IMemoryTypeIterator extracts memoryType indices from memoryTypeBits in arbitrary order
		//! which is used to give priority to memoryTypes in try-allocate usages where allocations may fail with some memoryTypes
		//! IMemoryTypeIterator will construct SAllocateInfo from object's memory requirements, allocateFlags and dedication using operator()
		class IMemoryTypeIterator
		{
			public:
				IMemoryTypeIterator(const IDeviceMemoryBacked::SDeviceMemoryRequirements& reqs, 
					core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags,
					IDeviceMemoryAllocation::E_EXTERNAL_HANDLE_TYPE handleType) : 
					m_allocateFlags(static_cast<uint32_t>(allocateFlags.value)), 
					m_reqs(reqs), 
					m_handleType(handleType)
				{}

				static inline uint32_t end() {return 32u;}

				IMemoryTypeIterator& operator++()
				{
					advance();
					return *this;
				}

				inline SAllocateInfo operator()(IDeviceMemoryBacked* dedication, system::external_handle_t external_handle)
				{
					SAllocateInfo ret;
					ret.allocationSize = m_reqs.size;
					ret.allocateFlags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS>(m_allocateFlags);
					ret.memoryTypeIndex = dereference();
					ret.dedication = dedication;
					ret.externalHandleType = m_handleType;
					ret.importHandle = external_handle;
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
		};

		//! DefaultMemoryTypeIterator will iterate through set bits of memoryTypeBits from LSB to MSB
		class DefaultMemoryTypeIterator : public IMemoryTypeIterator
		{
			public:
				DefaultMemoryTypeIterator(
					const IDeviceMemoryBacked::SDeviceMemoryRequirements& reqs, 
					core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags,
					IDeviceMemoryAllocation::E_EXTERNAL_HANDLE_TYPE handleType
				) : 
				IMemoryTypeIterator(reqs, allocateFlags, handleType)
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

		template<class memory_type_iterator_t=DefaultMemoryTypeIterator>
		// TODO(kevinyu) : Fix all example_tests if this api change to use SAllocateParams is approved
		inline SAllocation allocate(
			const IDeviceMemoryBacked::SDeviceMemoryRequirements& reqs, 
			const SAllocateParams& params)
		{
			for (memory_type_iterator_t memTypeIt(reqs, params.allocateFlags, params.externalHandleType); memTypeIt!=IMemoryTypeIterator::end(); ++memTypeIt)
			{
				SAllocateInfo allocateInfo = memTypeIt.operator()(params.dedication, params.externalHandle);
				auto allocation = allocate(allocateInfo);
				if (allocation.isValid())
					return allocation;
			}
			return {};
		}
};

}

#endif