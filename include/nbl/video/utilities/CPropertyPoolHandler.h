// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_C_PROPERTY_POOL_HANDLER_H_INCLUDED_
#define _NBL_VIDEO_C_PROPERTY_POOL_HANDLER_H_INCLUDED_

#include "nbl/asset/asset.h"

#include "nbl/video/IGPUCommandBuffer.h"
#include "nbl/video/alloc/StreamingTransientDataBuffer.h"
#include "nbl/video/utilities/IDescriptorSetCache.h"
#include "nbl/video/utilities/IPropertyPool.h"


namespace nbl::video
{

#if 0 // TODO: port
#define int int32_t
#define uint uint32_t
#include "nbl/builtin/glsl/property_pool/transfer.glsl"
#undef uint
#undef int
static_assert(NBL_BUILTIN_PROPERTY_POOL_INVALID==IPropertyPool::invalid);

// property pool factory is externally synchronized
// TODO: could rename to CSparseStreamingSystem/CSparseStreamingHandler
class NBL_API2 CPropertyPoolHandler final : public core::IReferenceCounted, public core::Unmovable
{
	public:
		//
		CPropertyPoolHandler(core::smart_refctd_ptr<ILogicalDevice>&& device);

        static inline constexpr auto MinimumPropertyAlignment = alignof(uint32_t);

		//
		inline ILogicalDevice* getDevice() {return m_device.get();}

		//
		inline const uint32_t getMaxPropertiesPerTransferDispatch() {return m_maxPropertiesPerPass;}

		//
		inline uint32_t getMaxScratchSize() const {return sizeof(nbl_glsl_property_pool_transfer_t)*m_maxPropertiesPerPass;}

        //
		inline IGPUComputePipeline* getPipeline() {return m_pipeline.get();}
		inline const IGPUComputePipeline* getPipeline() const {return m_pipeline.get();}

        //
		inline const IGPUDescriptorSetLayout* getCanonicalLayout() const { return m_dsCache->getCanonicalLayout(); }

        //
		struct TransferRequest
		{
			//
			enum E_FLAG : uint16_t
			{
				EF_NONE=0,
				EF_DOWNLOAD=NBL_BUILTIN_PROPERTY_POOL_TRANSFER_EF_DOWNLOAD,
				// this flag will make the `srcAddresses ? srcAddresses[0]:0` be used as the source address for all reads, effectively "filling" with uniform value
				EF_FILL=NBL_BUILTIN_PROPERTY_POOL_TRANSFER_EF_SRC_FILL,
				EF_BIT_COUNT=NBL_BUILTIN_PROPERTY_POOL_TRANSFER_EF_BIT_COUNT
			};
			//
			static inline constexpr uint32_t invalid_offset = ~0u;

			//
			inline void setFromPool(const IPropertyPool* pool, const uint16_t propertyID)
			{
				memblock = pool->getPropertyMemoryBlock(propertyID);
				elementSize = pool->getPropertySize(propertyID);
			}

			//
			inline bool isDownload() const {return flags&EF_DOWNLOAD;}

			//
			inline uint32_t getSourceElementCount() const
			{
				if (flags&EF_FILL)
					return 1u;
				return elementCount;
			}

			//
			asset::SBufferRange<IGPUBuffer> memblock = {};
			E_FLAG flags = EF_NONE;
			uint16_t elementSize = 0u;
			uint32_t elementCount = 0u;
			// the source or destination buffer depending on the transfer type
			asset::SBufferBinding<video::IGPUBuffer> buffer = {};
			// can be invalid, if invalid, treated like an implicit {0,1,2,3,...} iota view
			uint32_t srcAddressesOffset = IPropertyPool::invalid;
			uint32_t dstAddressesOffset = IPropertyPool::invalid;
		};
		// Fence must be not pending yet, `cmdbuf` must be already in recording state.
		[[nodiscard]] bool transferProperties(
			IGPUCommandBuffer* const cmdbuf, IGPUFence* const fence,
			const asset::SBufferBinding<video::IGPUBuffer>& scratch, const asset::SBufferBinding<video::IGPUBuffer>& addresses,
			const TransferRequest* const requestsBegin, const TransferRequest* const requestsEnd,
			system::logger_opt_ptr logger, const uint32_t baseDWORD=0u, const uint32_t endDWORD=~0ull
		);

		//
		struct UpStreamingRequest
		{
			inline UpStreamingRequest()
			{
			}
			//
			inline UpStreamingRequest(const UpStreamingRequest& other)
			{
				operator=(other);
			}
			//
			inline UpStreamingRequest& operator=(const UpStreamingRequest& other)
			{
				destination = other.destination;
				fill = other.fill;
				elementSize = other.elementSize;
				elementCount = other.elementCount;
				source.buffer = other.source.buffer;
				srcAddresses = other.srcAddresses;
				dstAddresses = other.dstAddresses;
				return *this;
			}

			//
			inline void setFromPool(const IPropertyPool* pool, const uint16_t propertyID)
			{
				destination = pool->getPropertyMemoryBlock(propertyID);
				elementSize = pool->getPropertySize(propertyID);
			}

			//
			inline uint32_t getElementDWORDs() const
			{
				return uint32_t(elementSize/sizeof(uint32_t))*elementCount;
			}

			//
			inline uint32_t getElementsToSkip(const uint32_t baseDWORD) const
			{
				return core::min(baseDWORD/uint32_t(elementSize/sizeof(uint32_t)),elementCount);
			}


			asset::SBufferRange<IGPUBuffer> destination = {};
			bool fill = false;
			uint16_t elementSize = 0u;
			uint32_t elementCount = 0u;
			union Source
			{
				Source()
				{
					data = nullptr;
					device2device = false;
				}
				~Source()
				{
					buffer.~SBufferBinding();
				}

				inline Source& operator=(const Source& other)
				{
					buffer = other.buffer;
					return *this;
				}

				// device
				asset::SBufferBinding<video::IGPUBuffer> buffer;
				// host
				struct
				{
					const void* data;
					uint64_t device2device;
				};
			} source;
			// can be nullptr, if null, treated like an implicit {0,1,2,3,...} iota view
			const uint32_t* srcAddresses = nullptr;
			const uint32_t* dstAddresses = nullptr;
		};
		// Fence must be not pending yet, `cmdbuf` must be already in recording state and be resettable.
		// `requests` will be consumed (destructively processed by sorting) and incremented by however many requests were fully processed
		// return value tells you how many DWORDs are remaining in the new first batch pointed to by `requests`
		[[nodiscard]] uint32_t transferProperties(
			StreamingTransientDataBufferMT<>* const upBuff, IGPUCommandBuffer* const cmdbuf, IGPUFence* const fence, IQueue* const queue,
			const asset::SBufferBinding<video::IGPUBuffer>& scratch, UpStreamingRequest*& requests, const uint32_t requestCount,
			uint32_t& waitSemaphoreCount, IGPUSemaphore* const*& semaphoresToWaitBeforeOverwrite, const asset::PIPELINE_STAGE_FLAGS*& stagesToWaitForPerSemaphore,
			system::logger_opt_ptr logger, const std::chrono::steady_clock::time_point& maxWaitPoint=std::chrono::steady_clock::now()+std::chrono::microseconds(500u)
		);

		// utility to help you fill out the tail move scatter request after the free, properly, returns if you actually need to transfer anything
		static inline bool freeProperties(IPropertyPool* pool, UpStreamingRequest* requests, const uint32_t* indicesBegin, const uint32_t* indicesEnd, uint32_t* srcAddresses, uint32_t* dstAddresses)
		{
			const auto oldHead = pool->getAllocated();
			const auto transferCount = pool->freeProperties(indicesBegin,indicesEnd,srcAddresses,dstAddresses);
			if (transferCount)
			{
				for (auto i=0u; i<pool->getPropertyCount(); i++)
				{
					requests[i].setFromPool(pool,i);
					requests[i].fill = false;
					requests[i].elementCount = transferCount;
					requests[i].source.buffer = {requests[i].destination.offset,requests[i].destination.buffer}; // copy from self to self
					requests[i].srcAddresses = srcAddresses;
					requests[i].dstAddresses = dstAddresses;
				}
				return true;
			}
			return false;
		}

    protected:
		~CPropertyPoolHandler() {}

		static inline constexpr auto MaxPropertiesPerDispatch = NBL_BUILTIN_PROPERTY_POOL_MAX_PROPERTIES_PER_DISPATCH;
		static inline constexpr auto DescriptorCacheSize = 128u;


		core::smart_refctd_ptr<ILogicalDevice> m_device;
		core::smart_refctd_ptr<IGPUComputePipeline> m_pipeline;
		// TODO: investigate using Push Descriptors for this
		class TransferDescriptorSetCache : public IDescriptorSetCache
		{
			public:
				using IDescriptorSetCache::IDescriptorSetCache;

				//
				uint32_t acquireSet(
					CPropertyPoolHandler* handler, const asset::SBufferBinding<video::IGPUBuffer>& scratch, const asset::SBufferBinding<video::IGPUBuffer>& addresses,
					const TransferRequest* requests, const uint32_t propertyCount
				);
		};
		core::smart_refctd_ptr<TransferDescriptorSetCache> m_dsCache;

		uint16_t m_maxPropertiesPerPass;
		uint32_t m_alignment;
};
#endif

}
#endif