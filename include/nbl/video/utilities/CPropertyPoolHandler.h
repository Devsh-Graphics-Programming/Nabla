// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_PROPERTY_POOL_HANDLER_H_INCLUDED__
#define __NBL_VIDEO_C_PROPERTY_POOL_HANDLER_H_INCLUDED__


#include "nbl/asset/asset.h"

#include "nbl/video/IGPUCommandBuffer.h"
#include "nbl/video/alloc/StreamingTransientDataBuffer.h"
#include "nbl/video/utilities/IDescriptorSetCache.h"
#include "nbl/video/utilities/IPropertyPool.h"


namespace nbl::video
{

class IPropertyPool;

// property pool factory is externally synchronized
class CPropertyPoolHandler final : public core::IReferenceCounted, public core::Unmovable
{
	public:
		//
		CPropertyPoolHandler(core::smart_refctd_ptr<ILogicalDevice>&& device);

        _NBL_STATIC_INLINE_CONSTEXPR auto MinimumPropertyAlignment = alignof(uint32_t);

		//
		inline ILogicalDevice* getDevice() {return m_device.get();}

		//
		inline const uint32_t getMaxPropertiesPerTransferDispatch() {return m_maxPropertiesPerPass;}

        //
		inline IGPUComputePipeline* getPipeline() {return m_pipeline.get();}
		inline const IGPUComputePipeline* getPipeline() const {return m_pipeline.get();}

        //
		inline const IGPUDescriptorSetLayout* getCanonicalLayout() const { return m_dsCache->getCanonicalLayout(); }
		
		//
		class download_future_t : public core::Uncopyable
		{
				friend class CPropertyPoolHandler;

				core::smart_refctd_ptr<ILogicalDevice> m_device;
				core::smart_refctd_ptr<StreamingTransientDataBufferMT<>> m_downBuff;
				core::smart_refctd_ptr<IGPUFence> m_fence;
				uint32_t m_reserved,m_allocCount;
				uint32_t* m_addresses,*m_sizes;
				
				download_future_t(
					core::smart_refctd_ptr<ILogicalDevice>&& _device,
					core::smart_refctd_ptr<StreamingTransientDataBufferMT<>>&& _downBuff,
					core::smart_refctd_ptr<IGPUFence>&& _fence,
					const uint32_t _reserved
				): m_device(std::move(_device)), m_downBuff(std::move(_downBuff)), m_fence(std::move(_fence)), m_reserved(_reserved), m_allocCount(0u)
				{
					if (m_reserved)
					{
						m_addresses = new uint32_t[m_reserved*2u];
						m_sizes = m_addresses+m_reserved;
					}
					else
					{
						m_addresses = nullptr;
						m_sizes = nullptr;
					}
				}

				void push(const uint32_t n, const uint32_t* addresses, const uint32_t* sizes)
				{
					std::copy_n(addresses,n,m_addresses+m_allocCount);
					std::copy_n(sizes,n,m_sizes+m_allocCount);
					m_allocCount += n;
				}

			public:
				download_future_t() : m_device(), m_downBuff(), m_fence(), m_reserved(0), m_allocCount(0), m_addresses(nullptr), m_sizes(nullptr) {}
				download_future_t(download_future_t&& other) : download_future_t()
				{
					operator=(std::move(other));
				}
				~download_future_t();
				
				inline download_future_t& operator=(download_future_t&& other)
				{
					std::swap(m_device,other.m_device);
					std::swap(m_downBuff,other.m_downBuff);
					std::swap(m_fence,other.m_fence);
					std::swap(m_reserved,other.m_reserved);
					std::swap(m_allocCount,other.m_allocCount);
					std::swap(m_addresses,other.m_addresses);
					std::swap(m_sizes,other.m_sizes);
					return *this;
				}

				bool wait();

				// downRequestIndex is an index into packed list of download only requests.
				// If you had transfer requests [u,u,d,u,d,d] then `downRequestIndex=1` maps to the 5th request
				const void* getData(const uint32_t downRequestIndex);
		};

		struct transfer_result_t
		{
			transfer_result_t(download_future_t&& _download, bool _transferSuccess) : download(std::move(_download)), transferSuccess(_transferSuccess) {}
			transfer_result_t(transfer_result_t&& other)
			{
				download = std::move(other.download);
				transferSuccess = other.transferSuccess;
				other.download = download_future_t();
				other.transferSuccess = false;
			}

			download_future_t download;
			bool transferSuccess;
		};


		// allocate and upload properties, indices need to be pre-initialized to `invalid_index`
		struct AllocationRequest
		{
			AllocationRequest() : pool(nullptr), outIndices{nullptr,nullptr}, data(nullptr) {}
			AllocationRequest(IPropertyPool* _pool, core::SRange<uint32_t> _outIndices, const void* const* _data) : pool(_pool), outIndices(_outIndices), data(_data) {}

			IPropertyPool* pool;
			core::SRange<uint32_t> outIndices;
			const void* const* data; 
		};

		// returns false if an allocation or part of a transfer has failed
		// while its possible to detect which allocation has failed, its not possible to know exactly what transfer failed
		transfer_result_t addProperties(
			StreamingTransientDataBufferMT<>* const upBuff, StreamingTransientDataBufferMT<>* const downBuff, IGPUCommandBuffer* const cmdbuf,
			IGPUFence* const fence, const AllocationRequest* const requestsBegin, const AllocationRequest* const requestsEnd, system::logger_opt_ptr logger,
			const std::chrono::high_resolution_clock::time_point maxWaitPoint=std::chrono::high_resolution_clock::now()+std::chrono::microseconds(1500u)
		);


        //
		struct TransferRequest
		{
			TransferRequest() : pool(nullptr), indices{nullptr,nullptr}, data(nullptr), propertyID(0xdeadbeefu) {}

			inline bool isDownload() const {return !data;}

			const IPropertyPool* pool;
			core::SRange<const uint32_t> indices;
			const void* data;
			uint32_t propertyID;
		};

		// Fence must be not pending yet
		[[nodiscard]] transfer_result_t transferProperties(
			StreamingTransientDataBufferMT<>* const upBuff, StreamingTransientDataBufferMT<>* const downBuff, IGPUCommandBuffer* const cmdbuf,
			IGPUFence* const fence, const TransferRequest* const requestsBegin, const TransferRequest* const requestsEnd,system::logger_opt_ptr logger,
			const std::chrono::high_resolution_clock::time_point maxWaitPoint=std::chrono::high_resolution_clock::now()+std::chrono::microseconds(500u)
		);

    protected:
		~CPropertyPoolHandler()
		{
			free(m_tmpIndexRanges);
			// pipelines drop themselves automatically
		}

		static inline constexpr auto IdealWorkGroupSize = 256u;
		static inline constexpr auto MaxPropertyTransfers = 512u;
		static inline constexpr auto DescriptorCacheSize = 128u;


		core::smart_refctd_ptr<ILogicalDevice> m_device;
		struct IndexUploadRange
		{
			IndexUploadRange() : contiguousPool(nullptr), source{nullptr,nullptr}, destOff(0xdeadbeefu) {}

			const IPropertyPool* contiguousPool;
			core::SRange<const uint32_t> source;
			uint32_t destOff;
		};
        IndexUploadRange* m_tmpIndexRanges;
		uint32_t* m_tmpAddresses,* m_tmpSizes,* m_alignments;
		uint8_t m_maxPropertiesPerPass;

		// TODO: investigate using Push Descriptors for this
		class TransferDescriptorSetCache : public IDescriptorSetCache
		{
			public:
				using IDescriptorSetCache::IDescriptorSetCache;

				//
				uint32_t acquireSet(
					CPropertyPoolHandler* handler,
					IGPUBuffer* const upBuff,
					IGPUBuffer* const downBuff,
					const TransferRequest* requests,
					const uint32_t propertyCount,
					const uint32_t firstSSBOSize,
					const uint32_t* uploadAddresses,
					const uint32_t* downloadAddresses
				);
		};
		core::smart_refctd_ptr<TransferDescriptorSetCache> m_dsCache;

		core::smart_refctd_ptr<IGPUComputePipeline> m_pipeline;
};


}

#endif