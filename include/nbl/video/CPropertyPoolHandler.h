// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_PROPERTY_POOL_HANDLER_H_INCLUDED__
#define __NBL_VIDEO_C_PROPERTY_POOL_HANDLER_H_INCLUDED__


#include "nbl/asset/asset.h"

#include "nbl/video/IDescriptorSetCache.h"


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
		bool addProperties(
			StreamingTransientDataBufferMT<>* const upBuff, StreamingTransientDataBufferMT<>* const downBuff, IGPUCommandBuffer* const cmdbuf,
			IGPUFence* const fence, const AllocationRequest* const requestsBegin, const AllocationRequest* const requestsEnd,
			system::logger_opt_ptr logger, const std::chrono::high_resolution_clock::time_point maxWaitPoint
		);
		bool addProperties(
			IGPUCommandBuffer* const cmdbuf, IGPUFence* const fence, const AllocationRequest* const requestsBegin, const AllocationRequest* const requestsEnd, system::logger_opt_ptr logger,
			const std::chrono::high_resolution_clock::time_point maxWaitPoint = std::chrono::high_resolution_clock::now() + std::chrono::microseconds(1500u)
		);

        //
		struct TransferRequest
		{
			TransferRequest() : download(false), pool(nullptr), indices{nullptr,nullptr}, propertyID(0xdeadbeefu)
			{
				readData = nullptr;
			}

			bool download;
			IPropertyPool* pool;
			core::SRange<const uint32_t> indices;
			uint32_t propertyID;
			union
			{
				const void* readData;
				void* writeData;
			};
		};
		// fence must be not pending yet
		bool transferProperties(
			StreamingTransientDataBufferMT<>* const upBuff, StreamingTransientDataBufferMT<>* const downBuff, IGPUCommandBuffer* const cmdbuf,
			IGPUFence* const fence, const TransferRequest* const requestsBegin, const TransferRequest* const requestsEnd,
			system::logger_opt_ptr logger, const std::chrono::high_resolution_clock::time_point maxWaitPoint
		);
		bool transferProperties(
			IGPUCommandBuffer* const cmdbuf, IGPUFence* const fence, const TransferRequest* const requestsBegin, const TransferRequest* const requestsEnd, system::logger_opt_ptr logger,
			const std::chrono::high_resolution_clock::time_point maxWaitPoint = std::chrono::high_resolution_clock::now() + std::chrono::microseconds(500u)
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
			IndexUploadRange() : source{nullptr,nullptr}, destOff(0xdeadbeefu) {}

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
					const TransferRequest* requests,
					const uint32_t indexCount,
					const uint32_t propertyCount,
					const uint32_t* uploadAddresses,
					const uint32_t* downloadAddresses
				);
		};
		core::smart_refctd_ptr<TransferDescriptorSetCache> m_dsCache;

		core::smart_refctd_ptr<IGPUComputePipeline> m_pipeline;
};


}

#endif