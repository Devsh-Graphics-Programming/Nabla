// Copyright (C) 2023-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_APPLICATION_TEMPLATES_BASIC_MULTI_QUEUE_APPLICATION_HPP_INCLUDED_
#define _NBL_EXAMPLES_APPLICATION_TEMPLATES_BASIC_MULTI_QUEUE_APPLICATION_HPP_INCLUDED_

// Build on top of the previous one
#include "MonoDeviceApplication.hpp"

namespace nbl::application_templates
{

// Virtual Inheritance because apps might end up doing diamond inheritance
class BasicMultiQueueApplication : public virtual MonoDeviceApplication
{
		using base_t = MonoDeviceApplication;

	public:
		using base_t::base_t;

		// So now we need to return "threadsafe" queues because the queues might get aliased and also used on multiple threads
		virtual video::CThreadSafeQueueAdapter* getComputeQueue() const
		{
			return m_device->getThreadSafeQueue(m_computeQueue.famIx,m_computeQueue.qIx);
		}
		virtual video::CThreadSafeQueueAdapter* getGraphicsQueue() const
		{
			if (m_graphicsQueue.famIx!=QueueAllocator::InvalidIndex)
				return m_device->getThreadSafeQueue(m_graphicsQueue.famIx,m_graphicsQueue.qIx);
			assert(!isComputeOnly());
			return nullptr;
		}

		// virtual to allow aliasing and total flexibility, as with the above
		virtual video::CThreadSafeQueueAdapter* getTransferUpQueue() const
		{
			return m_device->getThreadSafeQueue(m_transferUpQueue.famIx,m_transferUpQueue.qIx);
		}
		virtual video::CThreadSafeQueueAdapter* getTransferDownQueue() const
		{
			return m_device->getThreadSafeQueue(m_transferDownQueue.famIx,m_transferDownQueue.qIx);
		}

	protected:
		// This time we build upon the Mono-System and Mono-Logger application and add the creation of possibly multiple queues and creation of IUtilities
		virtual bool onAppInitialized(core::smart_refctd_ptr<system::ISystem>&& system) override
		{
			if (!base_t::onAppInitialized(std::move(system)))
				return false;
			
			using namespace core;
			m_utils = make_smart_refctd_ptr<video::IUtilities>(smart_refctd_ptr(m_device),smart_refctd_ptr(m_logger));
			if (!m_utils)
				return logFail("Failed to create nbl::video::IUtilities!");

			return true;
		}

		// Just to run destructors in a nice order
		virtual bool onAppTerminated() override
		{
			m_utils = nullptr;
			return base_t::onAppTerminated();
		}

		// overridable for future graphics queue using examples
		virtual bool isComputeOnly() const {return true;}

		using queue_flags_t = video::IQueue::FAMILY_FLAGS;
		// So because of lovely Intel GPUs that only have one queue, we can't really request anything different
		virtual core::vector<queue_req_t> getQueueRequirements() const override
		{
			queue_req_t singleQueueReq = {.requiredFlags=queue_flags_t::COMPUTE_BIT|queue_flags_t::TRANSFER_BIT,.disallowedFlags=queue_flags_t::NONE,.queueCount=1,.maxImageTransferGranularity={1,1,1}};
			if (!isComputeOnly())
				singleQueueReq.requiredFlags |= queue_flags_t::GRAPHICS_BIT;
			return {singleQueueReq};
		}

		// Their allocation and creation gets more complex
		class QueueAllocator final
		{
			public:
				QueueAllocator() = default;
				QueueAllocator(const queue_family_range_t& familyProperties) : m_remainingQueueCounts(familyProperties.size()), m_familyProperties(familyProperties)
				{
					for (uint8_t i=0u; i<m_familyProperties.size(); i++)
						m_remainingQueueCounts[i] = m_familyProperties[i].queueCount;
				}

				constexpr static inline uint8_t InvalidIndex = 0xff;

				// A little utility to try-allocate N queues from the same (but yet unknown) family
				// the unwantedFlags are listed in order of severity, most unwanted first
				uint8_t allocateFamily(const queue_req_t& originalReq, std::initializer_list<queue_flags_t> unwantedFlags)
				{
					for (auto flagCount=static_cast<int32_t>(unwantedFlags.size()); flagCount>=0; flagCount--)
					{
						queue_req_t req = originalReq;
						for (auto it=unwantedFlags.begin(); it!=(unwantedFlags.begin()+flagCount); it++)
							req.disallowedFlags |= *it;
						for (uint8_t i=0u; i<m_familyProperties.size(); i++)
						{
							if (req.familyMatches(m_familyProperties[i]) && m_remainingQueueCounts[i]>=req.queueCount)
							{
								m_remainingQueueCounts[i] -= req.queueCount;
								return i;
							}
						}
						// log compromises?
					}
					return InvalidIndex;
				}

				// to allow try-allocs
				inline void freeQueues(const uint8_t famIx, const uint8_t count)
				{
					assert(famIx<m_remainingQueueCounts.size());
					assert(count<=m_familyProperties[famIx].queueCount);
					m_remainingQueueCounts[famIx] += count;
				}

			private:
				core::vector<uint8_t> m_remainingQueueCounts;
				const queue_family_range_t& m_familyProperties;
		};

		virtual std::array<video::ILogicalDevice::SQueueCreationParams,video::ILogicalDevice::MaxQueueFamilies> getQueueCreationParameters(const queue_family_range_t& familyProperties) override
		{
			QueueAllocator queueAllocator(familyProperties);

			// This is a sort-of allocator of queue indices for distinct queues
			core::map<uint8_t,uint8_t> familyQueueCounts;

			// If we can't allocate any queue then we can alias to this
			uint8_t fallbackUsers = 0;
			const SQueueIndex fallbackQueue = {
				queueAllocator.allocateFamily({
						.requiredFlags = (isComputeOnly() ? queue_flags_t::NONE : queue_flags_t::GRAPHICS_BIT) | queue_flags_t::COMPUTE_BIT | queue_flags_t::TRANSFER_BIT,
						.disallowedFlags = queue_flags_t::NONE,
						.queueCount = 1,
						.maxImageTransferGranularity = {1,1,1}
					},
					{}
				),
				0
			};
			// Since we requested a device that has a compute capable queue family (unless `getQueueRequirements` got overriden) we're sure we'll get at least one family capable of Compute and Graphics (if not headless).
			assert(fallbackQueue.famIx!=QueueAllocator::InvalidIndex);
			// It's okay that the Fallback Queue is taking up space in the allocator, we need a Compute Queue with {1,1,1} granularity anyway
			familyQueueCounts[fallbackQueue.famIx]++;

			// Make sure we have a Compute Queue as we'll always need that
			queue_req_t computeQueueRequirement = {
				.requiredFlags = queue_flags_t::COMPUTE_BIT,
				.disallowedFlags = queue_flags_t::NONE,
				.queueCount = 1
			};
			// If we won't have a Graphics Queue, then we need our Compute Queue to be able to do transfers of any granularity
			if (isComputeOnly())
				computeQueueRequirement.maxImageTransferGranularity = {1,1,1};
			// Ideally we want the Compute Queue to be without Graphics and Transfer Capability
			// However we'll take a Graphics & Compute queue if there's room in those families for one extra aside from the Fallback.
			m_computeQueue.famIx = queueAllocator.allocateFamily(computeQueueRequirement,{queue_flags_t::GRAPHICS_BIT,queue_flags_t::TRANSFER_BIT,queue_flags_t::PROTECTED_BIT});
			// If no async Compute Queue exists, we'll alias it to the fallback queue
			if (m_computeQueue.famIx!=QueueAllocator::InvalidIndex)
				m_computeQueue.qIx = familyQueueCounts[m_computeQueue.famIx]++;
			else
			{
				// Going through this branch means no more queues can be allocated, even on the same family as Fallback. We have to alias.
				m_logger->log("Not enough queue counts in families, had to alias the Compute Queue to Fallback!", system::ILogger::ELL_PERFORMANCE);
				m_computeQueue = fallbackQueue;
				fallbackUsers++;
			}
			
			// Next we'll try to allocate the transfer queues from families don't support Graphics or Compute capabilities at all to ensure they're the DMA queues and not clogging up the main CP
			{
				constexpr queue_req_t TransferQueueRequirement = {.requiredFlags=queue_flags_t::TRANSFER_BIT,.disallowedFlags=queue_flags_t::NONE,.queueCount=1};
				// However if we can't get Queue Families without Graphics or Compute then we'll take whatever space is left so we can at least Async
				m_transferUpQueue.famIx = queueAllocator.allocateFamily(TransferQueueRequirement,{queue_flags_t::GRAPHICS_BIT,queue_flags_t::COMPUTE_BIT,queue_flags_t::PROTECTED_BIT});
				// In my opinion the Asynchronicity of the Upload queue is more important, so we assigned that first.
				// We don't need to do anything special to ensure the down transfer queue allocates on the same family as the up transfer queue
				m_transferDownQueue.famIx = queueAllocator.allocateFamily(TransferQueueRequirement,{queue_flags_t::GRAPHICS_BIT,queue_flags_t::COMPUTE_BIT,queue_flags_t::PROTECTED_BIT});
			}

			// If failed to allocate up-transfer queue, then alias it to the fallback or compute queue
			if (m_transferUpQueue.famIx!=QueueAllocator::InvalidIndex)
				m_transferUpQueue.qIx = familyQueueCounts[m_transferUpQueue.famIx]++;
			else
			{
				m_logger->log("Not enough queue counts in families, have to alias the Transfer-Up Queue!",system::ILogger::ELL_PERFORMANCE);
				// If we have a compute queue the we'll use it for Async, so best to use that for transfer, if it has a transfer capability
				if (m_physicalDevice->getQueueFamilyProperties()[m_computeQueue.famIx].queueFlags.hasFlags(queue_flags_t::TRANSFER_BIT))
				{
					m_logger->log("Aliasing Transfer-Up Queue to Compute!",system::ILogger::ELL_PERFORMANCE);
					m_transferUpQueue = m_computeQueue;
				}
				else // go to fallback
				{
					m_transferUpQueue = fallbackQueue;
					fallbackUsers++;
				}
			}

			// Failed to allocate down-transfer queue, then alias it to the up-transfer
			if (m_transferDownQueue.famIx!=QueueAllocator::InvalidIndex)
				m_transferDownQueue.qIx = familyQueueCounts[m_transferDownQueue.famIx]++;
			else
			{
				m_logger->log("Not enough queue counts in families, had to alias the Transfer-Up Queue to Transfer-Down!",system::ILogger::ELL_PERFORMANCE);
				m_transferDownQueue.famIx = m_transferUpQueue.famIx;
			}

			// Try to allocate a different queue than fallback for graphics and as little extra as possible
			if (!isComputeOnly())
			{
				queue_req_t graphicsQueueRequirement = {.requiredFlags=queue_flags_t::GRAPHICS_BIT,.disallowedFlags=queue_flags_t::NONE,.queueCount=1,.maxImageTransferGranularity={1,1,1}};
				m_graphicsQueue.famIx = queueAllocator.allocateFamily(graphicsQueueRequirement,{queue_flags_t::TRANSFER_BIT,queue_flags_t::SPARSE_BINDING_BIT,queue_flags_t::COMPUTE_BIT,queue_flags_t::PROTECTED_BIT});
				if (m_graphicsQueue.famIx!=QueueAllocator::InvalidIndex)
					m_graphicsQueue.qIx = familyQueueCounts[m_graphicsQueue.famIx]++;
				else
				{
					m_graphicsQueue = fallbackQueue;
					fallbackUsers++;
				}
			}

			// If the fallback queue has no users we can get rid of it
			if (fallbackUsers==0)
			{
				auto found = familyQueueCounts.find(fallbackQueue.famIx);
				if ((--found->second)==0)
					familyQueueCounts.erase(found);
				else
				{
					if (m_computeQueue.famIx==fallbackQueue.famIx)
						m_computeQueue.qIx--;
					if (m_transferUpQueue.famIx==fallbackQueue.famIx)
						m_transferUpQueue.qIx--;
					if (m_transferDownQueue.famIx==fallbackQueue.famIx)
						m_transferDownQueue.qIx--;
					if (m_graphicsQueue.famIx==fallbackQueue.famIx)
						m_graphicsQueue.qIx--;
				}
			}

			// Now after assigning all queues to families and indices, collate the creation parameters
			std::array<video::ILogicalDevice::SQueueCreationParams,video::ILogicalDevice::MaxQueueFamilies> retval = {};
			for (const auto& familyAndCount : familyQueueCounts)
				retval[familyAndCount.first].count = familyAndCount.second;
			return retval;
		}


		core::smart_refctd_ptr<video::IUtilities> m_utils;

	private:
		struct SQueueIndex
		{
			uint8_t famIx=QueueAllocator::InvalidIndex;
			uint8_t qIx=0;
		};
		SQueueIndex m_graphicsQueue={},m_computeQueue={},m_transferUpQueue={},m_transferDownQueue={};
};

}

#endif // _CAMERA_IMPL_