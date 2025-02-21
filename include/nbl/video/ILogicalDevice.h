#ifndef _NBL_VIDEO_I_LOGICAL_DEVICE_H_INCLUDED_
#define _NBL_VIDEO_I_LOGICAL_DEVICE_H_INCLUDED_

#include "nbl/asset/asset.h"
#include "nbl/asset/utils/ISPIRVOptimizer.h"
#include "nbl/asset/utils/CCompilerSet.h"

#include "nbl/video/SPhysicalDeviceFeatures.h"
#include "nbl/video/SPhysicalDeviceLimits.h"
#include "nbl/video/IDeferredOperation.h"
#include "nbl/video/IDeviceMemoryAllocator.h"
#include "nbl/video/IGPUPipelineCache.h"
#include "nbl/video/IGPUCommandBuffer.h"
#include "nbl/video/CThreadSafeQueueAdapter.h"
#include "nbl/video/CJITIncludeLoader.h"

#include "git_info.h"
#define NBL_LOG_FUNCTION m_logger.log
#include "nbl/logging_macros.h"

namespace nbl::video
{

class IPhysicalDevice;

class NBL_API2 ILogicalDevice : public core::IReferenceCounted, public IDeviceMemoryAllocator
{
    public:
        constexpr static inline uint8_t MaxQueueFamilies = 7;
        struct SQueueCreationParams
        {
            constexpr static inline uint8_t MaxQueuesInFamily = 15;

            core::bitflag<IQueue::CREATE_FLAGS> flags = IQueue::CREATE_FLAGS::NONE;
            uint8_t count : 4/*log2(MaxQueuesInFamily+1)*/ = 0;
            std::array<float,MaxQueuesInFamily> priorities = []()->auto{
                std::array<float,MaxQueuesInFamily> retval;retval.fill(IQueue::DEFAULT_QUEUE_PRIORITY);return retval;
            }();
        };
        struct SCreationParams
        {
            std::array<SQueueCreationParams,MaxQueueFamilies> queueParams = {};
            SPhysicalDeviceFeatures featuresToEnable = {};
            core::smart_refctd_ptr<asset::CCompilerSet> compilerSet = nullptr;
        };


        //! Basic getters
        inline const IPhysicalDevice* getPhysicalDevice() const { return m_physicalDevice; }

        inline const SPhysicalDeviceFeatures& getEnabledFeatures() const { return m_enabledFeatures; }

        E_API_TYPE getAPIType() const;

        //
        inline uint8_t getQueueCount(const uint32_t _familyIx) const
        {
            if (_familyIx<MaxQueueFamilies)
                return m_queueFamilyInfos[_familyIx].queueCount;
            return 0u;
        }

        inline IQueue* getQueue(uint32_t _familyIx, uint32_t _ix)
        {
            if (auto tsq=getThreadSafeQueue(_familyIx,_ix); tsq)
                return tsq->getUnderlyingQueue();
            return nullptr;
        }

        // Using the same queue as both a threadsafe queue and a normal queue invalidates the safety.
        inline CThreadSafeQueueAdapter* getThreadSafeQueue(uint32_t _familyIx, uint32_t _ix)
        {
            if (getQueueCount(_familyIx)==0)
                return nullptr;
            const uint32_t offset = m_queueFamilyInfos[_familyIx].firstQueueIndex;
            return (*m_queues)[offset+_ix];
        }


        //! sync validation
        inline core::bitflag<asset::PIPELINE_STAGE_FLAGS> getSupportedStageMask(const uint32_t queueFamilyIndex) const
        {
            if (getQueueCount(queueFamilyIndex)!=0)
                return m_queueFamilyInfos[queueFamilyIndex].supportedStages;
            return asset::PIPELINE_STAGE_FLAGS::NONE;
        }
        //! Use this to validate instead of `getSupportedStageMask(queueFamilyIndex)&stageMask`, it checks special values
        bool supportsMask(const uint32_t queueFamilyIndex, core::bitflag<asset::PIPELINE_STAGE_FLAGS> stageMask) const;
        
        inline core::bitflag<asset::ACCESS_FLAGS> getSupportedAccessMask(const uint32_t queueFamilyIndex) const
        {
            if (getQueueCount(queueFamilyIndex)!=0)
                return m_queueFamilyInfos[queueFamilyIndex].supportedAccesses;
            return asset::ACCESS_FLAGS::NONE;
        }
        //! Use this to validate instead of `getSupportedAccessMask(queueFamilyIndex)&accessMask`, it checks special values
        bool supportsMask(const uint32_t queueFamilyIndex, core::bitflag<asset::ACCESS_FLAGS> accessMask) const;

        //! NOTE/TODO: this is not yet finished
        bool validateMemoryBarrier(const uint32_t queueFamilyIndex, asset::SMemoryBarrier barrier) const;
        inline bool validateMemoryBarrier(const uint32_t queueFamilyIndex, const IGPUCommandBuffer::SOwnershipTransferBarrier& barrier, const bool concurrentSharing) const
        {
            // implicitly satisfied by our API:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkBufferMemoryBarrier2-srcQueueFamilyIndex-04087
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04070
            if (barrier.otherQueueFamilyIndex!=IQueue::FamilyIgnored)
            {
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkBufferMemoryBarrier2-srcStageMask-03851
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-srcStageMask-03854
                constexpr auto HostBit = asset::PIPELINE_STAGE_FLAGS::HOST_BIT;
                if (barrier.dep.srcStageMask.hasFlags(HostBit) || barrier.dep.dstStageMask.hasFlags(HostBit))
                {
                    NBL_LOG_ERROR("Invalid barrier, ownership transfer with host is not allowed");
                    return false;
                }

                // Will not check because it would involve a search:
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkBufferMemoryBarrier2-srcQueueFamilyIndex-04088
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkBufferMemoryBarrier2-srcQueueFamilyIndex-04089
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-image-04071
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-image-04072
            }
            return validateMemoryBarrier(queueFamilyIndex,barrier.dep);
        }
        template<typename ResourceBarrier>
        inline bool validateMemoryBarrier(const uint32_t queueFamilyIndex, const IGPUCommandBuffer::SBufferMemoryBarrier<ResourceBarrier>& barrier) const
        {
            const auto& range = barrier.range;
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkBufferMemoryBarrier2-buffer-parameter
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkBufferMemoryBarrier2-offset-01188
            if (!range.buffer || range.size == 0u) 
            {
                NBL_LOG_ERROR("No buffer was specified");
                return false;
            }
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkBufferMemoryBarrier2-offset-01187
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkBufferMemoryBarrier2-offset-01189
            const size_t remain = range.size!=IGPUCommandBuffer::SBufferMemoryBarrier<ResourceBarrier>{}.range.size ? range.size:1ull;
            if (range.offset+remain>range.buffer->getSize())
            {
                NBL_LOG_ERROR("Invalid range was specified");
                return false;
            }

            if constexpr(std::is_same_v<IGPUCommandBuffer::SOwnershipTransferBarrier,ResourceBarrier>)
                return validateMemoryBarrier(queueFamilyIndex,barrier.barrier,range.buffer->getCachedCreationParams().isConcurrentSharing());
            else
                return validateMemoryBarrier(queueFamilyIndex,barrier.barrier);
        }
        template<typename ResourceBarrier>
        bool validateMemoryBarrier(const uint32_t queueFamilyIndex, const IGPUCommandBuffer::SImageMemoryBarrier<ResourceBarrier>& barrier) const;

        
        //! Very important function, without it being called at the end of when you use a device, it will leak due to circular reference from resources used in the very last submit to a queue
        //! Alternatively to get rid of circular refs, you can call `waitIdle` individually on every queue you've ever used
        IQueue::RESULT waitIdle();

        //! Semaphore Stuff
        virtual core::smart_refctd_ptr<ISemaphore> createSemaphore(const uint64_t initialValue) = 0;
        // Waits for max timeout amout of time for the semaphores to reach a specific counter value
        // DOES NOT implicitly trigger Queue-refcount-resource release because of two reasons:
        // - the events may trigger loads of resource releases causing extra processing, whereas our `timeout` could be quite small
        // - the event handlers use `waitForSemaphores` themselves so don't want infinite recursion here
        virtual ISemaphore::WAIT_RESULT waitForSemaphores(const std::span<const ISemaphore::SWaitInfo> infos, const bool waitAll, const uint64_t timeout) = 0;
        // Forever waiting variant if you're confident that the fence will eventually be signalled
        // Does implicitly trigger Queue-refcount-resource-release
        inline ISemaphore::WAIT_RESULT blockForSemaphores(const std::span<const ISemaphore::SWaitInfo> infos, const bool waitAll=true)
        {
            using retval_t = ISemaphore::WAIT_RESULT;
            if (!infos.empty())
            {
                auto waitStatus = retval_t::TIMEOUT;
                while (waitStatus==retval_t::TIMEOUT)
                {
                    waitStatus = waitForSemaphores(infos,waitAll,999999999ull);
                    for (const auto& info : infos)
                    for (const auto& queue : *m_queues)
                        queue->cullResources(info.semaphore);
                }
                if (waitStatus == retval_t::DEVICE_LOST)
                {
                    NBL_LOG_ERROR("Device lost");
                    _NBL_DEBUG_BREAK_IF(true);
                }

                return waitStatus;
            }
            return retval_t::SUCCESS;
        }

        //! Event Stuff
        virtual core::smart_refctd_ptr<IEvent> createEvent(const IEvent::CREATE_FLAGS flags) = 0;

        //! ..
        virtual core::smart_refctd_ptr<IDeferredOperation> createDeferredOperation() = 0;


        //! Similar to VkMappedMemoryRange but no pNext
        struct MappedMemoryRange
        {
            MappedMemoryRange() : memory(nullptr), range{} {}
            MappedMemoryRange(IDeviceMemoryAllocation* mem, const size_t& off, const size_t& len) : memory(mem), range{off,len} {}

            inline bool valid() const
            {
                if (length==0ull || !memory)
                    return false;
                if (offset+length<length) // integer wraparound check
                    return false;
                if (offset+length>memory->getAllocationSize())
                    return false;
                return true;
            }

            IDeviceMemoryAllocation* memory;
            union
            {
                IDeviceMemoryAllocation::MemoryRange range;
                struct
                {
                    size_t offset;
                    size_t length;
                };
            };
        };
        // For memory allocations without the video::IDeviceMemoryAllocation::EMCF_COHERENT mapping capability flag you need to call this for the CPU writes to become GPU visible
        inline bool flushMappedMemoryRanges(uint32_t memoryRangeCount, const MappedMemoryRange* pMemoryRanges)
        {
            core::SRange<const MappedMemoryRange> ranges{ pMemoryRanges, pMemoryRanges + memoryRangeCount };
            return flushMappedMemoryRanges(ranges);
        }
        // Utility wrapper for the pointer based func
        inline bool flushMappedMemoryRanges(const std::span<const MappedMemoryRange>& ranges)
        {
            if (invalidMappedRanges(ranges))
            {
                NBL_LOG_ERROR("Invalid memory range");
                return false;
            }
            return flushMappedMemoryRanges_impl(ranges);
        }
        // For memory allocations without the video::IDeviceMemoryAllocation::EMCF_COHERENT mapping capability flag you need to call this for the GPU writes to become CPU visible
        inline bool invalidateMappedMemoryRanges(const uint32_t rangeCount, const MappedMemoryRange* pMemoryRanges)
        {
            return invalidateMappedMemoryRanges({pMemoryRanges,rangeCount});
        }
        //! Utility wrapper for the pointer based func
        inline bool invalidateMappedMemoryRanges(const std::span<const MappedMemoryRange> ranges)
        {
            if (invalidMappedRanges(ranges))
            {
                NBL_LOG_ERROR("Invalid memory range");
                return false;
            }
            return invalidateMappedMemoryRanges_impl(ranges);
        }
        //!

        //! Memory binding
        struct SBindBufferMemoryInfo
        {
            IGPUBuffer* buffer = nullptr;
            IDeviceMemoryBacked::SMemoryBinding binding = {};
        };
        // Binds memory allocation to provide the backing for the resource.
        /** Available only on Vulkan, in other API backends all resources create their own memory implicitly,
        so pooling or aliasing memory for different resources is not possible.
        There is no unbind, so once memory is bound it remains bound until you destroy the resource object.
        Actually all resource classes in other APIs implement both IDeviceMemoryBacked and IDeviceMemoryAllocation,
        so effectively the memory is pre-bound at the time of creation.
        \return true on success*/
        inline bool bindBufferMemory(const uint32_t count, const SBindBufferMemoryInfo* pBindInfos)
        {
            for (auto i=0u; i<count; i++)
            {
                auto& info = pBindInfos[i];
                if (!info.buffer)
                    return false;

                // TODO: @Cyprian other validation against device limits, esp deduction of max alignment from usages
                const size_t alignment = alignof(uint32_t);
                if (invalidAllocationForBind(info.buffer,info.binding,alignment))
                    return false;

                if (info.buffer->getCreationParams().usage.hasFlags(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) &&
                    !info.binding.memory->getAllocateFlags().hasFlags(IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT)
                )
                {
                    m_logger.log("Buffer %p created with EUF_SHADER_DEVICE_ADDRESS_BIT needs a Memory Allocation with EMAF_DEVICE_ADDRESS_BIT flag!",system::ILogger::ELL_ERROR,info.buffer);
                    return false;
                }
            }
            return bindBufferMemory_impl(count,pBindInfos);
        }
        //!
        struct SBindImageMemoryInfo
        {
            IGPUImage* image = nullptr;
            IDeviceMemoryBacked::SMemoryBinding binding = {};
        };
        //! The counterpart of @see bindBufferMemory for images
        [[deprecated]] inline bool bindImageMemory(uint32_t count, const SBindImageMemoryInfo* pBindInfos)
        {
            for (auto i=0u; i<count; i++)
            {
                auto& info = pBindInfos[i];
                if (!info.image)
                    return false;

                // TODO: @Cyprian other validation against device limits, esp deduction of max alignment from usages
                const size_t alignment = alignof(uint32_t);
                if (invalidAllocationForBind(info.image,info.binding,alignment))
                    return false;
            }
            return bindImageMemory_impl(count,pBindInfos);
        }
        inline bool bindImageMemory(const std::span<const SBindImageMemoryInfo> bindInfos)
        {
            return bindImageMemory(bindInfos.size(),bindInfos.data());
        }

        //! Descriptor Creation
        // Buffer (@see ICPUBuffer)
        inline core::smart_refctd_ptr<IGPUBuffer> createBuffer(IGPUBuffer::SCreationParams&& creationParams)
        {
            const auto maxSize = getPhysicalDeviceLimits().maxBufferSize;
            if (creationParams.size>maxSize)
            {
                m_logger.log("Failed to create Buffer, size %d larger than Device %p's limit!",system::ILogger::ELL_ERROR,creationParams.size,this,maxSize);
                return nullptr;
            }
            return createBuffer_impl(std::move(creationParams));
        }
        // Create a BufferView, to a shader; a fake 1D-like texture with no interpolation (@see ICPUBufferView)
        core::smart_refctd_ptr<IGPUBufferView> createBufferView(const asset::SBufferRange<const IGPUBuffer>& underlying, const asset::E_FORMAT _fmt);
        // Creates an Image (@see ICPUImage)
        inline core::smart_refctd_ptr<IGPUImage> createImage(IGPUImage::SCreationParams&& creationParams)
        {
            if (!IGPUImage::validateCreationParameters(creationParams))
            {
                m_logger.log("Failed to create Image, invalid creation parameters!",system::ILogger::ELL_ERROR);
                return nullptr;
            }
            // TODO: @Cyprian validation of creationParams against the device's limits (sample counts, etc.) see vkCreateImage
            return createImage_impl(std::move(creationParams));
        }
        // Create an ImageView that can actually be used by shaders (@see ICPUImageView)
        inline core::smart_refctd_ptr<IGPUImageView> createImageView(IGPUImageView::SCreationParams&& params)
        {
            if (!params.image->wasCreatedBy(this))
            {
                NBL_LOG_ERROR("The image was not created by this device");
                return nullptr;
            }
            // TODO: @Cyprian validation of params against the device's limits (sample counts, etc.) see vkCreateImage
            return createImageView_impl(std::move(params));
        }
        // Create a sampler object to use with ImageViews
        virtual core::smart_refctd_ptr<IGPUSampler> createSampler(const IGPUSampler::SParams& _params) = 0;
        // acceleration structures
        inline core::smart_refctd_ptr<IGPUBottomLevelAccelerationStructure> createBottomLevelAccelerationStructure(IGPUAccelerationStructure::SCreationParams&& params)
        {
            if (invalidCreationParams(params))
            {
                NBL_LOG_ERROR("Invalid creation parameters");
                return nullptr;
            }
            return createBottomLevelAccelerationStructure_impl(std::move(params));
        }
        inline core::smart_refctd_ptr<IGPUTopLevelAccelerationStructure> createTopLevelAccelerationStructure(IGPUTopLevelAccelerationStructure::SCreationParams&& params)
        {
            if (invalidCreationParams(params))
            {
                NBL_LOG_ERROR("Invalid creation parameters");
                return nullptr;
            }
            if (params.flags.hasFlags(IGPUAccelerationStructure::SCreationParams::FLAGS::MOTION_BIT) && (params.maxInstanceCount == 0u || params.maxInstanceCount > getPhysicalDeviceLimits().maxAccelerationStructureInstanceCount))
            {
                NBL_LOG_ERROR("Invalid creation parameters");
                return nullptr;
            }
            return createTopLevelAccelerationStructure_impl(std::move(params));
        }

        //! Acceleration Structure modifiers
        //
        struct AccelerationStructureBuildSizes
        {
            inline operator bool() const { return accelerationStructureSize!=(~0ull); }

            size_t accelerationStructureSize = ~0ull;
            size_t updateScratchSize = ~0ull;
            size_t buildScratchSize = ~0ull;
        };
        // fun fact: you can use garbage/invalid pointers/offset for the Device/Host addresses of the per-geometry data, just make sure what was supposed to be null is null
        template<class Geometry> requires nbl::is_any_of_v<Geometry,
            IGPUBottomLevelAccelerationStructure::Triangles<const IGPUBuffer>,
            IGPUBottomLevelAccelerationStructure::Triangles<const asset::ICPUBuffer>,
            IGPUBottomLevelAccelerationStructure::AABBs<const IGPUBuffer>,
            IGPUBottomLevelAccelerationStructure::AABBs<const asset::ICPUBuffer>
        >
        inline AccelerationStructureBuildSizes getAccelerationStructureBuildSizes(
            const core::bitflag<IGPUBottomLevelAccelerationStructure::BUILD_FLAGS> flags,
            const bool motionBlur,
            const std::span<Geometry> geometries,
            const uint32_t* const pMaxPrimitiveCounts
        ) const
        {
            if (invalidFeaturesForASBuild<typename Geometry::buffer_t>(motionBlur))
            {
                NBL_LOG_ERROR("Required features are not enabled");
                return {};
            }

            if (!IGPUBottomLevelAccelerationStructure::validBuildFlags(flags, m_enabledFeatures))
            {
                NBL_LOG_ERROR("Invalid build flags");
                return {};
            }

            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkGetAccelerationStructureBuildSizesKHR-pBuildInfo-03619
            if (geometries.empty() && !pMaxPrimitiveCounts)
            {
                NBL_LOG_ERROR("Invalid parameters, no geometry or primitives were specified");
                return {};
            }

            const auto& limits = getPhysicalDeviceLimits();
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03793
            if (geometries.size() > limits.maxAccelerationStructureGeometryCount)
            {
                NBL_LOG_ERROR("Geometry count exceeds device limit");
                return {};
            }

            // not sure of VUID
            uint32_t primsFree = limits.maxAccelerationStructurePrimitiveCount;
			for (auto i=0u; i<geometries.size(); i++)
            {
                if (pMaxPrimitiveCounts[i] > primsFree)
                {
                    NBL_LOG_ERROR("Primitive count exceeds device limit");
				    return {};
                }
                primsFree -= pMaxPrimitiveCounts[i];
            }

            return getAccelerationStructureBuildSizes_impl(flags,motionBlur,geometries,pMaxPrimitiveCounts);
        }
        inline AccelerationStructureBuildSizes getAccelerationStructureBuildSizes(
            const bool hostBuild,
            const core::bitflag<IGPUTopLevelAccelerationStructure::BUILD_FLAGS> flags,
            const bool motionBlur,
            const uint32_t maxInstanceCount
        ) const
        {
            if (invalidFeaturesForASBuild<IGPUBuffer>(motionBlur))
            {
                NBL_LOG_ERROR("Required features are not enabled");
                return {};
            }

            if (!IGPUTopLevelAccelerationStructure::validBuildFlags(flags))
            {
                NBL_LOG_ERROR("Invalid build flags");
                return {};
            }

            const auto& limits = getPhysicalDeviceLimits();
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkGetAccelerationStructureBuildSizesKHR-pBuildInfo-03785
            if (maxInstanceCount > limits.maxAccelerationStructureInstanceCount)
            {
                NBL_LOG_ERROR("maxInstanceCount exceeds device limits");
                return {};
            }

            return getAccelerationStructureBuildSizes_impl(hostBuild,flags,motionBlur,maxInstanceCount);
        }
        // little utility
        template<typename BufferType=IGPUBuffer>
        inline AccelerationStructureBuildSizes getAccelerationStructureBuildSizes(const core::bitflag<IGPUTopLevelAccelerationStructure::BUILD_FLAGS> flags, const bool motionBlur, const uint32_t maxInstanceCount) const
        {
            return getAccelerationStructureBuildSizes(std::is_same_v<std::remove_cv_t<BufferType>,asset::ICPUBuffer>,flags,motionBlur,maxInstanceCount);
        }

        //
        inline bool invalidAccelerationStructureForHostOperations(const IGPUAccelerationStructure* const as) const
        {
            if (!as)
                return true;
            const auto* memory = as->getCreationParams().bufferRange.buffer->getBoundMemory().memory;
            if (invalidMemoryForAccelerationStructureHostOperations(memory))
                return true;
            if (!memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_CACHED_BIT))
                m_logger.log("Acceleration Structures manipulated using Host Commands should always be bound to host cached memory, as the implementation may need to repeatedly read and write this memory during the execution of the command.",system::ILogger::ELL_PERFORMANCE);
            return false;
        }
        template<class AccelerationStructure> requires std::is_base_of_v<IGPUAccelerationStructure,AccelerationStructure>
        inline bool buildAccelerationStructures(
            IDeferredOperation* const deferredOperation, const std::span<const typename AccelerationStructure::HostBuildInfo> infos,
            const typename AccelerationStructure::DirectBuildRangeRangeInfos pDirectBuildRangeRangeInfos
        )
        {
            if (!acquireDeferredOperation(deferredOperation))
            {
                NBL_LOG_ERROR("Couldn't acquire deferred operation");
                return false;
            }

            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-accelerationStructureHostCommands-03581
            if (!m_enabledFeatures.accelerationStructureHostCommands)
            {
                NBL_LOG_ERROR("Feature `acceleration structure host commands` is not enabled");
                return false;
            }

            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-infoCount-arraylength
            if (infos.empty())
            {
                NBL_LOG_ERROR("Invalid parameters, infos must not be empty");
                return false;
            }

            uint32_t trackingReservation = 0; 
            uint32_t totalGeometryCount = 0; 
            for (auto i=0u; i<infos.size(); i++)
            {
                const auto toTrack = infos[i].valid(pDirectBuildRangeRangeInfos[i]);
                if (!toTrack)
                {
                    NBL_LOG_ERROR("Invalid info structure (infos[%u]) was given", i);
                    return false;
                }
                trackingReservation += toTrack;
                totalGeometryCount += infos[i].inputCount();
            }
            
            auto result = buildAccelerationStructures_impl(deferredOperation,infos,pDirectBuildRangeRangeInfos,totalGeometryCount);
            // track things created
            if (result==DEFERRABLE_RESULT::DEFERRED)
            {
                auto& tracking = deferredOperation->m_resourceTracking;
                tracking.resize(trackingReservation);
                auto oit = tracking.data();
                for (const auto& info : infos)
                    oit = info.fillTracking(oit);
            }
            return result!=DEFERRABLE_RESULT::SOME_ERROR;
        }
        // write out props, the length of the bugger pointed to by `data` must be `>=count*stride`
        inline bool writeAccelerationStructuresProperties(const std::span<const IGPUAccelerationStructure* const> accelerationStructures, const IQueryPool::TYPE type, size_t* data, const size_t stride=alignof(size_t))
        {
            if (stride < sizeof(size_t) || !core::is_aligned_to(stride, alignof(size_t)))
            {
                NBL_LOG_ERROR("Invalid stride, must be a multiply of (%llu-byte) alignment", alignof(size_t));
                return false;
            }
            switch (type)
            {
                case IQueryPool::TYPE::ACCELERATION_STRUCTURE_COMPACTED_SIZE: [[fallthrough]];
                case IQueryPool::TYPE::ACCELERATION_STRUCTURE_SERIALIZATION_SIZE: [[fallthrough]];
                case IQueryPool::TYPE::ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS: [[fallthrough]];
                case IQueryPool::TYPE::ACCELERATION_STRUCTURE_SIZE:
                    break;
                default:
                    NBL_LOG_ERROR("Invalid query pool type");
                    return false;
                    break;
            }
            // https://vulkan.lunarg.com/doc/view/1.3.290.0/windows/1.3-extensions/vkspec.html#VUID-vkWriteAccelerationStructuresPropertiesKHR-accelerationStructureHostCommands-03585
            if (!getEnabledFeatures().accelerationStructureHostCommands)
            {
                NBL_LOG_ERROR("Feature `acceleration structure` host commands is not enabled");
                return false;
            }
            // https://vulkan.lunarg.com/doc/view/1.3.290.0/windows/1.3-extensions/vkspec.html#VUID-vkWriteAccelerationStructuresPropertiesKHR-buffer-03733
            for (const auto& as : accelerationStructures)
            if (invalidAccelerationStructureForHostOperations(as))
            {
                NBL_LOG_ERROR("Invalid acceleration structure for host operations");
                return false;
            }
            // unfortunately cannot validate if they're built and if they're built with the right flags
            return writeAccelerationStructuresProperties_impl(accelerationStructures,type,data,stride);
        }
        // Host-side copy, DEFERRAL IS NOT OPTIONAL
        inline bool copyAccelerationStructure(IDeferredOperation* const deferredOperation, const IGPUAccelerationStructure::CopyInfo& copyInfo)
        {
            if (!acquireDeferredOperation(deferredOperation))
            {
                NBL_LOG_ERROR("Couldn't acquire deferred operation");
                return false;
            }
            if (invalidAccelerationStructureForHostOperations(copyInfo.src) || invalidAccelerationStructureForHostOperations(copyInfo.dst))
            {
                NBL_LOG_ERROR("Invalid Acceleration Structure for host operations");
                return false;
            }
            auto result = copyAccelerationStructure_impl(deferredOperation,copyInfo);
            if (result==DEFERRABLE_RESULT::DEFERRED)
                deferredOperation->m_resourceTracking.insert(deferredOperation->m_resourceTracking.begin(),{
                    core::smart_refctd_ptr<const IReferenceCounted>(copyInfo.src),
                    core::smart_refctd_ptr<const IReferenceCounted>(copyInfo.dst)
                });
            

            return result!=DEFERRABLE_RESULT::SOME_ERROR;
        }
        inline bool copyAccelerationStructureToMemory(IDeferredOperation* const deferredOperation, const IGPUAccelerationStructure::HostCopyToMemoryInfo& copyInfo)
        {
            if (!acquireDeferredOperation(deferredOperation))
            {
                NBL_LOG_ERROR("Couldn't acquire deferred operation");
                return false;
            }
            if (invalidAccelerationStructureForHostOperations(copyInfo.src))
            {
                NBL_LOG_ERROR("Invalid Acceleration Structure for host operations");
                return false;
            }
            if (!core::is_aligned_to(ptrdiff_t(copyInfo.dst.buffer->getPointer()) + copyInfo.dst.offset, 16u))
            {
                NBL_LOG_ERROR("Destination address is not aligned");
                return false;
            }
            auto result = copyAccelerationStructureToMemory_impl(deferredOperation,copyInfo);
            if (result==DEFERRABLE_RESULT::DEFERRED)
                deferredOperation->m_resourceTracking.insert(deferredOperation->m_resourceTracking.begin(),{
                    core::smart_refctd_ptr<const IReferenceCounted>(copyInfo.src),
                    core::smart_refctd_ptr<const IReferenceCounted>(copyInfo.dst.buffer)
                });
            return result!=DEFERRABLE_RESULT::SOME_ERROR;
        }
        inline bool copyAccelerationStructureFromMemory(IDeferredOperation* const deferredOperation, const IGPUAccelerationStructure::HostCopyFromMemoryInfo& copyInfo)
        {
            if (!acquireDeferredOperation(deferredOperation))
            {
                NBL_LOG_ERROR("Couldn't acquire deferred operation");
                return false;
            }
            if (!core::is_aligned_to(ptrdiff_t(copyInfo.src.buffer->getPointer()) + copyInfo.src.offset, 16u))
            {
                NBL_LOG_ERROR("Source address is not aligned");
                return false;
            }
            if (invalidAccelerationStructureForHostOperations(copyInfo.dst))
            {
                NBL_LOG_ERROR("Invalid acceleration structure for host operations");
                return false;
            }
            auto result = copyAccelerationStructureFromMemory_impl(deferredOperation,copyInfo);
            if (result==DEFERRABLE_RESULT::DEFERRED)
                deferredOperation->m_resourceTracking.insert(deferredOperation->m_resourceTracking.begin(),{
                    core::smart_refctd_ptr<const IReferenceCounted>(copyInfo.src.buffer),
                    core::smart_refctd_ptr<const IReferenceCounted>(copyInfo.dst)
                });
            return result!=DEFERRABLE_RESULT::SOME_ERROR;
        }


        //! Shaders
        struct SShaderCreationParameters
        {
            const asset::IShader* source;
            const asset::ISPIRVOptimizer* optimizer = nullptr;
            asset::IShaderCompiler::CCache* readCache = nullptr;
            asset::IShaderCompiler::CCache* writeCache = nullptr;
            std::span<const asset::IShaderCompiler::SMacroDefinition> extraDefines = {};
            hlsl::ShaderStage stage = hlsl::ShaderStage::ESS_ALL_OR_LIBRARY;
        };
        core::smart_refctd_ptr<asset::IShader> compileShader(const SShaderCreationParameters& creationParams);

        //! Layouts
        // Create a descriptor set layout (@see ICPUDescriptorSetLayout)
        core::smart_refctd_ptr<IGPUDescriptorSetLayout> createDescriptorSetLayout(const std::span<const IGPUDescriptorSetLayout::SBinding> bindings);
        // Create a pipeline layout (@see ICPUPipelineLayout)
        core::smart_refctd_ptr<IGPUPipelineLayout> createPipelineLayout(
            const std::span<const asset::SPushConstantRange> pcRanges={},
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0=nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1=nullptr,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2=nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3=nullptr
        )
        {
            if ((_layout0 && !_layout0->wasCreatedBy(this)))
            {
                NBL_LOG_ERROR("layout was not created by this device");
                return nullptr;
            }
            if (_layout1 && !_layout1->wasCreatedBy(this))
            {
                NBL_LOG_ERROR("layout was not created by this device");
                return nullptr;
            }
            if (_layout2 && !_layout2->wasCreatedBy(this))
            {
                NBL_LOG_ERROR("layout was not created by this device");
                return nullptr;
            }
            if (_layout3 && !_layout3->wasCreatedBy(this))
            {
                NBL_LOG_ERROR("layout was not created by this device");
                return nullptr;
            }
            // sanity check
            if (pcRanges.size() > getPhysicalDeviceLimits().maxPushConstantsSize * MaxStagesPerPipeline)
            {
                NBL_LOG_ERROR("Number of push constants ranges exceeds device limits");
                return nullptr;
            }
            core::bitflag<hlsl::ShaderStage> stages = hlsl::ShaderStage::ESS_UNKNOWN;
            uint32_t maxPCByte = 0u;
            for (auto range : pcRanges)
            {
                stages |= range.stageFlags;
                maxPCByte = core::max(range.offset+range.size,maxPCByte);
            }
            if (maxPCByte > getPhysicalDeviceLimits().maxPushConstantsSize)
            {
                NBL_LOG_ERROR("Push constants size exceeds device limit");
                return nullptr;
            }
            // TODO: validate `stages` against the supported ones as reported by enabled features
            return createPipelineLayout_impl(pcRanges,std::move(_layout0),std::move(_layout1),std::move(_layout2),std::move(_layout3));
        }

        //! Descriptor Sets
        inline core::smart_refctd_ptr<IDescriptorPool> createDescriptorPool(const IDescriptorPool::SCreateInfo& createInfo)
        {
            if (createInfo.maxSets == 0u)
            {
                NBL_LOG_ERROR("Invalid maxSets, must be greater than 0");
                return nullptr;
            }
            // its also not useful to have pools with zero descriptors
            uint32_t t = 0;
            for (; t<static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++t)
            if (createInfo.maxDescriptorCount[t])
                break;
            if (t == static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT))
            {
                NBL_LOG_ERROR("Invalid maxDescriptorCount, must greater than 0");
                return nullptr;
            }
            return createDescriptorPool_impl(createInfo);
        }
        // utility func
        inline core::smart_refctd_ptr<IDescriptorPool> createDescriptorPoolForDSLayouts(const IDescriptorPool::E_CREATE_FLAGS flags, const std::span<const IGPUDescriptorSetLayout* const> layouts, const uint32_t* setCounts=nullptr)
        {
            IDescriptorPool::SCreateInfo createInfo = {};
            createInfo.flags = flags;

            auto setCountsIt = setCounts;
            for (auto layout : layouts)
            {
                if (layout)
                {
                    // TODO: when creating the layouts, cache if they have any update after bindingings, and patch `createInfo.flags` with that here
                    const auto setCount = setCounts ? *(setCountsIt):1u;
                    createInfo.maxSets += setCount;
                    for (uint32_t t=0; t<static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++t)
                    {
                        const auto type = static_cast<asset::IDescriptor::E_TYPE>(t);
                        createInfo.maxDescriptorCount[t] += setCount*layout->getDescriptorRedirect(type).getTotalCount();
                    }
                }
                setCountsIt++;
            }
        
            return createDescriptorPool(createInfo);
        }
        // Fill out the descriptor sets with descriptors
        bool updateDescriptorSets(const std::span<const IGPUDescriptorSet::SWriteDescriptorSet> descriptorWrites, const std::span<const IGPUDescriptorSet::SCopyDescriptorSet> descriptorCopies);
        [[deprecated]] inline bool updateDescriptorSets(
            const uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* const pDescriptorWrites,
            const uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* const pDescriptorCopies
        )
        {
            return updateDescriptorSets({pDescriptorWrites,descriptorWriteCount},{pDescriptorCopies,descriptorCopyCount});
        }

        // should this be joined together with the existing updateDescriptorSets?
        bool nullifyDescriptors(const std::span<const IGPUDescriptorSet::SDropDescriptorSet> dropDescriptors);

        //! Renderpasses and Framebuffers
        core::smart_refctd_ptr<IGPURenderpass> createRenderpass(const IGPURenderpass::SCreationParams& params);
        inline core::smart_refctd_ptr<IGPUFramebuffer> createFramebuffer(IGPUFramebuffer::SCreationParams&& params)
        {
            // this validate already checks that Renderpass device creator matches with the images
            if (!params.validate())
            {
                NBL_LOG_ERROR("Invalid parameters were given");
                return nullptr;
            }

            if (params.width > getPhysicalDeviceLimits().maxFramebufferWidth ||
                params.height > getPhysicalDeviceLimits().maxFramebufferHeight ||
                params.layers > getPhysicalDeviceLimits().maxFramebufferLayers)
            {
                NBL_LOG_ERROR("Parameters exceed device limits");
                return nullptr;
            }

            if (!params.renderpass->wasCreatedBy(this))
            {
                NBL_LOG_ERROR("The renderpass was not created by this device");
                return nullptr;
            }

            // We won't support linear attachments
            auto anyNonOptimalTiling = [](const IGPUImageView* const* attachments, const uint32_t count)->bool
            {
                for (auto i=0u; i<count; i++)
                if (attachments[i]->getCreationParameters().image->getTiling()!=IGPUImage::TILING::OPTIMAL)
                    return true;
                return false;
            };
            if (anyNonOptimalTiling(params.depthStencilAttachments, params.renderpass->getDepthStencilAttachmentCount()))
            {
                NBL_LOG_ERROR("Linear attachments are not supported");
                return nullptr;
            }
            if (anyNonOptimalTiling(params.colorAttachments, params.renderpass->getColorAttachmentCount()))
            {
                NBL_LOG_ERROR("Linear attachments are not supported");
                return nullptr;
            }

            // Impossible to check:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBeginRenderPass2KHR.html#VUID-vkCmdBeginRenderPass2-initialLayout-03100
            // Not reasonably fast to check:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBeginRenderPass2KHR.html#VUID-vkCmdBeginRenderPass2-framebuffer-02533
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBeginRenderPass2KHR.html#VUID-vkCmdBeginRenderPass2-framebuffer-09046
            // Hard to check:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBeginRenderPass2KHR.html#VUID-vkCmdBeginRenderPass2-srcStageMask-06453
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBeginRenderPass2KHR.html#VUID-vkCmdBeginRenderPass2-srcStageMask-06454
            return createFramebuffer_impl(std::move(params));
        }

        //! Pipelines
        asset::ICPUPipelineCache::SCacheKey getPipelineCacheKey() const;

        // Create a pipeline cache object
        virtual core::smart_refctd_ptr<IGPUPipelineCache> createPipelineCache(const std::span<const uint8_t> initialData, const bool notThreadsafe=false) = 0;
        // utility
        inline core::smart_refctd_ptr<IGPUPipelineCache> createPipelineCache(const asset::ICPUPipelineCache* cpuCache, const bool notThreadsafe=false)
        {
            std::span<uint8_t> initialData = {};
            if (cpuCache)
            {
                const auto& entries = cpuCache->getEntries();
                auto found = entries.find(getPipelineCacheKey());
                if (found!=entries.end())
                {
                    initialData = {found->second.bin->data(),found->second.bin->size()};
                }
            }
            return createPipelineCache(initialData,notThreadsafe);
        }

        inline bool createComputePipelines(IGPUPipelineCache* const pipelineCache, const std::span<const IGPUComputePipeline::SCreationParams> params, core::smart_refctd_ptr<IGPUComputePipeline>* const output)
        {
            std::fill_n(output,params.size(),nullptr);
            IGPUComputePipeline::SCreationParams::SSpecializationValidationResult specConstantValidation = commonCreatePipelines(pipelineCache,params,[this](const IGPUShader::SSpecInfo& info)->bool
            {
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-stage-08771
                if (!info.shader->wasCreatedBy(this))
                {
                    NBL_LOG_ERROR("The shader was not created by this device");
                    return false;
                }
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-pNext-02755
                if (info.requiredSubgroupSize >= IGPUShader::SSpecInfo::SUBGROUP_SIZE::REQUIRE_4 && !getPhysicalDeviceLimits().requiredSubgroupSizeStages.hasFlags(info.shader->getStage()))
                {
                    NBL_LOG_ERROR("Invalid shader stage");
                    return false;
                }
                return true;
            });
            if (!specConstantValidation)
            {
                NBL_LOG_ERROR("Invalid parameters were given");
                return false;
            }

            createComputePipelines_impl(pipelineCache,params,output,specConstantValidation);
            
            bool retval = true;
            for (auto i=0u; i<params.size(); i++)
            {
                const char* debugName = params[i].shader.shader->getObjectDebugName();
                if (!output[i])
                {
                    NBL_LOG_ERROR("ComputeShader was not created (params[%u])" , i);
                    retval = false;
                }
                else if (debugName && debugName[0])
                    output[i]->setObjectDebugName(debugName);
            }
            return retval;
        }

        bool createGraphicsPipelines(
            IGPUPipelineCache* const pipelineCache,
            const std::span<const IGPUGraphicsPipeline::SCreationParams> params,
            core::smart_refctd_ptr<IGPUGraphicsPipeline>* const output
        );

        // queries
        inline core::smart_refctd_ptr<IQueryPool> createQueryPool(const IQueryPool::SCreationParams& params)
        {
            switch (params.queryType)
            {
                case IQueryPool::TYPE::PIPELINE_STATISTICS:
                    if (!getEnabledFeatures().pipelineStatisticsQuery)
                    {
                        NBL_LOG_ERROR("Feature `pipeline statistics` is not enabled");
                        return nullptr;
                    }
                    break;
                case IQueryPool::TYPE::ACCELERATION_STRUCTURE_COMPACTED_SIZE: [[fallthrough]];
                case IQueryPool::TYPE::ACCELERATION_STRUCTURE_SERIALIZATION_SIZE: [[fallthrough]];
                case IQueryPool::TYPE::ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS: [[fallthrough]];
                case IQueryPool::TYPE::ACCELERATION_STRUCTURE_SIZE:
                    if (!getEnabledFeatures().accelerationStructure)
                    {
                        NBL_LOG_ERROR("Feature `acceleration structure` is not enabled");
                        return nullptr;
                    }
                    break;
                case IQueryPool::TYPE::TIMESTAMP:
                    break;
                default:
                    NBL_LOG_ERROR("Unsupported query pool type");
                    return nullptr;
            }
            return createQueryPool_impl(params);
        }
        // `pData` must be sufficient to store the results (@see `IQueryPool::calcQueryResultsSize`)
        inline bool getQueryPoolResults(const IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount, void* const pData, const size_t stride, const core::bitflag<IQueryPool::RESULTS_FLAGS> flags)
        {
            if (!queryPool || !queryPool->wasCreatedBy(this))
            {
                NBL_LOG_ERROR("The queryPool was not created by this device");
                return false;
            }
            if (firstQuery + queryCount > queryPool->getCreationParameters().queryCount)
            {
                NBL_LOG_ERROR("Query index out of bounds");
                return false;
            }
            if (stride & ((flags.hasFlags(IQueryPool::RESULTS_FLAGS::_64_BIT) ? alignof(uint64_t) : alignof(uint32_t)) - 1))
            {
                auto aligment = (flags.hasFlags(IQueryPool::RESULTS_FLAGS::_64_BIT) ? alignof(uint64_t) : alignof(uint32_t));
                NBL_LOG_ERROR("Invalid stride, must be a multiply of the (%llu-byte) alignment", aligment);
                return false;
            }
            if (queryPool->getCreationParameters().queryType == IQueryPool::TYPE::TIMESTAMP && flags.hasFlags(IQueryPool::RESULTS_FLAGS::PARTIAL_BIT))
            {
                NBL_LOG_ERROR("Invalid query flag, partial results are not allowed for any timestamp pool");
                return false;
            }
            return getQueryPoolResults_impl(queryPool,firstQuery,queryCount,pData,stride,flags);
        }

        //! Commandbuffers
        inline core::smart_refctd_ptr<IGPUCommandPool> createCommandPool(const uint32_t familyIx, const core::bitflag<IGPUCommandPool::CREATE_FLAGS> flags)
        {
            if (getQueueCount(familyIx)!=0)
                return createCommandPool_impl(familyIx,flags);

            NBL_LOG_ERROR("Family index out of bounds");
            return nullptr;
        }

        // Not implemented stuff:
        //TODO: vkGetDescriptorSetLayoutSupport
        //vkGetPipelineCacheData //as pipeline cache method?? (why not)
        //vkMergePipelineCaches //as pipeline cache method (why not)

        // Vulkan: const VkDevice*
        virtual const void* getNativeHandle() const = 0;

    protected:
        ILogicalDevice(core::smart_refctd_ptr<const IAPIConnection>&& api, const IPhysicalDevice* const physicalDevice, const SCreationParams& params, const bool runningInRenderdoc);
        inline virtual ~ILogicalDevice()
        {
            // There's no point calling `waitIdle` here for two reasons:
            // - vtable already destroyed, you'll then call undefined function pointer for `waitIdle_impl`
            // - `waitIdle` must have been called already or a similar operation performed, otherwise you'll have circular references from the per-queue GC and you'll never enter this destructor
            if (m_queues)
            for (uint32_t i=0u; i<m_queues->size(); ++i)
                delete (*m_queues)[i];
        }

        virtual IQueue::RESULT waitIdle_impl() const = 0;

        virtual bool flushMappedMemoryRanges_impl(const std::span<const MappedMemoryRange> ranges) = 0;
        virtual bool invalidateMappedMemoryRanges_impl(const std::span<const MappedMemoryRange> ranges) = 0;

        virtual bool bindBufferMemory_impl(const uint32_t count, const SBindBufferMemoryInfo* pInfos) = 0;
        virtual bool bindImageMemory_impl(const uint32_t count, const SBindImageMemoryInfo* pInfos) = 0;

        virtual core::smart_refctd_ptr<IGPUBuffer> createBuffer_impl(IGPUBuffer::SCreationParams&& creationParams) = 0;
        virtual core::smart_refctd_ptr<IGPUBufferView> createBufferView_impl(const asset::SBufferRange<const IGPUBuffer>& underlying, const asset::E_FORMAT _fmt) = 0;
        virtual core::smart_refctd_ptr<IGPUImage> createImage_impl(IGPUImage::SCreationParams&& params) = 0;
        virtual core::smart_refctd_ptr<IGPUImageView> createImageView_impl(IGPUImageView::SCreationParams&& params) = 0;
        virtual core::smart_refctd_ptr<IGPUBottomLevelAccelerationStructure> createBottomLevelAccelerationStructure_impl(IGPUAccelerationStructure::SCreationParams&& params) = 0;
        virtual core::smart_refctd_ptr<IGPUTopLevelAccelerationStructure> createTopLevelAccelerationStructure_impl(IGPUTopLevelAccelerationStructure::SCreationParams&& params) = 0;

        virtual AccelerationStructureBuildSizes getAccelerationStructureBuildSizes_impl(
            const core::bitflag<IGPUBottomLevelAccelerationStructure::BUILD_FLAGS> flags, const bool motionBlur,
            const std::span<const IGPUBottomLevelAccelerationStructure::AABBs<const IGPUBuffer>> geometries, const uint32_t* const pMaxPrimitiveCounts
        ) const = 0;
        virtual AccelerationStructureBuildSizes getAccelerationStructureBuildSizes_impl(
            const core::bitflag<IGPUBottomLevelAccelerationStructure::BUILD_FLAGS> flags, const bool motionBlur,
            const std::span<const IGPUBottomLevelAccelerationStructure::AABBs<const asset::ICPUBuffer>> geometries, const uint32_t* const pMaxPrimitiveCounts
        ) const = 0;
        virtual AccelerationStructureBuildSizes getAccelerationStructureBuildSizes_impl(
            const core::bitflag<IGPUBottomLevelAccelerationStructure::BUILD_FLAGS> flags, const bool motionBlur,
            const std::span<const IGPUBottomLevelAccelerationStructure::Triangles<const IGPUBuffer>> geometries, const uint32_t* const pMaxPrimitiveCounts
        ) const = 0;
        virtual AccelerationStructureBuildSizes getAccelerationStructureBuildSizes_impl(
            const core::bitflag<IGPUBottomLevelAccelerationStructure::BUILD_FLAGS> flags, const bool motionBlur,
            const std::span<const IGPUBottomLevelAccelerationStructure::Triangles<const asset::ICPUBuffer>> geometries, const uint32_t* const pMaxPrimitiveCounts
        ) const = 0;
        virtual AccelerationStructureBuildSizes getAccelerationStructureBuildSizes_impl(
            const bool hostBuild, const core::bitflag<IGPUTopLevelAccelerationStructure::BUILD_FLAGS> flags,
            const bool motionBlur, const uint32_t maxInstanceCount
        ) const = 0;

        enum class DEFERRABLE_RESULT : uint8_t
        {
            DEFERRED,
            NOT_DEFERRED,
            SOME_ERROR
        };
        virtual DEFERRABLE_RESULT buildAccelerationStructures_impl(
            IDeferredOperation* const deferredOperation, const std::span<const IGPUBottomLevelAccelerationStructure::HostBuildInfo> infos,
            const IGPUBottomLevelAccelerationStructure::BuildRangeInfo* const* const ppBuildRangeInfos, const uint32_t totalGeometryCount
        ) = 0;
        virtual DEFERRABLE_RESULT buildAccelerationStructures_impl(
            IDeferredOperation* const deferredOperation, const std::span<const IGPUTopLevelAccelerationStructure::HostBuildInfo> infos,
            const IGPUTopLevelAccelerationStructure::BuildRangeInfo* const pBuildRangeInfos, const uint32_t totalGeometryCount
        ) = 0;
        virtual bool writeAccelerationStructuresProperties_impl(const std::span<const IGPUAccelerationStructure* const> accelerationStructures, const IQueryPool::TYPE type, size_t* data, const size_t stride) = 0;
        virtual DEFERRABLE_RESULT copyAccelerationStructure_impl(IDeferredOperation* const deferredOperation, const IGPUAccelerationStructure::CopyInfo& copyInfo) = 0;
        virtual DEFERRABLE_RESULT copyAccelerationStructureToMemory_impl(IDeferredOperation* const deferredOperation, const IGPUAccelerationStructure::HostCopyToMemoryInfo& copyInfo) = 0;
        virtual DEFERRABLE_RESULT copyAccelerationStructureFromMemory_impl(IDeferredOperation* const deferredOperation, const IGPUAccelerationStructure::HostCopyFromMemoryInfo& copyInfo) = 0;

        constexpr static inline auto MaxStagesPerPipeline = 6u;
        virtual core::smart_refctd_ptr<IGPUDescriptorSetLayout> createDescriptorSetLayout_impl(const std::span<const IGPUDescriptorSetLayout::SBinding> bindings, const uint32_t maxSamplersCount) = 0;
        virtual core::smart_refctd_ptr<IGPUPipelineLayout> createPipelineLayout_impl(
            const std::span<const asset::SPushConstantRange> pcRanges,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3
        ) = 0;

        virtual core::smart_refctd_ptr<IDescriptorPool> createDescriptorPool_impl(const IDescriptorPool::SCreateInfo& createInfo) = 0;

        struct SUpdateDescriptorSetsParams
        {
            std::span<const IGPUDescriptorSet::SWriteDescriptorSet> writes;
            std::span<const IGPUDescriptorSet::SCopyDescriptorSet> copies;
            const asset::IDescriptor::E_TYPE* pWriteTypes;
            uint32_t bufferCount = 0u;
            uint32_t bufferViewCount = 0u;
            uint32_t imageCount = 0u;       // combined image/samplers as well as samplers belong here, since they're written through a VkDescriptorImageInfo
            uint32_t accelerationStructureCount = 0u;
            uint32_t accelerationStructureWriteCount = 0u;
        };
        virtual void updateDescriptorSets_impl(const SUpdateDescriptorSetsParams& params) = 0;

        struct SDropDescriptorSetsParams
        {
            std::span<const IGPUDescriptorSet::SDropDescriptorSet> drops;
            uint32_t bufferCount = 0u;
            uint32_t bufferViewCount = 0u;
            uint32_t imageCount = 0u;
            uint32_t accelerationStructureCount = 0u;
            uint32_t accelerationStructureWriteCount = 0u;
        };
        virtual void nullifyDescriptors_impl(const SDropDescriptorSetsParams& params) = 0;

        virtual core::smart_refctd_ptr<IGPURenderpass> createRenderpass_impl(const IGPURenderpass::SCreationParams& params, IGPURenderpass::SCreationParamValidationResult&& validation) = 0;
        virtual core::smart_refctd_ptr<IGPUFramebuffer> createFramebuffer_impl(IGPUFramebuffer::SCreationParams&& params) = 0;

        template<typename CreationParams, typename ExtraLambda>
        inline CreationParams::SSpecializationValidationResult commonCreatePipelines(IGPUPipelineCache* const pipelineCache, const std::span<const CreationParams> params, ExtraLambda&& extra)
        {
            if (pipelineCache && !pipelineCache->wasCreatedBy(this))
            {
                NBL_LOG_ERROR("The pipelineCache was not created by this device");
                return {};
            }
            if (params.empty())
            {
                NBL_LOG_ERROR("No parameters were given");
                return {};
            }

            typename CreationParams::SSpecializationValidationResult retval = {.count=0,.dataSize=0};
            for (auto i=0; i<params.size(); i++)
            {
                const auto& ci = params[i];
                
                const auto validation = ci.valid();
                if (!validation)
                {
                    NBL_LOG_ERROR("Invalid parameters were given (params[%d])", i);
                    return {};
                }

                if (!ci.layout->wasCreatedBy(this))
                {
                    NBL_LOG_ERROR("The layout was not created by this device (params[%d])", i);
                    return {};
                }

                constexpr auto AllowDerivativesFlag = CreationParams::FLAGS::ALLOW_DERIVATIVES;
                if (ci.basePipeline)
                {
                    if (!ci.basePipeline->wasCreatedBy(this))
                    {
                        NBL_LOG_ERROR("Invalid basePipeline was specified (params[%d])", i);
                        return {};
                    }
                    if (!ci.basePipeline->getCreationFlags().hasFlags(AllowDerivativesFlag))
                    {
                        NBL_LOG_ERROR("Invalid basePipeline was specified (params[%d])", i);
                        return {};
                    }
                }
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComputePipelineCreateInfo.html#VUID-VkComputePipelineCreateInfo-flags-07985
                else if (ci.basePipelineIndex < -1 || ci.basePipelineIndex >= i || ci.basePipelineIndex >= 0 && !params[ci.basePipelineIndex].flags.hasFlags(AllowDerivativesFlag))
                {
                    NBL_LOG_ERROR("Invalid basePipeline was specified (params[%d])", i);
                    return {};
                }

                const auto& features = getEnabledFeatures();
                for (auto info : ci.getShaders())
                if (info.shader)
                {
                    const asset::IShader::E_SHADER_STAGE shaderStage = info.stage;

                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-stage-00704
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-stage-00705
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-stage-02091
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-stage-02092
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-stage-00706
                    switch (shaderStage)
                    {
                        case IGPUShader::E_SHADER_STAGE::ESS_TESSELLATION_CONTROL: [[fallthrough]];
                        case IGPUShader::E_SHADER_STAGE::ESS_TESSELLATION_EVALUATION:
                            if (!features.tessellationShader)
                            {
                                NBL_LOG_ERROR("Cannot create IGPUShader for %p, Tessellation Shader feature not enabled!", info.shader);
                                return nullptr;
                            }
                            break;
                        case IGPUShader::E_SHADER_STAGE::ESS_GEOMETRY:
                            if (!features.geometryShader)
                            {
                                NBL_LOG_ERROR("Cannot create IGPUShader for %p, Geometry Shader feature not enabled!", info.shader);
                                return nullptr;
                            }
                            break;
                        case IGPUShader::E_SHADER_STAGE::ESS_ALL_OR_LIBRARY: [[fallthrough]];
                        case IGPUShader::E_SHADER_STAGE::ESS_VERTEX: [[fallthrough]];
                        case IGPUShader::E_SHADER_STAGE::ESS_FRAGMENT: [[fallthrough]];
                        case IGPUShader::E_SHADER_STAGE::ESS_COMPUTE:
                            break;
                            // unsupported yet
                        case IGPUShader::E_SHADER_STAGE::ESS_TASK: [[fallthrough]];
                        case IGPUShader::E_SHADER_STAGE::ESS_MESH:
                            NBL_LOG_ERROR("Unsupported (yet) shader stage");
                            return nullptr;
                            break;
                        case IGPUShader::E_SHADER_STAGE::ESS_RAYGEN: [[fallthrough]];
                        case IGPUShader::E_SHADER_STAGE::ESS_ANY_HIT: [[fallthrough]];
                        case IGPUShader::E_SHADER_STAGE::ESS_CLOSEST_HIT: [[fallthrough]];
                        case IGPUShader::E_SHADER_STAGE::ESS_MISS: [[fallthrough]];
                        case IGPUShader::E_SHADER_STAGE::ESS_INTERSECTION: [[fallthrough]];
                        case IGPUShader::E_SHADER_STAGE::ESS_CALLABLE:
                            if (!features.rayTracingPipeline)
                            {
                                NBL_LOG_ERROR("Cannot create IGPUShader for %p, Raytracing Pipeline feature not enabled!", info.shader);
                                return nullptr;
                            }
                            break;
                        default:
                            // Implicit unsupported stages or weird multi-bit stage enum values
                            NBL_LOG_ERROR("Unknown Shader Stage %d", shaderStage);
                            return {};
                            break;
                    }

                    if (!extra(info))
                    {
                        NBL_LOG_ERROR("Invalid shader were specified (params[%d])", i);
                        return {};
                    }
                }

                retval += validation;
            }
            return retval;
        }
        virtual void createComputePipelines_impl(
            IGPUPipelineCache* const pipelineCache,
            const std::span<const IGPUComputePipeline::SCreationParams> createInfos,
            core::smart_refctd_ptr<IGPUComputePipeline>* const output,
            const IGPUComputePipeline::SCreationParams::SSpecializationValidationResult& validation
        ) = 0;
        virtual void createGraphicsPipelines_impl(
            IGPUPipelineCache* const pipelineCache,
            const std::span<const IGPUGraphicsPipeline::SCreationParams> params,
            core::smart_refctd_ptr<IGPUGraphicsPipeline>* const output,
            const IGPUGraphicsPipeline::SCreationParams::SSpecializationValidationResult& validation
        ) = 0;

        virtual core::smart_refctd_ptr<IQueryPool> createQueryPool_impl(const IQueryPool::SCreationParams& params) = 0;
        virtual bool getQueryPoolResults_impl(const IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount, void* const pData, const size_t stride, const core::bitflag<IQueryPool::RESULTS_FLAGS> flags) = 0;

        virtual core::smart_refctd_ptr<IGPUCommandPool> createCommandPool_impl(const uint32_t familyIx, const core::bitflag<IGPUCommandPool::CREATE_FLAGS> flags) = 0;


        // TODO: think what to move to `private`
        core::smart_refctd_ptr<asset::CCompilerSet> m_compilerSet;
        core::smart_refctd_ptr<const IAPIConnection> m_api;
        const IPhysicalDevice* const m_physicalDevice;
        const system::logger_opt_ptr m_logger;

        const SPhysicalDeviceFeatures m_enabledFeatures;

        using queues_array_t = core::smart_refctd_dynamic_array<CThreadSafeQueueAdapter*>;
        queues_array_t m_queues;
        struct QueueFamilyInfo
        {
            core::bitflag<asset::PIPELINE_STAGE_FLAGS> supportedStages = asset::PIPELINE_STAGE_FLAGS::NONE;
            core::bitflag<asset::ACCESS_FLAGS> supportedAccesses = asset::ACCESS_FLAGS::NONE;
            uint16_t queueCount = 0u;
            // index into flat array of `m_queues`
            uint16_t firstQueueIndex = 0u;
        };
        const std::array<QueueFamilyInfo,MaxQueueFamilies> m_queueFamilyInfos;
        
    private:
        const SPhysicalDeviceLimits& getPhysicalDeviceLimits() const;

        void addCommonShaderDefines(const bool runningInRenderDoc);

        inline bool invalidAllocationForBind(const IDeviceMemoryBacked* resource, const IDeviceMemoryBacked::SMemoryBinding& binding, const size_t alignment)
        {
            if (!resource->wasCreatedBy(this))
            {
                m_logger.log("Buffer or Image %p not compatible with Device %p !",system::ILogger::ELL_ERROR,resource,this);
                return true;
            }
            if (resource->getBoundMemory().isValid())
            {
                m_logger.log("Buffer or Image %p already has memory bound!",system::ILogger::ELL_ERROR,resource);
                return true;
            }
            if (!binding.isValid() || binding.memory->getOriginDevice()!=this)
            {
                m_logger.log("Memory Allocation %p and Offset %d not valid or not compatible with Device %p !",system::ILogger::ELL_ERROR,binding.memory,binding.offset,this);
                return true;
            }
            if (binding.offset&(alignment-1))
            {
                m_logger.log("Memory Allocation Offset %d not aligned to %d which is required by the Buffer or Image!",system::ILogger::ELL_ERROR,binding.offset,alignment);
                return true;
            }
            return false;
        }

        inline bool invalidMappedRanges(const std::span<const MappedMemoryRange> ranges)
        {
            for (auto& range : ranges)
            {
                if (!range.valid())
                    return true;
                if (range.memory->getOriginDevice()!=this)
                    return true;
            }
            return false;
        }

        inline bool acquireDeferredOperation(IDeferredOperation* const deferredOp)
        {
            if (!deferredOp || !deferredOp->wasCreatedBy(this) || deferredOp->isPending())
                return false;
            deferredOp->m_resourceTracking.clear();
            return getEnabledFeatures().accelerationStructureHostCommands;
        }

        inline bool invalidCreationParams(const IGPUAccelerationStructure::SCreationParams& params)
        {
            if (!getEnabledFeatures().accelerationStructure)
            {
                NBL_LOG_ERROR("Feature `acceleration structure` is not enabled");
                return true;
            }
            constexpr size_t MinAlignment = 256u;
            if (!params.bufferRange.isValid() || !params.bufferRange.buffer->wasCreatedBy(this) || (params.bufferRange.offset & (MinAlignment - 1)) != 0u)
            {
                NBL_LOG_ERROR("Invalid bufferRange was given");
                return true;
            }
            const auto bufferUsages = params.bufferRange.buffer->getCreationParams().usage;
            if (!bufferUsages.hasFlags(IGPUBuffer::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT))
            {
                NBL_LOG_ERROR("Invalid bufferRange was given");
                return true;
            }
            if (params.flags.hasFlags(IGPUAccelerationStructure::SCreationParams::FLAGS::MOTION_BIT) && !getEnabledFeatures().rayTracingMotionBlur)
            {
                NBL_LOG_ERROR("Feature `ray tracing motion blur` is not enabled");
                return true;
            }
            return false;
        }
        template<class BufferType>
        bool invalidFeaturesForASBuild(const bool motionBlur) const
        {
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkGetAccelerationStructureBuildSizesKHR-accelerationStructure-08933
            if (!m_enabledFeatures.accelerationStructure)
            {
                NBL_LOG_ERROR("Feature `acceleration structure` is not enabled");
                return true;
            }
			// not sure of VUID
            if (std::is_same_v<BufferType, asset::ICPUBuffer> && !m_enabledFeatures.accelerationStructureHostCommands)
            {
                NBL_LOG_ERROR("Feature `acceleration structure` host commands is not enabled");
				return true;
            }
            // not sure of VUID
            if (motionBlur && !m_enabledFeatures.rayTracingMotionBlur)
            {
                NBL_LOG_ERROR("Feature `ray tracing motion blur` is not enabled");
                return true;
            }

            return false;
        }
        static inline bool invalidMemoryForAccelerationStructureHostOperations(const IDeviceMemoryAllocation* const memory)
        {
            return !memory || !memory->isMappable() || !memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_DEVICE_LOCAL_BIT);
        }
};


template<typename ResourceBarrier>
inline bool ILogicalDevice::validateMemoryBarrier(const uint32_t queueFamilyIndex, const IGPUCommandBuffer::SImageMemoryBarrier<ResourceBarrier>& barrier) const
{
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-image-parameter
    if (!barrier.image)
    {
        NBL_LOG_ERROR("Invalid image handle");
        return false;
    }
    const auto& params = barrier.image->getCreationParameters();

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-subresourceRange-01486
    if (barrier.subresourceRange.baseMipLevel >= params.mipLevels)
    {
        NBL_LOG_ERROR("Invalid Mip level");
        return false;
    }
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-subresourceRange-01488
    if (barrier.subresourceRange.baseArrayLayer >= params.arrayLayers)
    {
        NBL_LOG_ERROR("Invalid array layer");
        return false;
    }
    // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-subresourceRange-01724
    // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-subresourceRange-01725

    const auto aspectMask = barrier.subresourceRange.aspectMask;
    if (asset::isDepthOrStencilFormat(params.format))
    {
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-image-03319
        constexpr auto DepthStencilAspects = IGPUImage::EAF_DEPTH_BIT|IGPUImage::EAF_STENCIL_BIT;
        if (aspectMask.value & (~DepthStencilAspects))
        {
            NBL_LOG_ERROR("Invalid aspect mask");
            return false;
        }
        if (!bool(aspectMask.value & DepthStencilAspects))
        {
            NBL_LOG_ERROR("Invalid aspect mask");
            return false;
        }
    }
    //https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-image-01671
    else if (aspectMask != IGPUImage::EAF_COLOR_BIT)
    {
        NBL_LOG_ERROR("Invalid aspect mask");
        return false;
    }
    
    const bool layoutTransform = barrier.oldLayout!=barrier.newLayout;
    bool ownershipTransfer = false;
    const asset::SMemoryBarrier* inner;
    if constexpr (std::is_same_v<ResourceBarrier,IGPUCommandBuffer::SOwnershipTransferBarrier>)
    {
        // TODO: better check for queue family ownership transfer
        const bool ownershipTransfer = barrier.barrier.otherQueueFamilyIndex!=IQueue::FamilyIgnored;
        inner = &barrier.barrier.dep;
    }
    else
        inner = &barrier.barrier;
    // TODO: WTF ? https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-srcStageMask-03855
    if (layoutTransform || ownershipTransfer)
    {
        const bool srcStageIsHost = inner->srcStageMask.hasFlags(asset::PIPELINE_STAGE_FLAGS::HOST_BIT);
        const auto logger = m_logger.get();
        auto mismatchedLayout = [&params,aspectMask,srcStageIsHost, logger]<bool dst>(const IGPUImage::LAYOUT layout) -> bool
        {
            switch (layout)
            {
                // Because we don't support legacy layout enums, we don't check these at all:
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-oldLayout-01208
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-oldLayout-01209
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-oldLayout-01210
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-oldLayout-01658
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-oldLayout-01659
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04065
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04066
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04067
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04068
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-aspectMask-08702
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-aspectMask-08703
                // and we check the following all at once:
                case IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL:
                    if (!dst && srcStageIsHost)
                    {
                        logger->log("Invalid srcStageMask, must not include HOST_BIT [%s - %s:%d]", system::ILogger::ELL_ERROR, __FUNCTION__, __FILE__, __LINE__);
                        return true;
                    }
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-03938
                    if (aspectMask && !params.usage.hasFlags(IGPUImage::E_USAGE_FLAGS::EUF_RENDER_ATTACHMENT_BIT))
                    {
                        logger->log("Invalid image usage [%s - %s:%d]", system::ILogger::ELL_ERROR, __FUNCTION__, __FILE__, __LINE__);
                        return true;
                    }
                    break;
                case IGPUImage::LAYOUT::READ_ONLY_OPTIMAL:
                    if (!dst && srcStageIsHost)
                    {
                        logger->log("Invalid srcStageMask, must not include HOST_BIT [%s - %s:%d]", system::ILogger::ELL_ERROR, __FUNCTION__, __FILE__, __LINE__);
                        return true;
                    }
                    {
                        constexpr auto ValidUsages = IGPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT|IGPUImage::E_USAGE_FLAGS::EUF_INPUT_ATTACHMENT_BIT;
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-oldLayout-01211
                        if (aspectMask.hasFlags(IGPUImage::EAF_STENCIL_BIT))
                        {
                            if (!bool(params.actualStencilUsage() & ValidUsages))
                            {
                                logger->log("Invalid stencil usages [%s - %s:%d]", system::ILogger::ELL_ERROR, __FUNCTION__, __FILE__, __LINE__);
                                return true;
                            }
                        }
                        else if (!bool(params.usage & ValidUsages))
                        {
                            logger->log("Invalid image usages [%s - %s:%d]", system::ILogger::ELL_ERROR, __FUNCTION__, __FILE__, __LINE__);
                            return true;
                        }
                    }
                    break;
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-oldLayout-01212
                case IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL:
                    if (!dst && srcStageIsHost)
                    {
                        logger->log("Invalid srcStageMask, must not include HOST_BIT [%s - %s:%d]", system::ILogger::ELL_ERROR, __FUNCTION__, __FILE__, __LINE__);
                        return true;
                    }
                    if (!params.usage.hasFlags(IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT))
                    {
                        logger->log("Invalid image usage [%s - %s:%d]", system::ILogger::ELL_ERROR, __FUNCTION__, __FILE__, __LINE__);
                        return true;
                    }
                    break;
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-oldLayout-01213
                case IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL:
                    if (!dst && srcStageIsHost)
                    {
                        logger->log("Invalid srcStageMask, must not include HOST_BIT [%s - %s:%d]", system::ILogger::ELL_ERROR, __FUNCTION__, __FILE__, __LINE__);
                        return true;
                    }
                    if(!params.usage.hasFlags(IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT))
                    {
                        logger->log("Invalid image usage [%s - %s:%d]", system::ILogger::ELL_ERROR, __FUNCTION__, __FILE__, __LINE__);
                        return true;
                    }
                    break;
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-oldLayout-01198
                case IGPUImage::LAYOUT::UNDEFINED: [[fallthrough]];
                case IGPUImage::LAYOUT::PREINITIALIZED:
                    if constexpr (dst)
                    {
                        logger->log("Invalid newLayout [%s - %s:%d]", system::ILogger::ELL_ERROR, __FUNCTION__, __FILE__, __LINE__);
                        return true;
                    }
                    break;
                // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-oldLayout-02088
                // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-07006
                    // Implied from being able to created an image with above required usage flags
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-attachmentFeedbackLoopLayout-07313
                default:
                    break;
            }
            return false;
        };
        // CANNOT CHECK: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-oldLayout-01197
        if (mismatchedLayout.operator()<false>(barrier.oldLayout) || mismatchedLayout.operator()<true>(barrier.newLayout))
            return false;
    }

    if constexpr (std::is_same_v<ResourceBarrier,IGPUCommandBuffer::SOwnershipTransferBarrier>)
        return validateMemoryBarrier(queueFamilyIndex,barrier.barrier,barrier.image->getCachedCreationParams().isConcurrentSharing());
    else
        return validateMemoryBarrier(queueFamilyIndex,barrier.barrier);
}

} // namespace nbl::video

#include "nbl/undef_logging_macros.h"
#endif //_NBL_VIDEO_I_LOGICAL_DEVICE_H_INCLUDED_