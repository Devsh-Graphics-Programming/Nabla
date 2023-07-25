#ifndef _NBL_VIDEO_I_LOGICAL_DEVICE_H_INCLUDED_
#define _NBL_VIDEO_I_LOGICAL_DEVICE_H_INCLUDED_

#include "nbl/asset/asset.h"
#include "nbl/asset/utils/ISPIRVOptimizer.h"
#include "nbl/asset/utils/CCompilerSet.h"

#include "nbl/video/SPhysicalDeviceFeatures.h"
#include "nbl/video/IDeferredOperation.h"
#include "nbl/video/IDeviceMemoryAllocator.h"
#include "nbl/video/IGPUPipelineCache.h"
#include "nbl/video/IGPUCommandBuffer.h"
#include "nbl/video/CThreadSafeQueueAdapter.h"
#include "nbl/video/ISwapchain.h"

namespace nbl::video
{

class IPhysicalDevice;

class NBL_API2 ILogicalDevice : public core::IReferenceCounted, public IDeviceMemoryAllocator
{
    public:
        struct SQueueCreationParams
        {
            IQueue::CREATE_FLAGS flags;
            uint32_t familyIndex;
            uint32_t count;
            const float* priorities;
        };
        struct SCreationParams
        {
            uint32_t queueParamsCount;
            const SQueueCreationParams* queueParams;
            SPhysicalDeviceFeatures featuresToEnable;
            core::smart_refctd_ptr<asset::CCompilerSet> compilerSet = nullptr;
        };


        //! Basic getters
        inline const IPhysicalDevice* getPhysicalDevice() const { return m_physicalDevice; }

        inline const SPhysicalDeviceFeatures& getEnabledFeatures() const { return m_enabledFeatures; }

        E_API_TYPE getAPIType() const;

        //
        inline IQueue* getQueue(uint32_t _familyIx, uint32_t _ix)
        {
            const uint32_t offset = m_queueFamilyInfos->operator[](_familyIx).firstQueueIndex;
            return (*m_queues)[offset+_ix]->getUnderlyingQueue();
        }

        // Using the same queue as both a threadsafe queue and a normal queue invalidates the safety.
        inline CThreadSafeQueueAdapter* getThreadSafeQueue(uint32_t _familyIx, uint32_t _ix)
        {
            const uint32_t offset = m_queueFamilyInfos->operator[](_familyIx).firstQueueIndex;
            return (*m_queues)[offset + _ix];
        }


        //! sync validation
        inline core::bitflag<asset::PIPELINE_STAGE_FLAGS> getSupportedStageMask(const uint32_t queueFamilyIndex) const
        {
            if (queueFamilyIndex>m_queueFamilyInfos->size())
                return asset::PIPELINE_STAGE_FLAGS::NONE;
            return m_queueFamilyInfos->operator[](queueFamilyIndex).supportedStages;
        }
        //! Use this to validate instead of `getSupportedStageMask(queueFamilyIndex)&stageMask`, it checks special values
        bool supportsMask(const uint32_t queueFamilyIndex, core::bitflag<asset::PIPELINE_STAGE_FLAGS> stageMask) const;
        
        inline core::bitflag<asset::ACCESS_FLAGS> getSupportedAccessMask(const uint32_t queueFamilyIndex) const
        {
            if (queueFamilyIndex>m_queueFamilyInfos->size())
                return asset::ACCESS_FLAGS::NONE;
            return m_queueFamilyInfos->operator[](queueFamilyIndex).supportedAccesses;
        }
        //! Use this to validate instead of `getSupportedAccessMask(queueFamilyIndex)&accessMask`, it checks special values
        bool supportsMask(const uint32_t queueFamilyIndex, core::bitflag<asset::ACCESS_FLAGS> accessMask) const;

        //! NOTE/TODO: this is not yet finished
        inline bool validateMemoryBarrier(const uint32_t queueFamilyIndex, asset::SMemoryBarrier barrier) const;
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
                if (barrier.dep.srcStageMask.hasFlags(HostBit)||barrier.dep.dstStageMask.hasFlags(HostBit))
                    return false;
                // spec doesn't require it now, but we do
                switch (barrier.ownershipOp)
                {
                    case IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::ACQUIRE:
                        if (barrier.dep.srcStageMask || barrier.dep.srcAccessMask)
                            return false;
                        break;
                    case IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::RELEASE:
                        if (barrier.dep.dstStageMask || barrier.dep.dstAccessMask)
                            return false;
                        break;
                    default:
                        break;
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
            if (!range.buffer || range.size==0u)
                return false;
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkBufferMemoryBarrier2-offset-01187
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkBufferMemoryBarrier2-offset-01189
            const size_t remain = range.size!=IGPUCommandBuffer::SBufferMemoryBarrier{}.range.size ? range.size:1ull;
            if (range.offset+remain>range.buffer->getSize())
                return false;

            if constexpr(std::is_same_v<IGPUCommandBuffer::SOwnershipTransferBarrier,ResourceBarrier>)
                return validateMemoryBarrier(queueFamilyIndex,barrier.barrier,range.buffer->getCachedCreationParams().isConcurrentSharing());
            else
                return validateMemoryBarrier(queueFamilyIndex,barrier.barrier);
        }
        template<typename ResourceBarrier>
        bool validateMemoryBarrier(const uint32_t queueFamilyIndex, const IGPUCommandBuffer::SImageMemoryBarrier<ResourceBarrier>& barrier) const;

        
        //! Sync
        virtual IQueue::RESULT waitIdle() const = 0;

        //! Semaphore Stuff
        virtual core::smart_refctd_ptr<ISemaphore> createSemaphore(const uint64_t initialValue) = 0;
        //
        struct SSemaphoreWaitInfo
        {
            const ISemaphore* semaphore;
            uint64_t value;
        };
        enum class WAIT_RESULT : uint8_t
        {
            TIMEOUT,
            SUCCESS,
            DEVICE_LOST,
            _ERROR
        };
        virtual WAIT_RESULT waitForSemaphores(const uint32_t count, const SSemaphoreWaitInfo* const infos, const bool waitAll, const uint64_t timeout) = 0;
        // Forever waiting variant if you're confident that the fence will eventually be signalled
        inline WAIT_RESULT blockForSemaphores(const uint32_t count, const SSemaphoreWaitInfo* const infos, const bool waitAll=true)
        {
            if (count)
            {
                auto waitStatus = WAIT_RESULT::TIMEOUT;
                while (waitStatus==WAIT_RESULT::TIMEOUT)
                    waitStatus = waitForSemaphores(count,infos,waitAll,999999999ull);
                return waitStatus;
            }
            return WAIT_RESULT::SUCCESS;
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
        inline bool flushMappedMemoryRanges(const core::SRange<const MappedMemoryRange>& ranges)
        {
            if (invalidMappedRanges(ranges))
                return false;
            return flushMappedMemoryRanges_impl(ranges);
        }
        // For memory allocations without the video::IDeviceMemoryAllocation::EMCF_COHERENT mapping capability flag you need to call this for the GPU writes to become CPU visible
        inline bool invalidateMappedMemoryRanges(uint32_t memoryRangeCount, const MappedMemoryRange* pMemoryRanges)
        {
            core::SRange<const MappedMemoryRange> ranges{ pMemoryRanges, pMemoryRanges + memoryRangeCount };
            return invalidateMappedMemoryRanges(ranges);
        }
        //! Utility wrapper for the pointer based func
        inline bool invalidateMappedMemoryRanges(const core::SRange<const MappedMemoryRange>& ranges)
        {
            if (invalidMappedRanges(ranges))
                return false;
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
                ) {
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
        inline bool bindImageMemory(uint32_t count, const SBindImageMemoryInfo* pBindInfos)
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

        //! Descriptor Creation
        // Buffer (@see ICPUBuffer)
        inline core::smart_refctd_ptr<IGPUBuffer> createBuffer(IGPUBuffer::SCreationParams&& creationParams)
        {
            const auto maxSize = getPhysicalDevice()->getLimits().maxBufferSize;
            if (creationParams.size>maxSize)
            {
                m_logger.log("Failed to create Buffer, size %d larger than Device %p's limit!",system::ILogger::ELL_ERROR,creationParams.size,this,maxSize);
                return nullptr;
            }
            return createBuffer_impl(std::move(creationParams));
        }
        // Create a BufferView, to a shader; a fake 1D-like texture with no interpolation (@see ICPUBufferView)
        inline core::smart_refctd_ptr<IGPUBufferView> createBufferView(const asset::SBufferRange<const IGPUBuffer>& underlying, const asset::E_FORMAT _fmt)
        {
            if (!underlying.isValid() || !underlying.buffer->wasCreatedBy(this))
                return nullptr;
            if (!getPhysicalDevice()->getBufferFormatUsages()[_fmt].bufferView)
                return nullptr;
            return createBufferView_impl(underlying,_fmt);
        }
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
                return nullptr;
            // TODO: @Cyprian validation of params against the device's limits (sample counts, etc.) see vkCreateImage
            return createImageView_impl(std::move(params));
        }
        // Create a sampler object to use with ImageViews
        virtual core::smart_refctd_ptr<IGPUSampler> createSampler(const IGPUSampler::SParams& _params) = 0;
        // acceleration structures
        inline core::smart_refctd_ptr<IGPUBottomLevelAccelerationStructure> createBottomLevelAccelerationStructure(IGPUAccelerationStructure::SCreationParams&& params)
        {
            if (invalidCreationParams(params))
                return nullptr;
            return createBottomLevelAccelerationStructure_impl(std::move(params));
        }
        inline core::smart_refctd_ptr<IGPUTopLevelAccelerationStructure> createTopLevelAccelerationStructure(IGPUTopLevelAccelerationStructure::SCreationParams&& params)
        {
            if (invalidCreationParams(params))
                return nullptr;
            if (params.flags.hasFlags(IGPUAccelerationStructure::SCreationParams::FLAGS::MOTION_BIT) && (params.maxInstanceCount==0u || params.maxInstanceCount>getPhysicalDevice()->getLimits().maxAccelerationStructureInstanceCount))
                return nullptr;
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
            IGPUBottomLevelAccelerationStructure::Triangles<IGPUBuffer>,
            IGPUBottomLevelAccelerationStructure::Triangles<asset::ICPUBuffer>,
            IGPUBottomLevelAccelerationStructure::AABBs<IGPUBuffer>,
            IGPUBottomLevelAccelerationStructure::AABBs<asset::ICPUBuffer>
        >
        inline AccelerationStructureBuildSizes getAccelerationStructureBuildSizes(
            const core::bitflag<IGPUBottomLevelAccelerationStructure::BUILD_FLAGS> flags, const bool motionBlur,
            const core::SRange<const Geometry>& geometries, const uint32_t* const pMaxPrimitiveCounts) const
        {
            if (invalidFeatures<Geometry::buffer_t>(motionBlur))
                return {};

            if (!IGPUBottomLevelAccelerationStructure::validBuildFlags(flags,m_enabledFeatures))
                return {};

            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkGetAccelerationStructureBuildSizesKHR-pBuildInfo-03619
            if (geometries.size() && !pMaxPrimitiveCounts)
                return {};

            const auto& limits = getPhysicalDevice()->getLimits();
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03793
            if (geometries.size()>limits.maxAccelerationStructureGeometryCount)
                return {};

            // not sure of VUID
            uint32_t primsFree = limits.maxAccelerationStructurePrimitiveCount;
			for (auto i=0u; i<geometries.size(); i++)
            {
			    if (pMaxPrimitiveCounts[i]>primsFree)
				    return {};
                primsFree -= pMaxPrimitiveCounts[i];
            }

            return getAccelerationStructureBuildSizes_impl(flags,motionBlur,geometries,pMaxPrimitiveCounts);
        }
        inline AccelerationStructureBuildSizes getAccelerationStructureBuildSizes(const bool hostBuild, const core::bitflag<IGPUTopLevelAccelerationStructure::BUILD_FLAGS> flags, const bool motionBlur, const uint32_t maxInstanceCount) const
        {
            if (invalidFeatures<IGPUBuffer>(motionBlur))
                return {};

            if (!IGPUTopLevelAccelerationStructure::validBuildFlags(flags))
                return {};

            const auto& limits = getPhysicalDevice()->getLimits();
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkGetAccelerationStructureBuildSizesKHR-pBuildInfo-03785
            if (maxInstanceCount>limits.maxAccelerationStructureInstanceCount)
                return {};

            return getAccelerationStructureBuildSizes_impl(hostBuild,flags,motionBlur,maxInstanceCount);
        }
        // little utility
        template<typename BufferType=IGPUBuffer>
        inline AccelerationStructureBuildSizes getAccelerationStructureBuildSizes(const core::bitflag<IGPUTopLevelAccelerationStructure::BUILD_FLAGS> flags, const bool motionBlur, const uint32_t maxInstanceCount) const
        {
            return getAccelerationStructureBuildSizes(std::is_same_v<BufferType,asset::ICPUBuffer>,flags,motionBlur,maxInstanceCount);
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
            IDeferredOperation* const deferredOperation, const core::SRange<const typename AccelerationStructure::HostBuildInfo>& infos,
            const typename AccelerationStructure::DirectBuildRangeRangeInfos pDirectBuildRangeRangeInfos
        )
        {
            if (!acquireDeferredOperation(deferredOperation))
                return false;

            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-accelerationStructureHostCommands-03581
            if (!m_enabledFeatures.accelerationStructureHostCommands)
                return false;

            uint32_t trackingReservation = 0; 
            for (auto i=0u; i<infos.size(); i++)
            {
                const auto toTrack = infos[i].valid(pDirectBuildRangeRangeInfos[i]);
                if (!toTrack)
                    return false;
                trackingReservation += toTrack;
            }
            
            auto result = buildAccelerationStructures_impl(deferredOperation,infos,pDirectBuildRangeRangeInfos);
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
        inline bool writeAccelerationStructuresProperties(const uint32_t count, const IGPUAccelerationStructure* const* const accelerationStructures, const IQueryPool::TYPE type, size_t* data, const size_t stride=alignof(size_t))
        {
            if (!core::is_aligned_to(stride,alignof(size_t)))
                return false;
            switch (type)
            {
                case IQueryPool::TYPE::ACCELERATION_STRUCTURE_COMPACTED_SIZE: [[fallthrough]];
                case IQueryPool::TYPE::ACCELERATION_STRUCTURE_SERIALIZATION_SIZE: [[fallthrough]];
                case IQueryPool::TYPE::ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS: [[fallthrough]];
                case IQueryPool::TYPE::ACCELERATION_STRUCTURE_SIZE:
                    break;
                default:
                    return false;
                    break;
            }
            if (!getEnabledFeatures().accelerationStructureHostCommands)
                return false;
            for (auto i=0u; i<count; i++)
            if (invalidAccelerationStructureForHostOperations(accelerationStructures[i]))
                return false;
            // unfortunately cannot validate if they're built and if they're built with the right flags
            return writeAccelerationStructuresProperties_impl(count,accelerationStructures,type,data,stride);
        }
        // Host-side copy, DEFERRAL IS NOT OPTIONAL
        inline bool copyAccelerationStructure(IDeferredOperation* const deferredOperation, const IGPUAccelerationStructure::CopyInfo& copyInfo)
        {
            if (!acquireDeferredOperation(deferredOperation))
                return false;
            if (invalidAccelerationStructureForHostOperations(copyInfo.src) || invalidAccelerationStructureForHostOperations(copyInfo.dst))
                return false;
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
                return false;
            if (invalidAccelerationStructureForHostOperations(copyInfo.src))
                return false;
            if (!core::is_aligned_to(ptrdiff_t(copyInfo.dst.buffer->getPointer())+copyInfo.dst.offset,16u))
                return false;
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
                return false;
            if (!core::is_aligned_to(ptrdiff_t(copyInfo.src.buffer->getPointer())+copyInfo.src.offset,16u))
                return false;
            if (invalidAccelerationStructureForHostOperations(copyInfo.dst))
                return false;
            auto result = copyAccelerationStructureFromMemory_impl(deferredOperation,copyInfo);
            if (result==DEFERRABLE_RESULT::DEFERRED)
                deferredOperation->m_resourceTracking.insert(deferredOperation->m_resourceTracking.begin(),{
                    core::smart_refctd_ptr<const IReferenceCounted>(copyInfo.src.buffer),
                    core::smart_refctd_ptr<const IReferenceCounted>(copyInfo.dst)
                });
            return result!=DEFERRABLE_RESULT::SOME_ERROR;
        }

        //! Shaders
        // these are the defines which shall be added to any IGPUShader which has its source as GLSL
        inline core::SRange<const char* const> getExtraShaderDefines() const
        {
            const char* const* begin = m_extraShaderDefines.data();
            return {begin,begin+m_extraShaderDefines.size()};
        }
        virtual core::smart_refctd_ptr<IGPUShader> createShader(core::smart_refctd_ptr<asset::ICPUShader>&& cpushader, const asset::ISPIRVOptimizer* optimizer=nullptr) = 0;
        core::smart_refctd_ptr<IGPUSpecializedShader> createSpecializedShader(const IGPUShader* _unspecialized, const asset::ISpecializedShader::SInfo& _specInfo)
        {
            if (!_unspecialized->wasCreatedBy(this))
                return nullptr;
            auto retval = createSpecializedShader_impl(_unspecialized,_specInfo);
            const auto path = _unspecialized->getFilepathHint();
            if (retval && !path.empty())
                retval->setObjectDebugName(path.c_str());
            return retval;
        }

        //! Layouts
        // Create a descriptor set layout (@see ICPUDescriptorSetLayout)
        core::smart_refctd_ptr<IGPUDescriptorSetLayout> createDescriptorSetLayout(const core::SRange<const IGPUDescriptorSetLayout::SBinding>& bindings);
        // Create a pipeline layout (@see ICPUPipelineLayout)
        core::smart_refctd_ptr<IGPUPipelineLayout> createPipelineLayout(
            const core::SRange<const asset::SPushConstantRange>& pcRanges={nullptr,nullptr},
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0=nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1=nullptr,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2=nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3=nullptr
        )
        {
            if (_layout0 && !_layout0->wasCreatedBy(this))
                return nullptr;
            if (_layout1 && !_layout1->wasCreatedBy(this))
                return nullptr;
            if (_layout2 && !_layout2->wasCreatedBy(this))
                return nullptr;
            if (_layout3 && !_layout3->wasCreatedBy(this))
                return nullptr;
            // sanity check
            if (pcRanges.size()>IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE*MaxStagesPerPipeline)
                return nullptr;
            core::bitflag<IGPUShader::E_SHADER_STAGE> stages = IGPUShader::ESS_UNKNOWN;
            uint32_t maxPCByte = 0u;
            for (auto range : pcRanges)
            {
                stages |= range.stageFlags;
                maxPCByte = core::max(range.offset+range.size,maxPCByte);
            }
            if (maxPCByte>IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE)
                return nullptr;
            // TODO: validate `stages` against the supported ones as reported by enabled features
            return createPipelineLayout_impl(pcRanges,std::move(_layout0),std::move(_layout1),std::move(_layout2),std::move(_layout3));
        }

        //! Descriptor Sets
        inline core::smart_refctd_ptr<IDescriptorPool> createDescriptorPool(const IDescriptorPool::SCreateInfo& createInfo)
        {
            if (createInfo.maxSets==0u)
                return nullptr;
            // its also not useful to have pools with zero descriptors
            uint32_t t = 0;
            for (; t<static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++t)
            if (createInfo.maxDescriptorCount[t])
                break;
            if (t==static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT))
                return nullptr;
            return createDescriptorPool_impl(createInfo);
        }
        // utility func
        inline core::smart_refctd_ptr<IDescriptorPool> createDescriptorPoolForDSLayouts(const IDescriptorPool::E_CREATE_FLAGS flags, const core::SRange<const IGPUDescriptorSetLayout*>& layouts, const uint32_t* setCounts=nullptr)
        {
            IDescriptorPool::SCreateInfo createInfo;
            createInfo.flags = flags;

            auto setCountsIt = setCounts;
            for (auto& layout : layouts)
            {
                if (!layout)
                    continue;
                const auto setCount = setCounts ? *(setCountsIt++):1u;
                createInfo.maxSets += setCount;
                for (uint32_t t=0; t<static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++t)
                {
                    const auto type = static_cast<asset::IDescriptor::E_TYPE>(t);
                    createInfo.maxDescriptorCount[t] += setCount*layout->getDescriptorRedirect(type).getTotalCount();
                }
            }
        
            return createDescriptorPool(createInfo);
        }
        // Fill out the descriptor sets with descriptors
        bool updateDescriptorSets(uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies);
















        //! Renderpasses and Framebuffers
        virtual core::smart_refctd_ptr<IGPURenderpass> createRenderpass(const IGPURenderpass::SCreationParams& params) = 0;
        core::smart_refctd_ptr<IGPUFramebuffer> createFramebuffer(IGPUFramebuffer::SCreationParams&& params)
        {
            if (!params.renderpass->wasCreatedBy(this))
                return nullptr;
            if (!IGPUFramebuffer::validate(params))
                return nullptr;
            return createFramebuffer_impl(std::move(params));
        }

        //! Pipelines
        // Create a pipeline cache object
        virtual core::smart_refctd_ptr<IGPUPipelineCache> createPipelineCache() { return nullptr; }
        











        core::smart_refctd_ptr<IGPUComputePipeline> createComputePipeline(
            IGPUPipelineCache* _pipelineCache,
            core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
            core::smart_refctd_ptr<IGPUSpecializedShader>&& _shader
        ) {
            if (_pipelineCache && !_pipelineCache->wasCreatedBy(this))
                return nullptr;
            if (!_layout->wasCreatedBy(this))
                return nullptr;
            if (!_shader->wasCreatedBy(this))
                return nullptr;
            const char* debugName = _shader->getObjectDebugName();
            auto retval = createComputePipeline_impl(_pipelineCache, std::move(_layout), std::move(_shader));
            if (retval && debugName[0])
                retval->setObjectDebugName(debugName);
            return retval;
        }

        bool createComputePipelines(
            IGPUPipelineCache* pipelineCache,
            core::SRange<const IGPUComputePipeline::SCreationParams> createInfos,
            core::smart_refctd_ptr<IGPUComputePipeline>* output
        ) {
            if (pipelineCache && !pipelineCache->wasCreatedBy(this))
                return false;
            for (const auto& ci : createInfos)
                if (!ci.layout->wasCreatedBy(this) || !ci.shader->wasCreatedBy(this))
                    return false;
            return createComputePipelines_impl(pipelineCache, createInfos, output);
        }

        bool createComputePipelines(
            IGPUPipelineCache* pipelineCache,
            uint32_t count,
            const IGPUComputePipeline::SCreationParams* createInfos,
            core::smart_refctd_ptr<IGPUComputePipeline>* output
        ) 
        {
            auto ci = core::SRange<const IGPUComputePipeline::SCreationParams>{createInfos, createInfos+count};
            return createComputePipelines(pipelineCache, ci, output);
        }

        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> createRenderpassIndependentPipeline(
            IGPUPipelineCache* _pipelineCache,
            core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
            IGPUSpecializedShader* const* _shaders, IGPUSpecializedShader* const* _shadersEnd,
            const asset::SVertexInputParams& _vertexInputParams,
            const asset::SBlendParams& _blendParams,
            const asset::SPrimitiveAssemblyParams& _primAsmParams,
            const asset::SRasterizationParams& _rasterParams
        ) {
            if (_pipelineCache && !_pipelineCache->wasCreatedBy(this))
                return nullptr;
            if (!_layout->wasCreatedBy(this))
                return nullptr;
            for (auto s = _shaders; s != _shadersEnd; ++s)
            {
                if (!(*s)->wasCreatedBy(this))
                    return nullptr;
            }
            return createRenderpassIndependentPipeline_impl(_pipelineCache, std::move(_layout), _shaders, _shadersEnd, _vertexInputParams, _blendParams, _primAsmParams, _rasterParams);
        }

        bool createRenderpassIndependentPipelines(
            IGPUPipelineCache* pipelineCache,
            core::SRange<const IGPURenderpassIndependentPipeline::SCreationParams> createInfos,
            core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>* output
        ) {
            if (pipelineCache && !pipelineCache->wasCreatedBy(this))
                return false;
            for (const IGPURenderpassIndependentPipeline::SCreationParams& ci : createInfos)
            {
                if (!ci.layout->wasCreatedBy(this))
                    return false;
                for (auto& s : ci.shaders)
                    if (s && !s->wasCreatedBy(this))
                        return false;
            }
            return createRenderpassIndependentPipelines_impl(pipelineCache, createInfos, output);
        }

        bool createRenderpassIndependentPipelines(
            IGPUPipelineCache* pipelineCache,
            uint32_t count,
            const IGPURenderpassIndependentPipeline::SCreationParams* createInfos,
            core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>* output
        )
        {
            auto ci = core::SRange<const IGPURenderpassIndependentPipeline::SCreationParams>{createInfos, createInfos+count};
            return createRenderpassIndependentPipelines(pipelineCache, ci, output);
        }

        core::smart_refctd_ptr<IGPUGraphicsPipeline> createGraphicsPipeline(IGPUPipelineCache* pipelineCache, IGPUGraphicsPipeline::SCreationParams&& params)
        {
            if (pipelineCache && !pipelineCache->wasCreatedBy(this))
                return nullptr;
            if (!params.renderpass->wasCreatedBy(this))
                return nullptr;
            if (!params.renderpassIndependent->wasCreatedBy(this))
                return nullptr;
            if (!IGPUGraphicsPipeline::validate(params))
                return nullptr;
            return createGraphicsPipeline_impl(pipelineCache, std::move(params));
        }

        bool createGraphicsPipelines(IGPUPipelineCache* pipelineCache, core::SRange<const IGPUGraphicsPipeline::SCreationParams> params, core::smart_refctd_ptr<IGPUGraphicsPipeline>* output)
        {
            if (pipelineCache && !pipelineCache->wasCreatedBy(this))
                return false;
            for (const auto& ci : params)
            {
                if (!ci.renderpass->wasCreatedBy(this))
                    return false;
                if (!ci.renderpassIndependent->wasCreatedBy(this))
                    return false;
                if (!IGPUGraphicsPipeline::validate(ci))
                    return false;
            }
            return createGraphicsPipelines_impl(pipelineCache, params, output);
        }

        bool createGraphicsPipelines(IGPUPipelineCache* pipelineCache, uint32_t count, const IGPUGraphicsPipeline::SCreationParams* params, core::smart_refctd_ptr<IGPUGraphicsPipeline>* output)
        {
            auto ci = core::SRange<const IGPUGraphicsPipeline::SCreationParams>{ params, params + count };
            return createGraphicsPipelines(pipelineCache, ci, output);
        }
        







        // queries
        inline core::smart_refctd_ptr<IQueryPool> createQueryPool(const IQueryPool::SCreationParams& params)
        {
            switch (params.queryType)
            {
                case IQueryPool::TYPE::PIPELINE_STATISTICS:
                    if (!getEnabledFeatures().pipelineStatisticsQuery)
                        return false;
                    break;
                case IQueryPool::TYPE::ACCELERATION_STRUCTURE_COMPACTED_SIZE: [[fallthrough]];
                case IQueryPool::TYPE::ACCELERATION_STRUCTURE_SERIALIZATION_SIZE: [[fallthrough]];
                case IQueryPool::TYPE::ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS: [[fallthrough]];
                case IQueryPool::TYPE::ACCELERATION_STRUCTURE_SIZE:
                    if (!getEnabledFeatures().accelerationStructure)
                        return false;
                    break;
                default:
                    return false;
            }
            return createQueryPool_impl(params);
        }
        // `pData` must be sufficient to store the results (@see `IQueryPool::calcQueryResultsSize`)
        inline bool getQueryPoolResults(const IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount, void* const pData, const size_t stride, const core::bitflag<IQueryPool::RESULTS_FLAGS> flags)
        {
            if (!queryPool || !queryPool->wasCreatedBy(this))
                return false;
            if (firstQuery+queryCount>=queryPool->getCreationParameters().queryCount)
                return false;
            if (stride&((flags.hasFlags(IQueryPool::RESULTS_FLAGS::_64_BIT) ? alignof(uint64_t):alignof(uint32_t))-1))
                return false;
            if (queryPool->getCreationParameters().queryType==IQueryPool::TYPE::TIMESTAMP && flags.hasFlags(IQueryPool::RESULTS_FLAGS::PARTIAL_BIT))
                return false;
            return getQueryPoolResults_impl(queryPool,firstQuery,queryCount,pData,stride,flags);
        }

        //! Commandbuffers
        inline core::smart_refctd_ptr<IGPUCommandPool> createCommandPool(const uint32_t familyIx, const core::bitflag<IGPUCommandPool::CREATE_FLAGS> flags)
        {
            if (familyIx>=m_queueFamilyInfos->size())
                return nullptr;
            return createCommandPool_impl(familyIx,flags);
        }

        // Not implemented stuff:
        //TODO: vkGetDescriptorSetLayoutSupport
        //vkGetPipelineCacheData //as pipeline cache method?? (why not)
        //vkMergePipelineCaches //as pipeline cache method (why not)

        // Vulkan: const VkDevice*
        virtual const void* getNativeHandle() const = 0;

    protected:
        ILogicalDevice(core::smart_refctd_ptr<const IAPIConnection>&& api, const IPhysicalDevice* const physicalDevice, const SCreationParams& params);
        inline virtual ~ILogicalDevice()
        {
            if (m_queues)
            for (uint32_t i=0u; i<m_queues->size(); ++i)
                delete (*m_queues)[i];
        }

        inline bool invalidMappedRanges(const core::SRange<const MappedMemoryRange>& ranges)
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
        virtual bool flushMappedMemoryRanges_impl(const core::SRange<const MappedMemoryRange>& ranges) = 0;
        virtual bool invalidateMappedMemoryRanges_impl(const core::SRange<const MappedMemoryRange>& ranges) = 0;

        inline bool invalidAllocationForBind(const IDeviceMemoryBacked* resource, const IDeviceMemoryBacked::SMemoryBinding& binding, const size_t alignment)
        {
            if (!resource->wasCreatedBy(this))
            {
                m_logger.log("Buffer or Image %p not compatible with Device %p !",system::ILogger::ELL_ERROR,resource,this);
                return true;
            }
            if (!resource->getBoundMemory().isValid())
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
        virtual bool bindBufferMemory_impl(const uint32_t count, const SBindBufferMemoryInfo* pInfos) = 0;
        virtual bool bindImageMemory_impl(const uint32_t count, const SBindImageMemoryInfo* pInfos) = 0;

        virtual core::smart_refctd_ptr<IGPUBuffer> createBuffer_impl(IGPUBuffer::SCreationParams&& creationParams) = 0;
        virtual core::smart_refctd_ptr<IGPUBufferView> createBufferView_impl(const asset::SBufferRange<const IGPUBuffer>& underlying, const asset::E_FORMAT _fmt) = 0;
        virtual core::smart_refctd_ptr<IGPUImage> createImage_impl(IGPUImage::SCreationParams&& params) = 0;
        virtual core::smart_refctd_ptr<IGPUImageView> createImageView_impl(IGPUImageView::SCreationParams&& params) = 0;
        inline bool invalidCreationParams(const IGPUAccelerationStructure::SCreationParams& params)
        {
            if (!getEnabledFeatures().accelerationStructure)
                return true;
            constexpr size_t MinAlignment = 256u;
            if (!params.bufferRange.isValid() || !params.bufferRange.buffer->wasCreatedBy(this) || (params.bufferRange.offset&(MinAlignment-1))!=0u)
                return true;
            const auto bufferUsages = params.bufferRange.buffer->getCreationParams().usage;
            if (!bufferUsages.hasFlags(IGPUBuffer::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT))
                return true;
            if (params.flags.hasFlags(IGPUAccelerationStructure::SCreationParams::FLAGS::MOTION_BIT) && !getEnabledFeatures().rayTracingMotionBlur)
                return true;
            return false;
        }
        virtual core::smart_refctd_ptr<IGPUBottomLevelAccelerationStructure> createBottomLevelAccelerationStructure_impl(IGPUAccelerationStructure::SCreationParams&& params) = 0;
        virtual core::smart_refctd_ptr<IGPUTopLevelAccelerationStructure> createTopLevelAccelerationStructure_impl(IGPUTopLevelAccelerationStructure::SCreationParams&& params) = 0;

        template<class Accelera>
        bool invalidFeatures(const bool motionBlur) const
        {
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkGetAccelerationStructureBuildSizesKHR-accelerationStructure-08933
            if (!m_enabledFeatures.accelerationStructure)
                return true;
			// not sure of VUID
			if (std::is_same_v<buffer_t,asset::ICPUBuffer> && !m_enabledFeatures.accelerationStructureHostCommands)
				return true;
            // not sure of VUID
            if (motionBlur && !m_enabledFeatures.rayTracingMotionBlur)
                return true;

            return false;
        }
        virtual AccelerationStructureBuildSizes getAccelerationStructureBuildSizes_impl(
            const core::bitflag<IGPUBottomLevelAccelerationStructure::BUILD_FLAGS> flags, const bool motionBlur,
            const core::SRange<const IGPUBottomLevelAccelerationStructure::AABBs<IGPUBuffer>>& geometries, const uint32_t* const pMaxPrimitiveCounts
        ) const = 0;
        virtual AccelerationStructureBuildSizes getAccelerationStructureBuildSizes_impl(
            const core::bitflag<IGPUBottomLevelAccelerationStructure::BUILD_FLAGS> flags, const bool motionBlur,
            const core::SRange<const IGPUBottomLevelAccelerationStructure::AABBs<asset::ICPUBuffer>>& geometries, const uint32_t* const pMaxPrimitiveCounts
        ) const = 0;
        virtual AccelerationStructureBuildSizes getAccelerationStructureBuildSizes_impl(
            const core::bitflag<IGPUBottomLevelAccelerationStructure::BUILD_FLAGS> flags, const bool motionBlur,
            const core::SRange<const IGPUBottomLevelAccelerationStructure::Triangles<IGPUBuffer>>& geometries, const uint32_t* const pMaxPrimitiveCounts
        ) const = 0;
        virtual AccelerationStructureBuildSizes getAccelerationStructureBuildSizes_impl(
            const core::bitflag<IGPUBottomLevelAccelerationStructure::BUILD_FLAGS> flags, const bool motionBlur,
            const core::SRange<const IGPUBottomLevelAccelerationStructure::Triangles<asset::ICPUBuffer>>& geometries, const uint32_t* const pMaxPrimitiveCounts
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
        inline bool acquireDeferredOperation(IDeferredOperation* const deferredOp)
        {
            if (!deferredOp || !deferredOp->wasCreatedBy(this) || deferredOp->isPending())
                return false;
            deferredOp->m_resourceTracking.clear();
            return getEnabledFeatures().accelerationStructureHostCommands;
        }
        static inline bool invalidMemoryForAccelerationStructureHostOperations(const IDeviceMemoryAllocation* const memory)
        {
            return !memory || !memory->isMappable() || !memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_DEVICE_LOCAL_BIT);
        }
        virtual DEFERRABLE_RESULT buildAccelerationStructures_impl(IDeferredOperation* const deferredOperation, const core::SRange<const IGPUBottomLevelAccelerationStructure::HostBuildInfo>& infos, const IGPUAccelerationStructure::BuildRangeInfo* const* const ppBuildRangeInfos) = 0;
        virtual DEFERRABLE_RESULT buildAccelerationStructures_impl(IDeferredOperation* const deferredOperation, const core::SRange<const IGPUTopLevelAccelerationStructure::HostBuildInfo>& infos, const IGPUAccelerationStructure::BuildRangeInfo* const pBuildRangeInfos) = 0;
        virtual bool writeAccelerationStructuresProperties_impl(const uint32_t count, const IGPUAccelerationStructure* const* const accelerationStructures, const IQueryPool::TYPE type, size_t* data, const size_t stride) = 0;
        virtual DEFERRABLE_RESULT copyAccelerationStructure_impl(IDeferredOperation* const deferredOperation, const IGPUAccelerationStructure::CopyInfo& copyInfo) = 0;
        virtual DEFERRABLE_RESULT copyAccelerationStructureToMemory_impl(IDeferredOperation* const deferredOperation, const IGPUAccelerationStructure::HostCopyToMemoryInfo& copyInfo) = 0;
        virtual DEFERRABLE_RESULT copyAccelerationStructureFromMemory_impl(IDeferredOperation* const deferredOperation, const IGPUAccelerationStructure::HostCopyFromMemoryInfo& copyInfo) = 0;

        virtual core::smart_refctd_ptr<IGPUSpecializedShader> createSpecializedShader_impl(const IGPUShader* _unspecialized, const asset::ISpecializedShader::SInfo& _specInfo) = 0;

        constexpr static inline auto MaxStagesPerPipeline = 6u;
        virtual core::smart_refctd_ptr<IGPUDescriptorSetLayout> createDescriptorSetLayout_impl(const core::SRange<const IGPUDescriptorSetLayout::SBinding>& bindings, const uint32_t maxSamplersCount) = 0;
        virtual core::smart_refctd_ptr<IGPUPipelineLayout> createPipelineLayout_impl(
            const core::SRange<const asset::SPushConstantRange>& pcRanges,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3
        ) = 0;

        virtual core::smart_refctd_ptr<IDescriptorPool> createDescriptorPool_impl(const IDescriptorPool::SCreateInfo& createInfo) = 0;









        virtual core::smart_refctd_ptr<IGPUFramebuffer> createFramebuffer_impl(IGPUFramebuffer::SCreationParams&& params) = 0;
        virtual void updateDescriptorSets_impl(
            const uint32_t descriptorWriteCount,
            const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites,
            uint32_t descriptorCopyCount,
            const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies
        ) = 0;

        virtual core::smart_refctd_ptr<IGPUComputePipeline> createComputePipeline_impl(
            IGPUPipelineCache* _pipelineCache,
            core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
            core::smart_refctd_ptr<IGPUSpecializedShader>&& _shader
        ) = 0;
        virtual bool createComputePipelines_impl(
            IGPUPipelineCache* pipelineCache,
            core::SRange<const IGPUComputePipeline::SCreationParams> createInfos,
            core::smart_refctd_ptr<IGPUComputePipeline>* output
        ) = 0;
        virtual core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> createRenderpassIndependentPipeline_impl(
            IGPUPipelineCache* _pipelineCache,
            core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
            IGPUSpecializedShader* const* _shaders, IGPUSpecializedShader* const* _shadersEnd,
            const asset::SVertexInputParams& _vertexInputParams,
            const asset::SBlendParams& _blendParams,
            const asset::SPrimitiveAssemblyParams& _primAsmParams,
            const asset::SRasterizationParams& _rasterParams
        ) = 0;
        virtual bool createRenderpassIndependentPipelines_impl(
            IGPUPipelineCache* pipelineCache,
            core::SRange<const IGPURenderpassIndependentPipeline::SCreationParams> createInfos,
            core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>* output
        ) = 0;
        virtual core::smart_refctd_ptr<IGPUGraphicsPipeline> createGraphicsPipeline_impl(IGPUPipelineCache* pipelineCache, IGPUGraphicsPipeline::SCreationParams&& params) = 0;
        virtual bool createGraphicsPipelines_impl(IGPUPipelineCache* pipelineCache, core::SRange<const IGPUGraphicsPipeline::SCreationParams> params, core::smart_refctd_ptr<IGPUGraphicsPipeline>* output) = 0;

        



        virtual core::smart_refctd_ptr<IQueryPool> createQueryPool_impl(const IQueryPool::SCreationParams& params) = 0;
        virtual bool getQueryPoolResults_impl(const IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount, void* const pData, const size_t stride, const core::bitflag<IQueryPool::RESULTS_FLAGS> flags) = 0;

        virtual core::smart_refctd_ptr<IGPUCommandPool> createCommandPool_impl(const uint32_t familyIx, const core::bitflag<IGPUCommandPool::CREATE_FLAGS> flags) = 0;


        void addCommonShaderDefines(std::ostringstream& pool, const bool runningInRenderDoc);

        template<typename... Args>
        inline void addShaderDefineToPool(std::ostringstream& pool, const char* define, Args&&... args)
        {
            const ptrdiff_t pos = pool.tellp();
            m_extraShaderDefines.push_back(reinterpret_cast<const char*>(pos));
            pool << define << " ";
            ((pool << std::forward<Args>(args)), ...);
        }
        inline void finalizeShaderDefinePool(std::ostringstream&& pool)
        {
            m_ShaderDefineStringPool.resize(static_cast<size_t>(pool.tellp())+m_extraShaderDefines.size());
            const auto data = ptrdiff_t(m_ShaderDefineStringPool.data());

            const auto str = pool.str();
            size_t nullCharsWritten = 0u;
            for (auto i=0u; i<m_extraShaderDefines.size(); i++)
            {
                auto& dst = m_extraShaderDefines[i];
                const auto len = (i!=(m_extraShaderDefines.size()-1u) ? ptrdiff_t(m_extraShaderDefines[i+1]):str.length())-ptrdiff_t(dst);
                const char* src = str.data()+ptrdiff_t(dst);
                dst += data+(nullCharsWritten++);
                memcpy(const_cast<char*>(dst),src,len);
                const_cast<char*>(dst)[len] = 0;
            }
        }


        core::smart_refctd_ptr<asset::CCompilerSet> m_compilerSet;
        core::smart_refctd_ptr<const IAPIConnection> m_api;
        const IPhysicalDevice* const m_physicalDevice;
        const system::logger_opt_ptr m_logger;

        SPhysicalDeviceFeatures m_enabledFeatures;

        core::vector<char> m_ShaderDefineStringPool;
        core::vector<const char*> m_extraShaderDefines;

        using queues_array_t = core::smart_refctd_dynamic_array<CThreadSafeQueueAdapter*>;
        queues_array_t m_queues;
        struct QueueFamilyInfo
        {
            core::bitflag<asset::PIPELINE_STAGE_FLAGS> supportedStages = asset::PIPELINE_STAGE_FLAGS::NONE;
            core::bitflag<asset::ACCESS_FLAGS> supportedAccesses = asset::ACCESS_FLAGS::NONE;
            // index into flat array of `m_queues`
            uint32_t firstQueueIndex = 0u;
        };
        using q_family_info_array_t = core::smart_refctd_dynamic_array<const QueueFamilyInfo>;
        q_family_info_array_t m_queueFamilyInfos;
};


template<typename ResourceBarrier>
inline bool ILogicalDevice::validateMemoryBarrier(const uint32_t queueFamilyIndex, const IGPUCommandBuffer::SImageMemoryBarrier<ResourceBarrier>& barrier) const
{
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-image-parameter
    if (!barrier.image)
        return false;
    const auto& params = barrier.image->getCreationParameters();

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-subresourceRange-01486
    if (barrier.subresourceRange.baseMipLevel>=params.mipLevels)
        return false;
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-subresourceRange-01488
    if (barrier.subresourceRange.baseArrayLayer>=params.arrayLayers)
        return false;
    // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-subresourceRange-01724
    // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-subresourceRange-01725

    const auto aspectMask = barrier.subresourceRange.aspectMask;
    if (asset::isDepthOrStencilFormat(params.format))
    {
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-image-03319
        constexpr auto DepthStencilAspects = IGPUImage::EAF_DEPTH_BIT|IGPUImage::EAF_STENCIL_BIT;
        if (aspectMask.value&(~DepthStencilAspects))
            return false;
        if (bool(aspectMask.value&DepthStencilAspects))
            return false;
    }
    //https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-image-01671
    else if (aspectMask!=IGPUImage::EAF_COLOR_BIT)
        return false;

    // TODO: better check for queue family ownership transfer
    const bool ownershipTransfer = barrier.barrier.srcQueueFamilyIndex!=barrier.barrier.dstQueueFamilyIndex;
    const bool layoutTransform = barrier.oldLayout!=barrier.newLayout;
    // TODO: WTF ? https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-srcStageMask-03855
    if (ownershipTransfer || layoutTransform)
    {
        const bool srcStageIsHost = barrier.barrier.barrier.srcStageMask.hasFlags(asset::PIPELINE_STAGE_FLAGS::HOST_BIT);
        auto mismatchedLayout = [&params,aspectMask,srcStageIsHost]<bool dst>(const IGPUImage::LAYOUT layout) -> bool
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
                    if (srcStageIsHost)
                        return true;
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-03938
                    if (aspectMask==IGPUImage::EAF_COLOR_BIT && !params.usage.hasFlags(IGPUImage::E_USAGE_FLAGS::EUF_COLOR_ATTACHMENT_BIT))
                        return true;
                    if (aspectMask.hasFlags(IGPUImage::EAF_DEPTH_BIT) && !params.usage.hasFlags(IGPUImage::E_USAGE_FLAGS::EUF_DEPTH_STENCIL_ATTACHMENT_BIT))
                        return true;
                    if (aspectMask.hasFlags(IGPUImage::EAF_STENCIL_BIT) && !params.actualStencilUsage().hasFlags(IGPUImage::E_USAGE_FLAGS::EUF_DEPTH_STENCIL_ATTACHMENT_BIT))
                        return true;
                    break;
                case IGPUImage::LAYOUT::READ_ONLY_OPTIMAL:
                    if (srcStageIsHost)
                        return true;
                    {
                        constexpr auto ValidUsages = IGPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT|IGPUImage::E_USAGE_FLAGS::EUF_INPUT_ATTACHMENT_BIT;
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-oldLayout-01211
                        if (aspectMask.hasFlags(IGPUImage::EAF_STENCIL_BIT))
                        {
                            if (!bool(params.actualStencilUsage()&ValidUsages))
                                return true;
                        }
                        else if (!bool(params.usage&ValidUsages))
                            return true;
                    }
                    break;
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-oldLayout-01212
                case IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL:
                    if (srcStageIsHost || !params.usage.hasFlags(IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT))
                        return true;
                    break;
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-oldLayout-01213
                case IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL:
                    if (srcStageIsHost || !params.usage.hasFlags(IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT))
                        return true;
                    break;
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageMemoryBarrier2-oldLayout-01198
                case IGPUImage::LAYOUT::UNDEFINED: [[fallthrough]]
                case IGPUImage::LAYOUT::PREINITIALIZED:
                    if constexpr (dst)
                        return true;
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
    return validateMemoryBarrier(queueFamilyIndex,barrier.barrier,barrier.image->getCachedCreationParams().isConcurrentSharing());
}

}


#endif