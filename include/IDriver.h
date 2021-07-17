// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_DRIVER_H_INCLUDED__
#define __NBL_I_DRIVER_H_INCLUDED__

#include "nbl/asset/asset.h"
#include "nbl/video/asset_traits.h"

#if 0
namespace nbl
{

namespace video
{
	class IGPUMeshDataFormatDesc;
	class IGPUObjectFromAssetConverter;
	class CPropertyPoolHandler;
}
}

#include "nbl/video/IGPUPipelineCache.h"
#include "nbl/video/IGPUImageView.h"
#include "nbl/asset/utils/ISPIRVOptimizer.h"

#include "IFrameBuffer.h"
#include "IVideoCapabilityReporter.h"
#include "IQueryObject.h"
#include "IGPUTimestampQuery.h"
#include "IDriverFence.h"

namespace nbl
{
namespace video
{

//! Interface to the functionality of the graphics API device which does not require the submission of GPU commands onto a queue.
/** This interface only deals with OpenGL and Vulkan concepts which do not require a command to be recorded in a command buffer
and then submitted to a command queue, i.e. functions which only require VkDevice or VkPhysicalDevice.
Examples of such functionality are the creation of buffers, textures, etc.*/
class IDriver : public virtual core::IReferenceCounted, public IVideoCapabilityReporter, public core::QuitSignalling
{
    protected:
        asset::IAssetManager* m_assetMgr;
		//core::smart_refctd_ptr<StreamingTransientDataBufferMT<> > defaultDownloadBuffer;
		//core::smart_refctd_ptr<StreamingTransientDataBufferMT<> > defaultUploadBuffer;

        inline IDriver(asset::IAssetManager* assmgr) : IVideoCapabilityReporter(), m_assetMgr(assmgr)//, defaultDownloadBuffer(nullptr), defaultUploadBuffer(nullptr) 
        {}

        virtual ~IDriver()
        {
        }
    public:
		struct SBindBufferMemoryInfo
		{
			IGPUBuffer* buffer;
			IDriverMemoryAllocation* memory;
			size_t offset;
		};
		struct SBindImageMemoryInfo
		{
			IGPUImage* image;
			IDriverMemoryAllocation* memory;
			size_t offset;
		};

        //! needs to be dropped (smart_refctd_ptr will do automatically) since its not refcounted by GPU driver internally
        /** Since not owned by any openGL context and hence not owned by driver.
        You normally need to call glFlush() after placing a fence
        \param whether to perform an implicit flush the first time CPU waiting,
        this only works if the first wait is from the same thread as the one which
        placed the fence. **/
		virtual core::smart_refctd_ptr<IDriverFence> placeFence(const bool& implicitFlushWaitSameThread = false) = 0;

        static inline IDriverMemoryBacked::SDriverMemoryRequirements getDeviceLocalGPUMemoryReqs()
        {
            IDriverMemoryBacked::SDriverMemoryRequirements reqs;
            reqs.vulkanReqs.alignment = 0;
            reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
            reqs.memoryHeapLocation = IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
            reqs.mappingCapability = IDriverMemoryAllocation::EMCF_CANNOT_MAP;
            reqs.prefersDedicatedAllocation = true;
            reqs.requiresDedicatedAllocation = true;
            return reqs;
        }
        static inline IDriverMemoryBacked::SDriverMemoryRequirements getSpilloverGPUMemoryReqs()
        {
            IDriverMemoryBacked::SDriverMemoryRequirements reqs;
            reqs.vulkanReqs.alignment = 0;
            reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
            reqs.memoryHeapLocation = IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL;
            reqs.mappingCapability = IDriverMemoryAllocation::EMCF_CANNOT_MAP;
            reqs.prefersDedicatedAllocation = true;
            reqs.requiresDedicatedAllocation = true;
            return reqs;
        }
        static inline IDriverMemoryBacked::SDriverMemoryRequirements getUpStreamingMemoryReqs()
        {
            IDriverMemoryBacked::SDriverMemoryRequirements reqs;
            reqs.vulkanReqs.alignment = 0;
            reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
            reqs.memoryHeapLocation = IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
            reqs.mappingCapability = IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_WRITE;
            reqs.prefersDedicatedAllocation = true;
            reqs.requiresDedicatedAllocation = true;
            return reqs;
        }
        static inline IDriverMemoryBacked::SDriverMemoryRequirements getDownStreamingMemoryReqs()
        {
            IDriverMemoryBacked::SDriverMemoryRequirements reqs;
            reqs.vulkanReqs.alignment = 0;
            reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
            reqs.memoryHeapLocation = IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL;
            reqs.mappingCapability = IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ|IDriverMemoryAllocation::EMCF_CACHED;
            reqs.prefersDedicatedAllocation = true;
            reqs.requiresDedicatedAllocation = true;
            return reqs;
        }
        static inline IDriverMemoryBacked::SDriverMemoryRequirements getCPUSideGPUVisibleGPUMemoryReqs()
        {
            IDriverMemoryBacked::SDriverMemoryRequirements reqs;
            reqs.vulkanReqs.alignment = 0;
            reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
            reqs.memoryHeapLocation = IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL;
            reqs.mappingCapability = IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ|IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_WRITE|IDriverMemoryAllocation::EMCF_COHERENT|IDriverMemoryAllocation::EMCF_CACHED;
            reqs.prefersDedicatedAllocation = true;
            reqs.requiresDedicatedAllocation = true;
            return reqs;
        }

        //! Best for Mesh data, UBOs, SSBOs, etc.
        virtual core::smart_refctd_ptr<IDriverMemoryAllocation> allocateDeviceLocalMemory(const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs) {return nullptr;}

        //! If cannot or don't want to use device local memory, then this memory can be used
        /** If the above fails (only possible on vulkan) or we have perfomance hitches due to video memory oversubscription.*/
        virtual core::smart_refctd_ptr<IDriverMemoryAllocation> allocateSpilloverMemory(const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs) {return nullptr;}

        //! Best for staging uploads to the GPU, such as resource streaming, and data to update the above memory with
        virtual core::smart_refctd_ptr<IDriverMemoryAllocation> allocateUpStreamingMemory(const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs) {return nullptr;}

        //! Best for staging downloads from the GPU, such as query results, Z-Buffer, video frames for recording, etc.
        virtual core::smart_refctd_ptr<IDriverMemoryAllocation> allocateDownStreamingMemory(const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs) {return nullptr;}

        //! Should be just as fast to play around with on the CPU as regular malloc'ed memory, but slowest to access with GPU
        virtual core::smart_refctd_ptr<IDriverMemoryAllocation> allocateCPUSideGPUVisibleMemory(const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs) {return nullptr;}


        //! For memory allocations without the video::IDriverMemoryAllocation::EMCF_COHERENT mapping capability flag you need to call this for the CPU writes to become GPU visible
        virtual void flushMappedMemoryRanges(uint32_t memoryRangeCount, const video::IDriverMemoryAllocation::MappedMemoryRange* pMemoryRanges) {}

        //! Utility wrapper for the pointer based func
        inline void flushMappedMemoryRanges(const core::vector<video::IDriverMemoryAllocation::MappedMemoryRange>& ranges)
        {
            this->flushMappedMemoryRanges(static_cast<uint32_t>(ranges.size()),ranges.data());
        }

        //! For memory allocations without the video::IDriverMemoryAllocation::EMCF_COHERENT mapping capability flag you need to call this for the GPU writes to become CPU visible (slow on OpenGL)
        virtual void invalidateMappedMemoryRanges(uint32_t memoryRangeCount, const video::IDriverMemoryAllocation::MappedMemoryRange* pMemoryRanges) {}

        //! Utility wrapper for the pointer based func
        inline void invalidateMappedMemoryRanges(const core::vector<video::IDriverMemoryAllocation::MappedMemoryRange>& ranges)
        {
            this->invalidateMappedMemoryRanges(static_cast<uint32_t>(ranges.size()),ranges.data());
        }


        //! Low level function used to implement the above, use with caution
        virtual core::smart_refctd_ptr<IGPUBuffer> createGPUBuffer(const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs, const bool canModifySubData=false) {return nullptr;}

		//! Binds memory allocation to provide the backing for the resource.
		/** Available only on Vulkan, in OpenGL all resources create their own memory implicitly,
		so pooling or aliasing memory for different resources is not possible.
		There is no unbind, so once memory is bound it remains bound until you destroy the resource object.
		Actually all resource classes in OpenGL implement both IDriverMemoryBacked and IDriverMemoryAllocation,
		so effectively the memory is pre-bound at the time of creation.
		\return true on success, always false under OpenGL.*/
		virtual bool bindBufferMemory(uint32_t bindInfoCount, const SBindBufferMemoryInfo* pBindInfos) { return false; }

        //! Creates the buffer, allocates memory dedicated memory and binds it at once.
        inline core::smart_refctd_ptr<IGPUBuffer> createDeviceLocalGPUBufferOnDedMem(size_t size)
        {
            auto reqs = getDeviceLocalGPUMemoryReqs();
            reqs.vulkanReqs.size = size;
            return this->createGPUBufferOnDedMem(reqs,false);
        }

        //! Creates the buffer, allocates memory dedicated memory and binds it at once.
        inline core::smart_refctd_ptr<IGPUBuffer> createSpilloverGPUBufferOnDedMem(size_t size)
        {
            auto reqs = getSpilloverGPUMemoryReqs();
            reqs.vulkanReqs.size = size;
            return this->createGPUBufferOnDedMem(reqs,false);
        }

        //! Creates the buffer, allocates memory dedicated memory and binds it at once.
        inline core::smart_refctd_ptr<IGPUBuffer> createUpStreamingGPUBufferOnDedMem(size_t size)
        {
            auto reqs = getUpStreamingMemoryReqs();
            reqs.vulkanReqs.size = size;
            return this->createGPUBufferOnDedMem(reqs,false);
        }

        //! Creates the buffer, allocates memory dedicated memory and binds it at once.
        inline core::smart_refctd_ptr<IGPUBuffer> createDownStreamingGPUBufferOnDedMem(size_t size)
        {
            auto reqs = getDownStreamingMemoryReqs();
            reqs.vulkanReqs.size = size;
            return this->createGPUBufferOnDedMem(reqs,false);
        }

        //! Creates the buffer, allocates memory dedicated memory and binds it at once.
        inline core::smart_refctd_ptr<IGPUBuffer> createCPUSideGPUVisibleGPUBufferOnDedMem(size_t size)
        {
            auto reqs = getCPUSideGPUVisibleGPUMemoryReqs();
            reqs.vulkanReqs.size = size;
            return this->createGPUBufferOnDedMem(reqs,false);
        }

        //! Low level function used to implement the above, use with caution
        virtual core::smart_refctd_ptr<IGPUBuffer> createGPUBufferOnDedMem(const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs, const bool canModifySubData=false) {return nullptr;}


		//! The counterpart of @see bindBufferMemory for images
		virtual bool bindImageMemory(uint32_t bindInfoCount, const SBindImageMemoryInfo* pBindInfos) { return false; }

		//! Creates the Image, allocates dedicated memory and binds it at once.
		inline core::smart_refctd_ptr<IGPUImage> createDeviceLocalGPUImageOnDedMem(IGPUImage::SCreationParams&& params)
		{
			auto reqs = getDeviceLocalGPUMemoryReqs();
			return this->createGPUImageOnDedMem(std::move(params),reqs);
		}

		//!
		virtual core::smart_refctd_ptr<IGPUImage> createGPUImageOnDedMem(IGPUImage::SCreationParams&& params, const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs) { return nullptr;}

		//!
		inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalGPUImageOnDedMem(IGPUImage::SCreationParams&& params, IGPUBuffer* srcBuffer, uint32_t regionCount, const IGPUImage::SBufferCopy* pRegions)
		{
			auto retval = createDeviceLocalGPUImageOnDedMem(std::move(params));
			this->copyBufferToImage(srcBuffer,retval.get(),regionCount,pRegions);
			return retval;
		}
		inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalGPUImageOnDedMem(IGPUImage::SCreationParams&& params, IGPUImage* srcImage, uint32_t regionCount, const IGPUImage::SImageCopy* pRegions)
		{
			auto retval = createDeviceLocalGPUImageOnDedMem(std::move(params));
			this->copyImage(srcImage,retval.get(),regionCount,pRegions);
			return retval;
		}

        virtual core::smart_refctd_ptr<IGPUPipelineCache> createGPUPipelineCache() { return nullptr; }

		//!
		virtual CPropertyPoolHandler* getDefaultPropertyPoolHandler() const = 0;

	//====================== THIS STUFF NEEDS A REWRITE =====================


        //these will have to be created by a query pool anyway
        virtual IQueryObject* createPrimitivesGeneratedQuery() {return nullptr;}
        virtual IQueryObject* createElapsedTimeQuery() {return nullptr;}
        virtual IGPUTimestampQuery* createTimestampQuery() {return nullptr;}



	//====================== THIS STUFF SHOULD BE IN A video::ICommandBuffer =====================
        //!
        virtual void fillBuffer(IGPUBuffer* buffer, size_t offset, size_t length, uint32_t value) {}

		//! TODO: make with VkBufferCopy and take a list of multiple copies to carry out (maybe rename to copyBufferRanges)
		virtual void copyBuffer(IGPUBuffer* readBuffer, IGPUBuffer* writeBuffer, size_t readOffset, size_t writeOffset, size_t length) {}

		//!
		virtual void copyImage(IGPUImage* srcImage, IGPUImage* dstImage, uint32_t regionCount, const IGPUImage::SImageCopy* pRegions) {}

		//!
		virtual void copyBufferToImage(IGPUBuffer* srcBuffer, IGPUImage* dstImage, uint32_t regionCount, const IGPUImage::SBufferCopy* pRegions) {}

		//!
		virtual void copyImageToBuffer(IGPUImage* srcImage, IGPUBuffer* dstBuffer, uint32_t regionCount, const IGPUImage::SBufferCopy* pRegions) {}
};

} // end namespace video
} // end namespace nbl
#endif

#endif

