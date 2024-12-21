#ifndef _NBL_VIDEO_I_PHYSICAL_DEVICE_H_INCLUDED_
#define _NBL_VIDEO_I_PHYSICAL_DEVICE_H_INCLUDED_


#include "nbl/core/util/bitflag.h"

#include "nbl/system/declarations.h"
#include "nbl/system/ISystem.h"

#include "nbl/asset/IImage.h" //for VkExtent3D only

#include "nbl/video/EApiType.h"
#include "nbl/video/debug/IDebugCallback.h"
#include "nbl/video/ILogicalDevice.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

#include "nbl/video/SPhysicalDeviceLimits.h"
#include "nbl/video/SPhysicalDeviceFeatures.h"


#include <type_traits>


namespace nbl::video
{

class NBL_API2 IPhysicalDevice : public core::Interface, public core::Unmovable
{
    public:
        //
        virtual E_API_TYPE getAPIType() const = 0;

        enum E_TYPE : uint8_t {
            ET_UNKNOWN = 0u,
            ET_INTEGRATED_GPU = 1u<<0u,
            ET_DISCRETE_GPU = 1u<<1u,
            ET_VIRTUAL_GPU = 1u<<2u,
            ET_CPU = 1u<<3u,
        };

        enum E_DRIVER_ID : uint32_t
        {
            EDI_UNKNOWN                     = 0u,
            EDI_AMD_PROPRIETARY             = 1u << 0u,
            EDI_AMD_OPEN_SOURCE             = 1u << 1u,
            EDI_MESA_RADV                   = 1u << 2u,
            EDI_NVIDIA_PROPRIETARY          = 1u << 3u,
            EDI_INTEL_PROPRIETARY_WINDOWS   = 1u << 4u,
            EDI_INTEL_OPEN_SOURCE_MESA      = 1u << 5u,
            EDI_IMAGINATION_PROPRIETARY     = 1u << 6u,
            EDI_QUALCOMM_PROPRIETARY        = 1u << 7u,
            EDI_ARM_PROPRIETARY             = 1u << 8u,
            EDI_GOOGLE_SWIFTSHADER          = 1u << 9u,
            EDI_GGP_PROPRIETARY             = 1u << 10u,
            EDI_BROADCOM_PROPRIETARY        = 1u << 11u,
            EDI_MESA_LLVMPIPE               = 1u << 12u,
            EDI_MOLTENVK                    = 1u << 13u,
            EDI_COREAVI_PROPRIETARY         = 1u << 14u,
            EDI_JUICE_PROPRIETARY           = 1u << 15u,
            EDI_VERISILICON_PROPRIETARY     = 1u << 16u,
            EDI_MESA_TURNIP                 = 1u << 17u,
            EDI_MESA_V3DV                   = 1u << 18u,
            EDI_MESA_PANVK                  = 1u << 19u,
            EDI_SAMSUNG_PROPRIETARY         = 1u << 20u,
            EDI_MESA_VENUS                  = 1u << 21u,
        };

        //
        struct APIVersion
        {
            uint32_t major : 8;
            uint32_t minor : 8;
            uint32_t subminor : 8;
            uint32_t patch : 8;

            inline auto operator <=> (uint32_t vkApiVersion) const { return vkApiVersion - VK_MAKE_API_VERSION(0, major, minor, patch); }
            inline auto operator <=> (const APIVersion& other) const 
            {
                if(major != other.major) return static_cast<uint32_t>(other.major - major);
                if(minor != other.minor) return static_cast<uint32_t>(other.minor - minor);
                if(subminor != other.subminor) return static_cast<uint32_t>(other.subminor - subminor);
                if(patch != other.patch) return static_cast<uint32_t>(other.patch - patch);
                return 0u;
            }
        };

        using SLimits = SPhysicalDeviceLimits;
        using SFeatures = SPhysicalDeviceFeatures;

        struct SProperties
        {
            /* Vulkan 1.0 Core  */
            APIVersion  apiVersion;
            uint32_t    driverVersion;
            uint32_t    vendorID;
            uint32_t    deviceID;
            E_TYPE      deviceType;
            char        deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
            uint8_t     pipelineCacheUUID[VK_UUID_SIZE];
            SLimits     limits; // Contains Limits on Vulkan 1.0 Core , 1.1, 1.2 and extensions
            
            /* Vulkan 1.1 Core  */
            uint8_t     deviceUUID[VK_UUID_SIZE];
            uint8_t     driverUUID[VK_UUID_SIZE];
            uint8_t     deviceLUID[VK_LUID_SIZE];
            uint32_t    deviceNodeMask;
            bool        deviceLUIDValid;

            /* Vulkan 1.2 Core  */
            E_DRIVER_ID driverID;
            char driverName[VK_MAX_DRIVER_NAME_SIZE];
            char driverInfo[VK_MAX_DRIVER_INFO_SIZE];
            APIVersion conformanceVersion;
        };

        const SProperties& getProperties() const { return m_initData.properties; }
        const SLimits& getLimits() const { return m_initData.properties.limits; }
        APIVersion getAPIVersion() const { return m_initData.properties.apiVersion; }
        const SFeatures& getFeatures() const { return m_initData.features; }

        struct MemoryType
        {
            core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS> propertyFlags = IDeviceMemoryAllocation::EMPF_NONE;
            uint32_t heapIndex = ~0u;
        };

        struct MemoryHeap
        {
            size_t size = 0u;
            core::bitflag<IDeviceMemoryAllocation::E_MEMORY_HEAP_FLAGS> flags = IDeviceMemoryAllocation::EMHF_NONE;
        };

        // Decision: do not expose as of this moment
        /* MemoryProperties2
            - VkPhysicalDeviceMemoryBudgetPropertiesEXT
                provided by VK_EXT_memory_budget
                devices not supporting this seem to be mostly mobile

                VkDeviceSize       heapBudget[VK_MAX_MEMORY_HEAPS];
                VkDeviceSize       heapUsage[VK_MAX_MEMORY_HEAPS];
        */
        //
        struct SMemoryProperties
        {
            uint32_t        memoryTypeCount = 0u;
            MemoryType      memoryTypes[VK_MAX_MEMORY_TYPES] = {};
            uint32_t        memoryHeapCount = 0u;
            MemoryHeap      memoryHeaps[VK_MAX_MEMORY_HEAPS] = {};
        };
        const SMemoryProperties& getMemoryProperties() const { return m_initData.memoryProperties; }
        
        //! Bit `i` in MemoryTypeBitss will be set if m_initData.memoryProperties.memoryTypes[i] has the `flags`
        uint32_t getMemoryTypeBitsFromMemoryTypeFlags(core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS> flags) const
        {
            uint32_t ret = 0u;
            for(uint32_t i=0; i<m_initData.memoryProperties.memoryTypeCount; ++i)
            if(m_initData.memoryProperties.memoryTypes[i].propertyFlags.hasFlags(flags))
                ret |= (1u << i);
            return ret;
        }

        //! DeviceLocal: most efficient for device access
        //! Requires EMPF_DEVICE_LOCAL_BIT from MemoryTypes
        uint32_t getDeviceLocalMemoryTypeBits() const
        {
            return getMemoryTypeBitsFromMemoryTypeFlags(IDeviceMemoryAllocation::EMPF_DEVICE_LOCAL_BIT);
        }
        //! DirectVRAMAccess: Mappable for read and write and device local, will often return 0, always check if the mask != 0
        //! Requires EMPF_DEVICE_LOCAL_BIT, EMPF_HOST_READABLE_BIT, EMPF_HOST_WRITABLE_BIT from MemoryTypes
        uint32_t getDirectVRAMAccessMemoryTypeBits() const
        {
            core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS> requiredFlags = IDeviceMemoryAllocation::EMPF_DEVICE_LOCAL_BIT|IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT|IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT;
            return getMemoryTypeBitsFromMemoryTypeFlags(requiredFlags);
        }
        //! HostVisible: Mappable for write/read
        //! Requires EMPF_HOST_WRITABLE_BIT OR EMPF_HOST_READABLE_BIT
        uint32_t getHostVisibleMemoryTypeBits() const
        {
            uint32_t hostWritable = getMemoryTypeBitsFromMemoryTypeFlags(IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT);
            uint32_t hostReadable = getMemoryTypeBitsFromMemoryTypeFlags(IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT);
            return hostWritable | hostReadable;
        }
        //! UpStreaming: Mappable for write and preferably device local
        //! Requires EMPF_HOST_WRITABLE_BIT
        //! Prefers EMPF_DEVICE_LOCAL_BIT
        uint32_t getUpStreamingMemoryTypeBits() const
        {
            uint32_t hostWritable = getMemoryTypeBitsFromMemoryTypeFlags(IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT);
            uint32_t deviceLocal = getMemoryTypeBitsFromMemoryTypeFlags(IDeviceMemoryAllocation::EMPF_DEVICE_LOCAL_BIT);
            uint32_t both = hostWritable & deviceLocal;
            if(both > 0)
                return both;
            else
                return hostWritable;
        }
        //! Mappable for read and preferably host cached
        //! Requires EMPF_HOST_READABLE_BIT
        //! Preferably EMPF_HOST_CACHED_BIT
        uint32_t getDownStreamingMemoryTypeBits() const
        {
            uint32_t hostReadable = getMemoryTypeBitsFromMemoryTypeFlags(IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT);
            uint32_t hostCached = getMemoryTypeBitsFromMemoryTypeFlags(IDeviceMemoryAllocation::EMPF_HOST_CACHED_BIT);
            uint32_t both = hostReadable & hostCached;
            if(both > 0)
                return both;
            else
                return hostReadable;
        }
        //! Spillover: Not host visible(read&write) and Not device local
        //! Excludes EMPF_DEVICE_LOCAL_BIT, EMPF_HOST_READABLE_BIT, EMPF_HOST_WRITABLE_BIT
        uint32_t getSpilloverMemoryTypeBits() const
        {
            uint32_t all = getMemoryTypeBitsFromMemoryTypeFlags(core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>(0u));
            uint32_t deviceLocal = getMemoryTypeBitsFromMemoryTypeFlags(IDeviceMemoryAllocation::EMPF_DEVICE_LOCAL_BIT);
            return all & (~deviceLocal);
        }
        //! HostVisibleSpillover: Same as Spillover but mappable for read&write
        //! Requires EMPF_HOST_READABLE_BIT, EMPF_HOST_WRITABLE_BIT
        //! Excludes EMPF_DEVICE_LOCAL_BIT
        uint32_t getHostVisibleSpilloverMemoryTypeBits() const
        {
            uint32_t hostWritable = getMemoryTypeBitsFromMemoryTypeFlags(IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT);
            uint32_t hostReadable = getMemoryTypeBitsFromMemoryTypeFlags(IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT);
            uint32_t deviceLocal = getMemoryTypeBitsFromMemoryTypeFlags(IDeviceMemoryAllocation::EMPF_DEVICE_LOCAL_BIT);
            return (hostWritable & hostReadable) & (~deviceLocal);
        }

        /* ImageFormatProperties2
                !! Exposing this ImageFormatProperties is not straightforward 
                !! because it depends on multiple parameters other than `format` including
                !! type, tiling, usage, flags and also whatever you put in it's pNextchain
                !! basically these could be only queried if we knew all the creation params of an image

                - VkAndroidHardwareBufferUsageANDROID,
                - VkExternalImageFormatProperties,
                - VkFilterCubicImageViewImageFormatPropertiesEXT,
                - VkImageCompressionPropertiesEXT,
                - VkSamplerYcbcrConversionImageFormatProperties,
                - VkTextureLODGatherFormatPropertiesAMD

            !! Same goes for `vkGetPhysicalDeviceSparseImageFormatProperties2`
        */
        struct SFormatBufferUsages
        {
            struct SUsage
            {
                uint16_t vertexAttribute : 1; // vertexAtrtibute binding
                uint16_t bufferView : 1; // samplerBuffer
                uint16_t storageBufferView : 1; // imageBuffer
                uint16_t storageBufferViewAtomic : 1; // imageBuffer
                uint16_t accelerationStructureVertex : 1;
                uint16_t storageBufferViewLoadWithoutFormat : 1;
                uint16_t storageBufferViewStoreWithoutFormat : 1;
                uint16_t opticalFlowImage : 1;
                uint16_t opticalFlowVector : 1;
                uint16_t opticalFlowCost : 1;

                SUsage()
                    : vertexAttribute(0)
                    , bufferView(0)
                    , storageBufferView(0)
                    , storageBufferViewAtomic(0)
                    , accelerationStructureVertex(0)
                    , storageBufferViewLoadWithoutFormat(0)
                    , storageBufferViewStoreWithoutFormat(0)
                    , opticalFlowImage(0)
                    , opticalFlowVector(0)
                    , opticalFlowCost(0)
                {}

                // Fields with 0 are deduced as false. User may patch it up later
                SUsage(core::bitflag<asset::IBuffer::E_USAGE_FLAGS> usages) 
                    : vertexAttribute(usages.hasFlags(asset::IBuffer::EUF_VERTEX_BUFFER_BIT))
                    , bufferView(usages.hasFlags(asset::IBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT))
                    , storageBufferView(usages.hasFlags(asset::IBuffer::EUF_STORAGE_TEXEL_BUFFER_BIT))
                    , storageBufferViewAtomic(0)
                    , accelerationStructureVertex(usages.hasFlags(asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT))
                    , storageBufferViewLoadWithoutFormat(0)
                    , storageBufferViewStoreWithoutFormat(0)
                    , opticalFlowImage(0)
                    , opticalFlowVector(0)
                    , opticalFlowCost(0)
                {}

                inline SUsage operator & (const SUsage& other) const
                {
                    SUsage result;
                    result.vertexAttribute = vertexAttribute & other.vertexAttribute;
                    result.bufferView = bufferView & other.bufferView;
                    result.storageBufferView = storageBufferView & other.storageBufferView;
                    result.storageBufferViewAtomic = storageBufferViewAtomic & other.storageBufferViewAtomic;
                    result.accelerationStructureVertex = accelerationStructureVertex & other.accelerationStructureVertex;
                    result.storageBufferViewLoadWithoutFormat = storageBufferViewLoadWithoutFormat & other.storageBufferViewLoadWithoutFormat;
                    result.storageBufferViewStoreWithoutFormat = storageBufferViewStoreWithoutFormat & other.storageBufferViewStoreWithoutFormat;
                    result.opticalFlowImage = opticalFlowImage & other.opticalFlowImage;
                    result.opticalFlowVector = opticalFlowVector & other.opticalFlowVector;
                    result.opticalFlowCost = opticalFlowCost & other.opticalFlowCost;
                    return result;
                }

                inline SUsage operator | (const SUsage& other) const
                {
                    SUsage result;
                    result.vertexAttribute = vertexAttribute | other.vertexAttribute;
                    result.bufferView = bufferView | other.bufferView;
                    result.storageBufferView = storageBufferView | other.storageBufferView;
                    result.storageBufferViewAtomic = storageBufferViewAtomic | other.storageBufferViewAtomic;
                    result.accelerationStructureVertex = accelerationStructureVertex | other.accelerationStructureVertex;
                    result.storageBufferViewLoadWithoutFormat = storageBufferViewLoadWithoutFormat | other.storageBufferViewLoadWithoutFormat;
                    result.storageBufferViewStoreWithoutFormat = storageBufferViewStoreWithoutFormat | other.storageBufferViewStoreWithoutFormat;
                    result.opticalFlowImage = opticalFlowImage | other.opticalFlowImage;
                    result.opticalFlowVector = opticalFlowVector | other.opticalFlowVector;
                    result.opticalFlowCost = opticalFlowCost | other.opticalFlowCost;
                    return result;
                }

                inline SUsage operator ^ (const SUsage& other) const
                {
                    SUsage result;
                    result.vertexAttribute = vertexAttribute ^ other.vertexAttribute;
                    result.bufferView = bufferView ^ other.bufferView;
                    result.storageBufferView = storageBufferView ^ other.storageBufferView;
                    result.storageBufferViewAtomic = storageBufferViewAtomic ^ other.storageBufferViewAtomic;
                    result.accelerationStructureVertex = accelerationStructureVertex ^ other.accelerationStructureVertex;
                    result.storageBufferViewLoadWithoutFormat = storageBufferViewLoadWithoutFormat ^ other.storageBufferViewLoadWithoutFormat;
                    result.storageBufferViewStoreWithoutFormat = storageBufferViewStoreWithoutFormat ^ other.storageBufferViewStoreWithoutFormat;
                    result.opticalFlowImage = opticalFlowImage ^ other.opticalFlowImage;
                    result.opticalFlowVector = opticalFlowVector ^ other.opticalFlowVector;
                    result.opticalFlowCost = opticalFlowCost ^ other.opticalFlowCost;
                    return result;
                }

                inline bool operator == (const SUsage& other) const
                {
                    return
                        (vertexAttribute == other.vertexAttribute) &&
                        (bufferView == other.bufferView) &&
                        (storageBufferView == other.storageBufferView) &&
                        (storageBufferViewAtomic == other.storageBufferViewAtomic) &&
                        (accelerationStructureVertex == other.accelerationStructureVertex) &&
                        (storageBufferViewLoadWithoutFormat == other.storageBufferViewLoadWithoutFormat) &&
                        (storageBufferViewStoreWithoutFormat == other.storageBufferViewStoreWithoutFormat) &&
                        (opticalFlowImage == other.opticalFlowImage) &&
                        (opticalFlowVector == other.opticalFlowVector) &&
                        (opticalFlowCost == other.opticalFlowCost);
                }
            };
            
            inline SUsage& operator[](const asset::E_FORMAT idx)
            {
                return m_usages[idx];
            }

            inline const SUsage& operator[](const asset::E_FORMAT idx) const
            {
                return m_usages[idx];
            }

            inline bool isSubsetOf(const SFormatBufferUsages& other) const
            {
                for (uint32_t i=0; i<asset::EF_COUNT; ++i)
                if ((m_usages[i]&other.m_usages[i])!=m_usages[i])
                    return false;
                return true;
            }

            SUsage m_usages[asset::EF_COUNT] = {};
        };
        const SFormatBufferUsages& getBufferFormatUsages() const { return m_initData.bufferUsages; };

        //

        struct SFormatImageUsages
        {
            struct SUsage
            {
                uint32_t sampledImage : 1; // samplerND
                uint32_t linearlySampledImage : 1; // samplerND with a sampler that has LINEAR or bitSrc with linear Filter
                uint32_t minmaxSampledImage : 1; // samplerND with a sampler that MINMAX
                // no cubic filter exposed
                uint32_t storageImage : 1; // imageND
                uint32_t storageImageAtomic : 1;
                uint32_t attachment : 1; // color, depth, stencil can be infferred from the format itself
                uint32_t attachmentBlend : 1;
                uint32_t blitSrc : 1;
                uint32_t blitDst : 1;
                uint32_t transferSrc : 1;
                uint32_t transferDst : 1;
                // TODO: chroma midpoint/cosited samples, ycbcr conversion, disjoint, fragment density map, shading rate,
                uint32_t videoDecodeOutput : 1;
                uint32_t videoDecodeDPB : 1;
                uint32_t videoEncodeInput : 1;
                uint32_t videoEncodeDPB : 1;
                uint32_t storageImageLoadWithoutFormat : 1;
                uint32_t storageImageStoreWithoutFormat : 1;
                uint32_t depthCompareSampledImage : 1;
                uint32_t hostImageTransfer : 1;
                // others
                uint32_t log2MaxSamples : 3; // 0 means cant use as a multisample image format

                constexpr SUsage()
                    : sampledImage(0)
                    , linearlySampledImage(0)
                    , minmaxSampledImage(0)
                    , storageImage(0)
                    , storageImageAtomic(0)
                    , attachment(0)
                    , attachmentBlend(0)
                    , blitSrc(0)
                    , blitDst(0)
                    , transferSrc(0)
                    , transferDst(0)
                    , videoDecodeOutput(0)
                    , videoDecodeDPB(0)
                    , videoEncodeInput(0)
                    , videoEncodeDPB(0)
                    , storageImageLoadWithoutFormat(0)
                    , storageImageStoreWithoutFormat(0)
                    , depthCompareSampledImage(0)
                    , hostImageTransfer(0)
                    , log2MaxSamples(0)
                {}

                // Fields with 0 are deduced as false. User may patch it up later
                constexpr SUsage(const core::bitflag<asset::IImage::E_USAGE_FLAGS> usages) :
                    sampledImage(usages.hasFlags(asset::IImage::EUF_SAMPLED_BIT)),
                    linearlySampledImage(0),
                    minmaxSampledImage(0),
                    storageImage(usages.hasFlags(asset::IImage::EUF_STORAGE_BIT)),
                    storageImageAtomic(0),
                    attachment(usages.hasFlags(IGPUImage::EUF_RENDER_ATTACHMENT_BIT)),
                    attachmentBlend(0),
                    blitSrc(0),
                    blitDst(0), // TODO: better deduction from render attachment and transfer?
                    transferSrc(usages.hasFlags(IGPUImage::EUF_TRANSFER_SRC_BIT)),
                    transferDst(usages.hasFlags(IGPUImage::EUF_TRANSFER_DST_BIT)),
                    videoDecodeOutput(0),
                    videoDecodeDPB(0),
                    videoEncodeInput(0),
                    videoEncodeDPB(0),
                    storageImageLoadWithoutFormat(0),
                    storageImageStoreWithoutFormat(0),
                    depthCompareSampledImage(0),
                    hostImageTransfer(0),
                    log2MaxSamples(0)
                {}

                constexpr explicit operator core::bitflag<asset::IImage::E_USAGE_FLAGS>() const
                {
                    using usage_flags_t = asset::IImage::E_USAGE_FLAGS;
                    core::bitflag<usage_flags_t> retval = usage_flags_t::EUF_NONE;
                    if (sampledImage)
                        retval |= usage_flags_t::EUF_SAMPLED_BIT;
                    if (storageImage)
                        retval |= usage_flags_t::EUF_STORAGE_BIT;
                    if (attachment || blitDst) // does also src imply?
                        retval |= usage_flags_t::EUF_RENDER_ATTACHMENT_BIT;
                    if (blitSrc || transferSrc)
                        retval |= usage_flags_t::EUF_TRANSFER_SRC_BIT;
                    if (blitDst || transferDst)
                        retval |= usage_flags_t::EUF_TRANSFER_DST_BIT;
                    return retval;
                }

                constexpr SUsage operator&(const SUsage& other) const
                {
                    SUsage result;
                    result.sampledImage = sampledImage & other.sampledImage;
                    result.linearlySampledImage = linearlySampledImage & other.linearlySampledImage;
                    result.minmaxSampledImage = minmaxSampledImage & other.minmaxSampledImage;
                    result.storageImage = storageImage & other.storageImage;
                    result.storageImageAtomic = storageImageAtomic & other.storageImageAtomic;
                    result.attachment = attachment & other.attachment;
                    result.attachmentBlend = attachmentBlend & other.attachmentBlend;
                    result.blitSrc = blitSrc & other.blitSrc;
                    result.blitDst = blitDst & other.blitDst;
                    result.transferSrc = transferSrc & other.transferSrc;
                    result.transferDst = transferDst & other.transferDst;
                    result.videoDecodeOutput = videoDecodeOutput & other.videoDecodeOutput;
                    result.videoDecodeDPB = videoDecodeDPB & other.videoDecodeDPB;
                    result.videoEncodeInput = videoEncodeInput & other.videoEncodeInput;
                    result.videoEncodeDPB = videoEncodeDPB & other.videoEncodeDPB;
                    result.storageImageLoadWithoutFormat = storageImageLoadWithoutFormat & other.storageImageLoadWithoutFormat;
                    result.storageImageStoreWithoutFormat = storageImageStoreWithoutFormat & other.storageImageStoreWithoutFormat;
                    result.depthCompareSampledImage = depthCompareSampledImage & other.depthCompareSampledImage;
                    result.hostImageTransfer = hostImageTransfer & other.hostImageTransfer;
                    result.log2MaxSamples = std::min(log2MaxSamples,other.log2MaxSamples);
                    return result;
                }

                constexpr SUsage operator|(const SUsage& other) const
                {
                    SUsage result;
                    result.sampledImage = sampledImage | other.sampledImage;
                    result.linearlySampledImage = linearlySampledImage | other.linearlySampledImage;
                    result.minmaxSampledImage = minmaxSampledImage | other.minmaxSampledImage;
                    result.storageImage = storageImage | other.storageImage;
                    result.storageImageAtomic = storageImageAtomic | other.storageImageAtomic;
                    result.attachment = attachment | other.attachment;
                    result.attachmentBlend = attachmentBlend | other.attachmentBlend;
                    result.blitSrc = blitSrc | other.blitSrc;
                    result.blitDst = blitDst | other.blitDst;
                    result.transferSrc = transferSrc | other.transferSrc;
                    result.transferDst = transferDst | other.transferDst;
                    result.videoDecodeOutput = videoDecodeOutput | other.videoDecodeOutput;
                    result.videoDecodeDPB = videoDecodeDPB | other.videoDecodeDPB;
                    result.videoEncodeInput = videoEncodeInput | other.videoEncodeInput;
                    result.videoEncodeDPB = videoEncodeDPB | other.videoEncodeDPB;
                    result.storageImageLoadWithoutFormat = storageImageLoadWithoutFormat | other.storageImageLoadWithoutFormat;
                    result.storageImageStoreWithoutFormat = storageImageStoreWithoutFormat | other.storageImageStoreWithoutFormat;
                    result.depthCompareSampledImage = depthCompareSampledImage | other.depthCompareSampledImage;
                    result.hostImageTransfer = hostImageTransfer | other.hostImageTransfer;
                    result.log2MaxSamples = std::max(log2MaxSamples,other.log2MaxSamples);
                    return result;
                }

                // TODO: do we even need this operator?
                constexpr SUsage operator^(const SUsage& other) const
                {
                    SUsage result;
                    result.sampledImage = sampledImage ^ other.sampledImage;
                    result.linearlySampledImage = linearlySampledImage ^ other.linearlySampledImage;
                    result.minmaxSampledImage = minmaxSampledImage ^ other.minmaxSampledImage;
                    result.storageImage = storageImage ^ other.storageImage;
                    result.storageImageAtomic = storageImageAtomic ^ other.storageImageAtomic;
                    result.attachment = attachment ^ other.attachment;
                    result.attachmentBlend = attachmentBlend ^ other.attachmentBlend;
                    result.blitSrc = blitSrc ^ other.blitSrc;
                    result.blitDst = blitDst ^ other.blitDst;
                    result.transferSrc = transferSrc ^ other.transferSrc;
                    result.transferDst = transferDst ^ other.transferDst;
                    result.videoDecodeOutput = videoDecodeOutput ^ other.videoDecodeOutput;
                    result.videoDecodeDPB = videoDecodeDPB ^ other.videoDecodeDPB;
                    result.videoEncodeInput = videoEncodeInput ^ other.videoEncodeInput;
                    result.transferDst = videoEncodeDPB ^ other.videoEncodeDPB;
                    // does this operator even make sense!?
                    result.storageImageLoadWithoutFormat = storageImageLoadWithoutFormat ^ other.storageImageLoadWithoutFormat;
                    result.storageImageStoreWithoutFormat = storageImageStoreWithoutFormat ^ other.storageImageStoreWithoutFormat;
                    result.depthCompareSampledImage = depthCompareSampledImage ^ other.depthCompareSampledImage;
                    result.hostImageTransfer = hostImageTransfer ^ other.hostImageTransfer;
                    result.log2MaxSamples = log2MaxSamples ^ other.log2MaxSamples;
                    return result;
                }

                constexpr bool operator==(const SUsage& other) const
                {
                    return
                        (sampledImage == other.sampledImage) &&
                        (linearlySampledImage == other.linearlySampledImage) &&
                        (minmaxSampledImage == other.minmaxSampledImage) &&
                        (storageImage == other.storageImage) &&
                        (storageImageAtomic == other.storageImageAtomic) &&
                        (attachment == other.attachment) &&
                        (attachmentBlend == other.attachmentBlend) &&
                        (blitSrc == other.blitSrc) &&
                        (blitDst == other.blitDst) &&
                        (transferSrc == other.transferSrc) &&
                        (transferDst == other.transferDst) &&
                        (videoDecodeOutput == other.videoDecodeOutput) &&
                        (videoDecodeDPB == other.videoDecodeDPB) &&
                        (videoEncodeInput == other.videoEncodeInput) &&
                        (videoEncodeDPB == other.videoEncodeDPB) &&
                        (storageImageLoadWithoutFormat == other.storageImageLoadWithoutFormat) &&
                        (storageImageStoreWithoutFormat == other.storageImageStoreWithoutFormat) &&
                        (depthCompareSampledImage == other.depthCompareSampledImage) &&
                        (hostImageTransfer == other.hostImageTransfer) &&
                        (log2MaxSamples == other.log2MaxSamples);
                }
            };
            
            inline SUsage& operator[](const asset::E_FORMAT idx)
            {
                return m_usages[idx];
            }

            inline const SUsage& operator[](const asset::E_FORMAT idx) const
            {
                return m_usages[idx];
            }

            inline bool isSubsetOf(const SFormatImageUsages& other) const
            {
                for (uint32_t i=0; i<asset::EF_COUNT; ++i)
                if((m_usages[i]&other.m_usages[i])!=m_usages[i])
                    return false;
                return true;
            }

            SUsage m_usages[asset::EF_COUNT] = {};
        };

        const SFormatImageUsages& getImageFormatUsagesLinearTiling() const { return m_initData.linearTilingUsages; }
        const SFormatImageUsages& getImageFormatUsagesOptimalTiling() const { return m_initData.optimalTilingUsages; }
        const SFormatImageUsages& getImageFormatUsages(const IGPUImage::TILING tiling) const {return tiling!=IGPUImage::TILING::OPTIMAL ? m_initData.linearTilingUsages:m_initData.optimalTilingUsages;}

        /* QueueFamilyProperties2
* 
                - VkQueueFamilyCheckpointProperties2NV, VkQueueFamilyCheckpointPropertiesNV [DON'T EXPOSE]: 
                    These extensions allows applications to insert markers in the command stream and associate them with custom data.
                    The one with the 2 suffix is provided by VK_KHR_synchronization2 other than VK_NV_device_diagnostic_checkpoints
                
                - VkQueueFamilyGlobalPriorityPropertiesKHR[FUTURE TODO]
                    Related to VK_KHR_global_priority (bool in features)
                        VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR = 128,
                        VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR = 256,
                        VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR = 512,
                        VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR = 1024,
                    This extension is basically for querying the queue family's supported global priorities 
                    In Vulkan, users can specify device-scope queue priorities.
                    In some cases it may be useful to extend this concept to a system-wide scope.
                    This device extension allows applications to query the global queue priorities supported by a queue family, and then set a priority when creating queues. The default queue priority is VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_EXT.


                [FUTURE TODO] [Currently Beta Extension in Vk Provided by VK_KHR_video_queue]
                - VkQueueFamilyQueryResultStatusProperties2KHR
                    Related to Queries 
                    `supported` reports VK_TRUE if query type VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR and use of VK_QUERY_RESULT_WITH_STATUS_BIT_KHR are supported.
                
                [FUTURE TODO]
                - VkVideoQueueFamilyProperties2KHR
                    videoCodecOperations is a bitmask of VkVideoCodecOperationFlagBitsKHR specifying supported video codec operation(s).
        */
        struct SQueueFamilyProperties
        {
            core::bitflag<IQueue::FAMILY_FLAGS> queueFlags;
            uint32_t queueCount;
            uint32_t timestampValidBits;
            asset::VkExtent3D minImageTransferGranularity;

            inline bool operator!=(const SQueueFamilyProperties& other) const
            {
                return queueFlags!=other.queueFlags || queueCount != other.queueCount || timestampValidBits != other.timestampValidBits || minImageTransferGranularity != other.minImageTransferGranularity;
            }
        };
        auto getQueueFamilyProperties() const 
        {
            return std::span<const SQueueFamilyProperties>(m_initData.qfamProperties->data(),m_initData.qfamProperties->data()+m_initData.qfamProperties->size());
        }

        struct SBufferFormatPromotionRequest {
            asset::E_FORMAT originalFormat = asset::EF_UNKNOWN;
            SFormatBufferUsages::SUsage usages = SFormatBufferUsages::SUsage();
        };

        struct SImageFormatPromotionRequest {
            asset::E_FORMAT originalFormat = asset::EF_UNKNOWN;
            SFormatImageUsages::SUsage usages = SFormatImageUsages::SUsage();
        };

        asset::E_FORMAT promoteBufferFormat(const SBufferFormatPromotionRequest req) const;
        asset::E_FORMAT promoteImageFormat(const SImageFormatPromotionRequest req, const IGPUImage::TILING tiling) const;

        //
        inline system::ISystem* getSystem() const {return m_initData.system.get();}
        
        IDebugCallback* getDebugCallback() const;

        core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice(ILogicalDevice::SCreationParams&& params)
        {
            if (!validateLogicalDeviceCreation(params))
                return nullptr;
            return createLogicalDevice_impl(std::move(params));
        }

    protected:
        struct SInitData final
        {
            core::smart_refctd_ptr<system::ISystem> system;
            IAPIConnection* api; // dumb pointer to avoid circ ref

            SProperties properties = {};
            SFeatures features = {};
            SMemoryProperties memoryProperties = {};

            using qfam_props_array_t = core::smart_refctd_dynamic_array<const SQueueFamilyProperties>;
            qfam_props_array_t qfamProperties;

            SFormatImageUsages linearTilingUsages = {};
            SFormatImageUsages optimalTilingUsages = {};
            SFormatBufferUsages bufferUsages = {};
        };
        inline IPhysicalDevice(SInitData&& _initData) : m_initData(std::move(_initData)) {}

        // ILogicalDevice creation
        bool validateLogicalDeviceCreation(const ILogicalDevice::SCreationParams& params) const;
        virtual core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(ILogicalDevice::SCreationParams&& params) = 0;

        // static utils
        static inline uint32_t getMaxInvocationsPerComputeUnitsFromDriverID(E_DRIVER_ID driverID)
        {
            const bool isIntelGPU = (driverID == E_DRIVER_ID::EDI_INTEL_OPEN_SOURCE_MESA || driverID == E_DRIVER_ID::EDI_INTEL_PROPRIETARY_WINDOWS);
            const bool isAMDGPU = (driverID == E_DRIVER_ID::EDI_AMD_OPEN_SOURCE || driverID == E_DRIVER_ID::EDI_AMD_PROPRIETARY);
            const bool isNVIDIAGPU = (driverID == E_DRIVER_ID::EDI_NVIDIA_PROPRIETARY);
            if (isNVIDIAGPU)
                return 32u * 64u; // Hopper (32 Threads/Warp *  Warps/SM)
            else if (isAMDGPU)
                return 32u * 1024u; // RDNA2 (32 Threads/Wave * 1024 Waves/CU) https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/
            // https://www.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top/thread-mapping.html
            // https://www.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top/intel-processors-with-intel-uhd-graphics.html
            // https://www.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top/xe-arch.html
            // Intel(R) Iris(R) Xe: Maximum Worgroups on a Subslice = 16 & maxComputeWorkGroupInvocations = 1024
            else if (isIntelGPU)
                return 16u * 1024u;
            else
                return 32u * 1024u;
        }

        static inline uint32_t getMaxComputeUnitsFromDriverID(E_DRIVER_ID driverID)
        {
            const bool isIntelGPU = (driverID == E_DRIVER_ID::EDI_INTEL_OPEN_SOURCE_MESA || driverID == E_DRIVER_ID::EDI_INTEL_PROPRIETARY_WINDOWS);
            const bool isAMDGPU = (driverID == E_DRIVER_ID::EDI_AMD_OPEN_SOURCE || driverID == E_DRIVER_ID::EDI_AMD_PROPRIETARY);
            const bool isNVIDIAGPU = (driverID == E_DRIVER_ID::EDI_NVIDIA_PROPRIETARY);
            if (isNVIDIAGPU) // NVIDIA SM
                return 144u; // RTX 4090
            else if (isAMDGPU) // AMD Compute Units
                return 220u; // AMD Instinct (TM) MI250X
            else if (isIntelGPU) // Intel DSS (or XC = new abbrevation) or is it 128 ?
                return 64u; // Iris Xe HPG (DG2) https://www.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top/xe-arch.html
            else
                return 220u; // largest from above
        }

        // Format Promotion
        struct SBufferFormatPromotionRequestHash
        {
            // pack into 64bit for easy hashing 
            inline uint64_t operator()(const SBufferFormatPromotionRequest& r) const;
        };

        struct SBufferFormatPromotionRequestEqualTo
        {
            inline bool operator()(const SBufferFormatPromotionRequest& l, const SBufferFormatPromotionRequest& r) const;
        };

        struct SImageFormatPromotionRequestHash
        {
            // pack into 64bit for easy hashing 
            inline uint64_t operator()(const SImageFormatPromotionRequest& r) const;
        };

        struct SImageFormatPromotionRequestEqualTo
        {
            inline bool operator()(const SImageFormatPromotionRequest& l, const SImageFormatPromotionRequest& r) const;
        };

        // TODO: use `using`
        typedef core::unordered_map<SBufferFormatPromotionRequest, asset::E_FORMAT, 
            SBufferFormatPromotionRequestHash, 
            SBufferFormatPromotionRequestEqualTo> format_buffer_cache_t;
        typedef core::unordered_map<SImageFormatPromotionRequest, asset::E_FORMAT, 
            SImageFormatPromotionRequestHash, 
            SImageFormatPromotionRequestEqualTo> format_image_cache_t;

        struct format_promotion_cache_t
        {
            format_buffer_cache_t buffers;
            format_image_cache_t optimalTilingImages;
            format_image_cache_t linearTilingImages;
        };


        // data members
        const SInitData m_initData;
        mutable format_promotion_cache_t m_formatPromotionCache;
};

}

namespace std
{
    template<>
    struct hash<nbl::video::IPhysicalDevice::SFormatImageUsages::SUsage>
    {
        inline uint32_t operator()(const nbl::video::IPhysicalDevice::SFormatImageUsages::SUsage& i) const
        {
            return
                i.sampledImage |
                (i.storageImage << 1) |
                (i.storageImageAtomic << 2) |
                (i.attachment << 3) |
                (i.attachmentBlend << 4) |
                (i.blitSrc << 5) |
                (i.blitDst << 6) |
                (i.transferSrc << 7) |
                (i.transferDst << 8) |
                (i.log2MaxSamples << 9);
        }
    };

    template<>
    struct hash<nbl::video::IPhysicalDevice::SFormatBufferUsages::SUsage>
    {
        inline uint32_t operator()(const nbl::video::IPhysicalDevice::SFormatBufferUsages::SUsage& b) const
        {
            return
                b.vertexAttribute |
                (b.bufferView << 1) |
                (b.storageBufferView << 2) |
                (b.storageBufferViewAtomic << 3) |
                (b.accelerationStructureVertex << 4);
        }
    };
}

namespace nbl::video
{
    inline uint64_t IPhysicalDevice::SBufferFormatPromotionRequestHash::operator()(const SBufferFormatPromotionRequest& r) const {
        uint64_t msb = uint64_t(std::hash<IPhysicalDevice::SFormatBufferUsages::SUsage>()(r.usages));
        return (msb << 32u) | r.originalFormat;
    }

    inline uint64_t IPhysicalDevice::SImageFormatPromotionRequestHash::operator()(const SImageFormatPromotionRequest& r) const {
        uint64_t msb = uint64_t(std::hash<IPhysicalDevice::SFormatImageUsages::SUsage>()(r.usages));
        return (msb << 32u) | r.originalFormat;
    }

    inline bool IPhysicalDevice::SBufferFormatPromotionRequestEqualTo::operator()(const SBufferFormatPromotionRequest& l, const SBufferFormatPromotionRequest& r) const
    {
        return l.originalFormat == r.originalFormat && l.usages == r.usages;
    }

     inline bool IPhysicalDevice::SImageFormatPromotionRequestEqualTo::operator()(const SImageFormatPromotionRequest& l, const SImageFormatPromotionRequest& r) const
    {
        return l.originalFormat == r.originalFormat && l.usages == r.usages;
    }
    

}
#endif
