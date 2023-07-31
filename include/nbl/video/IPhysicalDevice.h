#ifndef _NBL_VIDEO_I_PHYSICAL_DEVICE_H_INCLUDED_
#define _NBL_VIDEO_I_PHYSICAL_DEVICE_H_INCLUDED_


#include "nbl/core/util/bitflag.h"

#include "nbl/system/declarations.h"
#include "nbl/system/ISystem.h"

#include "nbl/asset/IImage.h" //for VkExtent3D only
#include "nbl/asset/ISpecializedShader.h"

#include "nbl/video/EApiType.h"
#include "nbl/video/debug/IDebugCallback.h"
#include "nbl/video/ILogicalDevice.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

#include "SPhysicalDeviceLimits.h"
#include "SPhysicalDeviceFeatures.h"

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

        const SProperties& getProperties() const { return m_properties; }
        const SLimits& getLimits() const { return m_properties.limits; }
        APIVersion getAPIVersion() const { return m_properties.apiVersion; }
        const SFeatures& getFeatures() const { return m_features; }

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
        const SMemoryProperties& getMemoryProperties() const { return m_memoryProperties; }
        
        //! Bit `i` in MemoryTypeBitss will be set if m_memoryProperties.memoryTypes[i] has the `flags`
        uint32_t getMemoryTypeBitsFromMemoryTypeFlags(core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS> flags) const
        {
            uint32_t ret = 0u;
            for(uint32_t i = 0; i < m_memoryProperties.memoryTypeCount; ++i)
                if(m_memoryProperties.memoryTypes[i].propertyFlags.hasFlags(flags))
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
            core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS> requiredFlags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>(IDeviceMemoryAllocation::EMPF_DEVICE_LOCAL_BIT) | IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT | IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT;
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

        /* FormatProperties2 
                - VkDrmFormatModifierPropertiesListEXT(linux stuff)
                - VkDrmFormatModifierPropertiesList2EXT(linux stuff)

                [TODO][SOON] Add new flags to our own enum and implement for all backends
                - VkFormatProperties3: (available in Vulkan Core 1.1)
                    Basically same as VkFromatProperties but the flag type is VkFormatFeatureFlagBits2
                    VkFormatFeatureFlagBits2 is basically compensating for the fuckup when `VkFormatFeatureFlagBits` could only have 31 flags
                    this type is VkFlags64 and added two extra flags, namely:
                        1. VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT_KHR and VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT_KHR 
                            indicate that an implementation supports respectively reading and writing
                            a given VkFormat through storage operations without specifying the format in the shader.

                        2. VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT_KHR indicates that an implementation supports 
                            depth comparison performed by OpImage*Dref* instructions on a given VkFormat.
                            Previously the result of executing a OpImage*Dref* instruction on an image view,
                            where the format was not one of the depth/stencil formats with a depth component,
                            was undefined. This bit clarifies on which formats such instructions can be used.

                - VkVideoDecodeH264ProfileEXT(video stuff)
                - VkVideoDecodeH265ProfileEXT(video stuff)
                - VkVideoEncodeH264ProfileEXT(video stuff)
                - VkVideoEncodeH265ProfileEXT(video stuff)
                - VkVideoProfileKHR (video stuff)
                - VkVideoProfilesKHR(video stuff)
        */

        //
        struct SFormatBufferUsages
        {
            struct SUsage
            {
                uint8_t vertexAttribute : 1u; // vertexAtrtibute binding
                uint8_t bufferView : 1u; // samplerBuffer
                uint8_t storageBufferView : 1u; // imageBuffer
                uint8_t storageBufferViewAtomic : 1u; // imageBuffer
                uint8_t accelerationStructureVertex : 1u;

                SUsage()
                    : vertexAttribute(0)
                    , bufferView(0)
                    , storageBufferView(0)
                    , storageBufferViewAtomic(0)
                    , accelerationStructureVertex(0)
                {}

                SUsage(core::bitflag<asset::IBuffer::E_USAGE_FLAGS> usages) 
                    : vertexAttribute(usages.hasFlags(asset::IBuffer::EUF_VERTEX_BUFFER_BIT))
                    , bufferView(usages.hasFlags(asset::IBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT))
                    , storageBufferView(usages.hasFlags(asset::IBuffer::EUF_STORAGE_TEXEL_BUFFER_BIT))
                    , accelerationStructureVertex(usages.hasFlags(asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT))
                    // Deduced as false. User may patch it up later
                    , storageBufferViewAtomic(0)
                {}

                inline SUsage operator & (const SUsage& other) const
                {
                    SUsage result;
                    result.vertexAttribute = vertexAttribute & other.vertexAttribute;
                    result.bufferView = bufferView & other.bufferView;
                    result.storageBufferView = storageBufferView & other.storageBufferView;
                    result.storageBufferViewAtomic = storageBufferViewAtomic & other.storageBufferViewAtomic;
                    result.accelerationStructureVertex = accelerationStructureVertex & other.accelerationStructureVertex;
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
                    return result;
                }

                inline bool operator<(const SUsage& other) const
                {
                    if (vertexAttribute && !other.vertexAttribute) return false;
                    if (bufferView && !other.bufferView) return false;
                    if (storageBufferView && !other.storageBufferView) return false;
                    if (storageBufferViewAtomic && !other.storageBufferViewAtomic) return false;
                    if (accelerationStructureVertex && !other.accelerationStructureVertex) return false;
                    return true;
                }

                inline bool operator == (const SUsage& other) const
                {
                    return
                        (vertexAttribute == other.vertexAttribute) &&
                        (bufferView == other.bufferView) &&
                        (storageBufferView == other.storageBufferView) &&
                        (storageBufferViewAtomic == other.storageBufferViewAtomic) &&
                        (accelerationStructureVertex == other.accelerationStructureVertex);
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
                for(uint32_t i = 0; i < asset::EF_COUNT; ++i)
                    if(!(m_usages[i] < other.m_usages[i]))
                        return false;
                return true;
            }

            SUsage m_usages[asset::EF_COUNT] = {};
        };
        const SFormatBufferUsages& getBufferFormatUsages() const { return m_bufferUsages; };

        //

        struct SFormatImageUsages
        {
            // TODO: should memset everything to 0 on default constructor?

            struct SUsage
            {
                uint16_t sampledImage : 1u; // samplerND
                uint16_t storageImage : 1u; // imageND
                uint16_t storageImageAtomic : 1u;
                uint16_t attachment : 1u; // color, depth, stencil can be infferred from the format itself
                uint16_t attachmentBlend : 1u;
                uint16_t blitSrc : 1u;
                uint16_t blitDst : 1u;
                uint16_t transferSrc : 1u;
                uint16_t transferDst : 1u;
                uint16_t log2MaxSamples : 3u; // 0 means cant use as a multisample image format

                SUsage()
                    : sampledImage(0)
                    , storageImage(0)
                    , storageImageAtomic(0)
                    , attachment(0)
                    , attachmentBlend(0)
                    , blitSrc(0)
                    , blitDst(0)
                    , transferSrc(0)
                    , transferDst(0)
                    , log2MaxSamples(0)
                {}

                SUsage(core::bitflag<IGPUImage::E_USAGE_FLAGS> usages):
                    log2MaxSamples(0),
                    sampledImage(usages.hasFlags(IGPUImage::EUF_SAMPLED_BIT)),
                    storageImage(usages.hasFlags(IGPUImage::EUF_STORAGE_BIT)),
                    transferSrc(usages.hasFlags(IGPUImage::EUF_TRANSFER_SRC_BIT)),
                    transferDst(usages.hasFlags(IGPUImage::EUF_TRANSFER_DST_BIT)),
                    attachment(usages.hasFlags(IGPUImage::EUF_RENDER_ATTACHMENT_BIT)),
                    attachmentBlend(usages.hasFlags(IGPUImage::EUF_RENDER_ATTACHMENT_BIT)/*&& TODO: is Float or Normalized Format*/),
                    // Deduced as false. User may patch it up later
                    blitSrc(0),
                    blitDst(0),
                    storageImageAtomic(0)
                {}

                inline SUsage operator & (const SUsage& other) const
                {
                    SUsage result;
                    result.sampledImage = sampledImage & other.sampledImage;
                    result.storageImage = storageImage & other.storageImage;
                    result.storageImageAtomic = storageImageAtomic & other.storageImageAtomic;
                    result.attachment = attachment & other.attachment;
                    result.attachmentBlend = attachmentBlend & other.attachmentBlend;
                    result.blitSrc = blitSrc & other.blitSrc;
                    result.blitDst = blitDst & other.blitDst;
                    result.transferSrc = transferSrc & other.transferSrc;
                    result.transferDst = transferDst & other.transferDst;
                    result.log2MaxSamples = log2MaxSamples & other.log2MaxSamples;
                    return result;
                }

                inline SUsage operator | (const SUsage& other) const
                {
                    SUsage result;
                    result.sampledImage = sampledImage | other.sampledImage;
                    result.storageImage = storageImage | other.storageImage;
                    result.storageImageAtomic = storageImageAtomic | other.storageImageAtomic;
                    result.attachment = attachment | other.attachment;
                    result.attachmentBlend = attachmentBlend | other.attachmentBlend;
                    result.blitSrc = blitSrc | other.blitSrc;
                    result.blitDst = blitDst | other.blitDst;
                    result.transferSrc = transferSrc | other.transferSrc;
                    result.transferDst = transferDst | other.transferDst;
                    result.log2MaxSamples = log2MaxSamples | other.log2MaxSamples;
                    return result;
                }

                inline SUsage operator ^ (const SUsage& other) const
                {
                    SUsage result;
                    result.sampledImage = sampledImage ^ other.sampledImage;
                    result.storageImage = storageImage ^ other.storageImage;
                    result.storageImageAtomic = storageImageAtomic ^ other.storageImageAtomic;
                    result.attachment = attachment ^ other.attachment;
                    result.attachmentBlend = attachmentBlend ^ other.attachmentBlend;
                    result.blitSrc = blitSrc ^ other.blitSrc;
                    result.blitDst = blitDst ^ other.blitDst;
                    result.transferSrc = transferSrc ^ other.transferSrc;
                    result.transferDst = transferDst ^ other.transferDst;
                    result.log2MaxSamples = log2MaxSamples ^ other.log2MaxSamples;
                    return result;
                }

                inline bool operator<(const SUsage& other) const
                {
                    if (sampledImage && !other.sampledImage) return false;
                    if (storageImage && !other.storageImage) return false;
                    if (storageImageAtomic && !other.storageImageAtomic) return false;
                    if (attachment && !other.attachment) return false;
                    if (attachmentBlend && !other.attachmentBlend) return false;
                    if (blitSrc && !other.blitSrc) return false;
                    if (blitDst && !other.blitDst) return false;
                    if (transferSrc && !other.transferSrc) return false;
                    if (transferDst && !other.transferDst) return false;
                    if (other.log2MaxSamples < log2MaxSamples) return false;
                    return true;
                }

                inline bool operator == (const SUsage& other) const
                {
                    return
                        (sampledImage == other.sampledImage) &&
                        (storageImage == other.storageImage) &&
                        (storageImageAtomic == other.storageImageAtomic) &&
                        (attachment == other.attachment) &&
                        (attachmentBlend == other.attachmentBlend) &&
                        (blitSrc == other.blitSrc) &&
                        (blitDst == other.blitDst) &&
                        (transferSrc == other.transferSrc) &&
                        (transferDst == other.transferDst) &&
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
                for(uint32_t i = 0; i < asset::EF_COUNT; ++i)
                    if(!(m_usages[i] < other.m_usages[i]))
                        return false;
                return true;
            }

            SUsage m_usages[asset::EF_COUNT] = {};
        };

        const SFormatImageUsages& getImageFormatUsagesLinearTiling() const { return m_linearTilingUsages; }
        const SFormatImageUsages& getImageFormatUsagesOptimalTiling() const { return m_optimalTilingUsages; }
        const SFormatImageUsages& getImageFormatUsages(const IGPUImage::TILING tiling) const {return tiling!=IGPUImage::TILING::OPTIMAL ? m_linearTilingUsages:m_optimalTilingUsages;}

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
            using citer_t = qfam_props_array_t::pointee::const_iterator;
            return core::SRange<const SQueueFamilyProperties, citer_t, citer_t>(
                m_qfamProperties->cbegin(),
                m_qfamProperties->cend()
            );
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
        inline system::ISystem* getSystem() const {return m_system.get();}
        
        virtual IDebugCallback* getDebugCallback() const = 0;

        core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice(ILogicalDevice::SCreationParams&& params)
        {
            if (!validateLogicalDeviceCreation(params))
                return nullptr;
            return createLogicalDevice_impl(std::move(params));
        }

    protected:
        IPhysicalDevice(core::smart_refctd_ptr<system::ISystem>&& s, IAPIConnection* api);

        virtual core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(ILogicalDevice::SCreationParams&& params) = 0;

        bool validateLogicalDeviceCreation(const ILogicalDevice::SCreationParams& params) const;

        static inline uint32_t getMaxInvocationsPerComputeUnitsFromDriverID(E_DRIVER_ID driverID)
        {
            const bool isIntelGPU = (driverID == E_DRIVER_ID::EDI_INTEL_OPEN_SOURCE_MESA || driverID == E_DRIVER_ID::EDI_INTEL_PROPRIETARY_WINDOWS);
            const bool isAMDGPU = (driverID == E_DRIVER_ID::EDI_AMD_OPEN_SOURCE || driverID == E_DRIVER_ID::EDI_AMD_PROPRIETARY);
            const bool isNVIDIAGPU = (driverID == E_DRIVER_ID::EDI_NVIDIA_PROPRIETARY);
            if (isNVIDIAGPU)
                return 32u * 48u; // RTX 3090 (32 Threads/Warp * 48 Warp/SM)
            else if (isAMDGPU)
                return 32u * 1024u; // RX 6900XT (64 Threads/Wave * 1024 Waves/CU) https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/
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
                return 82u; // RTX 3090
            else if (isAMDGPU) // AMD Compute Units
                return 80u; // RX 6900XT
            else if (isIntelGPU) // Intel DSS (or XC = new abbrevation)
                return 64u; // Iris Xe HPG (DG2) https://www.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top/xe-arch.html
            else
                return 82u; // largest from above
        }

        static inline void getMinMaxSubgroupSizeFromDriverID(E_DRIVER_ID driverID, uint32_t& minSubgroupSize, uint32_t& maxSubgroupSize)
        {
            const bool isIntelGPU = (driverID == E_DRIVER_ID::EDI_INTEL_OPEN_SOURCE_MESA || driverID == E_DRIVER_ID::EDI_INTEL_PROPRIETARY_WINDOWS);
            const bool isAMDGPU = (driverID == E_DRIVER_ID::EDI_AMD_OPEN_SOURCE || driverID == E_DRIVER_ID::EDI_AMD_PROPRIETARY);
            const bool isNVIDIAGPU = (driverID == E_DRIVER_ID::EDI_NVIDIA_PROPRIETARY);
            if(isIntelGPU)
            {
                minSubgroupSize = 8u;
                maxSubgroupSize = 32u;
            }
            else if(isAMDGPU)
            {
                minSubgroupSize = 32u;
                maxSubgroupSize = 64u;
            }
            else if(isNVIDIAGPU)
            {
                minSubgroupSize = 32u;
                maxSubgroupSize = 32u;
            }
            else
            {
                minSubgroupSize = 4u;
                maxSubgroupSize = 64u;
            }
        }
        
        IAPIConnection* m_api; // dumb pointer to avoid circ ref
        core::smart_refctd_ptr<system::ISystem> m_system;

        SProperties m_properties = {};
        SFeatures m_features = {};
        SMemoryProperties m_memoryProperties = {};

        using qfam_props_array_t = core::smart_refctd_dynamic_array<SQueueFamilyProperties>;
        qfam_props_array_t m_qfamProperties;

        SFormatImageUsages m_linearTilingUsages = {};
        SFormatImageUsages m_optimalTilingUsages = {};
        SFormatBufferUsages m_bufferUsages = {};

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
