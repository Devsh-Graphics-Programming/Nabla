#ifndef __NBL_VIDEO_I_PHYSICAL_DEVICE_H_INCLUDED__
#define __NBL_VIDEO_I_PHYSICAL_DEVICE_H_INCLUDED__

#include "nbl/system/declarations.h"

#include <type_traits>

#include "nbl/asset/IImage.h" //for VkExtent3D only
#include "nbl/asset/ISpecializedShader.h"

#include "nbl/system/ISystem.h"

#include "nbl/video/EApiType.h"
#include "nbl/video/debug/IDebugCallback.h"
#include "nbl/video/ILogicalDevice.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

#include "SPhysicalDeviceLimits.h"
#include "SPhysicalDeviceFeatures.h"

namespace nbl::video
{

class NBL_API2 IPhysicalDevice : public core::Interface, public core::Unmovable
{
    public:
        //
        virtual E_API_TYPE getAPIType() const = 0;

        enum E_TYPE : uint8_t {
            ET_UNKNOWN = 0,
            ET_INTEGRATED_GPU = 1,
            ET_DISCRETE_GPU = 2,
            ET_VIRTUAL_GPU = 3,
            ET_CPU = 4,
        };

        enum E_DRIVER_ID : uint8_t
        {
            EDI_UNKNOWN = 0,
            EDI_AMD_PROPRIETARY = 1,
            EDI_AMD_OPEN_SOURCE = 2,
            EDI_MESA_RADV = 3,
            EDI_NVIDIA_PROPRIETARY = 4,
            EDI_INTEL_PROPRIETARY_WINDOWS = 5,
            EDI_INTEL_OPEN_SOURCE_MESA = 6,
            EDI_IMAGINATION_PROPRIETARY = 7,
            EDI_QUALCOMM_PROPRIETARY = 8,
            EDI_ARM_PROPRIETARY = 9,
            EDI_GOOGLE_SWIFTSHADER = 10,
            EDI_GGP_PROPRIETARY = 11,
            EDI_BROADCOM_PROPRIETARY = 12,
            EDI_MESA_LLVMPIPE = 13,
            EDI_MOLTENVK = 14,
            EDI_COREAVI_PROPRIETARY = 15,
            EDI_JUICE_PROPRIETARY = 16,
            EDI_VERISILICON_PROPRIETARY = 17,
            EDI_MESA_TURNIP = 18,
            EDI_MESA_V3DV = 19,
            EDI_MESA_PANVK = 20,
            EDI_SAMSUNG_PROPRIETARY = 21,
            EDI_MESA_VENUS = 22,
        };

        //
        struct APIVersion
        {
            uint32_t major : 5;
            uint32_t minor : 5;
            uint32_t patch : 22;
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

            /* Vulkan 1.2 Core  or VK_KHR_driver_properties */
            E_DRIVER_ID driverID;
            char driverName[VK_MAX_DRIVER_NAME_SIZE];
            char driverInfo[VK_MAX_DRIVER_INFO_SIZE];
            VkConformanceVersion conformanceVersion;
        };

        const SProperties& getProperties() const { return m_properties; }
        const SLimits& getLimits() const { return m_properties.limits; }
        APIVersion getAPIVersion() const { return m_properties.apiVersion; }
        const SFeatures& getFeatures() const { return m_features; }

        struct MemoryType
        {
            core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS> propertyFlags;
            uint32_t heapIndex;
        };

        struct MemoryHeap
        {
            size_t size;
            core::bitflag<IDeviceMemoryAllocation::E_MEMORY_HEAP_FLAGS> flags;
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
            MemoryType      memoryTypes[VK_MAX_MEMORY_TYPES];
            uint32_t        memoryHeapCount = 0u;
            MemoryHeap      memoryHeaps[VK_MAX_MEMORY_HEAPS];
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
        struct SFormatBufferUsage
        {
            uint8_t isInitialized : 1u;

            uint8_t vertexAttribute : 1u; // vertexAtrtibute binding
            uint8_t bufferView : 1u; // samplerBuffer
            uint8_t storageBufferView : 1u; // imageBuffer
            uint8_t storageBufferViewAtomic : 1u; // imageBuffer
            uint8_t accelerationStructureVertex : 1u;

            SFormatBufferUsage()
                : isInitialized(0)
            {}

            SFormatBufferUsage(core::bitflag<asset::IBuffer::E_USAGE_FLAGS> usages)
                : isInitialized(1),
                vertexAttribute(usages.hasFlags(asset::IBuffer::EUF_VERTEX_BUFFER_BIT)),
                bufferView(usages.hasFlags(asset::IBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT)),
                storageBufferView(usages.hasFlags(asset::IBuffer::EUF_STORAGE_TEXEL_BUFFER_BIT)),
                accelerationStructureVertex(usages.hasFlags(asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT)),
                // Deduced as false. User may patch it up later
                storageBufferViewAtomic(0)
            {}

            inline SFormatBufferUsage operator & (const SFormatBufferUsage& other) const
            {
                SFormatBufferUsage result;
                result.vertexAttribute = vertexAttribute & other.vertexAttribute;
                result.bufferView = bufferView & other.bufferView;
                result.storageBufferView = storageBufferView & other.storageBufferView;
                result.storageBufferViewAtomic = storageBufferViewAtomic & other.storageBufferViewAtomic;
                result.accelerationStructureVertex = accelerationStructureVertex & other.accelerationStructureVertex;
                return result;
            }

            inline SFormatBufferUsage operator | (const SFormatBufferUsage& other) const
            {
                SFormatBufferUsage result;
                result.vertexAttribute = vertexAttribute | other.vertexAttribute;
                result.bufferView = bufferView | other.bufferView;
                result.storageBufferView = storageBufferView | other.storageBufferView;
                result.storageBufferViewAtomic = storageBufferViewAtomic | other.storageBufferViewAtomic;
                result.accelerationStructureVertex = accelerationStructureVertex | other.accelerationStructureVertex;
                return result;
            }

            inline SFormatBufferUsage operator ^ (const SFormatBufferUsage& other) const
            {
                SFormatBufferUsage result;
                result.vertexAttribute = vertexAttribute ^ other.vertexAttribute;
                result.bufferView = bufferView ^ other.bufferView;
                result.storageBufferView = storageBufferView ^ other.storageBufferView;
                result.storageBufferViewAtomic = storageBufferViewAtomic ^ other.storageBufferViewAtomic;
                result.accelerationStructureVertex = accelerationStructureVertex ^ other.accelerationStructureVertex;
                return result;
            }

            inline bool operator<(const SFormatBufferUsage& other) const
            {
                if (vertexAttribute && !other.vertexAttribute) return false;
                if (bufferView && !other.bufferView) return false;
                if (storageBufferView && !other.storageBufferView) return false;
                if (storageBufferViewAtomic && !other.storageBufferViewAtomic) return false;
                if (accelerationStructureVertex && !other.accelerationStructureVertex) return false;
                return true;
            }

            inline bool operator == (const SFormatBufferUsage& other) const
            {
                return
                    (vertexAttribute == other.vertexAttribute) &&
                    (bufferView == other.bufferView) &&
                    (storageBufferView == other.storageBufferView) &&
                    (storageBufferViewAtomic == other.storageBufferViewAtomic) &&
                    (accelerationStructureVertex == other.accelerationStructureVertex);
            }
        };
        virtual const SFormatBufferUsage& getBufferFormatUsages(const asset::E_FORMAT format) = 0;

        //
        struct SFormatImageUsage
        {
            uint8_t isInitialized : 1u; // TODO: get rid of this

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

            SFormatImageUsage()
                : isInitialized(0)
            {}

            SFormatImageUsage(core::bitflag<asset::IImage::E_USAGE_FLAGS> usages)
                : isInitialized(1), 
                log2MaxSamples(0),
                sampledImage(usages.hasFlags(asset::IImage::EUF_SAMPLED_BIT)),
                storageImage(usages.hasFlags(asset::IImage::EUF_STORAGE_BIT)),
                transferSrc(usages.hasFlags(asset::IImage::EUF_TRANSFER_SRC_BIT)),
                transferDst(usages.hasFlags(asset::IImage::EUF_TRANSFER_DST_BIT)),
                attachment((usages & (core::bitflag(asset::IImage::EUF_COLOR_ATTACHMENT_BIT) | asset::IImage::EUF_DEPTH_STENCIL_ATTACHMENT_BIT)).value != 0),
                attachmentBlend(usages.hasFlags(asset::IImage::EUF_COLOR_ATTACHMENT_BIT)), // TODO: should conservatively deduct to be false
                // Deduced as false. User may patch it up later
                blitSrc(0),
                blitDst(0),
                storageImageAtomic(0)
            {}

            inline SFormatImageUsage operator & (const SFormatImageUsage& other) const
            {
                SFormatImageUsage result;
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

            inline SFormatImageUsage operator | (const SFormatImageUsage& other) const
            {
                SFormatImageUsage result;
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

            inline SFormatImageUsage operator ^ (const SFormatImageUsage& other) const
            {
                SFormatImageUsage result;
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

            inline bool operator<(const SFormatImageUsage& other) const
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

            inline bool operator == (const SFormatImageUsage& other) const
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

        virtual const SFormatImageUsage& getImageFormatUsagesLinear(const asset::E_FORMAT format) = 0;
        virtual const SFormatImageUsage& getImageFormatUsagesOptimal(const asset::E_FORMAT format) = 0;

        //
        enum E_QUEUE_FLAGS : uint32_t
        {
            EQF_GRAPHICS_BIT = 0x01,
            EQF_COMPUTE_BIT = 0x02,
            EQF_TRANSFER_BIT = 0x04,
            EQF_SPARSE_BINDING_BIT = 0x08,
            EQF_PROTECTED_BIT = 0x10
        };

        
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
            core::bitflag<E_QUEUE_FLAGS> queueFlags;
            uint32_t queueCount;
            uint32_t timestampValidBits;
            asset::VkExtent3D minImageTransferGranularity;

            inline bool operator!=(const SQueueFamilyProperties& other) const
            {
                return queueFlags.value != other.queueFlags.value || queueCount != other.queueCount || timestampValidBits != other.timestampValidBits || minImageTransferGranularity != other.minImageTransferGranularity;
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

        // these are the defines which shall be added to any IGPUShader which has its source as GLSL
        inline core::SRange<const char* const> getExtraGLSLDefines() const
        {
            const char* const* begin = m_extraGLSLDefines.data();
            return {begin,begin+m_extraGLSLDefines.size()};
        }

        template<typename FORMAT_USAGE>
        struct FormatPromotionRequest
        {
            asset::E_FORMAT originalFormat = asset::EF_UNKNOWN;
            FORMAT_USAGE usages = FORMAT_USAGE(0);

            struct hash
            {
                // pack into 64bit for easy hashing 
                uint64_t operator()(const FormatPromotionRequest<FORMAT_USAGE>& r) const
                {
                    uint64_t msb = uint64_t(std::hash<FORMAT_USAGE>()(r.usages));
                    return (msb << 32u) | r.originalFormat;
                }
            };

            struct equal_to
            {
                bool operator()(const FormatPromotionRequest<FORMAT_USAGE>& l, const FormatPromotionRequest<FORMAT_USAGE>& r) const
                {
                    return l.originalFormat == r.originalFormat && l.usages == r.usages;
                }
            };
        };

        asset::E_FORMAT promoteBufferFormat(const FormatPromotionRequest<video::IPhysicalDevice::SFormatBufferUsage> req);
        asset::E_FORMAT promoteImageFormat(const FormatPromotionRequest<video::IPhysicalDevice::SFormatImageUsage> req, const IGPUImage::E_TILING tiling);

        //
        inline system::ISystem* getSystem() const {return m_system.get();}
        inline asset::IGLSLCompiler* getGLSLCompiler() const {return m_GLSLCompiler.get();}

        virtual IDebugCallback* getDebugCallback() = 0;

        // TODO: shouldn't this be in SFeatures?
        virtual bool isSwapchainSupported() const = 0;

        core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice(const ILogicalDevice::SCreationParams& params)
        {
            if (!validateLogicalDeviceCreation(params))
                return nullptr;

            return createLogicalDevice_impl(params);
        }

    protected:
        IPhysicalDevice(core::smart_refctd_ptr<system::ISystem>&& s, core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc);

        virtual core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(const ILogicalDevice::SCreationParams& params) = 0;

        bool validateLogicalDeviceCreation(const ILogicalDevice::SCreationParams& params) const;

        void addCommonGLSLDefines(std::ostringstream& pool, const bool runningInRenderDoc);

        template<typename... Args>
        inline void addGLSLDefineToPool(std::ostringstream& pool, const char* define, Args&&... args)
        {
            const ptrdiff_t pos = pool.tellp();
            m_extraGLSLDefines.push_back(reinterpret_cast<const char*>(pos));
            pool << define << " ";
            ((pool << std::forward<Args>(args)), ...);
        }
        inline void finalizeGLSLDefinePool(std::ostringstream&& pool)
        {
            m_GLSLDefineStringPool.resize(static_cast<size_t>(pool.tellp())+m_extraGLSLDefines.size());
            const auto data = ptrdiff_t(m_GLSLDefineStringPool.data());

            const auto str = pool.str();
            size_t nullCharsWritten = 0u;
            for (auto i=0u; i<m_extraGLSLDefines.size(); i++)
            {
                auto& dst = m_extraGLSLDefines[i];
                const auto len = (i!=(m_extraGLSLDefines.size()-1u) ? ptrdiff_t(m_extraGLSLDefines[i+1]):str.length())-ptrdiff_t(dst);
                const char* src = str.data()+ptrdiff_t(dst);
                dst += data+(nullCharsWritten++);
                memcpy(const_cast<char*>(dst),src,len);
                const_cast<char*>(dst)[len] = 0;
            }
        }

        
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

        core::smart_refctd_ptr<system::ISystem> m_system;
        core::smart_refctd_ptr<asset::IGLSLCompiler> m_GLSLCompiler;

        SProperties m_properties;
        SFeatures m_features;
        SMemoryProperties m_memoryProperties;
        using qfam_props_array_t = core::smart_refctd_dynamic_array<SQueueFamilyProperties>;
        qfam_props_array_t m_qfamProperties;

        SFormatImageUsage m_linearTilingUsages[asset::EF_UNKNOWN] = {};
        SFormatImageUsage m_optimalTilingUsages[asset::EF_UNKNOWN] = {};
        SFormatBufferUsage m_bufferUsages[asset::EF_UNKNOWN] = {};

        core::vector<char> m_GLSLDefineStringPool;
        core::vector<const char*> m_extraGLSLDefines;

        typedef core::unordered_map<FormatPromotionRequest<video::IPhysicalDevice::SFormatBufferUsage>, asset::E_FORMAT, FormatPromotionRequest<video::IPhysicalDevice::SFormatBufferUsage>::hash, FormatPromotionRequest<video::IPhysicalDevice::SFormatBufferUsage>::equal_to> format_buffer_cache_t;
        typedef core::unordered_map<FormatPromotionRequest<video::IPhysicalDevice::SFormatImageUsage>, asset::E_FORMAT, FormatPromotionRequest<video::IPhysicalDevice::SFormatImageUsage>::hash, FormatPromotionRequest<video::IPhysicalDevice::SFormatImageUsage>::equal_to> format_image_cache_t;

        struct format_promotion_cache_t
        {
            format_buffer_cache_t buffers;
            format_image_cache_t optimalTilingImages;
            format_image_cache_t linearTilingImages;
        } m_formatPromotionCache;
};

}

namespace std
{
    template<>
    struct std::hash<nbl::video::IPhysicalDevice::SFormatImageUsage>
    {
        inline uint32_t operator()(const nbl::video::IPhysicalDevice::SFormatImageUsage& i) const
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
    struct std::hash<nbl::video::IPhysicalDevice::SFormatBufferUsage>
    {
        inline uint32_t operator()(const nbl::video::IPhysicalDevice::SFormatBufferUsage& b) const
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

#endif
