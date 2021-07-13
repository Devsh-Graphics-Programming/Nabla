#include "nbl/video/IPhysicalDevice.h"

#include <volk.h>

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl {
namespace video
{

class CVulkanPhysicalDevice final : public IPhysicalDevice
{
public:
    CVulkanPhysicalDevice(VkPhysicalDevice _vkphd, core::smart_refctd_ptr<io::IFileSystem>&& fs, core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc)
        : IPhysicalDevice(std::move(fs), std::move(glslc)), m_vkphysdev(_vkphd)
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(m_vkphysdev, &props);
        // TODO fill m_properties

        VkPhysicalDeviceFeatures features;
        vkGetPhysicalDeviceFeatures(m_vkphysdev, &features);
        // TODO fill m_features

        uint32_t qfamCount = 0u;
        vkGetPhysicalDeviceQueueFamilyProperties(m_vkphysdev, &qfamCount, nullptr);
        core::vector<VkQueueFamilyProperties> qfamprops(qfamCount);
        vkGetPhysicalDeviceQueueFamilyProperties(m_vkphysdev, &qfamCount, qfamprops.data());

        m_qfamProperties = core::make_refctd_dynamic_array<qfam_props_array_t>(qfamCount);
        for (uint32_t i = 0u; i < qfamCount; ++i)
        {
            const auto& vkqf = qfamprops[i];
            auto& qf = (*m_qfamProperties)[i];

            qf.queueCount = vkqf.queueCount;
            qf.queueFlags = static_cast<E_QUEUE_FLAGS>(vkqf.queueFlags);
            qf.timestampValidBits = vkqf.timestampValidBits;
            qf.minImageTransferGranularity = { vkqf.minImageTransferGranularity.width, vkqf.minImageTransferGranularity.height, vkqf.minImageTransferGranularity.depth };
        }
    }

    inline VkPhysicalDevice getInternalObject() const { return m_vkphysdev; }

    E_API_TYPE getAPIType() const override { return EAT_VULKAN; }

protected:
    core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(const ILogicalDevice::SCreationParams& params) override
    {
        VkDeviceCreateInfo ci;
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        // TODO:
        //ci.enabledExtensionCount
        //ci.enabledLayerCount
        //ci.pEnabledFeatures
        //ci.ppEnabledExtensionNames
        //ci.ppEnabledLayerNames
        ci.pNext = nullptr;
        ci.queueCreateInfoCount = params.queueParamsCount;

        core::vector<VkDeviceQueueCreateInfo> qcis(ci.queueCreateInfoCount);
        for (uint32_t i = 0u; i < qcis.size(); ++i)
        {
            const auto& qparams = params.queueCreateInfos[i];
            auto& qci = qcis[i];

            qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qci.pNext = nullptr;
            qci.queueCount = qparams.count;
            qci.queueFamilyIndex = qparams.familyIndex;
            qci.flags = static_cast<VkDeviceQueueCreateFlags>(qparams.flags);
            qci.pQueuePriorities = qparams.priorities;
        }
        ci.pQueueCreateInfos = qcis.data();

        VkDevice vkdev = VK_NULL_HANDLE;
        vkCreateDevice(m_vkphysdev, &ci, nullptr, &vkdev);

        // TODO uncomment when CVKLogicalDevice has all pure virtual methods implemented
        //return core::make_smart_refctd_ptr<CVKLogicalDevice>(vkdev, params);
        return nullptr;
    }

private:
    VkPhysicalDevice m_vkphysdev;
};

}
}
