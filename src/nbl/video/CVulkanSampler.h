#ifndef __NBL_VIDEO_C_VULKAN_SAMPLER_H_INCLUDED__

#include "nbl/video/IGPUSampler.h"

#include <volk.h>

namespace nbl::video
{

class CVulkanSampler : public IGPUSampler
{
public:
    // Todo(achal): Constructor & desctructor

    inline VkSampler getInternalObject() const { return m_sampler; }

private:
    VkSampler m_sampler;
};

}

#define __NBL_VIDEO_C_VULKAN_SAMPLER_H_INCLUDED__
#endif
