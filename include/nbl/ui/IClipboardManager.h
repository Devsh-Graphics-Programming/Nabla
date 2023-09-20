#ifndef _NBL_I_CLIPBOARD_MANAGER_H_INCLUDED_
#define _NBL_I_CLIPBOARD_MANAGER_H_INCLUDED_

#include "nbl/core/IReferenceCounted.h"

#include "nbl/asset/ICPUImage.h"

namespace nbl::ui
{

class IClipboardManager : public core::IReferenceCounted
{
    public:
        struct SImageClipboardRegion
        {
            VkOffset2D srcOffset;
            VkOffset2D dstOffset;
            VkExtent2D extent;
        };

        NBL_API2 virtual std::string getClipboardText() = 0;
        NBL_API2 virtual bool setClipboardText(const std::string_view& data) = 0;

        // virtual core::smart_refctd_ptr<asset::ICPUImage> getClipboardImage() = 0;
        // virtual bool setClipboardImage(asset::ICPUImage* image, const SImageClipboardRegion& data) = 0;
         
        NBL_API2 virtual ~IClipboardManager() = default;

    protected:
        NBL_API2 inline IClipboardManager() = default;
};

}

#endif
