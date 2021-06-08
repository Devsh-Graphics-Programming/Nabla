#ifndef __NBL_I_CLIPBOARD_MANAGER_H_INCLUDED__
#define __NBL_I_CLIPBOARD_MANAGER_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/system/ISystem.h"
#include "nbl/asset/ICPUImage.h"

namespace nbl {
namespace ui
{

class IClipboardManager : public core::IReferenceCounted
{
public:
    virtual std::string getClipboardText() = 0;
    virtual bool setClipboardText(const std::string_view& data) = 0;
    virtual core::smart_refctd_ptr<asset::ICPUImage> getClipboardImage() = 0;
    virtual bool setClipboardImage(asset::ICPUImage* image, asset::ICPUImage::SImageCopy data) = 0;
         
    virtual ~IClipboardManager() = default;

protected:
    IClipboardManager(core::smart_refctd_ptr<system::ISystem>&& sys) : m_sys(std::move(sys)) {}

    core::smart_refctd_ptr<system::ISystem> m_sys;
};

}
}

#endif
