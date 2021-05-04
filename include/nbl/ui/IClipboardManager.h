#ifndef __NBL_I_CLIPBOARD_MANAGER_H_INCLUDED__
#define __NBL_I_CLIPBOARD_MANAGER_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/system/ISystem.h"

namespace nbl {
namespace ui
{

class IClipboardManager : public core::IReferenceCounted
{
public:
    virtual const char* getClipboard() = 0;
    virtual bool setClipboard(const char* contents) = 0;

    virtual ~IClipboardManager() = default;

protected:
    IClipboardManager(core::smart_refctd_ptr<system::ISystem>&& sys) : m_sys(std::move(sys)) {}

    core::smart_refctd_ptr<system::ISystem> m_sys;
};

}
}

#endif
