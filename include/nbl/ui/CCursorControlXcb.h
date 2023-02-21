#ifndef __NBL_SYSTEM_C_CURSOR_CONTROL_XCB_H_INCLUDED__
#define __NBL_SYSTEM_C_CURSOR_CONTROL_XCB_H_INCLUDED__


#include "nbl/ui/ICursorControl.h"
#include "nbl/ui/XcbConnection.h"

#ifdef _NBL_PLATFORM_LINUX_

namespace nbl::ui
{
class NBL_API2 CCursorControlXcb final : public ICursorControl
{
        core::smart_refctd_ptr<XcbConnection> m_xcbConnection;

    public:
        inline CCursorControlXcb(
            core::smart_refctd_ptr<XcbConnection>&& xcbConnection) : 
            m_xcbConnection(std::move(xcbConnection)) {}

        void setVisible(bool visible) override;
        bool isVisible() const override;

        void setPosition(SPosition pos) override;
        void setRelativePosition(IWindow* window, SRelativePosition pos) override;

        SPosition getPosition() override;
        SRelativePosition getRelativePosition(IWindow* window) override;
};
}

#endif

#endif