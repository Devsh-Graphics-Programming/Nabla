#ifndef __NBL_SYSTEM_C_CURSOR_CONTROL_XCB_H_INCLUDED__
#define __NBL_SYSTEM_C_CURSOR_CONTROL_XCB_H_INCLUDED__

#ifdef _NBL_PLATFORM_LINUX_

#include "nbl/ui/ICursorControl.h"

namespace nbl::ui
{

class XCBConnection;
class NBL_API2 CCursorControlXCB final : public ICursorControl
{
    public:
        inline CCursorControlXCB() {}

        void setVisible(bool visible) override;
        bool isVisible() const override;

        void setPosition(SPosition pos) override;
        void setRelativePosition(IWindow* window, SRelativePosition pos) override;

        SPosition getPosition() override;
        SRelativePosition getRelativePosition(IWindow* window) override;
    private: 
        // core::smart_refctd_ptr<XCBConnection> m_connection;
};
}

#endif

#endif