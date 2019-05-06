#ifndef _IRR_EXT_CEGUI_INCLUDED_
#define _IRR_EXT_CEGUI_INCLUDED_

#include "irrlicht.h"
#include <CEGUI/CEGUI.h>

namespace irr
{
namespace ext
{
namespace cegui
{

class GUIManager;
GUIManager* createGUIManager(video::IVideoDriver* driver);

class GUIManager: public core::IReferenceCounted
{
public:
    GUIManager(video::IVideoDriver* driver);
    ~GUIManager();

    void init();
    void shutdown();
    void render();

private:
    video::IVideoDriver* Driver = nullptr;
};

} // namespace cegui
} // namespace ext
} // namespace irr

#endif // _IRR_EXT_CEGUI_INCLUDED_
