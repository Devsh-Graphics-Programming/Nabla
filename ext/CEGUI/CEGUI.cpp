#include "CEGUI.h"

namespace irr
{
namespace ext
{
namespace cegui
{

GUIManager* createGUIManager(video::IVideoDriver* driver)
{
    return new GUIManager(driver);
}

GUIManager::GUIManager(video::IVideoDriver* driver)
    :   Driver(driver)
{

}

GUIManager::~GUIManager()
{

}

} // namespace cegui
} // namespace ext
} // namespace irr
