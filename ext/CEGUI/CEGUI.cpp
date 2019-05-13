/*

MIT License

Copyright (c) 2019 InnerPiece Technology Co., Ltd.
https://innerpiece.io

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include "CEGUI.h"
#include "CEGUIOpenGLState.h"
#include <CEGUI/CEGUI.h>
#include <CEGUI/CommonDialogs/ColourPicker/ColourPicker.h>

using namespace CEGUI;

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
    :   Driver(driver),
        Renderer(OpenGL3Renderer::create(Sizef(
            float(Driver->getScreenSize().Width),
            float(Driver->getScreenSize().Height)
        )))
{

}

void GUIManager::init()
{
    initOpenGLState();
    Renderer.enableExtraStateSettings(true);

    System::create(Renderer);

    DefaultResourceProvider* rp = static_cast<DefaultResourceProvider*>(
        System::getSingleton().getResourceProvider());
    rp->setResourceGroupDirectory("schemes", "../../media/cegui_alfisko/schemes/");
    rp->setResourceGroupDirectory("imagesets", "../../media/cegui_alfisko/imagesets/");
    rp->setResourceGroupDirectory("fonts", "../../media/cegui_alfisko/fonts/");
    rp->setResourceGroupDirectory("layouts", "../../media/cegui_alfisko/layouts/");
    rp->setResourceGroupDirectory("looknfeels", "../../media/cegui_alfisko/looknfeel/");
    rp->setResourceGroupDirectory("schemas", "../../media/cegui_alfisko/xml_schemas/");

    ImageManager::setImagesetDefaultResourceGroup("imagesets");
    Font::setDefaultResourceGroup("fonts");
    Scheme::setDefaultResourceGroup("schemes");
    WidgetLookManager::setDefaultResourceGroup("looknfeels");
    WindowManager::setDefaultResourceGroup("layouts");

    XMLParser* parser = System::getSingleton().getXMLParser();
    if (parser->isPropertyPresent("SchemaDefaultResourceGroup"))
        parser->setProperty("SchemaDefaultResourceGroup", "schemas");

    SchemeManager::getSingleton().createFromFile("Alfisko.scheme", "schemes");
    FontManager::getSingleton().createFromFile("Cousine-Regular.font");
    SchemeManager::getSingleton().createFromFile("AlfiskoCommonDialogs.scheme",
        "schemes");

    System::getSingleton().getDefaultGUIContext().setDefaultFont(
        "Cousine-Regular");
    System::getSingleton()
        .getDefaultGUIContext()
        .getMouseCursor()
        .setDefaultImage("Alfisko/MouseArrow");
    System::getSingleton().getDefaultGUIContext().setDefaultTooltipType(
        "Alfisko/Tooltip");
    System::getSingleton().notifyDisplaySizeChanged(Sizef(
        float(Driver->getScreenSize().Width),
        float(Driver->getScreenSize().Height)
    ));
}

void GUIManager::destroy()
{
    destroyOpenGLState();
}

void GUIManager::render()
{
    saveOpenGLState();
    CEGUI::System::getSingleton().renderAllGUIContexts();
    restoreOpenGLState();
}

void GUIManager::createRootWindowFromLayout(const std::string& layout)
{
    RootWindow = WindowManager::getSingleton().loadLayoutFromString(layout);
    System::getSingleton().getDefaultGUIContext().setRootWindow(RootWindow);
}

GUIManager::~GUIManager()
{

}

} // namespace cegui
} // namespace ext
} // namespace irr
