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
#include "CEGUIOpenGLClip.h"

using namespace CEGUI;

namespace irr
{
namespace ext
{
namespace cegui
{

GUIManager* createGUIManager(IrrlichtDevice* device)
{
    auto gui = new GUIManager(device->getVideoDriver());
    device->setEventReceiver(gui);
    return gui;
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
    rp->setResourceGroupDirectory("schemes", "./../../media/cegui_alfisko/schemes/");
    rp->setResourceGroupDirectory("imagesets", "./../../media/cegui_alfisko/imagesets/");
    rp->setResourceGroupDirectory("fonts", "./../../media/cegui_alfisko/fonts/");
    rp->setResourceGroupDirectory("layouts", "./../../media/cegui_alfisko/layouts/");
    rp->setResourceGroupDirectory("looknfeels", "./../../media/cegui_alfisko/looknfeel/");
    rp->setResourceGroupDirectory("schemas", "./../../media/cegui_alfisko/xml_schemas/");

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
    initialiseCEGUICommonDialogs();
}

void GUIManager::destroy()
{
    destroyOpenGLState();
}

void GUIManager::render()
{
    saveOpenGLState();
    setOpenGLClip();
    CEGUI::System::getSingleton().renderAllGUIContexts();
    restoreOpenGLState();
}

bool GUIManager::OnEvent(const SEvent& event)
{
    CEGUI::GUIContext& context = CEGUI::System::getSingleton().getDefaultGUIContext();

    switch (event.EventType) {
        case irr::EET_KEY_INPUT_EVENT:
        {
            if (event.KeyInput.PressedDown)
            {
                context.injectKeyDown(toCEGUIKey(event.KeyInput.Key));
                context.injectChar(event.KeyInput.Char);
            }
            else
            {
                context.injectKeyUp(toCEGUIKey(event.KeyInput.Key));
            }
        } break;


        case irr::EET_MOUSE_INPUT_EVENT:
        {
            // if (event.MouseInput.ButtonStates)
            context.injectMousePosition(event.MouseInput.X, event.MouseInput.Y);

        } break;

        default: return false;
    }
    return true;
}

void GUIManager::createRootWindowFromLayout(const std::string& layout)
{
    RootWindow = WindowManager::getSingleton().loadLayoutFromString(layout);
    System::getSingleton().getDefaultGUIContext().setRootWindow(RootWindow);
}

CEGUI::ColourPicker* GUIManager::createColourPicker(
    bool alternativeLayout,
    const char* parent,
    const char* title,
    const char* name)
{
    assert(parent);
    assert(name);
    static const auto defaultColor = CEGUI::Colour(1.0f, 1.0f, 1.0f, 1.0f);

    Window* layout = WindowManager::getSingleton().loadLayoutFromFile(
        alternativeLayout ? "CPAlternativeLayout.layout"
                           : "CPMainLayout.layout");

    if (RootWindow && layout) {
        auto window = static_cast<DefaultWindow*>(RootWindow->getChild(parent));
        window->addChild(layout);
        // window->setSize(USize(UDim(1.0f, 0.0f), UDim(1.0f, 0.0f)));

        if (title) {
            auto bar = static_cast<DefaultWindow*>(layout->getChild("TitleLabel"));
            bar->setText(title);
        }

        static std::array<char, 3> s{ 'R', 'G', 'B' };

        for (const auto& v : s) {
            static_cast<Slider*>(layout->getChild(std::string("Slider") + v))
                ->subscribeEvent(Slider::EventValueChanged, [layout](void) -> void {
                    auto sliderR = static_cast<Slider*>(layout->getChild("SliderR"))
                                       ->getCurrentValue();
                    auto sliderG = static_cast<Slider*>(layout->getChild("SliderG"))
                                       ->getCurrentValue();
                    auto sliderB = static_cast<Slider*>(layout->getChild("SliderB"))
                                       ->getCurrentValue();

                    const Colour c(sliderR / 255.0f, sliderG / 255.0f, sliderB / 255.0f,
                        1.0f);

                    static_cast<ColourPicker*>(
                        layout->getChild("ColorPickerContainer/MyPicker"))
                        ->setColour(c);

                    if (layout->isChild("LabelShared")) {
                        Window* label_shared = layout->getChild("LabelShared");

                        std::ostringstream ss;
                        ss << '(' << "[colour='FFFFFFFF']"
                           << "[colour='FFFF0000']" << std::setprecision(3) << sliderR
                           << ", "
                           << "[colour='FF00FF00']" << std::setprecision(3) << sliderG
                           << ", "
                           << "[colour='FF0000FF']" << std::setprecision(3) << sliderB
                           << "[colour='FFFFFFFF']" << ')';

                        static_cast<DefaultWindow*>(label_shared)->setText(ss.str());
                    } else {
                        auto* labelR = layout->getChild("SliderR/Label");
                        auto* labelG = layout->getChild("SliderG/Label");
                        auto* labelB = layout->getChild("SliderB/Label");

                        static_cast<DefaultWindow*>(labelR)->setText(
                            String("[colour='FFFF0000']") + toStringFloat(sliderR, 0));
                        static_cast<DefaultWindow*>(labelG)->setText(
                            String("[colour='FF00FF00']") + toStringFloat(sliderG, 0));
                        static_cast<DefaultWindow*>(labelB)->setText(
                            String("[colour='FF0000FF']") + toStringFloat(sliderB, 0));
                    }
                });
        }

        auto cpicker_window = static_cast<DefaultWindow*>(layout->getChild("ColorPickerContainer"));
        ColourPicker* picker = static_cast<ColourPicker*>(WindowManager::getSingleton().createWindow(
            "Alfisko/CPColourPicker", "MyPicker"));
        picker->setTooltipText("Left-click to open the color picker.");

        cpicker_window->addChild(picker);

        picker->subscribeEvent(ColourPicker::EventAcceptedColour,
            [layout, picker](void) -> void {
                const auto color = picker->getColour();
                static_cast<Slider*>(layout->getChild("SliderR"))
                    ->setCurrentValue(color.getRed() * 255.0f);
                static_cast<Slider*>(layout->getChild("SliderG"))
                    ->setCurrentValue(color.getGreen() * 255.0f);
                static_cast<Slider*>(layout->getChild("SliderB"))
                    ->setCurrentValue(color.getBlue() * 255.0f);
            });

        picker->subscribeEvent(
            ColourPicker::EventOpenedPicker, [layout, picker](void) -> void {
                auto sliderR = static_cast<Slider*>(layout->getChild("SliderR"))
                                   ->getCurrentValue();
                auto sliderG = static_cast<Slider*>(layout->getChild("SliderG"))
                                   ->getCurrentValue();
                auto sliderB = static_cast<Slider*>(layout->getChild("SliderB"))
                                   ->getCurrentValue();
            });

        picker->setInheritsAlpha(false);
        picker->setPosition(UVector2(UDim(0.0f, 0.0f), UDim(0.0f, 0.0f)));
        picker->setSize(USize(UDim(1.0f, 0.0f), UDim(1.0f, 0.0f)));
        picker->setColour(defaultColor);

        auto pickerWindow = static_cast<CEGUI::ColourPicker*>(layout);
        ColourPickers[name] = pickerWindow;
        return pickerWindow;
    }

    return nullptr;
}

GUIManager::~GUIManager()
{

}

} // namespace cegui
} // namespace ext
} // namespace irr
