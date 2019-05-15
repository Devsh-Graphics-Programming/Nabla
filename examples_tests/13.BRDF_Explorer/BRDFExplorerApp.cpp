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

#include "BRDFExplorerApp.h"
#include "../../ext/CEGUI/CEGUI.h"

namespace irr
{

BRDFExplorerApp::BRDFExplorerApp(IrrlichtDevice* device)
    :   Driver(device->getVideoDriver()),
        GUI(ext::cegui::createGUIManager(device))
{
    TextureSlotMap = {
        { ETEXTURE_SLOT::TEXTURE_AO,
        std::make_tuple("AOTextureBuffer", // Texture buffer name
            "AOTexture", // Texture name
            "MaterialParamsWindow/AOWindow/ImageButton") },

        { ETEXTURE_SLOT::TEXTURE_BUMP,
            std::make_tuple("BumpTextureBuffer", // Texture buffer name
                "BumpTexture", // Texture name
                "MaterialParamsWindow/BumpWindow/ImageButton") },

        { ETEXTURE_SLOT::TEXTURE_SLOT_1,
            std::make_tuple("T1TextureBuffer", // Texture buffer name
                "T1Texture", // Texture name
                "TextureViewWindow/Texture0Window/Texture") },

        { ETEXTURE_SLOT::TEXTURE_SLOT_2,
            std::make_tuple("T2TextureBuffer", // Texture buffer name
                "T2Texture", // Texture name
                "TextureViewWindow/Texture1Window/Texture") },

        { ETEXTURE_SLOT::TEXTURE_SLOT_3,
            std::make_tuple("T3TextureBuffer", // Texture buffer name
                "T3Texture", // Texture name
                "TextureViewWindow/Texture2Window/Texture") },

        { ETEXTURE_SLOT::TEXTURE_SLOT_4,
            std::make_tuple("T4TextureBuffer", // Texture buffer name
                "T4Texture", // Texture name
                "TextureViewWindow/Texture3Window/Texture") },
    };

    GUI->init();
    GUI->createRootWindowFromLayout(
        ext::cegui::readWindowLayout("../../media/brdf_explorer/MainWindow.layout")
    );
    GUI->createColourPicker(false, "LightParamsWindow/ColorWindow", "Color", "pickerLightColor");
    GUI->createColourPicker(true, "MaterialParamsWindow/EmissiveWindow", "Emissive", "pickerEmissiveColor");
    GUI->createColourPicker(true, "MaterialParamsWindow/AlbedoWindow", "Albedo", "pickerAlbedoColor");

    // Fill all the available texture slots using the default (no texture) image
    const auto image_default = ext::cegui::loadImage("../../media/brdf_explorer/DefaultEmpty.png");

    loadTextureSlot(ETEXTURE_SLOT::TEXTURE_AO, image_default.buffer, image_default.w,
        image_default.h);
    loadTextureSlot(ETEXTURE_SLOT::TEXTURE_BUMP, image_default.buffer, image_default.w,
        image_default.h);
    loadTextureSlot(ETEXTURE_SLOT::TEXTURE_SLOT_1, image_default.buffer, image_default.w,
        image_default.h);
    loadTextureSlot(ETEXTURE_SLOT::TEXTURE_SLOT_2, image_default.buffer, image_default.w,
        image_default.h);
    loadTextureSlot(ETEXTURE_SLOT::TEXTURE_SLOT_3, image_default.buffer, image_default.w,
        image_default.h);
    loadTextureSlot(ETEXTURE_SLOT::TEXTURE_SLOT_4, image_default.buffer, image_default.w,
        image_default.h);

    auto root = GUI->getRootWindow();
    // Material window: Subscribe to sliders' events and set its default value to
    // 0.0.
    GUI->registerSliderEvent(
        "MaterialParamsWindow/RefractionIndexWindow/Slider", sliderRIRange, 0.01f,
        [root](const ::CEGUI::EventArgs&) {
            auto roughness = static_cast<::CEGUI::Slider*>(
                root->getChild(
                    "MaterialParamsWindow/RefractionIndexWindow/Slider"))
                                 ->getCurrentValue();
            root->getChild(
                    "MaterialParamsWindow/RefractionIndexWindow/LabelPercent")
                ->setText(ext::cegui::toStringFloat(roughness, 2));
        });

    GUI->registerSliderEvent(
        "MaterialParamsWindow/MetallicWindow/Slider", sliderMetallicRange, 0.01f,
        [root](const ::CEGUI::EventArgs&) {
            auto roughness = static_cast<::CEGUI::Slider*>(
                root->getChild("MaterialParamsWindow/MetallicWindow/Slider"))
                                 ->getCurrentValue();
            root->getChild("MaterialParamsWindow/MetallicWindow/LabelPercent")
                ->setText(ext::cegui::toStringFloat(roughness, 2));
        });

    GUI->registerSliderEvent(
        "MaterialParamsWindow/RoughnessWindow/Slider", sliderRoughness1Range,
        0.01f, [this](const ::CEGUI::EventArgs&) {
            auto root = GUI->getRootWindow();

            const auto v = static_cast<::CEGUI::Slider*>(
                root->getChild("MaterialParamsWindow/RoughnessWindow/Slider"))
                               ->getCurrentValue();
            const auto s = ext::cegui::toStringFloat(v, 2);

            root->getChild("MaterialParamsWindow/RoughnessWindow/LabelPercent1")
                ->setText(s);

            if (IsIsotropic) {
                root->getChild("MaterialParamsWindow/RoughnessWindow/LabelPercent2")
                    ->setText(s);
                static_cast<::CEGUI::Slider*>(
                    root->getChild("MaterialParamsWindow/RoughnessWindow/Slider2"))
                    ->setCurrentValue(v);
            }
        });

    GUI->registerSliderEvent(
        "MaterialParamsWindow/RoughnessWindow/Slider2", sliderRoughness2Range,
        0.01f, [root](const ::CEGUI::EventArgs&) {
            auto roughness = static_cast<::CEGUI::Slider*>(
                root->getChild("MaterialParamsWindow/RoughnessWindow/Slider2"))
                                 ->getCurrentValue();
            root->getChild("MaterialParamsWindow/RoughnessWindow/LabelPercent2")
                ->setText(ext::cegui::toStringFloat(roughness, 2));
        });

    // Set the sliders' text objects to their default value (whatever value the
    // slider was set to).
    {
        // Roughness slider, first one
        root->getChild("MaterialParamsWindow/RoughnessWindow/LabelPercent2")
            ->setText(ext::cegui::toStringFloat(
                static_cast<::CEGUI::Slider*>(
                    root->getChild("MaterialParamsWindow/RoughnessWindow/Slider2"))
                    ->getCurrentValue(),
                2));

        // Roughness slider, second one
        root->getChild("MaterialParamsWindow/RoughnessWindow/LabelPercent1")
            ->setText(ext::cegui::toStringFloat(
                static_cast<::CEGUI::Slider*>(
                    root->getChild("MaterialParamsWindow/RoughnessWindow/Slider"))
                    ->getCurrentValue(),
                2));

        // Refractive index slider
        root->getChild("MaterialParamsWindow/RefractionIndexWindow/LabelPercent")
            ->setText(ext::cegui::toStringFloat(
                static_cast<::CEGUI::Slider*>(
                    root->getChild(
                        "MaterialParamsWindow/RefractionIndexWindow/Slider"))
                    ->getCurrentValue(),
                2));

        // Metallic slider
        root->getChild("MaterialParamsWindow/MetallicWindow/LabelPercent")
            ->setText(ext::cegui::toStringFloat(
                static_cast<::CEGUI::Slider*>(
                    root->getChild("MaterialParamsWindow/MetallicWindow/Slider"))
                    ->getCurrentValue(),
                2));

        // Bump-mapping's height slider
        root->getChild("MaterialParamsWindow/BumpWindow/LabelPercent")
            ->setText(ext::cegui::toStringFloat(
                static_cast<::CEGUI::Slider*>(
                    root->getChild("MaterialParamsWindow/BumpWindow/Spinner"))
                    ->getCurrentValue(),
                2));
    }

    // Isotropic checkbox
    auto isotropic = static_cast<::CEGUI::ToggleButton*>(
    root->getChild("MaterialParamsWindow/RoughnessWindow/Checkbox"));
    isotropic->subscribeEvent(
        ::CEGUI::ToggleButton::EventSelectStateChanged,
        [this](const ::CEGUI::EventArgs& e) -> void {
            auto root = GUI->getRootWindow();

            const ::CEGUI::WindowEventArgs& we = static_cast<const ::CEGUI::WindowEventArgs&>(e);
            IsIsotropic = static_cast<::CEGUI::ToggleButton*>(we.window)->isSelected();
            static_cast<::CEGUI::Slider*>(
                root->getChild("MaterialParamsWindow/RoughnessWindow/Slider2"))
                ->setDisabled(IsIsotropic);

            if (IsIsotropic) {
                root->getChild("MaterialParamsWindow/RoughnessWindow/LabelPercent2")
                    ->setText(ext::cegui::toStringFloat(
                        static_cast<::CEGUI::Slider*>(
                            root->getChild(
                                "MaterialParamsWindow/RoughnessWindow/Slider"))
                            ->getCurrentValue(),
                        2));

                static_cast<::CEGUI::Slider*>(
                    root->getChild("MaterialParamsWindow/RoughnessWindow/Slider2"))
                    ->setCurrentValue(
                        static_cast<::CEGUI::Slider*>(
                            root->getChild(
                                "MaterialParamsWindow/RoughnessWindow/Slider"))
                            ->getCurrentValue());
            }
        });

    /*
    // AO texturing & bump-mapping texturing window
    auto button_browse_AO = static_cast<::CEGUI::PushButton*>(
        root->getChild("MaterialParamsWindow/AOWindow/Button"));
    button_browse_AO->subscribeEvent(::CEGUI::PushButton::EventClicked,
        AOTextureBrowseEvent);
    static_cast<::CEGUI::DefaultWindow*>(
        root->getChild("MaterialParamsWindow/AOWindow/ImageButton"))
        ->subscribeEvent(::CEGUI::Window::EventMouseClick, AOTextureBrowseEvent);
    static_cast<::CEGUI::Editbox*>(root->getChild("MaterialParamsWindow/AOWindow/Editbox"))
        ->subscribeEvent(::CEGUI::Editbox::EventTextAccepted,
            AOTextureBrowseEvent_EditBox);

    auto button_browse_bump_map = static_cast<::CEGUI::PushButton*>(
        root->getChild("MaterialParamsWindow/BumpWindow/Button"));
    button_browse_bump_map->subscribeEvent(::CEGUI::PushButton::EventClicked,
        BumpTextureBrowseEvent);
    static_cast<::CEGUI::DefaultWindow*>(
        root->getChild("MaterialParamsWindow/BumpWindow/ImageButton"))
        ->subscribeEvent(::CEGUI::Window::EventMouseClick, BumpTextureBrowseEvent);
    static_cast<::CEGUI::Editbox*>(
        root->getChild("MaterialParamsWindow/BumpWindow/Editbox"))
        ->subscribeEvent(::CEGUI::Editbox::EventTextAccepted,
            BumpTextureBrowseEvent_EditBox);
    */

    GUI->registerSliderEvent(
        "MaterialParamsWindow/BumpWindow/Spinner", sliderBumpHeightRange, 1.0f,
        [root](const ::CEGUI::EventArgs&) {
            auto roughness = static_cast<::CEGUI::Slider*>(
                root->getChild("MaterialParamsWindow/BumpWindow/Spinner"))
                                 ->getCurrentValue();
            root->getChild("MaterialParamsWindow/BumpWindow/LabelPercent")
                ->setText(ext::cegui::toStringFloat(roughness, 2));
        });
}

void BRDFExplorerApp::renderGUI()
{
    GUI->render();
}

void BRDFExplorerApp::loadTextureSlot(ETEXTURE_SLOT slot,
    const unsigned char* buffer,
    unsigned w,
    unsigned h)
{
    auto tupl = TextureSlotMap[slot];
    auto root = GUI->getRootWindow();
    auto& renderer = GUI->getRenderer();
    ::CEGUI::Texture& texture = !renderer.isTextureDefined(std::get<1>(tupl))
        ? renderer.createTexture(std::get<1>(tupl), ::CEGUI::Sizef(w, h))
        : renderer.getTexture(std::get<1>(tupl));

    texture.loadFromMemory(buffer, ::CEGUI::Sizef(w, h), ::CEGUI::Texture::PF_RGBA);

    ::CEGUI::BasicImage& image = !::CEGUI::ImageManager::getSingleton().isDefined(std::get<0>(tupl))
        ? static_cast<::CEGUI::BasicImage&>(::CEGUI::ImageManager::getSingleton().create(
              "BasicImage", std::get<0>(tupl)))
        : static_cast<::CEGUI::BasicImage&>(
              ::CEGUI::ImageManager::getSingleton().get(std::get<0>(tupl)));
    image.setTexture(&texture);
    image.setArea(::CEGUI::Rectf(0, 0, w, h));
    image.setAutoScaled(::CEGUI::AutoScaledMode::ASM_Both);

    static const std::vector<const char*> property = { "NormalImage", "HoverImage", "PushedImage" };

    for (const auto& v : property) {
        root->getChild(std::get<2>(tupl))->setProperty(v, std::get<0>(tupl));
    }
}

BRDFExplorerApp::~BRDFExplorerApp()
{

}

} // namespace irr
