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
#include "../../ext/CEGUI/ExtCEGUI.h"

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
        [this](const ::CEGUI::EventArgs& e) {
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

    // AO texturing & bump-mapping texturing window
    auto button_browse_AO = static_cast<::CEGUI::PushButton*>(
        root->getChild("MaterialParamsWindow/AOWindow/Button"));

    button_browse_AO->subscribeEvent(::CEGUI::PushButton::EventClicked,
        ::CEGUI::Event::Subscriber(&BRDFExplorerApp::eventAOTextureBrowse, this));

    static_cast<::CEGUI::DefaultWindow*>(
        root->getChild("MaterialParamsWindow/AOWindow/ImageButton"))
        ->subscribeEvent(::CEGUI::Window::EventMouseClick,
            ::CEGUI::Event::Subscriber(&BRDFExplorerApp::eventAOTextureBrowse, this));

    static_cast<::CEGUI::Editbox*>(root->getChild("MaterialParamsWindow/AOWindow/Editbox"))
        ->subscribeEvent(::CEGUI::Editbox::EventTextAccepted,
            ::CEGUI::Event::Subscriber(&BRDFExplorerApp::eventAOTextureBrowse_EditBox, this));

    auto button_browse_bump_map = static_cast<::CEGUI::PushButton*>(
        root->getChild("MaterialParamsWindow/BumpWindow/Button"));

    button_browse_bump_map->subscribeEvent(::CEGUI::PushButton::EventClicked,
        ::CEGUI::Event::Subscriber(&BRDFExplorerApp::eventBumpTextureBrowse, this));

    static_cast<::CEGUI::DefaultWindow*>(root->getChild("MaterialParamsWindow/BumpWindow/ImageButton"))
        ->subscribeEvent(::CEGUI::Window::EventMouseClick,
            ::CEGUI::Event::Subscriber(&BRDFExplorerApp::eventBumpTextureBrowse, this));

    static_cast<::CEGUI::Editbox*>(root->getChild("MaterialParamsWindow/BumpWindow/Editbox"))
        ->subscribeEvent(::CEGUI::Editbox::EventTextAccepted,
            ::CEGUI::Event::Subscriber(&BRDFExplorerApp::eventBumpTextureBrowse_EditBox, this));

    GUI->registerSliderEvent(
        "MaterialParamsWindow/BumpWindow/Spinner", sliderBumpHeightRange, 1.0f,
        [root](const ::CEGUI::EventArgs&) {
            auto roughness = static_cast<::CEGUI::Slider*>(
                root->getChild("MaterialParamsWindow/BumpWindow/Spinner"))
                                 ->getCurrentValue();
            root->getChild("MaterialParamsWindow/BumpWindow/LabelPercent")
                ->setText(ext::cegui::toStringFloat(roughness, 2));
        });
    initDropdown();
    initTooltip();

    // Setting up the texture preview window
    std::array<::CEGUI::PushButton*, 4> texturePreviewIcon = {
        static_cast<::CEGUI::PushButton*>(
            root->getChild("TextureViewWindow/Texture0Window/Texture")),
        static_cast<::CEGUI::PushButton*>(
            root->getChild("TextureViewWindow/Texture1Window/Texture")),
        static_cast<::CEGUI::PushButton*>(
            root->getChild("TextureViewWindow/Texture2Window/Texture")),
        static_cast<::CEGUI::PushButton*>(
            root->getChild("TextureViewWindow/Texture3Window/Texture"))
    };

    for (const auto& v : texturePreviewIcon)
    {
        v->subscribeEvent(::CEGUI::PushButton::EventClicked,
            ::CEGUI::Event::Subscriber(&BRDFExplorerApp::eventTextureBrowse, this));
    }

    // Setting up the master windows & their default opacity
    auto* window_material = static_cast<::CEGUI::FrameWindow*>(root->getChild("MaterialParamsWindow"));
    window_material->subscribeEvent(
        ::CEGUI::FrameWindow::EventCloseClicked, [root](const ::CEGUI::EventArgs&) {
            static_cast<::CEGUI::FrameWindow*>(
                root->getChild("MaterialParamsWindow"))
                ->setVisible(false);
        });
    GUI->setOpacity("MaterialParamsWindow", defaultOpacity);

    auto* window_light = static_cast<::CEGUI::FrameWindow*>(root->getChild("LightParamsWindow"));
    window_light->subscribeEvent(::CEGUI::FrameWindow::EventCloseClicked,
        [root](const CEGUI::EventArgs&) {
            static_cast<CEGUI::FrameWindow*>(
                root->getChild("LightParamsWindow"))
                ->setVisible(false);
        });
    GUI->setOpacity("LightParamsWindow", defaultOpacity);

    auto* window_texture = static_cast<::CEGUI::FrameWindow*>(root->getChild("TextureViewWindow"));
    window_texture->subscribeEvent(
        ::CEGUI::FrameWindow::EventCloseClicked, [root](const ::CEGUI::EventArgs&) {
            static_cast<::CEGUI::FrameWindow*>(
                root->getChild("TextureViewWindow"))
                ->setVisible(false);
        });
    GUI->setOpacity("TextureViewWindow", defaultOpacity);
}

void BRDFExplorerApp::initDropdown()
{
    static const std::vector<const char*> drop_ID = {
        "Constant", "Texture 0", "Texture 1", "Texture 2", "Texture 3"
    };
    const auto default_halignment = ::CEGUI::HA_RIGHT;
    const auto default_width = ::CEGUI::UDim(0.5f, 0.0f);
    const auto default_position = ::CEGUI::UVector2(::CEGUI::UDim(0.0f, 0.0f), ::CEGUI::UDim(0.125f, 0.0f));

    auto root = GUI->getRootWindow();

    auto* albedo_drop = GUI->createDropDownList(
        "MaterialParamsWindow/AlbedoDropDownList", "DropDown_Albedo", drop_ID,
        [this](const ::CEGUI::EventArgs&) {
            auto root = GUI->getRootWindow();
            auto* list = static_cast<::CEGUI::Combobox*>(root->getChild(
                "MaterialParamsWindow/AlbedoDropDownList/DropDown_Albedo"));
            list->setProperty("NormalEditTextColour", GUI->WhiteProperty);

            root->getChild("MaterialParamsWindow/AlbedoWindow")
                ->setDisabled(list->getSelectedItem()->getText() != "Constant");
        });

    albedo_drop->setHorizontalAlignment(default_halignment);
    albedo_drop->setWidth(default_width);
    albedo_drop->setPosition(default_position);

    auto* roughness_drop = GUI->createDropDownList(
        "MaterialParamsWindow/RoughnessDropDownList", "DropDown_Roughness",
        drop_ID, [this](const ::CEGUI::EventArgs&) {
            auto root = GUI->getRootWindow();
            auto* list = static_cast<CEGUI::Combobox*>(root->getChild(
                "MaterialParamsWindow/RoughnessDropDownList/DropDown_Roughness"));
            list->setProperty("NormalEditTextColour", GUI->WhiteProperty);

            root->getChild("MaterialParamsWindow/RoughnessWindow")
                ->setDisabled(list->getSelectedItem()->getText() != "Constant");
        });

    roughness_drop->setHorizontalAlignment(default_halignment);
    roughness_drop->setWidth(default_width);
    roughness_drop->setPosition(default_position);

    auto* ri_drop = GUI->createDropDownList(
        "MaterialParamsWindow/RIDropDownList", "DropDown_RI", drop_ID,
        [this](const ::CEGUI::EventArgs&) {
            auto root = GUI->getRootWindow();

            auto* list = static_cast<::CEGUI::Combobox*>(
                root->getChild("MaterialParamsWindow/RIDropDownList/DropDown_RI"));
            list->setProperty("NormalEditTextColour", GUI->WhiteProperty);

            root->getChild("MaterialParamsWindow/RefractionIndexWindow")
                ->setDisabled(list->getSelectedItem()->getText() != "Constant");
        });

    ri_drop->setHorizontalAlignment(default_halignment);
    ri_drop->setWidth(default_width);
    ri_drop->setPosition(default_position);

    auto* metallic_drop = GUI->createDropDownList(
        "MaterialParamsWindow/MetallicDropDownList", "DropDown_Metallic", drop_ID,
        [this](const ::CEGUI::EventArgs&) {
            auto root = GUI->getRootWindow();

            auto* list = static_cast<::CEGUI::Combobox*>(root->getChild(
                "MaterialParamsWindow/MetallicDropDownList/DropDown_Metallic"));
            list->setProperty("NormalEditTextColour", GUI->WhiteProperty);

            root->getChild("MaterialParamsWindow/MetallicWindow")
                ->setDisabled(list->getSelectedItem()->getText() != "Constant");
        });

    metallic_drop->setHorizontalAlignment(default_halignment);
    metallic_drop->setWidth(default_width);
    metallic_drop->setPosition(default_position);
}

void BRDFExplorerApp::initTooltip()
{
    auto root = GUI->getRootWindow();

    static_cast<CEGUI::DefaultWindow*>(
        root->getChild("MaterialParamsWindow/BumpWindow/ImageButton"))
        ->setTooltipText("Left-click to select a bump-mapping texture.");
    static_cast<CEGUI::DefaultWindow*>(
        root->getChild("MaterialParamsWindow/AOWindow/ImageButton"))
        ->setTooltipText("Left-click to select an AO texture.");
    static_cast<CEGUI::DefaultWindow*>(
        root->getChild("TextureViewWindow/Texture0Window/Texture"))
        ->setTooltipText("Left-click to select a new texture.");
    static_cast<CEGUI::DefaultWindow*>(
        root->getChild("TextureViewWindow/Texture1Window/Texture"))
        ->setTooltipText("Left-click to select a new texture.");
    static_cast<CEGUI::DefaultWindow*>(
        root->getChild("TextureViewWindow/Texture2Window/Texture"))
        ->setTooltipText("Left-click to select a new texture.");
    static_cast<CEGUI::DefaultWindow*>(
        root->getChild("TextureViewWindow/Texture3Window/Texture"))
        ->setTooltipText("Left-click to select a new texture.");
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

void BRDFExplorerApp::updateTooltip(const char* name, const char* text)
{
    std::string s(text);
    ext::cegui::Replace(s, "\\", "\\\\");

    static_cast<CEGUI::DefaultWindow*>(GUI->getRootWindow()->getChild(name))
        ->setTooltipText(s.c_str());
}

void BRDFExplorerApp::showErrorMessage(const char* title, const char* message)
{
    auto root = GUI->getRootWindow();
    if (!root->isChild("MessageBoxRoot")) {
        CEGUI::Window* layout = CEGUI::WindowManager::getSingleton().loadLayoutFromFile(
            "MessageBox.layout");
        layout->setVisible(false);
        layout->setAlwaysOnTop(true);
        layout->setSize(
            CEGUI::USize(CEGUI::UDim(0.5, 0.0f), CEGUI::UDim(0.2f, 0.0f)));
        layout->setHorizontalAlignment(CEGUI::HA_CENTRE);
        layout->setVerticalAlignment(CEGUI::VA_CENTRE);

        static_cast<CEGUI::PushButton*>(
            layout->getChild("FrameWindow/ButtonWindow/Button"))
            ->subscribeEvent(
                CEGUI::PushButton::EventClicked,
                [root](const CEGUI::EventArgs&) {
                    root->getChild("MessageBoxRoot")->setVisible(false);
                });

        root->addChild(layout);
    }

    auto header = static_cast<CEGUI::DefaultWindow*>(root->getChild("MessageBoxRoot"));
    header->setVisible(true);
    header->activate();

    auto frame = static_cast<CEGUI::FrameWindow*>(header->getChild("FrameWindow"));
    frame->setText(title);
    static_cast<CEGUI::DefaultWindow*>(frame->getChild("Label"))
        ->setText(message);
}

void BRDFExplorerApp::eventAOTextureBrowse(const ::CEGUI::EventArgs&)
{
    const auto p = GUI->openFileDialog(FileDialogTitle, FileDialogFilters);

    if (p.first) {
        auto box = static_cast<CEGUI::Editbox*>(
            GUI->getRootWindow()->getChild("MaterialParamsWindow/AOWindow/Editbox"));

        const auto image = ext::cegui::loadImage(p.second.c_str());
        loadTextureSlot(ETEXTURE_SLOT::TEXTURE_AO, image.buffer, image.w, image.h);

        box->setText(p.second);
        updateTooltip(
            "MaterialParamsWindow/AOWindow/ImageButton",
            ext::cegui::ssprintf("%s (%ix%i)\nLeft-click to select a new texture.",
                p.second.c_str(), image.w, image.h)
                .c_str());
    }
}

void BRDFExplorerApp::eventAOTextureBrowse_EditBox(const ::CEGUI::EventArgs&)
{
    auto box = static_cast<CEGUI::Editbox*>(
        GUI->getRootWindow()->getChild("MaterialParamsWindow/AOWindow/Editbox"));

    if (ext::cegui::Exists(box->getText().c_str())) {
        const auto image = ext::cegui::loadImage(box->getText().c_str());
        loadTextureSlot(ETEXTURE_SLOT::TEXTURE_AO, image.buffer, image.w, image.h);

        updateTooltip(
            "MaterialParamsWindow/AOWindow/ImageButton",
            irr::ext::cegui::ssprintf("%s (%ix%i)\nLeft-click to select a new texture.",
                box->getText().c_str(), image.w, image.h)
                .c_str());
    } else {
        std::string s;
        s += std::string(box->getText().c_str()) + ": The file couldn't be opened.";
        ext::cegui::Replace(s, "\\", "\\\\");
        showErrorMessage("Error", s.c_str());
    }
}

void BRDFExplorerApp::eventBumpTextureBrowse(const ::CEGUI::EventArgs&)
{
    const auto p = GUI->openFileDialog(FileDialogTitle, FileDialogFilters);

    if (p.first) {
        auto box = static_cast<CEGUI::Editbox*>(
            GUI->getRootWindow()->getChild("MaterialParamsWindow/BumpWindow/Editbox"));
        const auto image = irr::ext::cegui::loadImage(p.second.c_str());
        loadTextureSlot(ETEXTURE_SLOT::TEXTURE_BUMP, image.buffer, image.w, image.h);

        box->setText(p.second);
        updateTooltip(
            "MaterialParamsWindow/BumpWindow/ImageButton",
            ext::cegui::ssprintf("%s (%ix%i)\nLeft-click to select a new texture.",
                p.second.c_str(), image.w, image.h)
                .c_str());
    }
}

void BRDFExplorerApp::eventBumpTextureBrowse_EditBox(const ::CEGUI::EventArgs&)
{
    auto box = static_cast<CEGUI::Editbox*>(
        GUI->getRootWindow()->getChild("MaterialParamsWindow/BumpWindow/Editbox"));

    if (ext::cegui::Exists(box->getText().c_str())) {
        const auto image = ext::cegui::loadImage(box->getText().c_str());
        loadTextureSlot(ETEXTURE_SLOT::TEXTURE_BUMP, image.buffer, image.w, image.h);

        updateTooltip(
            "MaterialParamsWindow/BumpWindow/ImageButton",
            ext::cegui::ssprintf("%s (%ix%i)\nLeft-click to select a new texture.",
                box->getText().c_str(), image.w, image.h)
                .c_str());
    } else {
        std::string s;
        s += std::string(box->getText().c_str()) + ": The file couldn't be opened.";
        ext::cegui::Replace(s, "\\", "\\\\");
        showErrorMessage("Error", s.c_str());
    }
}

void BRDFExplorerApp::eventTextureBrowse(const CEGUI::EventArgs& e)
{
    const CEGUI::WindowEventArgs& we = static_cast<const CEGUI::WindowEventArgs&>(e);
    const auto parent = static_cast<CEGUI::PushButton*>(we.window)->getParent()->getName();
    const auto p = GUI->openFileDialog(FileDialogTitle, FileDialogFilters);


    const auto path_label = ext::cegui::ssprintf("TextureViewWindow/%s/LabelWindow/Label", parent.c_str());
    const auto path_texture = ext::cegui::ssprintf("TextureViewWindow/%s/Texture", parent.c_str());

    if (p.first) {
        auto box = static_cast<CEGUI::Editbox*>(GUI->getRootWindow()->getChild(path_label));
        const auto v = ext::cegui::Split(p.second, '\\');

        ETEXTURE_SLOT type;
        if (parent == "Texture0Window")
            type = ETEXTURE_SLOT::TEXTURE_SLOT_1;
        else if (parent == "Texture1Window")
            type = ETEXTURE_SLOT::TEXTURE_SLOT_2;
        else if (parent == "Texture2Window")
            type = ETEXTURE_SLOT::TEXTURE_SLOT_3;
        else if (parent == "Texture3Window")
            type = ETEXTURE_SLOT::TEXTURE_SLOT_4;

        const auto image = ext::cegui::loadImage(p.second.c_str());
        loadTextureSlot(type, image.buffer, image.w, image.h);

        box->setText(v[v.size() - 1]);
        updateTooltip(
            path_texture.c_str(),
            ext::cegui::ssprintf("%s (%ix%i)\nLeft-click to select a new texture.",
                p.second.c_str(), image.w, image.h)
                .c_str());
    }
}


BRDFExplorerApp::~BRDFExplorerApp()
{

}

} // namespace irr
