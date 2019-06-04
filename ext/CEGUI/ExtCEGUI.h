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

#ifndef _IRR_EXT_CEGUI_INCLUDED_
#define _IRR_EXT_CEGUI_INCLUDED_

#include "irrlicht.h"
#include "Helpers.h"
#include <CEGUI/widgets/Slider.h>
#include <CEGUI/RendererModules/OpenGL/GL3Renderer.h>
#include <CEGUI/CommonDialogs/ColourPicker/ColourPicker.h>
#include <map>
#include <functional>

#include "CEGUIOpenGLState.h"

namespace irr
{

class IrrlichtDevice;

namespace ext
{
namespace cegui
{


class GUIManager;
GUIManager* createGUIManager(IrrlichtDevice* device);

using TEventHandler = std::function<void(const ::CEGUI::EventArgs&)>;

class GUIManager: public core::IReferenceCounted, public IEventReceiver
{
    public:
        // White (1.0f, 1.0f, 1.0f) color, but it's a CEGUI::String property
        // (which is the editable property seen in CEED). Needed for setProperty().
        const ::CEGUI::String WhiteProperty = ::CEGUI::PropertyHelper<::CEGUI::ColourRect>::toString(
            ::CEGUI::ColourRect(::CEGUI::Colour(1.0f, 1.0f, 1.0f, 1.0f)));

    public:
        GUIManager(IrrlichtDevice* device);
        ~GUIManager();

        void init();
		void destroy();
        void render();
        bool OnEvent(const SEvent& event) override;

        void createRootWindowFromLayout(const std::string& layout);
        auto getRootWindow() const { return RootWindow; }
        auto& getRenderer() const { return Renderer; }

        std::pair<bool, std::string>  openFileDialog(
            const char* title,
            const std::vector<std::string>& filters
        );

        ::CEGUI::ColourPicker* createColourPicker(
            bool alternativeLayout,
            const char* parent,
            const char* title,
            const char* name
        );

        ::CEGUI::Window* createDropDownList(
            const char* name,
            const char* title,
            const std::vector<const char*>& list,
            const TEventHandler& f = [](const ::CEGUI::EventArgs&) {}
        );

        void registerSliderEvent(
            const char* name,
            float max,
            float step,
            const TEventHandler& func
        );

        void setOpacity(const char* name, float opacity);

    private:
        IrrlichtDevice* Device = nullptr;
        video::IVideoDriver* Driver = nullptr;
		CEGUIOpenGLState GLStateManager;
        ::CEGUI::OpenGL3Renderer& Renderer;
        ::CEGUI::Window* RootWindow;
        std::map<const char*, ::CEGUI::ColourPicker*> ColourPickers;
};

} // namespace cegui
} // namespace ext
} // namespace irr

#endif // _IRR_EXT_CEGUI_INCLUDED_
