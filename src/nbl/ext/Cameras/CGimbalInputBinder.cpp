#include "nbl/macros.h"

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include "nbl/ext/Cameras/CGimbalInputBinder.hpp"

namespace nbl::ui
{

uint32_t CGimbalInputBinder::SCollectedVirtualEvents::totalCount() const
{
    return keyboardCount + mouseCount + imguizmoCount;
}

void CGimbalInputBinder::clearActiveBindings()
{
    updateKeyboardMapping([](auto& map) { map.clear(); });
    updateMouseMapping([](auto& map) { map.clear(); });
    updateImguizmoMapping([](auto& map) { map.clear(); });
}

void CGimbalInputBinder::clearBindingLayout()
{
    clearActiveBindings();
}

void CGimbalInputBinder::copyActiveBindingsFromLayout(const IGimbalBindingLayout& layout)
{
    updateKeyboardMapping([&](auto& map) { map = sanitizeMapping(layout.getKeyboardVirtualEventMap()); });
    updateMouseMapping([&](auto& map) { map = sanitizeMapping(layout.getMouseVirtualEventMap()); });
    updateImguizmoMapping([&](auto& map) { map = sanitizeMapping(layout.getImguizmoVirtualEventMap()); });
}

void CGimbalInputBinder::copyBindingLayoutFrom(const IGimbalBindingLayout& layout)
{
    copyActiveBindingsFromLayout(layout);
}

void CGimbalInputBinder::copyActiveBindingsToLayout(IGimbalBindingLayout& layout) const
{
    layout.updateKeyboardMapping([&](auto& map) { map = sanitizeMapping(getKeyboardVirtualEventMap()); });
    layout.updateMouseMapping([&](auto& map) { map = sanitizeMapping(getMouseVirtualEventMap()); });
    layout.updateImguizmoMapping([&](auto& map) { map = sanitizeMapping(getImguizmoVirtualEventMap()); });
}

void CGimbalInputBinder::copyBindingLayoutTo(IGimbalBindingLayout& layout) const
{
    copyActiveBindingsToLayout(layout);
}

CGimbalInputBinder::SCollectedVirtualEvents CGimbalInputBinder::collectVirtualEvents(
    const std::chrono::microseconds nextPresentationTimeStamp,
    const SUpdateParameters parameters)
{
    beginInputProcessing(nextPresentationTimeStamp);

    SCollectedVirtualEvents output;
    uint32_t keyboardPotentialCount = 0u;
    uint32_t mousePotentialCount = 0u;
    uint32_t imguizmoPotentialCount = 0u;

    processKeyboard(nullptr, keyboardPotentialCount, {});
    processMouse(nullptr, mousePotentialCount, {});
    processImguizmo(nullptr, imguizmoPotentialCount, {});

    output.events.resize(keyboardPotentialCount + mousePotentialCount + imguizmoPotentialCount);
    auto* dst = output.events.data();

    if (keyboardPotentialCount)
    {
        output.keyboardCount = keyboardPotentialCount;
        processKeyboard(dst, output.keyboardCount, parameters.keyboardEvents);
        dst += output.keyboardCount;
    }

    if (mousePotentialCount)
    {
        output.mouseCount = mousePotentialCount;
        processMouse(dst, output.mouseCount, parameters.mouseEvents);
        dst += output.mouseCount;
    }

    if (imguizmoPotentialCount)
    {
        output.imguizmoCount = imguizmoPotentialCount;
        processImguizmo(dst, output.imguizmoCount, parameters.imguizmoEvents);
    }

    endInputProcessing();
    output.events.resize(output.totalCount());
    return output;
}

} // namespace nbl::ui
