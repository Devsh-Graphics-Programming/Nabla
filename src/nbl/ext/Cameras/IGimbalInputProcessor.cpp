#include "nbl/macros.h"

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include "nbl/ext/Cameras/IGimbalInputProcessor.hpp"

#include <algorithm>

#include "nbl/ext/Cameras/CCameraMathUtilities.hpp"

namespace nbl::ui
{

void IGimbalInputProcessor::beginInputProcessing(const std::chrono::microseconds nextPresentationTimeStamp)
{
    m_nextPresentationTimeStamp = nextPresentationTimeStamp;
    m_frameDeltaSeconds = clampFrameDeltaTimeSeconds(m_nextPresentationTimeStamp, m_lastVirtualUpTimeStamp);
}

void IGimbalInputProcessor::endInputProcessing()
{
    m_lastVirtualUpTimeStamp = m_nextPresentationTimeStamp;
}

void IGimbalInputProcessor::process(gimbal_event_t* output, uint32_t& count, const SUpdateParameters parameters)
{
    count = 0u;
    uint32_t vKeyboardEventsCount = {}, vMouseEventsCount = {}, vImguizmoEventsCount = {};

    if (output)
    {
        processKeyboard(output, vKeyboardEventsCount, parameters.keyboardEvents);
        output += vKeyboardEventsCount;
        processMouse(output, vMouseEventsCount, parameters.mouseEvents);
        output += vMouseEventsCount;
        processImguizmo(output, vImguizmoEventsCount, parameters.imguizmoEvents);
    }
    else
    {
        processKeyboard(nullptr, vKeyboardEventsCount, {});
        processMouse(nullptr, vMouseEventsCount, {});
        processImguizmo(nullptr, vImguizmoEventsCount, {});
    }

    count = vKeyboardEventsCount + vMouseEventsCount + vImguizmoEventsCount;
}

void IGimbalInputProcessor::processKeyboard(gimbal_event_t* output, uint32_t& count, std::span<const input_keyboard_event_t> events)
{
    processBindingMap(
        m_keyboardVirtualEventMap,
        output,
        count,
        [&](auto& map)
        {
            for (const auto& keyboardEvent : events)
            {
                if (keyboardEvent.action == input_keyboard_event_t::ECA_PRESSED)
                    setBindingActiveState(map, keyboardEvent.keyCode, true);
                else if (keyboardEvent.action == input_keyboard_event_t::ECA_RELEASED)
                    setBindingActiveState(map, keyboardEvent.keyCode, false);
            }
        });
}

void IGimbalInputProcessor::processMouse(gimbal_event_t* output, uint32_t& count, std::span<const input_mouse_event_t> events)
{
    processBindingMap(
        m_mouseVirtualEventMap,
        output,
        count,
        [&](auto& map)
        {
            for (const auto& mouseEvent : events)
            {
                switch (mouseEvent.type)
                {
                    case input_mouse_event_t::EET_CLICK:
                        updateMouseButtonState(map, mouseEvent.clickEvent);
                        break;

                    case input_mouse_event_t::EET_SCROLL:
                        requestMagnitudeUpdateWithSignedComponents(
                            ZeroPivot,
                            hlsl::float32_t2(
                                static_cast<float>(mouseEvent.scrollEvent.verticalScroll),
                                mouseEvent.scrollEvent.horizontalScroll),
                            SInputProcessorBindingGroups::MouseScroll,
                            map);
                        break;

                    case input_mouse_event_t::EET_MOVEMENT:
                        requestMagnitudeUpdateWithSignedComponents(
                            ZeroPivot,
                            hlsl::float32_t2(
                                mouseEvent.movementEvent.relativeMovementX,
                                mouseEvent.movementEvent.relativeMovementY),
                            SInputProcessorBindingGroups::MouseRelativeMovement,
                            map);
                        break;

                    default:
                        break;
                }
            }
        });
}

void IGimbalInputProcessor::processImguizmo(gimbal_event_t* output, uint32_t& count, std::span<const input_imguizmo_event_t> events)
{
    processBindingMap(
        m_imguizmoVirtualEventMap,
        output,
        count,
        [&](auto& map)
        {
            for (const auto& ev : events)
            {
                const auto& deltaWorldTRS = ev;

                hlsl::SRigidTransformComponents<hlsl::float32_t> world = {};
                if (!hlsl::CCameraMathUtilities::tryExtractRigidTransformComponents(deltaWorldTRS, world))
                    continue;

                requestMagnitudeUpdateWithSignedComponents(
                    ZeroPivot,
                    world.translation,
                    SInputProcessorBindingGroups::ImguizmoTranslation,
                    map);

                const auto dRotationRad = hlsl::CCameraMathUtilities::getCameraOrientationEulerRadians(world.orientation);
                requestMagnitudeUpdateWithSignedComponents(
                    ZeroPivot,
                    dRotationRad,
                    SInputProcessorBindingGroups::ImguizmoRotation,
                    map);

                requestMagnitudeUpdateWithSignedComponents(
                    UnitPivot,
                    world.scale,
                    SInputProcessorBindingGroups::ImguizmoScale,
                    map);
            }
        });
}

double IGimbalInputProcessor::clampFrameDeltaTimeSeconds(
    const std::chrono::microseconds nextPresentationTimeStamp,
    const std::chrono::microseconds lastVirtualUpTimeStamp)
{
    const auto deltaSeconds = std::chrono::duration<double>(
        nextPresentationTimeStamp - lastVirtualUpTimeStamp).count();
    if (deltaSeconds < 0.0)
        return 0.0;
    return std::min(deltaSeconds, MaxFrameDeltaSeconds);
}

bool IGimbalInputProcessor::tryGetMouseButtonCode(
    const ui::E_MOUSE_BUTTON button,
    ui::E_MOUSE_CODE& outCode)
{
    switch (button)
    {
        case ui::EMB_LEFT_BUTTON:    outCode = ui::EMC_LEFT_BUTTON; return true;
        case ui::EMB_RIGHT_BUTTON:   outCode = ui::EMC_RIGHT_BUTTON; return true;
        case ui::EMB_MIDDLE_BUTTON:  outCode = ui::EMC_MIDDLE_BUTTON; return true;
        case ui::EMB_BUTTON_4:       outCode = ui::EMC_BUTTON_4; return true;
        case ui::EMB_BUTTON_5:       outCode = ui::EMC_BUTTON_5; return true;
        default:
            return false;
    }
}

} // namespace nbl::ui
