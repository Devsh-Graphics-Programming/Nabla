// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/CCameraInputBindingUtilities.hpp"

#include <tuple>

namespace nbl::ext::cameras
{

namespace
{

using virtual_event_t = CVirtualGimbalEvent::VirtualEventType;
using keyboard_axis_group_t = std::array<virtual_event_t, 4u>;
using mouse_axis_group_t = std::array<virtual_event_t, 4u>;
using scalar_axis_pair_t = std::array<virtual_event_t, 2u>;

struct SKeyboardPresetSpec final
{
    keyboard_axis_group_t wasd = {
        CVirtualGimbalEvent::None,
        CVirtualGimbalEvent::None,
        CVirtualGimbalEvent::None,
        CVirtualGimbalEvent::None
    };
    double wasdScale = IGimbalBindingLayout::CHashInfo::DefaultMagnitudeScale;
    scalar_axis_pair_t qe = {
        CVirtualGimbalEvent::None,
        CVirtualGimbalEvent::None
    };
    double qeScale = IGimbalBindingLayout::CHashInfo::DefaultMagnitudeScale;
    keyboard_axis_group_t ijkl = {
        CVirtualGimbalEvent::None,
        CVirtualGimbalEvent::None,
        CVirtualGimbalEvent::None,
        CVirtualGimbalEvent::None
    };
    double ijklScale = IGimbalBindingLayout::CHashInfo::DefaultMagnitudeScale;
};

struct SMousePresetSpec final
{
    mouse_axis_group_t relative = {
        CVirtualGimbalEvent::None,
        CVirtualGimbalEvent::None,
        CVirtualGimbalEvent::None,
        CVirtualGimbalEvent::None
    };
    double relativeScale = IGimbalBindingLayout::CHashInfo::DefaultMagnitudeScale;
    scalar_axis_pair_t scroll = {
        CVirtualGimbalEvent::None,
        CVirtualGimbalEvent::None
    };
    double scrollScale = IGimbalBindingLayout::CHashInfo::DefaultMagnitudeScale;
};

struct SCameraInputBindingEventGroups final
{
    static inline constexpr std::array FpsMove = {
        CVirtualGimbalEvent::MoveForward,
        CVirtualGimbalEvent::MoveBackward,
        CVirtualGimbalEvent::MoveLeft,
        CVirtualGimbalEvent::MoveRight
    };
    static inline constexpr std::array OrbitTranslate = {
        CVirtualGimbalEvent::MoveUp,
        CVirtualGimbalEvent::MoveDown,
        CVirtualGimbalEvent::MoveLeft,
        CVirtualGimbalEvent::MoveRight
    };
    static inline constexpr std::array OrbitZoom = {
        CVirtualGimbalEvent::MoveForward,
        CVirtualGimbalEvent::MoveBackward
    };
    static inline constexpr std::array VerticalMove = {
        CVirtualGimbalEvent::MoveDown,
        CVirtualGimbalEvent::MoveUp
    };
    static inline constexpr std::array PathRigProgressAndU = {
        CVirtualGimbalEvent::MoveForward,
        CVirtualGimbalEvent::MoveBackward,
        CVirtualGimbalEvent::MoveLeft,
        CVirtualGimbalEvent::MoveRight
    };
    static inline constexpr std::array PathRigV = VerticalMove;
    static inline constexpr std::array TurntableMove = {
        CVirtualGimbalEvent::MoveForward,
        CVirtualGimbalEvent::MoveBackward,
        CVirtualGimbalEvent::PanLeft,
        CVirtualGimbalEvent::PanRight
    };
    static inline constexpr std::array LookYawPitch = {
        CVirtualGimbalEvent::TiltDown,
        CVirtualGimbalEvent::TiltUp,
        CVirtualGimbalEvent::PanLeft,
        CVirtualGimbalEvent::PanRight
    };
    static inline constexpr std::array Roll = {
        CVirtualGimbalEvent::RollLeft,
        CVirtualGimbalEvent::RollRight
    };
    static inline constexpr std::array PanOnly = {
        CVirtualGimbalEvent::None,
        CVirtualGimbalEvent::None,
        CVirtualGimbalEvent::PanLeft,
        CVirtualGimbalEvent::PanRight
    };
    static inline constexpr std::array RelativeLook = {
        CVirtualGimbalEvent::PanRight,
        CVirtualGimbalEvent::PanLeft,
        CVirtualGimbalEvent::TiltUp,
        CVirtualGimbalEvent::TiltDown
    };
    static inline constexpr std::array RelativeOrbitTranslate = {
        CVirtualGimbalEvent::MoveRight,
        CVirtualGimbalEvent::MoveLeft,
        CVirtualGimbalEvent::MoveUp,
        CVirtualGimbalEvent::MoveDown
    };
    static inline constexpr std::array RelativeTopDown = {
        CVirtualGimbalEvent::PanRight,
        CVirtualGimbalEvent::PanLeft,
        CVirtualGimbalEvent::MoveUp,
        CVirtualGimbalEvent::MoveDown
    };
};

struct SCameraInteractionBindingSpec
{
    SKeyboardPresetSpec keyboard = {};
    SMousePresetSpec mouse = {};
};

struct SCameraMappedInteractionBindingSpec
{
    IGimbalBindingLayout::keyboard_to_virtual_events_t keyboard;
    IGimbalBindingLayout::mouse_to_virtual_events_t mouse;
};

template<typename Map, typename Codes>
bool containsBindingForAnyCode(const Map& preset, const Codes& codes)
{
    for (const auto code : codes)
    {
        if (preset.find(code) != preset.end())
            return true;
    }
    return false;
}

template<typename Map, typename... Codes>
bool containsBindingForAnyCodeGroups(const Map& preset, const Codes&... codes)
{
    return (containsBindingForAnyCode(preset, codes) || ...);
}

constexpr size_t interactionFamilyIndex(const ECameraInteractionFamily family)
{
    return static_cast<size_t>(family);
}

template<typename Map, typename Codes, typename Events>
void appendBindingSpec(Map& preset, const Codes& codes, const Events& events, const double magnitudeScale)
{
    for (size_t i = 0u; i < codes.size() && i < events.size(); ++i)
    {
        const auto event = events[i];
        if (event == CVirtualGimbalEvent::None)
            continue;
        preset.emplace(codes[i], IGimbalBindingLayout::CHashInfo(event, magnitudeScale));
    }
}

template<typename Map, typename Codes>
void appendMirroredBindingSpec(Map& preset, const Codes& codes, const virtual_event_t event, const double magnitudeScale)
{
    if (event == CVirtualGimbalEvent::None)
        return;

    std::array<virtual_event_t, std::tuple_size_v<Codes>> duplicatedEvents = {};
    duplicatedEvents.fill(event);
    appendBindingSpec(preset, codes, duplicatedEvents, magnitudeScale);
}

IGimbalBindingLayout::keyboard_to_virtual_events_t buildKeyboardPreset(const SKeyboardPresetSpec& spec)
{
    IGimbalBindingLayout::keyboard_to_virtual_events_t preset;
    appendBindingSpec(preset, SCameraInputBindingPhysicalGroups::KeyboardWasdCodes, spec.wasd, spec.wasdScale);
    appendBindingSpec(preset, SCameraInputBindingPhysicalGroups::KeyboardQeCodes, spec.qe, spec.qeScale);
    appendBindingSpec(preset, SCameraInputBindingPhysicalGroups::KeyboardIjklCodes, spec.ijkl, spec.ijklScale);
    return preset;
}

IGimbalBindingLayout::mouse_to_virtual_events_t buildMousePreset(const SMousePresetSpec& spec)
{
    IGimbalBindingLayout::mouse_to_virtual_events_t preset;
    appendBindingSpec(preset, SCameraInputBindingPhysicalGroups::RelativeMouseCodes, spec.relative, spec.relativeScale);
    appendMirroredBindingSpec(preset, SCameraInputBindingPhysicalGroups::PositiveScrollCodes, spec.scroll[0], spec.scrollScale);
    appendMirroredBindingSpec(preset, SCameraInputBindingPhysicalGroups::NegativeScrollCodes, spec.scroll[1], spec.scrollScale);
    return preset;
}

double getDefaultImguizmoMagnitudeScale(const virtual_event_t event)
{
    if (CVirtualGimbalEvent::isTranslationEvent(event))
        return CCameraInputBindingUtilities::SInputMagnitudeDefaults::ImguizmoTranslationUnitsPerWorldUnit;
    if (CVirtualGimbalEvent::isRotationEvent(event))
        return CCameraInputBindingUtilities::SInputMagnitudeDefaults::ImguizmoRotationUnitsPerRadian;
    if (CVirtualGimbalEvent::isScaleEvent(event))
        return CCameraInputBindingUtilities::SInputMagnitudeDefaults::ImguizmoScaleUnitsPerFactor;
    return IGimbalBindingLayout::CHashInfo::DefaultMagnitudeScale;
}

IGimbalBindingLayout::imguizmo_to_virtual_events_t makeImguizmoPreset(const uint32_t allowedVirtualEvents)
{
    IGimbalBindingLayout::imguizmo_to_virtual_events_t preset;
    for (const auto event : CVirtualGimbalEvent::VirtualEventsTypeTable)
    {
        if (event == CVirtualGimbalEvent::None)
            continue;
        if ((allowedVirtualEvents & event) != event)
            continue;
        preset.emplace(event, IGimbalBindingLayout::CHashInfo(event, getDefaultImguizmoMagnitudeScale(event)));
    }
    return preset;
}

constexpr SCameraInteractionBindingSpec EmptyInteractionBindingSpec = {};

constexpr SKeyboardPresetSpec FpsKeyboardSpec = {
    SCameraInputBindingEventGroups::FpsMove,
    CCameraInputBindingUtilities::SInputMagnitudeDefaults::KeyboardHeldUnitsPerSecond,
    {},
    IGimbalBindingLayout::CHashInfo::DefaultMagnitudeScale,
    SCameraInputBindingEventGroups::LookYawPitch,
    CCameraInputBindingUtilities::SInputMagnitudeDefaults::KeyboardHeldUnitsPerSecond
};

constexpr SKeyboardPresetSpec FreeKeyboardSpec = {
    FpsKeyboardSpec.wasd,
    FpsKeyboardSpec.wasdScale,
    SCameraInputBindingEventGroups::Roll,
    CCameraInputBindingUtilities::SInputMagnitudeDefaults::KeyboardHeldUnitsPerSecond,
    FpsKeyboardSpec.ijkl,
    FpsKeyboardSpec.ijklScale
};

constexpr SKeyboardPresetSpec OrbitKeyboardSpec = {
    SCameraInputBindingEventGroups::OrbitTranslate,
    CCameraInputBindingUtilities::SInputMagnitudeDefaults::KeyboardHeldUnitsPerSecond,
    SCameraInputBindingEventGroups::OrbitZoom,
    CCameraInputBindingUtilities::SInputMagnitudeDefaults::KeyboardHeldUnitsPerSecond,
    {},
    IGimbalBindingLayout::CHashInfo::DefaultMagnitudeScale
};

constexpr SKeyboardPresetSpec TargetRigKeyboardSpec = {
    FpsKeyboardSpec.wasd,
    FpsKeyboardSpec.wasdScale,
    SCameraInputBindingEventGroups::VerticalMove,
    CCameraInputBindingUtilities::SInputMagnitudeDefaults::KeyboardHeldUnitsPerSecond,
    FpsKeyboardSpec.ijkl,
    FpsKeyboardSpec.ijklScale
};

constexpr SKeyboardPresetSpec TurntableKeyboardSpec = {
    SCameraInputBindingEventGroups::TurntableMove,
    CCameraInputBindingUtilities::SInputMagnitudeDefaults::KeyboardHeldUnitsPerSecond,
    {},
    IGimbalBindingLayout::CHashInfo::DefaultMagnitudeScale,
    FpsKeyboardSpec.ijkl,
    FpsKeyboardSpec.ijklScale
};

constexpr SKeyboardPresetSpec TopDownKeyboardSpec = {
    OrbitKeyboardSpec.wasd,
    OrbitKeyboardSpec.wasdScale,
    OrbitKeyboardSpec.qe,
    OrbitKeyboardSpec.qeScale,
    SCameraInputBindingEventGroups::PanOnly,
    CCameraInputBindingUtilities::SInputMagnitudeDefaults::KeyboardHeldUnitsPerSecond
};

constexpr SKeyboardPresetSpec PathKeyboardSpec = {
    SCameraInputBindingEventGroups::PathRigProgressAndU,
    CCameraInputBindingUtilities::SInputMagnitudeDefaults::KeyboardHeldUnitsPerSecond,
    SCameraInputBindingEventGroups::PathRigV,
    CCameraInputBindingUtilities::SInputMagnitudeDefaults::KeyboardHeldUnitsPerSecond,
    {},
    IGimbalBindingLayout::CHashInfo::DefaultMagnitudeScale
};

constexpr SMousePresetSpec FpsMouseSpec = {
    SCameraInputBindingEventGroups::RelativeLook,
    CCameraInputBindingUtilities::SInputMagnitudeDefaults::RelativeMouseUnitsPerStep,
    {},
    IGimbalBindingLayout::CHashInfo::DefaultMagnitudeScale
};

constexpr SMousePresetSpec OrbitMouseSpec = {
    SCameraInputBindingEventGroups::RelativeOrbitTranslate,
    CCameraInputBindingUtilities::SInputMagnitudeDefaults::RelativeMouseUnitsPerStep,
    SCameraInputBindingEventGroups::OrbitZoom,
    CCameraInputBindingUtilities::SInputMagnitudeDefaults::ScrollUnitsPerStep
};

constexpr SMousePresetSpec TargetRigMouseSpec = {
    FpsMouseSpec.relative,
    FpsMouseSpec.relativeScale,
    OrbitMouseSpec.scroll,
    OrbitMouseSpec.scrollScale
};

constexpr SMousePresetSpec TopDownMouseSpec = {
    SCameraInputBindingEventGroups::RelativeTopDown,
    CCameraInputBindingUtilities::SInputMagnitudeDefaults::RelativeMouseUnitsPerStep,
    OrbitMouseSpec.scroll,
    OrbitMouseSpec.scrollScale
};

constexpr SMousePresetSpec PathMouseSpec = {
    SCameraInputBindingEventGroups::RelativeOrbitTranslate,
    CCameraInputBindingUtilities::SInputMagnitudeDefaults::RelativeMouseUnitsPerStep,
    SCameraInputBindingEventGroups::OrbitZoom,
    CCameraInputBindingUtilities::SInputMagnitudeDefaults::ScrollUnitsPerStep
};

constexpr SCameraInteractionBindingSpec FpsInteractionBindingSpec = {
    FpsKeyboardSpec,
    FpsMouseSpec
};

constexpr SCameraInteractionBindingSpec FreeInteractionBindingSpec = {
    FreeKeyboardSpec,
    FpsMouseSpec
};

constexpr SCameraInteractionBindingSpec OrbitInteractionBindingSpec = {
    OrbitKeyboardSpec,
    OrbitMouseSpec
};

constexpr SCameraInteractionBindingSpec TargetRigInteractionBindingSpec = {
    TargetRigKeyboardSpec,
    TargetRigMouseSpec
};

constexpr SCameraInteractionBindingSpec TurntableInteractionBindingSpec = {
    TurntableKeyboardSpec,
    TargetRigMouseSpec
};

constexpr SCameraInteractionBindingSpec TopDownInteractionBindingSpec = {
    TopDownKeyboardSpec,
    TopDownMouseSpec
};

constexpr SCameraInteractionBindingSpec PathInteractionBindingSpec = {
    PathKeyboardSpec,
    PathMouseSpec
};

template<typename Map, typename SpecArray, typename Builder>
auto makePresetCache(const SpecArray& specs, Builder&& builder)
{
    std::array<Map, std::tuple_size_v<SpecArray>> cache = {};
    for (size_t i = 0u; i < specs.size(); ++i)
        cache[i] = builder(specs[i]);
    return cache;
}

SCameraMappedInteractionBindingSpec mapInteractionBindingSpec(const SCameraInteractionBindingSpec& spec)
{
    return {
        .keyboard = buildKeyboardPreset(spec.keyboard),
        .mouse = buildMousePreset(spec.mouse)
    };
}

constexpr std::array<SCameraInteractionBindingSpec, 8u> InteractionFamilyPresetSpecs = {{
    EmptyInteractionBindingSpec,
    FpsInteractionBindingSpec,
    FreeInteractionBindingSpec,
    OrbitInteractionBindingSpec,
    TargetRigInteractionBindingSpec,
    TurntableInteractionBindingSpec,
    TopDownInteractionBindingSpec,
    PathInteractionBindingSpec
}};

const SCameraMappedInteractionBindingSpec& interactionBindingPresetForKind(const ICamera::CameraKind kind)
{
    const auto familyIx = interactionFamilyIndex(CCameraKindUtilities::getCameraInteractionFamily(kind));
    static const auto cache = makePresetCache<SCameraMappedInteractionBindingSpec>(
        InteractionFamilyPresetSpecs,
        [](const SCameraInteractionBindingSpec& spec) { return mapInteractionBindingSpec(spec); });
    return cache[familyIx < cache.size() ? familyIx : 0u];
}

} // namespace

bool CCameraInputBindingUtilities::hasMouseRelativeMovementBinding(const IGimbalBindingLayout::mouse_to_virtual_events_t& mousePreset)
{
    return containsBindingForAnyCodeGroups(mousePreset, SCameraInputBindingPhysicalGroups::RelativeMouseCodes);
}

bool CCameraInputBindingUtilities::hasMouseScrollBinding(const IGimbalBindingLayout::mouse_to_virtual_events_t& mousePreset)
{
    return containsBindingForAnyCodeGroups(
        mousePreset,
        SCameraInputBindingPhysicalGroups::PositiveScrollCodes,
        SCameraInputBindingPhysicalGroups::NegativeScrollCodes);
}

const IGimbalBindingLayout::keyboard_to_virtual_events_t& CCameraInputBindingUtilities::getDefaultCameraKeyboardMappingPreset(const ICamera::CameraKind kind)
{
    return interactionBindingPresetForKind(kind).keyboard;
}

const IGimbalBindingLayout::keyboard_to_virtual_events_t& CCameraInputBindingUtilities::getDefaultCameraKeyboardMappingPreset(const ICamera& camera)
{
    return getDefaultCameraKeyboardMappingPreset(camera.getKind());
}

const IGimbalBindingLayout::mouse_to_virtual_events_t& CCameraInputBindingUtilities::getDefaultCameraMouseMappingPreset(const ICamera::CameraKind kind)
{
    return interactionBindingPresetForKind(kind).mouse;
}

const IGimbalBindingLayout::mouse_to_virtual_events_t& CCameraInputBindingUtilities::getDefaultCameraMouseMappingPreset(const ICamera& camera)
{
    return getDefaultCameraMouseMappingPreset(camera.getKind());
}

IGimbalBindingLayout::imguizmo_to_virtual_events_t CCameraInputBindingUtilities::buildDefaultCameraImguizmoMappingPreset(const uint32_t allowedVirtualEvents)
{
    return makeImguizmoPreset(allowedVirtualEvents);
}

IGimbalBindingLayout::imguizmo_to_virtual_events_t CCameraInputBindingUtilities::buildDefaultCameraImguizmoMappingPreset(const ICamera& camera)
{
    return buildDefaultCameraImguizmoMappingPreset(camera.getAllowedVirtualEvents());
}

SCameraInputBindingPreset CCameraInputBindingUtilities::buildDefaultCameraInputBindingPreset(
    const ICamera::CameraKind kind,
    const uint32_t allowedVirtualEvents)
{
    SCameraInputBindingPreset preset;
    preset.keyboard = getDefaultCameraKeyboardMappingPreset(kind);
    preset.mouse = getDefaultCameraMouseMappingPreset(kind);
    preset.imguizmo = buildDefaultCameraImguizmoMappingPreset(allowedVirtualEvents);
    return preset;
}

SCameraInputBindingPreset CCameraInputBindingUtilities::buildDefaultCameraInputBindingPreset(const ICamera& camera)
{
    return buildDefaultCameraInputBindingPreset(camera.getKind(), camera.getAllowedVirtualEvents());
}

void CCameraInputBindingUtilities::applyDefaultCameraInputBindingPreset(
    IGimbalBindingLayout& layout,
    const ICamera::CameraKind kind,
    const uint32_t allowedVirtualEvents)
{
    const auto preset = buildDefaultCameraInputBindingPreset(kind, allowedVirtualEvents);
    layout.updateKeyboardMapping([&](auto& map) { map = preset.keyboard; });
    layout.updateMouseMapping([&](auto& map) { map = preset.mouse; });
    layout.updateImguizmoMapping([&](auto& map) { map = preset.imguizmo; });
}

void CCameraInputBindingUtilities::applyDefaultCameraInputBindingPreset(IGimbalBindingLayout& layout, const ICamera& camera)
{
    applyDefaultCameraInputBindingPreset(layout, camera.getKind(), camera.getAllowedVirtualEvents());
}

} // namespace nbl::ext::cameras
