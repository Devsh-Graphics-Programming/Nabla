// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/CCameraMouseKeyboardPresets.hpp"

namespace nbl::ext::cameras
{

namespace
{

using defaults_t = CCameraMouseKeyboardPresets::SDefaults;

void bindKeys(SMouseKeyboardAxisBinding& slot, const ui::E_KEY_CODE positive, const ui::E_KEY_CODE negative, const hlsl::float64_t rate)
{
    slot.positiveKey = positive;
    slot.negativeKey = negative;
    slot.keyRate = rate;
}

void bindMouseMovementX(SMouseKeyboardAxisBinding& slot, const hlsl::float64_t gain)
{
    slot.mouseMovementGain.x = gain;
}

void bindMouseMovementY(SMouseKeyboardAxisBinding& slot, const hlsl::float64_t gain)
{
    slot.mouseMovementGain.y = gain;
}

// vertical and horizontal scroll alike
void bindScroll(SMouseKeyboardAxisBinding& slot, const hlsl::float64_t gain)
{
    slot.mouseScrollGain = hlsl::float64_t2(gain);
}

// K/I pitch and L/J yaw from the keys, mouse X yaw and mouse Y pitch
void bindLook(SCameraMouseKeyboardBinding& binding)
{
    bindKeys(binding[ECameraControlAxis::RotateX], ui::EKC_K, ui::EKC_I, defaults_t::KeyboardAngleRate);
    bindKeys(binding[ECameraControlAxis::RotateY], ui::EKC_L, ui::EKC_J, defaults_t::KeyboardAngleRate);
    bindMouseMovementX(binding[ECameraControlAxis::RotateY], defaults_t::MouseAngleGain);
    bindMouseMovementY(binding[ECameraControlAxis::RotateX], defaults_t::MouseAngleGain);
}

SCameraMouseKeyboardBinding makeFpsBinding()
{
    SCameraMouseKeyboardBinding binding = {};
    bindKeys(binding[ECameraControlAxis::TranslateZ], ui::EKC_W, ui::EKC_S, defaults_t::KeyboardLengthRate);
    bindKeys(binding[ECameraControlAxis::TranslateX], ui::EKC_D, ui::EKC_A, defaults_t::KeyboardLengthRate);
    bindLook(binding);
    return binding;
}

SCameraMouseKeyboardBinding makeFreeBinding()
{
    auto binding = makeFpsBinding();
    bindKeys(binding[ECameraControlAxis::RotateZ], ui::EKC_E, ui::EKC_Q, defaults_t::KeyboardAngleRate);
    return binding;
}

// W/S yaw and D/A pitch around the target, Q/E and the wheel change the distance, mouse X pitches and mouse Y yaws
SCameraMouseKeyboardBinding makeOrbitAngleBinding()
{
    SCameraMouseKeyboardBinding binding = {};
    bindKeys(binding[ECameraControlAxis::RotateY], ui::EKC_W, ui::EKC_S, defaults_t::KeyboardAngleRate);
    bindKeys(binding[ECameraControlAxis::RotateX], ui::EKC_D, ui::EKC_A, defaults_t::KeyboardAngleRate);
    bindKeys(binding[ECameraControlAxis::Distance], ui::EKC_Q, ui::EKC_E, defaults_t::KeyboardLengthRate);
    bindMouseMovementX(binding[ECameraControlAxis::RotateX], defaults_t::MouseAngleGain);
    bindMouseMovementY(binding[ECameraControlAxis::RotateY], defaults_t::MouseAngleGain);
    bindScroll(binding[ECameraControlAxis::Distance], defaults_t::ScrollLengthGain);
    return binding;
}

// W/S and D/A pan the target, Q/E and the wheel change the distance, the mouse pans
SCameraMouseKeyboardBinding makeOrbitPanBinding()
{
    SCameraMouseKeyboardBinding binding = {};
    bindKeys(binding[ECameraControlAxis::TranslateY], ui::EKC_W, ui::EKC_S, defaults_t::KeyboardLengthRate);
    bindKeys(binding[ECameraControlAxis::TranslateX], ui::EKC_D, ui::EKC_A, defaults_t::KeyboardLengthRate);
    bindKeys(binding[ECameraControlAxis::Distance], ui::EKC_Q, ui::EKC_E, defaults_t::KeyboardLengthRate);
    bindMouseMovementX(binding[ECameraControlAxis::TranslateX], defaults_t::MouseLengthGain);
    bindMouseMovementY(binding[ECameraControlAxis::TranslateY], defaults_t::MouseLengthGain);
    bindScroll(binding[ECameraControlAxis::Distance], defaults_t::ScrollLengthGain);
    return binding;
}

// W/S, D/A and E/Q move the target through the camera frame, the wheel moves it forward, IJKL and the mouse look
SCameraMouseKeyboardBinding makeDollyBinding()
{
    SCameraMouseKeyboardBinding binding = {};
    bindKeys(binding[ECameraControlAxis::TranslateZ], ui::EKC_W, ui::EKC_S, defaults_t::KeyboardLengthRate);
    bindKeys(binding[ECameraControlAxis::TranslateX], ui::EKC_D, ui::EKC_A, defaults_t::KeyboardLengthRate);
    bindKeys(binding[ECameraControlAxis::TranslateY], ui::EKC_E, ui::EKC_Q, defaults_t::KeyboardLengthRate);
    bindLook(binding);
    bindScroll(binding[ECameraControlAxis::TranslateZ], defaults_t::ScrollLengthGain);
    return binding;
}

// W/S and D/A move the target on the ground plane, the wheel moves it forward, E/Q change the distance, IJKL and the mouse look
SCameraMouseKeyboardBinding makeChaseBinding()
{
    SCameraMouseKeyboardBinding binding = {};
    bindKeys(binding[ECameraControlAxis::TranslateZ], ui::EKC_W, ui::EKC_S, defaults_t::KeyboardLengthRate);
    bindKeys(binding[ECameraControlAxis::TranslateX], ui::EKC_D, ui::EKC_A, defaults_t::KeyboardLengthRate);
    bindKeys(binding[ECameraControlAxis::Distance], ui::EKC_E, ui::EKC_Q, defaults_t::KeyboardLengthRate);
    bindLook(binding);
    bindScroll(binding[ECameraControlAxis::TranslateZ], defaults_t::ScrollLengthGain);
    return binding;
}

// W/S and the wheel change the distance, D/A yaw, K/I pitch, the mouse looks.
// TODO: the old preset also had L/J yaw next to D/A; one key pair per axis leaves no slot for it.
SCameraMouseKeyboardBinding makeTurntableBinding()
{
    SCameraMouseKeyboardBinding binding = {};
    bindKeys(binding[ECameraControlAxis::Distance], ui::EKC_W, ui::EKC_S, defaults_t::KeyboardLengthRate);
    bindKeys(binding[ECameraControlAxis::RotateY], ui::EKC_D, ui::EKC_A, defaults_t::KeyboardAngleRate);
    bindKeys(binding[ECameraControlAxis::RotateX], ui::EKC_K, ui::EKC_I, defaults_t::KeyboardAngleRate);
    bindMouseMovementX(binding[ECameraControlAxis::RotateY], defaults_t::MouseAngleGain);
    bindMouseMovementY(binding[ECameraControlAxis::RotateX], defaults_t::MouseAngleGain);
    bindScroll(binding[ECameraControlAxis::Distance], defaults_t::ScrollLengthGain);
    return binding;
}

// W/S and D/A pan the target, Q/E and the wheel change the distance, L/J and mouse X yaw, mouse Y pans
SCameraMouseKeyboardBinding makeTopDownBinding()
{
    SCameraMouseKeyboardBinding binding = {};
    bindKeys(binding[ECameraControlAxis::TranslateY], ui::EKC_W, ui::EKC_S, defaults_t::KeyboardLengthRate);
    bindKeys(binding[ECameraControlAxis::TranslateX], ui::EKC_D, ui::EKC_A, defaults_t::KeyboardLengthRate);
    bindKeys(binding[ECameraControlAxis::Distance], ui::EKC_Q, ui::EKC_E, defaults_t::KeyboardLengthRate);
    bindKeys(binding[ECameraControlAxis::RotateY], ui::EKC_L, ui::EKC_J, defaults_t::KeyboardAngleRate);
    bindMouseMovementX(binding[ECameraControlAxis::RotateY], defaults_t::MouseAngleGain);
    bindMouseMovementY(binding[ECameraControlAxis::TranslateY], defaults_t::MouseLengthGain);
    bindScroll(binding[ECameraControlAxis::Distance], defaults_t::ScrollLengthGain);
    return binding;
}

// W/S and the wheel advance `s`, D/A change `u`, E/Q change `v`, mouse X and Y change `u` and `v`.
// The built-in path model reads `s` and `roll` as angles and `u` and `v` as lengths.
SCameraMouseKeyboardBinding makePathBinding()
{
    SCameraMouseKeyboardBinding binding = {};
    bindKeys(binding[ECameraControlAxis::PathS], ui::EKC_W, ui::EKC_S, defaults_t::KeyboardAngleRate);
    bindKeys(binding[ECameraControlAxis::PathU], ui::EKC_D, ui::EKC_A, defaults_t::KeyboardLengthRate);
    bindKeys(binding[ECameraControlAxis::PathV], ui::EKC_E, ui::EKC_Q, defaults_t::KeyboardLengthRate);
    bindMouseMovementX(binding[ECameraControlAxis::PathU], defaults_t::MouseLengthGain);
    bindMouseMovementY(binding[ECameraControlAxis::PathV], defaults_t::MouseLengthGain);
    bindScroll(binding[ECameraControlAxis::PathS], defaults_t::ScrollAngleGain);
    return binding;
}

} // namespace

SCameraMouseKeyboardBinding CCameraMouseKeyboardPresets::makeDefaultBinding(const ICamera::CameraKind kind)
{
    switch (CCameraKindUtilities::getCameraInteractionFamily(kind))
    {
        case ECameraInteractionFamily::Fps:
            return makeFpsBinding();
        case ECameraInteractionFamily::Free:
            return makeFreeBinding();
        case ECameraInteractionFamily::Orbit:
            // the same keys pan the target on Arcball and Isometric and orbit it on Orbit and DollyZoom
            if (kind == ICamera::CameraKind::Arcball || kind == ICamera::CameraKind::Isometric)
                return makeOrbitPanBinding();
            return makeOrbitAngleBinding();
        case ECameraInteractionFamily::TargetRig:
            if (kind == ICamera::CameraKind::Chase)
                return makeChaseBinding();
            return makeDollyBinding();
        case ECameraInteractionFamily::Turntable:
            return makeTurntableBinding();
        case ECameraInteractionFamily::TopDown:
            return makeTopDownBinding();
        case ECameraInteractionFamily::Path:
            return makePathBinding();
        case ECameraInteractionFamily::None:
        default:
            return {};
    }
}

SCameraMouseKeyboardBinding CCameraMouseKeyboardPresets::makeDefaultBinding(const ICamera& camera)
{
    return makeDefaultBinding(camera.getKind());
}

} // namespace nbl::ext::cameras
