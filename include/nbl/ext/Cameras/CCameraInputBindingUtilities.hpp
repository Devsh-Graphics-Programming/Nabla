#ifndef _NBL_C_CAMERA_INPUT_BINDING_UTILITIES_HPP_
#define _NBL_C_CAMERA_INPUT_BINDING_UTILITIES_HPP_

#include <array>
#include <tuple>

#include "CCameraKindUtilities.hpp"
#include "ICamera.hpp"
#include "IGimbalBindingLayout.hpp"

namespace nbl::ui
{

/// @brief Reusable keyboard, mouse, and ImGuizmo binding preset grouped for one camera kind.
struct SCameraInputBindingPreset
{
    IGimbalBindingLayout::keyboard_to_virtual_events_t keyboard;
    IGimbalBindingLayout::mouse_to_virtual_events_t mouse;
    IGimbalBindingLayout::imguizmo_to_virtual_events_t imguizmo;
};

/// @brief Shared physical input bundles reused by default presets and smoke probing.
struct SCameraInputBindingPhysicalGroups final
{
    static inline constexpr std::array KeyboardWasdCodes = {
        ui::E_KEY_CODE::EKC_W,
        ui::E_KEY_CODE::EKC_S,
        ui::E_KEY_CODE::EKC_A,
        ui::E_KEY_CODE::EKC_D
    };
    static inline constexpr std::array KeyboardQeCodes = {
        ui::E_KEY_CODE::EKC_Q,
        ui::E_KEY_CODE::EKC_E
    };
    static inline constexpr std::array KeyboardIjklCodes = {
        ui::E_KEY_CODE::EKC_I,
        ui::E_KEY_CODE::EKC_K,
        ui::E_KEY_CODE::EKC_J,
        ui::E_KEY_CODE::EKC_L
    };
    static inline constexpr auto KeyboardProbeMoveCodes = KeyboardWasdCodes;
    static inline constexpr auto KeyboardProbeLookCodes = KeyboardIjklCodes;
    static inline constexpr std::array KeyboardProbeExtraCodes = {
        ui::E_KEY_CODE::EKC_U,
        ui::E_KEY_CODE::EKC_O
    };
    static inline constexpr std::array KeyboardProbeCodes = {
        ui::E_KEY_CODE::EKC_W,
        ui::E_KEY_CODE::EKC_S,
        ui::E_KEY_CODE::EKC_A,
        ui::E_KEY_CODE::EKC_D,
        ui::E_KEY_CODE::EKC_Q,
        ui::E_KEY_CODE::EKC_E,
        ui::E_KEY_CODE::EKC_I,
        ui::E_KEY_CODE::EKC_K,
        ui::E_KEY_CODE::EKC_J,
        ui::E_KEY_CODE::EKC_L,
        ui::E_KEY_CODE::EKC_U,
        ui::E_KEY_CODE::EKC_O
    };
    static inline constexpr std::array RelativeMouseCodes = {
        ui::E_MOUSE_CODE::EMC_RELATIVE_POSITIVE_MOVEMENT_X,
        ui::E_MOUSE_CODE::EMC_RELATIVE_NEGATIVE_MOVEMENT_X,
        ui::E_MOUSE_CODE::EMC_RELATIVE_POSITIVE_MOVEMENT_Y,
        ui::E_MOUSE_CODE::EMC_RELATIVE_NEGATIVE_MOVEMENT_Y
    };
    static inline constexpr std::array PositiveScrollCodes = {
        ui::E_MOUSE_CODE::EMC_VERTICAL_POSITIVE_SCROLL,
        ui::E_MOUSE_CODE::EMC_HORIZONTAL_POSITIVE_SCROLL
    };
    static inline constexpr std::array NegativeScrollCodes = {
        ui::E_MOUSE_CODE::EMC_VERTICAL_NEGATIVE_SCROLL,
        ui::E_MOUSE_CODE::EMC_HORIZONTAL_NEGATIVE_SCROLL
    };
};

struct CCameraInputBindingUtilities final
{
public:
    /// @brief Default gains used by the shared binding presets.
    ///
    /// These gains convert producer-local units into virtual-event magnitudes:
    /// held keys into units per second, mouse deltas into units per mouse
    /// step, scroll into units per scroll step, and ImGuizmo deltas into
    /// virtual translation, rotation, or scale magnitudes.
    struct SInputMagnitudeDefaults final
    {
        static inline constexpr double KeyboardHeldUnitsPerSecond = 1000.0;
        static inline constexpr double RelativeMouseUnitsPerStep = 1.0;
        static inline constexpr double ScrollUnitsPerStep = 1.0;
        static inline constexpr double ImguizmoTranslationUnitsPerWorldUnit = 1.0;
        static inline constexpr double ImguizmoRotationUnitsPerRadian = 1.0;
        static inline constexpr double ImguizmoScaleUnitsPerFactor = 1.0;
    };

    static bool hasMouseRelativeMovementBinding(const IGimbalBindingLayout::mouse_to_virtual_events_t& mousePreset);

    static bool hasMouseScrollBinding(const IGimbalBindingLayout::mouse_to_virtual_events_t& mousePreset);

    static const IGimbalBindingLayout::keyboard_to_virtual_events_t& getDefaultCameraKeyboardMappingPreset(core::ICamera::CameraKind kind);

    static const IGimbalBindingLayout::keyboard_to_virtual_events_t& getDefaultCameraKeyboardMappingPreset(const core::ICamera& camera);

    static const IGimbalBindingLayout::mouse_to_virtual_events_t& getDefaultCameraMouseMappingPreset(core::ICamera::CameraKind kind);

    static const IGimbalBindingLayout::mouse_to_virtual_events_t& getDefaultCameraMouseMappingPreset(const core::ICamera& camera);

    static IGimbalBindingLayout::imguizmo_to_virtual_events_t buildDefaultCameraImguizmoMappingPreset(uint32_t allowedVirtualEvents);

    static IGimbalBindingLayout::imguizmo_to_virtual_events_t buildDefaultCameraImguizmoMappingPreset(const core::ICamera& camera);

    static SCameraInputBindingPreset buildDefaultCameraInputBindingPreset(core::ICamera::CameraKind kind, uint32_t allowedVirtualEvents);

    static SCameraInputBindingPreset buildDefaultCameraInputBindingPreset(const core::ICamera& camera);

    static void applyDefaultCameraInputBindingPreset(
        IGimbalBindingLayout& layout,
        core::ICamera::CameraKind kind,
        uint32_t allowedVirtualEvents);

    static void applyDefaultCameraInputBindingPreset(IGimbalBindingLayout& layout, const core::ICamera& camera);
};

} // namespace nbl::ui

#endif // _NBL_C_CAMERA_INPUT_BINDING_UTILITIES_HPP_
