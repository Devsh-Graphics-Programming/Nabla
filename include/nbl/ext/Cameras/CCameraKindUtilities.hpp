#ifndef _C_CAMERA_KIND_UTILITIES_HPP_
#define _C_CAMERA_KIND_UTILITIES_HPP_

#include <array>
#include <string_view>

#include "CCameraPathMetadata.hpp"
#include "ICamera.hpp"

namespace nbl::core
{

/// @brief Interaction family used to group camera kinds with matching control semantics.
enum class ECameraInteractionFamily : uint8_t
{
    None,
    Fps,
    Free,
    Orbit,
    TargetRig,
    Turntable,
    TopDown,
    Path
};

/// @brief Shared metadata for one concrete `CameraKind`.
struct SCameraKindTraits final
{
    ICamera::CameraKind kind = ICamera::CameraKind::Unknown;
    std::string_view label = "Unknown";
    std::string_view description = "Unspecified camera behavior";
    ECameraInteractionFamily interactionFamily = ECameraInteractionFamily::None;
};

struct CCameraKindUtilities final
{
public:
    static inline constexpr const SCameraKindTraits& getCameraKindTraits(const ICamera::CameraKind kind)
    {
        const auto ix = static_cast<size_t>(kind);
        if (ix >= CameraKindTraitsTable.size())
            return CameraKindTraitsTable[0u];
        return CameraKindTraitsTable[ix];
    }

    static inline constexpr std::string_view getCameraKindLabel(const ICamera::CameraKind kind)
    {
        return getCameraKindTraits(kind).label;
    }

    static inline constexpr std::string_view getCameraKindDescription(const ICamera::CameraKind kind)
    {
        return getCameraKindTraits(kind).description;
    }

    static inline constexpr ECameraInteractionFamily getCameraInteractionFamily(const ICamera::CameraKind kind)
    {
        return getCameraKindTraits(kind).interactionFamily;
    }

private:
    static inline constexpr std::array<SCameraKindTraits, static_cast<size_t>(ICamera::CameraKind::Path) + 1u> CameraKindTraitsTable = {{
        {
            .kind = ICamera::CameraKind::Unknown,
            .label = "Unknown",
            .description = "Unspecified camera behavior",
            .interactionFamily = ECameraInteractionFamily::None
        },
        {
            .kind = ICamera::CameraKind::FPS,
            .label = "FPS",
            .description = "First-person WASD + mouse look",
            .interactionFamily = ECameraInteractionFamily::Fps
        },
        {
            .kind = ICamera::CameraKind::Free,
            .label = "Free",
            .description = "Free-fly 6DOF with full rotation",
            .interactionFamily = ECameraInteractionFamily::Free
        },
        {
            .kind = ICamera::CameraKind::Orbit,
            .label = "Orbit",
            .description = "Orbit around target with dolly",
            .interactionFamily = ECameraInteractionFamily::Orbit
        },
        {
            .kind = ICamera::CameraKind::Arcball,
            .label = "Arcball",
            .description = "Arcball trackball around target",
            .interactionFamily = ECameraInteractionFamily::Orbit
        },
        {
            .kind = ICamera::CameraKind::Turntable,
            .label = "Turntable",
            .description = "Turntable yaw/pitch around target",
            .interactionFamily = ECameraInteractionFamily::Turntable
        },
        {
            .kind = ICamera::CameraKind::TopDown,
            .label = "TopDown",
            .description = "Fixed pitch top-down pan",
            .interactionFamily = ECameraInteractionFamily::TopDown
        },
        {
            .kind = ICamera::CameraKind::Isometric,
            .label = "Isometric",
            .description = "Fixed isometric view with pan",
            .interactionFamily = ECameraInteractionFamily::Orbit
        },
        {
            .kind = ICamera::CameraKind::Chase,
            .label = "Chase",
            .description = "Target follow with chase controls",
            .interactionFamily = ECameraInteractionFamily::TargetRig
        },
        {
            .kind = ICamera::CameraKind::Dolly,
            .label = "Dolly",
            .description = "Rig truck/dolly with look-at",
            .interactionFamily = ECameraInteractionFamily::TargetRig
        },
        {
            .kind = ICamera::CameraKind::DollyZoom,
            .label = "Dolly Zoom",
            .description = "Orbit with dolly-zoom FOV",
            .interactionFamily = ECameraInteractionFamily::Orbit
        },
        {
            .kind = ICamera::CameraKind::Path,
            .label = SCameraPathRigMetadata::KindLabel,
            .description = SCameraPathRigMetadata::KindDescription,
            .interactionFamily = ECameraInteractionFamily::Path
        }
    }};
};

} // namespace nbl::core

#endif // _C_CAMERA_KIND_UTILITIES_HPP_
