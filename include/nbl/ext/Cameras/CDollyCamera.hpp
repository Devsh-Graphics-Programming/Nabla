#ifndef _C_DOLLY_CAMERA_HPP_
#define _C_DOLLY_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::core
{

/// @brief Target-relative camera that translates the target in the full local camera basis.
///
/// Translation uses the current right/up/forward basis. Rotation updates orbit
/// yaw and pitch while the camera pose is rebuilt from the maintained
/// target-relative offset.
class CDollyCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CDollyCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(position, target)
    {
        m_orbitUv.y = std::clamp(m_orbitUv.y, MinPitch, MaxPitch);
        applyPose();
    }
    ~CDollyCamera() = default;

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    /// @brief Apply one frame of local-frame dolly translation plus orbit rotation.
    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        if (referenceFrame)
        {
            CReferenceTransform reference = {};
            SCameraTargetRelativeState resolvedState = {};
            if (!tryExtractReferenceTransform(reference, referenceFrame) ||
                !tryResolveReferenceTargetRelativeState(reference, resolvedState))
            {
                return false;
            }

            resolvedState.orbitUv.y = std::clamp(resolvedState.orbitUv.y, MinPitch, MaxPitch);
            adoptTargetRelativeState(resolvedState);
        }

        const auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        const auto deltaRotation = scaleVirtualRotation(impulse.dVirtualRotation);

        const auto deltaTranslation = scaleVirtualTranslation(impulse.dVirtualTranslate);
        const auto basis = computeBasis(m_orbitUv, m_distance);
        const auto delta = hlsl::CCameraMathUtilities::transformLocalVectorToWorldBasis(deltaTranslation, basis.right, basis.up, basis.forward);

        m_targetPosition += delta;
        m_orbitUv.x += deltaRotation.y;
        m_orbitUv.y = std::clamp(m_orbitUv.y + deltaRotation.x, MinPitch, MaxPitch);

        return applyPose();
    }

    virtual uint32_t getAllowedVirtualEvents() const override { return AllowedVirtualEvents; }
    virtual CameraKind getKind() const override { return CameraKind::Dolly; }
    /// @brief Return the stable user-facing identifier for this concrete camera kind.
    virtual std::string_view getIdentifier() const override { return "Dolly Camera"; }

private:
    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
    static inline constexpr double MaxPitch = SCameraTargetRelativeRigDefaults::DollyPitchLimitRad;
    static inline constexpr double MinPitch = -MaxPitch;
};

}

#endif
