#ifndef _C_TOPDOWN_CAMERA_HPP_
#define _C_TOPDOWN_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::core
{

/// @brief Target-relative camera constrained to look straight down at the tracked target.
///
/// Yaw may still rotate the view around the vertical axis, while pitch is fixed to
/// the top-down angle and translation moves the tracked target in the view plane.
class CTopDownCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CTopDownCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(position, target)
    {
        m_orbitUv.y = TopDownPitch;
        applyPose();
    }
    ~CTopDownCamera() = default;

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    /// @brief Apply one frame of top-down yaw rotation, planar translation, and distance changes.
    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        if (referenceFrame)
        {
            CReferenceTransform reference = {};
            SCameraTargetRelativeState resolvedState = {};
            if (!tryExtractReferenceTransform(reference, referenceFrame) ||
                !tryResolveReferenceTopDownState(reference, resolvedState))
            {
                return false;
            }

            adoptTargetRelativeState(resolvedState);
        }

        const auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        const auto deltaRotation = scaleVirtualRotation(impulse.dVirtualRotation);
        const auto deltaTranslation = scaleVirtualTranslation(impulse.dVirtualTranslate);
        const double deltaDistance = scaleUnscaledVirtualTranslation(impulse.dVirtualTranslate.z);

        m_orbitUv.x += deltaRotation.y;
        m_orbitUv.y = TopDownPitch;
        m_distance = std::clamp<float>(m_distance + static_cast<float>(deltaDistance), MinDistance, MaxDistance);

        const auto basis = computeBasis(m_orbitUv, m_distance);
        applyPlanarTargetTranslation(deltaTranslation, basis);

        return applyPose();
    }

    virtual uint32_t getAllowedVirtualEvents() const override { return AllowedVirtualEvents; }
    virtual CameraKind getKind() const override { return CameraKind::TopDown; }
    /// @brief Return the stable user-facing identifier for this concrete camera kind.
    virtual std::string_view getIdentifier() const override { return "Top-Down Camera"; }

private:
    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
    static inline constexpr double TopDownPitch = SCameraTargetRelativeRigDefaults::TopDownPitchRad;
};

}

#endif
