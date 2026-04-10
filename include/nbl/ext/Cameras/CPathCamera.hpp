#ifndef _C_PATH_CAMERA_HPP_
#define _C_PATH_CAMERA_HPP_

#include <algorithm>
#include <utility>

#include "CCameraPathUtilities.hpp"
#include "CSphericalTargetCamera.hpp"

namespace nbl::core
{

/// @brief Path-rig camera driven by typed `PathState` plus an injected path model.
///
/// The public runtime path stays event-only through `manipulate(...)`.
/// `CPathCamera` only interprets the accumulated impulse through `m_pathModel`
/// instead of hardcoding one default target-relative mapping in the method body.
class CPathCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;
    using path_model_t = SCameraPathModel;
    using path_limits_t = PathStateLimits;

    /// @brief Construct the path rig with the shared default path model and default limits.
    CPathCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : CPathCamera(position, target, CCameraPathUtilities::makeDefaultPathModel(), CCameraPathUtilities::makeDefaultPathLimits())
    {
    }

    /// @brief Construct the path rig with a caller-provided model and default limits.
    CPathCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target, path_model_t pathModel)
        : CPathCamera(position, target, std::move(pathModel), CCameraPathUtilities::makeDefaultPathLimits())
    {
    }

    /// @brief Construct the path rig with the shared default model and caller-provided limits.
    CPathCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target, path_limits_t pathLimits)
        : CPathCamera(position, target, CCameraPathUtilities::makeDefaultPathModel(), pathLimits)
    {
    }

    /// @brief Construct the path rig with fully caller-provided model and path-state limits.
    CPathCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target, path_model_t pathModel, path_limits_t pathLimits)
        : base_t(position, target)
    {
        initializePathRig(position, std::move(pathModel), pathLimits);
    }

    ~CPathCamera() = default;

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    /// @brief Consume virtual events through the active path model and update the runtime pose from the resulting path state.
    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr) override
    {
        if (virtualEvents.empty() && !referenceFrame)
            return false;

        PathState nextPathState = m_pathState;
        CReferenceTransform reference = {};
        const CReferenceTransform* resolvedReference = nullptr;
        if (referenceFrame)
        {
            if (!m_gimbal.extractReferenceTransform(&reference, referenceFrame))
                return false;
            resolvedReference = &reference;
            if (!m_pathModel.resolveState ||
                !m_pathModel.resolveState(
                    m_targetPosition,
                    hlsl::float64_t3(reference.frame[3]),
                    m_pathLimits,
                    nullptr,
                    nextPathState))
            {
                return false;
            }
        }

        const auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);
        const SCameraPathControlContext context = {
            .currentState = nextPathState,
            .translation = scaleVirtualTranslation(impulse.dVirtualTranslate),
            .rotation = scaleVirtualRotation(impulse.dVirtualRotation),
            .targetPosition = m_targetPosition,
            .reference = resolvedReference,
            .limits = m_pathLimits
        };

        if (!m_pathModel.controlLaw || !m_pathModel.integrate)
            return false;

        const auto stateDelta = m_pathModel.controlLaw(context);
        if (!m_pathModel.integrate(nextPathState, stateDelta, m_pathLimits, nextPathState))
            return false;

        const auto previousPathState = m_pathState;
        m_pathState = nextPathState;
        bool manipulated = false;
        if (refreshFromPathState(&manipulated))
            return manipulated;

        m_pathState = previousPathState;
        refreshFromPathState();
        return false;
    }

    virtual uint32_t getAllowedVirtualEvents() const override { return AllowedVirtualEvents; }
    virtual CameraKind getKind() const override { return CameraKind::Path; }
    virtual uint32_t getGoalStateMask() const override { return base_t::getGoalStateMask() | base_t::GoalStatePath; }

    /// @brief Query the current typed path state.
    virtual bool tryGetPathState(PathState& out) const override
    {
        out = m_pathState;
        return true;
    }

    /// @brief Query the active path-state limits used by this camera instance.
    virtual bool tryGetPathStateLimits(PathStateLimits& out) const override
    {
        out = m_pathLimits;
        return true;
    }

    /// @brief Query the derived spherical-target state corresponding to the current path-state evaluation.
    virtual bool tryGetSphericalTargetState(typename base_t::SphericalTargetState& out) const override
    {
        out.target = m_targetPosition;
        out.distance = m_distance;
        out.orbitUv = m_orbitUv;
        out.minDistance = static_cast<float>(m_pathLimits.minDistance);
        out.maxDistance = static_cast<float>(m_pathLimits.maxDistance);
        return true;
    }

    /// @brief Replace only the tracked target position and rebuild the current path pose against it.
    virtual bool trySetSphericalTarget(const hlsl::float64_t3& targetPosition) override
    {
        if (m_targetPosition == targetPosition)
            return true;

        const auto previousTarget = m_targetPosition;
        m_targetPosition = targetPosition;
        if (refreshFromPathState())
            return true;

        m_targetPosition = previousTarget;
        refreshFromPathState();
        return false;
    }

    /// @brief Replace the current path state after sanitizing it through the active path model.
    virtual bool trySetPathState(const PathState& state) override
    {
        if (!m_pathModel.resolveState)
            return false;

        PathState sanitized = {};
        if (!m_pathModel.resolveState(m_targetPosition, m_gimbal.getPosition(), m_pathLimits, &state, sanitized))
            return false;

        const bool exact = CCameraPathUtilities::pathStatesNearlyEqual(sanitized, state, SCameraPathDefaults::ExactComparisonThresholds);
        const auto previousState = m_pathState;
        m_pathState = sanitized;
        if (refreshFromPathState())
            return exact;

        m_pathState = previousState;
        refreshFromPathState();
        return false;
    }

    /// @brief Replace the derived path distance while preserving the rest of the typed path state.
    virtual bool trySetSphericalDistance(float distance) override
    {
        SCameraPathDistanceUpdateResult distanceUpdate = {};
        if (!m_pathModel.updateDistance)
        {
            return false;
        }

        const auto previousState = m_pathState;
        if (!m_pathModel.updateDistance(distance, m_pathLimits, m_pathState, &distanceUpdate))
            return false;
        if (!refreshFromPathState())
        {
            m_pathState = previousState;
            refreshFromPathState();
            return false;
        }

        return distanceUpdate.exact;
    }

    virtual std::string_view getIdentifier() const override { return SCameraPathDefaults::Identifier; }

    /// @brief Return the currently installed path model.
    inline const path_model_t& getPathModel() const
    {
        return m_pathModel;
    }

    /// @brief Return the current path-state limits enforced by this camera instance.
    inline const path_limits_t& getPathStateLimits() const
    {
        return m_pathLimits;
    }

    /// @brief Replace the active path-state limits after sanitizing the current path state against them.
    inline bool setPathStateLimits(path_limits_t pathLimits)
    {
        if (!CCameraPathUtilities::sanitizePathLimits(pathLimits) || !m_pathModel.resolveState)
            return false;

        PathState sanitizedState = {};
        if (!m_pathModel.resolveState(m_targetPosition, m_gimbal.getPosition(), pathLimits, &m_pathState, sanitizedState))
            return false;

        const auto previousLimits = m_pathLimits;
        const auto previousState = m_pathState;
        m_pathLimits = pathLimits;
        m_pathState = sanitizedState;
        if (refreshFromPathState())
            return true;

        m_pathLimits = previousLimits;
        m_pathState = previousState;
        refreshFromPathState();
        return false;
    }

    /// @brief Replace the active path model after validating that it can resolve the current path state.
    inline bool setPathModel(path_model_t pathModel)
    {
        if (!isPathModelComplete(pathModel))
            return false;

        PathState sanitized = {};
        if (!pathModel.resolveState(m_targetPosition, m_gimbal.getPosition(), m_pathLimits, &m_pathState, sanitized))
            return false;

        const auto previousModel = m_pathModel;
        const auto previousState = m_pathState;
        m_pathModel = std::move(pathModel);
        m_pathState = sanitized;
        if (refreshFromPathState())
            return true;

        m_pathModel = previousModel;
        m_pathState = previousState;
        refreshFromPathState();
        return false;
    }

private:
    static inline constexpr auto AllowedVirtualEvents =
        CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::RollLeft | CVirtualGimbalEvent::RollRight;

    /// @brief Check whether a path model provides all callbacks required by the runtime camera.
    static inline bool isPathModelComplete(const path_model_t& pathModel)
    {
        return pathModel.resolveState && pathModel.controlLaw && pathModel.integrate && pathModel.evaluate && pathModel.updateDistance;
    }

    /// @brief Attempt to initialize the runtime path state and pose from one model/limit pair.
    inline bool tryInitializePathRig(const hlsl::float64_t3& position, path_model_t pathModel, path_limits_t pathLimits)
    {
        if (!CCameraPathUtilities::sanitizePathLimits(pathLimits))
            return false;

        if (!isPathModelComplete(pathModel))
            return false;

        PathState resolvedState = {};
        if (!pathModel.resolveState(m_targetPosition, position, pathLimits, nullptr, resolvedState))
            return false;

        m_pathLimits = pathLimits;
        m_pathModel = std::move(pathModel);
        m_pathState = resolvedState;
        return refreshFromPathState();
    }

    /// @brief Initialize the path rig with graceful fallback to the shared default model and limits.
    inline void initializePathRig(const hlsl::float64_t3& position, path_model_t pathModel, path_limits_t pathLimits)
    {
        path_limits_t sanitizedLimits = pathLimits;
        const bool hasCustomLimits = CCameraPathUtilities::sanitizePathLimits(sanitizedLimits);
        if (!hasCustomLimits)
            sanitizedLimits = CCameraPathUtilities::makeDefaultPathLimits();

        if (tryInitializePathRig(position, std::move(pathModel), sanitizedLimits))
            return;

        if (tryInitializePathRig(position, CCameraPathUtilities::makeDefaultPathModel(), sanitizedLimits))
            return;

        m_pathLimits = CCameraPathUtilities::makeDefaultPathLimits();
        m_pathModel = CCameraPathUtilities::makeDefaultPathModel();
        m_pathState = CCameraPathUtilities::makeDefaultPathState(m_pathLimits.minU);
        m_pathModel.resolveState(m_targetPosition, position, m_pathLimits, nullptr, m_pathState);
        refreshFromPathState();
    }

    path_model_t m_pathModel = CCameraPathUtilities::makeDefaultPathModel();
    path_limits_t m_pathLimits = CCameraPathUtilities::makeDefaultPathLimits();
    PathState m_pathState = CCameraPathUtilities::makeDefaultPathState(CCameraPathUtilities::makeDefaultPathLimits().minU);

    /// @brief Evaluate the current path state into a canonical pose and write it back to the runtime gimbal.
    bool refreshFromPathState(bool* outManipulated = nullptr)
    {
        if (!m_pathModel.evaluate)
            return false;

        SCameraCanonicalPathState canonicalPathState = {};
        if (!m_pathModel.evaluate(m_targetPosition, m_pathState, m_pathLimits, canonicalPathState))
            return false;

        m_distance = canonicalPathState.targetRelative.distance;
        m_orbitUv = canonicalPathState.targetRelative.orbitUv;

        m_gimbal.begin();
        {
            m_gimbal.setPosition(canonicalPathState.pose.position);
            m_gimbal.setOrientation(canonicalPathState.pose.orientation);
        }
        m_gimbal.end();

        const bool manipulated = bool(m_gimbal.getManipulationCounter());
        if (manipulated)
            m_gimbal.updateView();

        if (outManipulated)
            *outManipulated = manipulated;
        return true;
    }
};

}

#endif
