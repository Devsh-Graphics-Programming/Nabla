// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_IGIMBAL_HPP_
#define _NBL_IGIMBAL_HPP_

#include <cstdint>

#include "CCameraMathUtilities.hpp"

namespace nbl::ext::cameras
{
    /// @brief World-space pose: position plus unit orientation.
    ///
    /// The gimbal stores nothing but the pose and a count of the manipulations applied to it. Basis vectors and
    /// the world matrix are computed from the orientation on every call. Every setter reports whether the stored
    /// pose changed, and a changed pose advances the manipulation counter by one.
    class IGimbal
    {
    public:
        using quaternion_t = hlsl::math::quaternion<hlsl::float64_t>;

        IGimbal(const IGimbal&) = default;
        IGimbal(IGimbal&&) noexcept = default;
        IGimbal& operator=(const IGimbal&) = default;
        IGimbal& operator=(IGimbal&&) noexcept = default;

        /// @brief Store `pose` with its orientation normalized.
        explicit IGimbal(const SCameraRigPose& pose)
            : m_pose{ .position = pose.position, .orientation = hlsl::normalize(pose.orientation) } {}

        /// @brief Replace the world-space position.
        /// @return whether the stored position changed.
        inline bool setPosition(const hlsl::float64_t3& position)
        {
            return countManipulation(writePosition(position));
        }

        /// @brief Replace the orientation, storing it normalized.
        /// @return whether the stored orientation changed.
        inline bool setOrientation(const quaternion_t& orientation)
        {
            return countManipulation(writeOrientation(orientation));
        }

        /// @brief Replace position and orientation. Counts as one manipulation even when both change.
        /// @return whether either changed.
        inline bool setPose(const SCameraRigPose& pose)
        {
            const bool positionChanged = writePosition(pose.position);
            const bool orientationChanged = writeOrientation(pose.orientation);
            return countManipulation(positionChanged || orientationChanged);
        }

        /// @brief Decompose one rigid world-space transform (basis in the columns, translation in the last
        /// column) and store it as the pose.
        /// @return whether `rigidTransform` was rigid and therefore applied; a rejected transform leaves the pose untouched.
        inline bool setPose(const hlsl::float64_t4x4& rigidTransform)
        {
            SCameraRigPose pose = {};
            if (!CCameraMathUtilities::tryExtractRigidPoseFromTransform(rigidTransform, pose.position, pose.orientation))
                return false;

            setPose(pose);
            return true;
        }

        /// @brief Position in world space.
        inline const hlsl::float64_t3& getPosition() const { return m_pose.position; }

        /// @brief Unit orientation.
        inline const quaternion_t& getOrientation() const { return m_pose.orientation; }

        /// @brief Position and orientation as one value.
        inline const SCameraRigPose& getPose() const { return m_pose; }

        /// @brief Orthonormal local basis rotated into world space, as three named vectors.
        inline SCameraBasis<hlsl::float64_t> getBasis() const { return CCameraMathUtilities::getOrientationBasis(m_pose.orientation); }

        /// @brief Local +X in world space.
        inline hlsl::float64_t3 getRight() const { return m_pose.orientation.transformVector(CCameraMathUtilities::getCameraWorldRight<hlsl::float64_t>(), true); }

        /// @brief Local +Y in world space.
        inline hlsl::float64_t3 getUp() const { return m_pose.orientation.transformVector(CCameraMathUtilities::getCameraWorldUp<hlsl::float64_t>(), true); }

        /// @brief Local +Z in world space.
        inline hlsl::float64_t3 getForward() const { return m_pose.orientation.transformVector(CCameraMathUtilities::getCameraWorldForward<hlsl::float64_t>(), true); }

        /// @brief Rigid local-to-world matrix: basis vectors in the columns, position in the last column.
        inline hlsl::float64_t3x4 getWorldMatrix() const
        {
            const auto transform = CCameraMathUtilities::composeTransformMatrix(m_pose.position, m_pose.orientation);
            return hlsl::float64_t3x4(transform[0], transform[1], transform[2]);
        }

        /// @brief Number of manipulations that changed the pose since construction. Never decreases.
        ///
        /// Two reads returning the same value mean the pose did not change in between, which is what lets
        /// derived data be cached against it.
        inline uint64_t getManipulationCounter() const { return m_manipulationCounter; }

    private:
        inline bool writePosition(const hlsl::float64_t3& position)
        {
            if (m_pose.position == position)
                return false;

            m_pose.position = position;
            return true;
        }

        inline bool writeOrientation(const quaternion_t& orientation)
        {
            const auto normalized = hlsl::normalize(orientation);
            if (m_pose.orientation.data == normalized.data)
                return false;

            m_pose.orientation = normalized;
            return true;
        }

        inline bool countManipulation(const bool changed)
        {
            if (changed)
                ++m_manipulationCounter;
            return changed;
        }

        SCameraRigPose m_pose;
        uint64_t m_manipulationCounter = 0ull;
    };
} // namespace nbl::ext::cameras

#endif // _NBL_IGIMBAL_HPP_
