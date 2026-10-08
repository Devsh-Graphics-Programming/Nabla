// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_I_CAMERA_WITH_PROJECTIONS_HPP_
#define _NBL_I_CAMERA_WITH_PROJECTIONS_HPP_

#include "nbl/core/decl/smart_refctd_ptr.h"
#include "CPlanarProjection.hpp"
#include "ICamera.hpp"

namespace nbl::ext::cameras
{

/// @brief One camera paired with the projections it is viewed through.
///
/// The camera provides the view matrix and each projection entry provides a projection matrix,
/// so one camera can be shown through several viewports, or switch projection presets, without
/// being replaced. Model transforms are left to the caller.
class ICameraWithProjections : virtual public core::IReferenceCounted
{
protected:
    ICameraWithProjections(core::smart_refctd_ptr<ICamera>&& camera)
        : m_camera(std::move(camera)) {}
    virtual ~ICameraWithProjections() = default;

    core::smart_refctd_ptr<ICamera> m_camera;
public:
    /// @brief Return the number of projection entries.
    virtual uint32_t getProjectionCount() const = 0;
    /// @brief Return one projection entry by index.
    virtual const CPlanarProjection& getProjection(uint32_t index) const = 0;

    /// @brief Replace the camera. Keeps the current one and returns `false` when `camera` is null.
    bool setCamera(core::smart_refctd_ptr<ICamera>&& camera);

    /// @brief Return the camera.
    ICamera* getCamera();

    /// @brief Left-handed world-to-view matrix of the camera, promoted to 4x4.
    hlsl::float64_t4x4 getViewMatrix() const;

    /// @brief Projection matrix of one entry, as last built by `CPlanarProjection::update`.
    hlsl::float64_t4x4 getProjectionMatrix(uint32_t projectionIx) const;

    /// @brief Projection matrix of one entry multiplied by the view matrix.
    hlsl::float64_t4x4 getViewProjectionMatrix(uint32_t projectionIx) const;
};

} // namespace nbl::ext::cameras

#endif // _NBL_I_CAMERA_WITH_PROJECTIONS_HPP_
