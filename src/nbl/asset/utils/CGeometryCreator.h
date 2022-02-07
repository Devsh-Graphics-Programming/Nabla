// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_GEOMETRY_CREATOR_H_INCLUDED__
#define __NBL_ASSET_C_GEOMETRY_CREATOR_H_INCLUDED__

#include "nbl/asset/utils/IGeometryCreator.h"

namespace nbl
{
namespace asset
{
//! class for creating geometry on the fly
class CGeometryCreator : public IGeometryCreator
{
public:
#include "nbl/nblpack.h"
    struct CubeVertex
    {
        float pos[3];
        uint8_t color[4];  // normalized
        uint8_t uv[2];
        int8_t normal[3];
        uint8_t dummy[3];

        void setPos(float x, float y, float z)
        {
            pos[0] = x;
            pos[1] = y;
            pos[2] = z;
        }
        void translate(float dx, float dy, float dz)
        {
            pos[0] += dx;
            pos[1] += dy;
            pos[2] += dz;
        }
        void setColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
        {
            color[0] = r;
            color[1] = g;
            color[2] = b;
            color[3] = a;
        }
        void setNormal(int8_t x, int8_t y, int8_t z)
        {
            normal[0] = x;
            normal[1] = y;
            normal[2] = z;
        }
        void setUv(uint8_t u, uint8_t v)
        {
            uv[0] = u;
            uv[1] = v;
        }
    } PACK_STRUCT;

public:
    CGeometryCreator(IMeshManipulator* const _defaultMeshManipulator);

private:
    struct RectangleVertex
    {
        RectangleVertex(const core::vector3df_SIMD& _pos, const video::SColor& _color, const core::vector2du32_SIMD _uv, const core::vector3df_SIMD _normal)
        {
            memcpy(pos, _pos.pointer, sizeof(float) * 3);
            _color.toOpenGLColor(color);
            uv[0] = _uv.x;
            uv[1] = _uv.y;
            normal[0] = _normal.x;
            normal[1] = _normal.y;
            normal[2] = _normal.z;
        }
        float pos[3];
        uint8_t color[4];
        uint8_t uv[2];
        uint8_t dummy[2];
        float normal[3];
    } PACK_STRUCT;

    typedef RectangleVertex DiskVertex;

    struct ConeVertex
    {
        inline ConeVertex(const core::vectorSIMDf& _pos, const CQuantNormalCache::value_type_t<EF_A2B10G10R10_SNORM_PACK32>& _nml, const video::SColor& _color)
            : normal{_nml}
        {
            memcpy(pos, _pos.pointer, 12);
            _color.toOpenGLColor(color);
        }

        float pos[3];
        uint8_t color[4];
        CQuantNormalCache::value_type_t<EF_A2B10G10R10_SNORM_PACK32> normal;
    } PACK_STRUCT;

    struct CylinderVertex
    {
        CylinderVertex()
            : pos{0.f, 0.f, 0.f}, color{0u, 0u, 0u, 0u}, uv{0.f, 0.f}, normal() {}

        float pos[3];
        uint8_t color[4];
        float uv[2];
        CQuantNormalCache::value_type_t<EF_A2B10G10R10_SNORM_PACK32> normal;
    } PACK_STRUCT;

    struct IcosphereVertex
    {
        IcosphereVertex()
            : pos{0.f, 0.f, 0.f}, normals{0.f, 0.f, 0.f}, uv{0.f, 0.f} {}

        float pos[3];
        float normals[3];
        float uv[2];
    } PACK_STRUCT;
#include "nbl/nblunpack.h"

    using SphereVertex = CylinderVertex;
    using ArrowVertex = CylinderVertex;

    //smart_refctd_ptr?
    IMeshManipulator* const defaultMeshManipulator;

public:
    return_type createCubeMesh(const core::vector3df& size) const override;

    return_type createArrowMesh(const uint32_t tesselationCylinder,
        const uint32_t tesselationCone, const float height,
        const float cylinderHeight, const float width0,
        const float width1, const video::SColor vtxColor0,
        const video::SColor vtxColor1,
        IMeshManipulator* const meshManipulatorOverride = nullptr) const override;

    return_type createSphereMesh(float radius, uint32_t polyCountX, uint32_t polyCountY, IMeshManipulator* const meshManipulatorOverride = nullptr) const override;

    return_type createCylinderMesh(float radius, float length, uint32_t tesselation,
        const video::SColor& color = 0xffffffff,
        IMeshManipulator* const meshManipulatorOverride = nullptr) const override;

    return_type createConeMesh(float radius, float length, uint32_t tesselation,
        const video::SColor& colorTop = 0xffffffff,
        const video::SColor& colorBottom = 0xffffffff,
        float oblique = 0.f,
        IMeshManipulator* const meshManipulatorOverride = nullptr) const override;

    return_type createRectangleMesh(const core::vector2df_SIMD& _size = core::vector2df_SIMD(0.5f, 0.5f)) const override;

    return_type createDiskMesh(float radius, uint32_t tesselation) const override;

    return_type createIcoSphere(float radius = 1.0f, uint32_t subdivision = 1, bool smooth = false) const override;
};

}  // end namespace asset
}  // end namespace nbl

#endif
