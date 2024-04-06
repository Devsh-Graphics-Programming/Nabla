// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_IES_PROFILE_H_INCLUDED__
#define __NBL_ASSET_C_IES_PROFILE_H_INCLUDED__

#include "nbl/core/Types.h"
#include <sstream>

namespace nbl 
{
    namespace asset 
    {
        class CIESProfile 
        {
            public:
                using IES_STORAGE_FORMAT = double;

                _NBL_STATIC_INLINE_CONSTEXPR size_t CDC_DEFAULT_TEXTURE_WIDTH = 1024;
                _NBL_STATIC_INLINE_CONSTEXPR size_t CDC_DEFAULT_TEXTURE_HEIGHT = 1024;

                _NBL_STATIC_INLINE_CONSTEXPR IES_STORAGE_FORMAT MAX_VANGLE = 180.0;
                _NBL_STATIC_INLINE_CONSTEXPR IES_STORAGE_FORMAT MAX_HANGLE = 360.0;

                _NBL_STATIC_INLINE_CONSTEXPR auto UI16_MAX_D = 65535.0;
                _NBL_STATIC_INLINE_CONSTEXPR auto IES_TEXTURE_STORAGE_FORMAT = asset::EF_R16_UNORM;

                enum Version : uint8_t
                {
                    V_1995,
                    V_2002,
                    V_SIZE
                };

                enum PhotometricType : uint8_t 
                {
                    TYPE_NONE,
                    TYPE_C,
                    TYPE_B,
                    TYPE_A
                };

                enum LuminairePlanesSymmetry : uint8_t
                {
                    ISOTROPIC,                  //! Only one horizontal angle present and a luminaire is assumed to be laterally axial symmetric
                    QUAD_SYMETRIC,              //! The luminaire is assumed to be symmetric in each quadrant
                    HALF_SYMETRIC,              //! The luminaire is assumed to be symmetric about the 0 to 180 degree plane
                    OTHER_HALF_SYMMETRIC,       //! HALF_SYMETRIC case for legacy V_1995 version where horizontal angles are in range [90, 270], in that case the parser patches horizontal angles to be HALF_SYMETRIC
                    NO_LATERAL_SYMMET           //! The luminaire is assumed to exhibit no lateral symmet
                };

                CIESProfile() = default;
                ~CIESProfile() = default;

                const IES_STORAGE_FORMAT& getHAnglesOffset() const { return hAnglesOffset; }
                const core::vector<IES_STORAGE_FORMAT>& getHoriAngles() const { return hAngles; }
                const core::vector<IES_STORAGE_FORMAT>& getVertAngles() const { return vAngles; }
                const core::vector<IES_STORAGE_FORMAT>& getData() const { return data; }
                const auto getSymmetry() const { return symmetry; }

                IES_STORAGE_FORMAT getValue(size_t i, size_t j) const { return data[vAngles.size() * i + j]; }
                IES_STORAGE_FORMAT getMaxValue() const { return maxValue; }

                const IES_STORAGE_FORMAT& getIntegralFromGrid() const { return integral; }
                const IES_STORAGE_FORMAT& getAvgEmmision() const { return avgEmmision; }

                template<class ExecutionPolicy>
                core::smart_refctd_ptr<asset::ICPUImageView> createIESTexture(ExecutionPolicy&& policy, const size_t& width = CDC_DEFAULT_TEXTURE_WIDTH, const size_t& height = CDC_DEFAULT_TEXTURE_HEIGHT) const;
                core::smart_refctd_ptr<asset::ICPUImageView> createIESTexture(const size_t& width = CDC_DEFAULT_TEXTURE_WIDTH, const size_t& height = CDC_DEFAULT_TEXTURE_HEIGHT) const;

            private:
                CIESProfile(PhotometricType type, size_t hSize, size_t vSize)
                    : type(type), version(V_SIZE), hAngles(hSize), vAngles(vSize), data(hSize* vSize) {}

                // TODO for @Hazard, I would move it into separate file, we may use this abstraction somewhere too
                //! Returns spherical coordinates with physics convention in radians
                /*
                    https://en.wikipedia.org/wiki/Spherical_coordinate_system#/media/File:3D_Spherical.svg
                    Retval.first is "theta" polar angle in range [0, PI] & Retval.second "phi" is azimuthal angle
                    in range [-PI, PI] range

                    Cartesian coordinates obtained from the spherical coordinates in Nabla
                    are assumed to have radius equal to 1 and therefore always are

                    x = cos(phi)*sin(theta)
                    y = sin(phi)*sin(theta)
                    z = cos(theta)
                */

                static inline std::pair<float, float> sphericalDirToRadians(const core::vectorSIMDf& dir);
                
                //! Octahedral coordinate mapping is following
                /*
                    center is Z-
                    U+ from center is X+
                    V+ from center is Y+

                    when viewed as a texture, the net folds, and the apex where the seams join is Z+
                */
                
                static inline core::vectorSIMDf octahdronUVToDir(const float& u, const float& v);
                
                void addHoriAngle(IES_STORAGE_FORMAT hAngle)
                {
                    hAngles.push_back(hAngle);
                    data.resize(getHoriAngles().size() * vAngles.size());
                }

                void setValue(size_t i, size_t j, IES_STORAGE_FORMAT val) { data[vAngles.size() * i + j] = val; }

                const IES_STORAGE_FORMAT sample(IES_STORAGE_FORMAT vAngle, IES_STORAGE_FORMAT hAngle) const;

                PhotometricType type;
                Version version;
                LuminairePlanesSymmetry symmetry;

                core::vector<IES_STORAGE_FORMAT> hAngles;                   //! The angular displacement indegreesfrom straight down, a value represents spherical coordinate "theta" with physics convention
                IES_STORAGE_FORMAT hAnglesOffset;                           //! The real horizontal angle provided by IES data is (hAngles[index] + hAnglesOffset) - the reason behind it is we patch 1995 IES OTHER_HALF_SYMETRIC case to be HALF_SYMETRIC
                core::vector<IES_STORAGE_FORMAT> vAngles;                   //! Measurements in degrees of angular displacement measured counterclockwise in a horizontal plane for Type C photometry and clockwise for Type A and B photometry, a value represents spherical coordinate "phi" with physics convention
                core::vector<IES_STORAGE_FORMAT> data;                      //! Candela values
                IES_STORAGE_FORMAT maxValue = {};                           //! Max value from this->data vector
                
                mutable IES_STORAGE_FORMAT integral = {};                  //! Total energy emitted
                mutable IES_STORAGE_FORMAT avgEmmision = {};                //! this->integral / <size of the emission domain where non zero values>

                friend class CIESProfileParser;
        };
    }
}

#endif // __NBL_ASSET_C_IES_PROFILE_H_INCLUDED__