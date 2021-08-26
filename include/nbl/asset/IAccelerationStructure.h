// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_I_ACCELERATION_STRUCTURE_H_INCLUDED_
#define _NBL_ASSET_I_ACCELERATION_STRUCTURE_H_INCLUDED_

#include "nbl/asset/IDescriptor.h"
#include "nbl/asset/ECommonEnums.h"
#include "nbl/asset/IBuffer.h"

namespace nbl::asset
{
template<class BufferType>
class IAccelerationStructure : public IDescriptor
{
    public:
        enum E_TYPE : uint32_t
        {
            ET_TOP_LEVEL = 0,
            ET_BOTTOM_LEVEL = 1,
            ET_GENERIC = 2,
        };
        
        enum E_CREATE_FLAGS : uint32_t
        {
            ECF_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT   = 0x1u << 0u,
            ECF_MOTION_BIT_NV                       = 0x1u << 1u, // (erfan): extension, for later ?
        };

        struct SCreationParams
        {
            E_CREATE_FLAGS  flags;
            E_TYPE          type;
            SBufferRange<BufferType> bufferRange;
            bool operator==(const SCreationParams& rhs) const
            {
                return flags == rhs.flags && type == rhs.type;
            }
            bool operator!=(const SCreationParams& rhs) const
            {
				return !operator==(rhs);
            }
        };

        inline const auto& getCreationParameters() const
        {
            return params;
        }

        //!
        inline static bool validateCreationParameters(const SCreationParams& _params)
        {
            if(!_params.bufferRange.isValid()) {
                return false;
            }
            return true;
        }

        //!
        E_CATEGORY getTypeCategory() const override { return EC_ACCELERATION_STRUCTURE; }

    protected:
        IAccelerationStructure() :	params{ static_cast<E_CREATE_FLAGS>(0u), static_cast<E_TYPE>(0u), SBufferRange<BufferType>{} }
        {
        }
        IAccelerationStructure(SCreationParams&& _params) : params(std::move(_params)) {}

        virtual ~IAccelerationStructure()
        {}

        SCreationParams params;

    private:
};
} // end namespace nbl::asset

#endif


