// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_SCENE_H_INCLUDED_
#define _NBL_ASSET_I_SCENE_H_INCLUDED_


#include "nbl/asset/IMorphTargets.h"


namespace nbl::asset
{
// This is incredibly temporary, lots of things are going to change
class NBL_API2 IScene : public virtual core::IReferenceCounted
{
    public:

    protected:
        virtual ~IScene() = default;
};
}

#endif