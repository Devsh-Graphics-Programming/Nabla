// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/core/IReferenceCounted.h"

using namespace nbl;
using namespace core;

IReferenceCounted::~IReferenceCounted()
{
    _NBL_DEBUG_BREAK_IF(ReferenceCounter != 0);
}
