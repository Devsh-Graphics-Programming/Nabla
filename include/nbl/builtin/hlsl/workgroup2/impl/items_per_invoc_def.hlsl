// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

DEFINE_ASSIGN(uint16_t, ItemsPerInvocationProductLog2, MAX(int16_t,VIRTUAL_WG_SIZE WorkgroupSizeLog2-VIRTUAL_WG_SIZE SubgroupSizeLog2*VIRTUAL_WG_SIZE levels,0))
DEFINE_ASSIGN(uint16_t, value0, BaseItemsPerInvocation)
DEFINE_ASSIGN(uint16_t, value1, uint16_t(0x1u) << SELECT(uint16_t,(VIRTUAL_WG_SIZE levels==3),MIN(uint16_t,ItemsPerInvocationProductLog2,2),ItemsPerInvocationProductLog2))
DEFINE_ASSIGN(uint16_t, value2, uint16_t(0x1u) << MAX(int16_t,ItemsPerInvocationProductLog2-2,0))