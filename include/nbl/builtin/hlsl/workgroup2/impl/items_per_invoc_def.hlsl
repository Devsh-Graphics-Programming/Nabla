// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

DEFINE_ASSIGN(uint16_t, ItemsPerInvocationProductLog2, DEFINE_MPL_MAX_V(int16_t,DEFINE_VIRTUAL_WG_T(WorkgroupSizeLog2)-DEFINE_VIRTUAL_WG_T(SubgroupSizeLog2)*DEFINE_VIRTUAL_WG_T(levels),0))
DEFINE_ASSIGN(uint16_t, value0, BaseItemsPerInvocation)
DEFINE_ASSIGN(uint16_t, value1, uint16_t(0x1u) << DEFINE_COND_VAL(uint16_t,(DEFINE_VIRTUAL_WG_T(levels)==3),DEFINE_MPL_MIN_V(uint16_t,DEFINE_ITEMS_INVOC_T(ItemsPerInvocationProductLog2),2),DEFINE_ITEMS_INVOC_T(ItemsPerInvocationProductLog2)))
DEFINE_ASSIGN(uint16_t, value2, uint16_t(0x1u) << DEFINE_MPL_MAX_V(int16_t,DEFINE_ITEMS_INVOC_T(ItemsPerInvocationProductLog2)-2,0))