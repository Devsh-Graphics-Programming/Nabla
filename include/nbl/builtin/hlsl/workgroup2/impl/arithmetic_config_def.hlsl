// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

DEFINE_ASSIGN(uint16_t, WorkgroupSizeLog2, _WorkgroupSizeLog2)
DEFINE_ASSIGN(uint16_t, WorkgroupSize, uint16_t(0x1u) << DEFINE_CONFIG_T(WorkgroupSizeLog2))
DEFINE_ASSIGN(uint16_t, SubgroupSizeLog2, _SubgroupSizeLog2)
DEFINE_ASSIGN(uint16_t, SubgroupSize, uint16_t(0x1u) << DEFINE_CONFIG_T(SubgroupSizeLog2))

DEFINE_ASSIGN(uint16_t, LevelCount, DEFINE_VIRTUAL_WG_T(levels))
DEFINE_ASSIGN(uint16_t, VirtualWorkgroupSize, uint16_t(0x1u) << DEFINE_VIRTUAL_WG_T(value))

DEFINE_ASSIGN(uint16_t, ItemsPerInvocation_0, DEFINE_ITEMS_INVOC_T(value0))
DEFINE_ASSIGN(uint16_t, ItemsPerInvocation_1, DEFINE_ITEMS_INVOC_T(value1))
DEFINE_ASSIGN(uint16_t, ItemsPerInvocation_2, DEFINE_ITEMS_INVOC_T(value2))

DEFINE_ASSIGN(uint16_t, LevelInputCount_1, DEFINE_COND_VAL(uint16_t,(DEFINE_CONFIG_T(LevelCount)==3),
    DEFINE_MPL_MAX_V(uint16_t, (DEFINE_CONFIG_T(VirtualWorkgroupSize)>>DEFINE_CONFIG_T(SubgroupSizeLog2)), DEFINE_CONFIG_T(SubgroupSize)),
    DEFINE_CONFIG_T(SubgroupSize)*DEFINE_CONFIG_T(ItemsPerInvocation_1)))
DEFINE_ASSIGN(uint16_t, LevelInputCount_2, DEFINE_COND_VAL(uint16_t,(DEFINE_CONFIG_T(LevelCount)==3),DEFINE_CONFIG_T(SubgroupSize)*DEFINE_CONFIG_T(ItemsPerInvocation_2),0))
DEFINE_ASSIGN(uint16_t, VirtualInvocationsAtLevel1, DEFINE_CONFIG_T(LevelInputCount_1) / DEFINE_CONFIG_T(ItemsPerInvocation_1))

DEFINE_ASSIGN(uint16_t, __padding, DEFINE_COND_VAL(uint16_t,(DEFINE_CONFIG_T(LevelCount)==3),DEFINE_CONFIG_T(SubgroupSize)-1,0))
DEFINE_ASSIGN(uint16_t, __channelStride_1, DEFINE_COND_VAL(uint16_t,(DEFINE_CONFIG_T(LevelCount)==3),DEFINE_CONFIG_T(VirtualInvocationsAtLevel1),DEFINE_CONFIG_T(SubgroupSize)) + DEFINE_CONFIG_T(__padding))
DEFINE_ASSIGN(uint16_t, __channelStride_2, DEFINE_COND_VAL(uint16_t,(DEFINE_CONFIG_T(LevelCount)==3),DEFINE_CONFIG_T(SubgroupSize),0))

// user specified the shared mem size of Scalars
DEFINE_ASSIGN(uint32_t, SharedScratchElementCount, DEFINE_COND_VAL(uint16_t,(DEFINE_CONFIG_T(LevelCount)==1),
    0,
    DEFINE_COND_VAL(uint16_t,(DEFINE_CONFIG_T(LevelCount)==3),
        DEFINE_CONFIG_T(LevelInputCount_2)+(DEFINE_CONFIG_T(SubgroupSize)*DEFINE_CONFIG_T(ItemsPerInvocation_1))-1,
        0
        ) + DEFINE_CONFIG_T(LevelInputCount_1)
    ))
