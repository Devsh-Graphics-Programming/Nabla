// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

DEFINE_ASSIGN(uint16_t, WorkgroupSizeLog2, _WorkgroupSizeLog2)
DEFINE_ASSIGN(uint16_t, WorkgroupSize, uint16_t(0x1u) << WorkgroupSizeLog2)
DEFINE_ASSIGN(uint16_t, SubgroupSizeLog2, _SubgroupSizeLog2)
DEFINE_ASSIGN(uint16_t, SubgroupSize, uint16_t(0x1u) << SubgroupSizeLog2)

DEFINE_ASSIGN(uint16_t, LevelCount, VIRTUAL_WG_SIZE levels)
DEFINE_ASSIGN(uint16_t, VirtualWorkgroupSize, uint16_t(0x1u) << VIRTUAL_WG_SIZE value)

DEFINE_ASSIGN(uint16_t, ItemsPerInvocation_0, ITEMS_PER_INVOC value0)
DEFINE_ASSIGN(uint16_t, ItemsPerInvocation_1, ITEMS_PER_INVOC value1)
DEFINE_ASSIGN(uint16_t, ItemsPerInvocation_2, ITEMS_PER_INVOC value2)

DEFINE_ASSIGN(uint16_t, LevelInputCount_1, SELECT(uint16_t,(LevelCount==3),
    MAX(uint16_t, (VirtualWorkgroupSize>>SubgroupSizeLog2), SubgroupSize),
    SubgroupSize*ItemsPerInvocation_1))
DEFINE_ASSIGN(uint16_t, LevelInputCount_2, SELECT(uint16_t,(LevelCount==3),SubgroupSize*ItemsPerInvocation_2,0))
DEFINE_ASSIGN(uint16_t, VirtualInvocationsAtLevel1, LevelInputCount_1 / ItemsPerInvocation_1)

DEFINE_ASSIGN(uint16_t, __padding, SELECT(uint16_t,(LevelCount==3),SubgroupSize-1,0))
DEFINE_ASSIGN(uint16_t, __channelStride_1, SELECT(uint16_t,(LevelCount==3),VirtualInvocationsAtLevel1,SubgroupSize) + __padding)
DEFINE_ASSIGN(uint16_t, __channelStride_2, SELECT(uint16_t,(LevelCount==3),SubgroupSize,0))

// user specified the shared mem size of Scalars
DEFINE_ASSIGN(uint32_t, SharedScratchElementCount, SELECT(uint16_t,(LevelCount==1),
    0,
    SELECT(uint16_t,(LevelCount==3),
        LevelInputCount_2+(SubgroupSize*ItemsPerInvocation_1)-1,
        0
        ) + LevelInputCount_1
    ))
