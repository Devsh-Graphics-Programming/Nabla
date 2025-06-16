// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

DEFINE_ASSIGN(uint16_t, WorkgroupSizeLog2, _WorkgroupSizeLog2)
DEFINE_ASSIGN(uint16_t, SubgroupSizeLog2, _SubgroupSizeLog2)
DEFINE_ASSIGN(uint16_t, levels, SELECT(uint16_t,(_WorkgroupSizeLog2>_SubgroupSizeLog2),SELECT(uint16_t,(_WorkgroupSizeLog2>_SubgroupSizeLog2*2+2),3,2),1))
DEFINE_ASSIGN(uint16_t, value, MAX(uint16_t, _SubgroupSizeLog2*levels, _WorkgroupSizeLog2))
