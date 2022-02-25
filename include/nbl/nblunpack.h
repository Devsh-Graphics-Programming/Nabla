// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

// include this file to switch back to default alignment
// file belongs to nblpack.h, see there for more info

// Default alignment
#if defined(_MSC_VER) || defined(__GNUC__) || defined (__clang__)
#	pragma pack( pop, packing )
#else
#	error compiler not supported
#endif

#undef PACK_STRUCT

