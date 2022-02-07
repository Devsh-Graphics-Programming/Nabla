// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

// include this file right before the data structures to be 1-aligned
// Always include the nblunpack.h file right after the last type declared
// like this, and do not put any other types with different alignment
// in between!

// byte-align structures
#if defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__)
#ifdef _MSC_VER
#pragma warning(disable : 4103)
#endif
#pragma pack(push, packing)
#pragma pack(1)
// TODO: Remove PACK_STRUCT from the engine
#define PACK_STRUCT
#else
#error compiler not supported
#endif
