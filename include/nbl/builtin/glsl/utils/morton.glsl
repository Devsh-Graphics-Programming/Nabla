// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

uint decodeMorton2d4bComponent(in uint x) 
{
    x &= 0x55555555u;
    x = (x ^ (x >>  1u)) & 0x33333333u;
    x = (x ^ (x >>  2u)) & 0x0f0f0f0fu;
    return x;
}

uint decodeMorton2d8bComponent(in uint x) 
{
    x &= 0x55555555u;
    x = (x ^ (x >>  1u)) & 0x33333333u;
    x = (x ^ (x >>  2u)) & 0x0f0f0f0fu;
    x = (x ^ (x >>  4u)) & 0x00ff00ffu;
    return x;
}

uvec2 decodeMorton2d4b(in uint x)
{
    return uvec2(decodeMorton2d4bComponent(x), decodeMorton2d4bComponent(x >> 1u));
}

uvec2 decodeMorton2d8b(in uint x)
{
    return uvec2(decodeMorton2d8bComponent(x), decodeMorton2d8bComponent(x >> 1u));
}