#ifndef _NBL_MATH_TYPELESS_ARITHMETIC_INCLUDED_
#define _NBL_MATH_TYPELESS_ARITHMETIC_INCLUDED_

// TODO: change header name to binary_operator_functions.glsl

int nbl_glsl_identityFunction(in int x) {return x;}
uint nbl_glsl_identityFunction(in uint x) {return x;}
float nbl_glsl_identityFunction(in float x) {return x;}

// no float version of these because denormalized bit patterns may flush to 0
int nbl_glsl_and(in int x, in int y) {return x&y;}
uint nbl_glsl_and(in uint x, in uint y) {return x&y;}

int nbl_glsl_xor(in int x, in int y) {return x^y;}
uint nbl_glsl_xor(in uint x, in uint y) {return x^y;}

int nbl_glsl_or(in int x, in int y) {return x|y;}
uint nbl_glsl_or(in uint x, in uint y) {return x|y;}


int nbl_glsl_add(in int x, in int y) {return x+y;}
uint nbl_glsl_add(in uint x, in uint y) {return x+y;}
float nbl_glsl_add(in float x, in float y) {return x+y;}

int nbl_glsl_mul(in int x, in int y) {return x*y;}
uint nbl_glsl_mul(in uint x, in uint y) {return x*y;}
float nbl_glsl_mul(in float x, in float y) {return x*y;}

#endif
