#ifndef _NBL_BUILTIN_HLSL_BINOPS_INCLUDED_
#define _NBL_BUILTIN_HLSL_BINOPS_INCLUDED_

namespace nbl
{
namespace hlsl
{

#define COMPOUND_ASSIGN(NAME,OP) template<typename T> struct assign_ ## NAME ## _t { \
    void operator()(NBL_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs) \
    { \
        lhs = lhs OP rhs; \
    } \
}
COMPOUND_ASSIGN(add,+);
COMPOUND_ASSIGN(subtract,-);
COMPOUND_ASSIGN(mul,*);
COMPOUND_ASSIGN(div,/);

#undef COMPOUND_ASSIGN

}
}

#endif