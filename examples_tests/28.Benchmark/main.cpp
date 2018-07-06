#define WITH_COMPUTE_SHADER
//#define WITH_VERTEX_SHADER

#if defined(WITH_COMPUTE_SHADER)
#include "main_cs.h"
#elif defined(WITH_VERTEX_SHADER)
#include "main_vs.h"
#else
int main() 
{
    return 0;
}
#endif