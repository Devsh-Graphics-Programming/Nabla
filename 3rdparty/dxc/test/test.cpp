#include <iostream>

#if __cplusplus >= 201703L
#pragma message("'__cplusplus' is " _CRT_STRINGIZE(__cplusplus))
#error C++ compiler with too high standard for DXC compilation. Your compiler must be capable of using C++11 or C++14 standard!
#endif 

int main() { return 0; }
