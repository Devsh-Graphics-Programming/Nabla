#include <irrlicht.h>
#include <omp.h>
#include <IReferenceCounted.h>
#include <cassert>
#include <random>

uint32_t getRandom(uint32_t min, uint32_t max)
{
    static thread_local std::mt19937 generator;
    return std::uniform_int_distribution<uint32_t>{min,max}(generator);
}
#define RANDOM_MIN (1u<<18)
#define RANDOM_MAX (1u<<19)

class RefCounted : public irr::IReferenceCounted {};

#define TEST_COUNT_AFTER
int main()
{
    auto r = new RefCounted();
    omp_set_num_threads(256);
#pragma omp parallel
{
    #pragma omp for
    for (int i = 0; i<256; ++i)
    {
        r->grab();

#ifndef TEST_COUNT_AFTER
        if (omp_get_thread_num() == 0)
        {
            r->drop();
        }
#endif TEST_COUNT_AFTER
    }

    #pragma omp for
  	for (int i = 0; i < 256; ++i)
    {
        const size_t cnt = getRandom(RANDOM_MIN, RANDOM_MAX);
        size_t ctr[2]{ 0u, 0u };
        while (ctr[0] < cnt || ctr[1] < cnt)
        {
            switch (getRandom(0u, 1u))
            {
            case 0:
                if (ctr[0] < cnt)
                {
                    r->grab();
                    ctr[0]++;
                }
                break;
            case 1:
                if (ctr[1] < ctr[0])
                {
                    r->drop();
                    ctr[1]++;
                }
                break;
            }
        }
        r->drop();
    }
}//omp parallel
#ifdef TEST_COUNT_AFTER
    assert(r->getReferenceCount() == 1);
    assert(r->drop());
#endif
}
