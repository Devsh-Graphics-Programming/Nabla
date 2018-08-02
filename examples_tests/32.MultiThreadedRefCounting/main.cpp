#include <irrlicht.h>
#include <omp.h>
#include <IReferenceCounted.h>
#include <cassert>
#include <random>

class RefCounted : public irr::IReferenceCounted {};

#define TEST_COUNT_AFTER
int main()
{
    auto r = new RefCounted();
    omp_set_num_threads(256);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i<256; i++)
    {
        r->grab();

#ifndef TEST_COUNT_AFTER
        if (omp_get_thread_num() == 0)
        {
            r->drop();
        }
#endif TEST_COUNT_AFTER

//#pragma omp barrier
//cannot be nested in parallel for

        const size_t cnt = rand();
        size_t ctr[2]{ 0u, 0u };
        while (ctr[0] < cnt || ctr[1] < cnt)
        {
            switch (rand()%2)
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
#ifdef TEST_COUNT_AFTER
    assert(r->getReferenceCount() == 1);
    assert(r->drop());
#endif
}