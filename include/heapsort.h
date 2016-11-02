// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_HEAPSORT_H_INCLUDED__
#define __IRR_HEAPSORT_H_INCLUDED__

#include "irrTypes.h"

namespace irr
{
namespace core
{

//! Sinks an element into the heap.
template<class T>
inline void heapsink(T*array, s32 element, s32 max)
{
	while ((element<<1) < max) // there is a left child
	{
		s32 j = (element<<1);

		if (j+1 < max && array[j] < array[j+1])
			j = j+1; // take right child

		if (array[element] < array[j])
		{
			T t = array[j]; // swap elements
			array[j] = array[element];
			array[element] = t;
			element = j;
		}
		else
			return;
	}
}


//! Sorts an array with size 'size' using heapsort.
template<class T>
inline void heapsort(T* array_, s32 size)
{
	// for heapsink we pretent this is not c++, where
	// arrays start with index 0. So we decrease the array pointer,
	// the maximum always +2 and the element always +1

	T* virtualArray = array_ - 1;
	s32 virtualSize = size + 2;
	s32 i;

	// build heap

	for (i=((size-1)/2); i>=0; --i)
		heapsink(virtualArray, i+1, virtualSize-1);

	// sort array, leave out the last element (0)
	for (i=size-1; i>0; --i)
	{
		T t = array_[0];
		array_[0] = array_[i];
		array_[i] = t;
		heapsink(virtualArray, 1, i + 1);
	}
}
/*
template<size_t elementBytes>
inline void radixsort(void* array, const size_t& count, const size_t bucketBits)
{
    if (!bucketBits)
        return;

    const size_t kHist = 0x1<<bucketBits;
	size_t histogram[kHist];

	void* arrayTmp = malloc(count*elementBytes);
	void* ptrIn = array;
	void* ptrOut = arrayTmp;

    const size_t numberOfPasses = (8+bucketBits-1)/bucketBits;
    for (size_t i=0; i<numberOfPasses; i++)
    {
        memset(histogram,0,kHist*sizeof(size_t));

        for (size_t j=0; j<count; j++)
        {
            size_t fi = reinterpret_cast<uint8_t*>(ptrIn)[j]>>(elementBytes-i*bucketBits);
            histogram[fi]++;
        }

        size_t sum=0;
        for (size_t j=0; j<kHist; j++)
        {
            size_t tsum = histogram[j]+sum;
            histogram[j] = sum-1;
            sum = tsum;
        }

        for (size_t j=0; j<count; j++)
        {
            uint8_t key = reinterpret_cast<uint8_t*>(ptrIn)[j];

            size_t fi = key>>(elementBytes-i*bucketBits);
            reinterpret_cast<uint8_t*>(ptrOut)[++histogram[fi]] = key;
        }
    }
}

template<> inline void radixsort<1>(void* array, const size_t& size, const size_t bucketBits)
{
}
*/
} // end namespace core
} // end namespace irr

#endif

