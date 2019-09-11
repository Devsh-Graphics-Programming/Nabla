// Copyright (C) 2014 Mateusz 'DevSH' Kielan
// This file is part of the "Irrlicht Engine".
// Contributed from "Build a World"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_SPLINES_H_INCLUDED__
#define __IRR_SPLINES_H_INCLUDED__

#include "IrrCompileConfig.h"
#include <cmath>       /* sqrt */
#include "vectorSIMD.h"

#include <vector>


namespace irr
{
namespace core
{


class ISpline// : public AllocationOverrideDefault
{
    public:
        virtual ~ISpline() {}

        //
        virtual bool        isLooping() const {return isLoop;}
        virtual size_t      getSegmentCount() const = 0;
        virtual float       getSplineLength() const = 0;

        //
        virtual void        getSegmentLengths(float* outSegLens) const = 0;
        virtual float       getSegmentLength(const uint32_t& segmentID) const = 0;
        virtual float       getSegmentParameterRange(const uint32_t& segmentID) const = 0;

        //get position
        //this function returns the id of the segment you might have moved into
        virtual uint32_t    getPos(vectorSIMDf& pos, float& distanceAlongSeg, const uint32_t& segmentID, float* paramHint=NULL, const float& accuracyThresh=0.00390625f) const = 0;
        virtual bool        getPos_fromParameter(vectorSIMDf& pos, const uint32_t& segmentID, const float& parameter) const = 0;

        //to get direction to look in
        virtual bool        getUnnormDirection(vectorSIMDf& tan, const uint32_t& segmentID, const float& distanceAlongSeg) const = 0;
        virtual bool        getUnnormDirection_fromParameter(vectorSIMDf& tan, const uint32_t& segmentID, const float& parameter) const = 0;

        //baw specific
        virtual bool      canGiveParameterUntilBlockChange() const {return false;}
        // pass in current position
        virtual float           getParameterUntilBlockChange(const uint32_t& segmentID, const float& param) = 0;
        virtual core::vector<float>   getBlockChangesInSegment(const uint32_t& segmentID, float startParam=0.f) = 0;

        //is the distance and parameter the same?
        virtual bool            isArcLengthPrecise() const = 0;
        ///virtual float           parameterToDistance(const float& param) const = 0;
        ///virtual float           distanceToParameter(const float& dist) const = 0;
    protected:
        ISpline(bool loop) : isLoop(loop) {}

        const bool isLoop;
    private:
};


class CLinearSpline : public ISpline
{
    public:
        CLinearSpline(vectorSIMDf* controlPoints, const size_t& count, const bool loop = false) : ISpline(loop)
        {
            //assert(count<0x80000000u && count);
            for (size_t i=1; i<count; i++)
                segments.push_back(Segment(controlPoints[i-1],controlPoints[i]));

            if (isLoop)
                segments.push_back(Segment(controlPoints[count-1],controlPoints[0]));
            finalize();
        }
        CLinearSpline(vectorSIMDf* controlPoints, float* customDistances, const size_t& count, const bool loop = false) : ISpline(loop)
        {
            //assert(count<0x80000000u);
            for (size_t i=1; i<count; i++)
                segments.push_back(Segment(controlPoints[i-1],controlPoints[i],customDistances[i-1]));

            if (isLoop)
                segments.push_back(Segment(controlPoints[count-1],controlPoints[0],customDistances[count-1]));
            finalize();
        }

        //
        virtual size_t      getSegmentCount() const
        {
            return segments.size();
        }
        virtual float       getSplineLength() const
        {
            return splineLen;
        }

        //
        virtual void        getSegmentLengths(float* outSegLens) const
        {
            for (size_t i=0; i<segments.size(); i++)
                outSegLens[i] = segments[i].length;
        }
        virtual float       getSegmentLength(const uint32_t& segmentID) const
        {
            //assert(segmentID<segments.size());
            return segments[segmentID].length;
        }
        virtual float       getSegmentParameterRange(const uint32_t& segmentID) const
        {
            return getSegmentLength(segmentID);
        }

        //get position
        //this function returns the id of the segment you might have moved into - 0xdeadbeefu is an error code
        virtual uint32_t    getPos(vectorSIMDf& pos, float& distanceAlongSeg, const uint32_t& segmentID, float* paramHint=NULL, const float& accuracyThresh=0.00390625f) const
        {
            if (distanceAlongSeg<0.f)
                return 0xdeadbeefu;

            uint32_t actualSeg;
            if (isLoop)
            {
                actualSeg = segmentID%segments.size();
                while (distanceAlongSeg>=segments[actualSeg].length)
                {
                    distanceAlongSeg -= segments[actualSeg].length;
                    actualSeg++;
                    if (actualSeg==segments.size())
                        actualSeg = 0;
                }
            }
            else
            {
                if (segmentID>=segments.size())
                    return 0xdeadbeefu;

                actualSeg = segmentID;
                while (distanceAlongSeg>=segments[actualSeg].length)
                {
                    distanceAlongSeg -= segments[actualSeg].length;
                    actualSeg++;
                    if (actualSeg==segments.size())
                        return 0xdeadbeefu;
                }
            }

            pos = segments[actualSeg].posHelper(distanceAlongSeg);
            if (paramHint)
                *paramHint = distanceAlongSeg;

            return actualSeg;
        }
        virtual bool        getPos_fromParameter(vectorSIMDf& pos, const uint32_t& segmentID, const float& parameter) const
        {
            if (segmentID>=segments.size()||parameter>segments[segmentID].length)
                return false;

            pos = segments[segmentID].posHelper(parameter);

            return true;
        }

        //to get direction to look in
        virtual bool        getUnnormDirection(vectorSIMDf& tan, const uint32_t& segmentID, const float& distanceAlongSeg) const
        {
            if (segmentID>=segments.size()||distanceAlongSeg>segments[segmentID].length)
                return false;

            tan = segments[segmentID].directionHelper(distanceAlongSeg);

            return true;
        }

        virtual bool        getUnnormDirection_fromParameter(vectorSIMDf& tan, const uint32_t& segmentID, const float& parameter) const
        {
            return getUnnormDirection(tan,segmentID,parameter);
        }

        //baw specific
        virtual bool      canGiveParameterUntilBlockChange() const {return true;}
        // pass in current position
        virtual float           getParameterUntilBlockChange(const uint32_t& segmentID, const float& param)
        {
            if (segmentID>=segments.size()||param>=segments[segmentID].length)
                return -1.f;

            return segments[segmentID].findNextBlockChange(param);
        }
        virtual core::vector<float>   getBlockChangesInSegment(const uint32_t& segmentID, float startParam=0.f)
        {
            core::vector<float> changes;
            if (segmentID>=segments.size())
                return changes;

            const Segment& seg = segments[segmentID];
            while (true)
            {
                float fnd = seg.findNextBlockChange(startParam);
                if (fnd<0.f)
                    return changes;

                startParam = fnd;
                reinterpret_cast<uint32_t&>(startParam)++;
                changes.push_back(fnd);
            }
        }


        virtual bool            isArcLengthPrecise() const {return true;}
        ///virtual float           parameterToDistance(const float& param) const {return param;}
        ///virtual float           distanceToParameter(const float& dist) const {return dist;}
    private:
        void finalize()
        {
            double lenDouble = 0;
            for (size_t i=0; i<segments.size(); i++)
            {
                lenDouble += segments[i].length;
            }
            splineLen = lenDouble;
        }


        struct Segment
        {
            Segment(const vectorSIMDf& startPt,const vectorSIMDf& endPt)
            {
                weights[0] = endPt-startPt;
                length = weights[0].getLengthAsFloat();
                weights[0] /= length;

                weights[1] = startPt;
            }
            Segment(const vectorSIMDf& startPt,const vectorSIMDf& endPt, const float& customLen)
            {
                weights[0] = (endPt-startPt);
                length = customLen;
                weights[0] /= length;

                weights[1] = startPt;
            }

            inline vectorSIMDf posHelper(const float& distanceAlongSeg) const
            {
                return weights[0]*distanceAlongSeg+weights[1];
            }
            inline vectorSIMDf directionHelper(const float& distanceAlongSeg) const
            {
                return weights[0];
            }
            inline float findNextBlockChange(const float& param) const
            {
                vectorSIMDf startingNegFrac = posHelper(param);
                startingNegFrac = floor(startingNegFrac)-startingNegFrac;
                vectorSIMDf dir = directionHelper(param);
                float changes[3];
                for (uint32_t i=0; i<3; i++)
                    changes[i] = findChange(startingNegFrac.pointer[i],dir.pointer[i]);

                float smallest;
                if (reinterpret_cast<uint32_t*>(changes)[0]<=reinterpret_cast<uint32_t*>(changes)[1])
                {
                    if (reinterpret_cast<uint32_t*>(changes)[2]<=reinterpret_cast<uint32_t*>(changes)[0])
                        smallest = changes[2];
                    else
                        smallest = changes[0];
                }
                else if (reinterpret_cast<uint32_t*>(changes)[2]<=reinterpret_cast<uint32_t*>(changes)[1])
                {
                    smallest = changes[2];
                }
                else
                    smallest = changes[1];

                smallest += param;
                if (smallest<length)
                    return smallest;

                return -1.f;
            }
            inline float findChange(const float& currentNegFrac, const float& changePerParam) const
            {
                if (currentNegFrac==0.f||currentNegFrac==1.f)
                    return 0.f;

                if (changePerParam < -FLT_MIN)
                    return currentNegFrac/changePerParam;
                else if (changePerParam > FLT_MIN)
                    return (1.f+currentNegFrac)/changePerParam;

                return -1.f;
            }

            float length;
            vectorSIMDf weights[2];
        };

        core::vector<Segment> segments;
        float splineLen;
};


//! Loop code is wrong for now, unable to calculate A_0 so the gradients match all the way around the loop
class CQuadraticSpline : public ISpline
{
    public:
        CQuadraticSpline(vectorSIMDf* controlPoints, const size_t& count, const bool loop = false, float tightness = 1.1107f) : ISpline(loop)
        {
			assert(count>1ull);
            vectorSIMDf startGradient;
            if (isLoop)
				startGradient = controlPoints[1]-controlPoints[count-1ull];
            else
				startGradient = controlPoints[1]-controlPoints[0];
         
			float currentApproxLen = (controlPoints[1]-controlPoints[0]).getLengthAsFloat();
			segments.push_back(Segment(controlPoints[0], controlPoints[1], startGradient, currentApproxLen, tightness));

            for (size_t i=2; i<count; i++)
            {
                vectorSIMDf startPt = controlPoints[i-1ull];
                vectorSIMDf endPt = controlPoints[i];
                currentApproxLen = (endPt-startPt).getLengthAsFloat();
                segments.push_back(Segment(segments.back(),startPt,endPt,currentApproxLen, tightness));
            }

			if (isLoop)
			{
				vectorSIMDf startPt = controlPoints[count-1ull];
				vectorSIMDf endPt = controlPoints[0];
				currentApproxLen = (endPt-startPt).getLengthAsFloat();
				segments.push_back(Segment(segments.back(),startPt,endPt,currentApproxLen, tightness));
			}

            finalize();
        }

        //
        virtual size_t      getSegmentCount() const
        {
            return segments.size();
        }
        virtual float       getSplineLength() const
        {
            return splineLen;
        }

        //
        virtual void        getSegmentLengths(float* outSegLens) const
        {
            for (size_t i=0; i<segments.size(); i++)
                outSegLens[i] = segments[i].length;
        }
        virtual float       getSegmentLength(const uint32_t& segmentID) const
        {
            //assert(segmentID<segments.size());
            return segments[segmentID].length;
        }
        virtual float       getSegmentParameterRange(const uint32_t& segmentID) const
        {
            return segments[segmentID].parameterLength;
        }

        //get position
        //this function returns the id of the segment you might have moved into - 0xdeadbeefu is an error code
        virtual uint32_t    getPos(vectorSIMDf& pos, float& distanceAlongSeg, const uint32_t& segmentID, float* paramHint=NULL, const float& accuracyThresh=0.00390625f) const
        {
            if (distanceAlongSeg<0.f)
                return 0xdeadbeefu;

            uint32_t actualSeg;
            if (isLoop)
            {
                actualSeg = segmentID%segments.size();
                while (distanceAlongSeg>=segments[actualSeg].length)
                {
                    distanceAlongSeg -= segments[actualSeg].length;
                    actualSeg++;
                    if (actualSeg==segments.size())
                        actualSeg = 0;
                }
            }
            else
            {
                if (segmentID>=segments.size())
                    return 0xdeadbeefu;

                actualSeg = segmentID;
                while (distanceAlongSeg>=segments[actualSeg].length)
                {
                    distanceAlongSeg -= segments[actualSeg].length;
                    actualSeg++;
                    if (actualSeg==segments.size())
                        return 0xdeadbeefu;
                }
            }

            if (paramHint)
            {
                if (actualSeg!=segmentID)
                    *paramHint = -1.f;

                *paramHint = segments[actualSeg].getParameterFromArcLen(distanceAlongSeg,*paramHint,accuracyThresh);
				//assert(*paramHint < segments[actualSeg].parameterLength);
                pos = segments[actualSeg].posHelper(*paramHint);
            }
            else
                pos = segments[actualSeg].posHelper(segments[actualSeg].getParameterFromArcLen(distanceAlongSeg,-1.f,accuracyThresh));

            return actualSeg;
        }
        virtual bool        getPos_fromParameter(vectorSIMDf& pos, const uint32_t& segmentID, const float& parameter) const
        {
            if (segmentID>=segments.size()||parameter>segments[segmentID].parameterLength)
                return false;

            pos = segments[segmentID].posHelper(parameter);

            return true;
        }

        //to get direction to look in
        virtual bool        getUnnormDirection(vectorSIMDf& tan, const uint32_t& segmentID, const float& distanceAlongSeg) const
        {
            if (segmentID>=segments.size()||distanceAlongSeg>segments[segmentID].length)
                return false;

            tan = segments[segmentID].directionHelper(segments[segmentID].getParameterFromArcLen(distanceAlongSeg,-1.f,0.00390625f));

            return true;
        }
        virtual bool        getUnnormDirection_fromParameter(vectorSIMDf& tan, const uint32_t& segmentID, const float& parameter) const
        {
            if (segmentID>=segments.size()||parameter>segments[segmentID].parameterLength)
                return false;

            tan = segments[segmentID].directionHelper(parameter);

            return true;
        }

        //baw specific -- to be implemented later
        virtual bool      canGiveParameterUntilBlockChange() const {return false;}
        // pass in current position
        virtual float           getParameterUntilBlockChange(const uint32_t& segmentID, const float& param)
        {
            //if (segmentID>=segments.size()||param>=segments[segmentID].parameterLength)
                return -1.f;

            //return segments[segmentID].findNextBlockChange(param);
        }
        virtual core::vector<float>   getBlockChangesInSegment(const uint32_t& segmentID, float startParam=0.f)
        {
            core::vector<float> changes;
            //if (segmentID>=segments.size())
                return changes;
/*
            const Segment& seg = segments[segmentID];
            while (true)
            {
                float fnd = seg.findNextBlockChange(startParam);
                if (fnd<0.f)
                    return changes;

                startParam = fnd;
                reinterpret_cast<uint32_t&>(startParam)++;
                changes.push_back(fnd);
            }*/
        }


        virtual bool            isArcLengthPrecise() const {return true;}/**
        virtual float           parameterToDistance(const float& param) const
        {
        }
        virtual float           distanceToParameter(const float& dist) const
        {
        }**/

    protected:
        CQuadraticSpline(bool loop) : ISpline(loop) {}

        void finalize()
        {
            double lenDouble = 0;
            for (size_t i=0; i<segments.size(); i++)
            {
                lenDouble += segments[i].length;
            }
            splineLen = lenDouble;
        }


        class Segment
        {
            public:
				Segment(const vectorSIMDf& startPt, const vectorSIMDf& endPt, const vectorSIMDf& startGradient, float currentApproxLen, float tightness)
				{
					/// ad^2+bd+y0 = y1
					weights[2] = startPt;
					/// ad+b = (y1-y0)/d = w
					auto approxMidGradient = (endPt-startPt)/currentApproxLen;

					float K;
					auto comparelambda = [&]() -> bool
					{
						/// The differential 2ad+b is continuous
						/// b = Kf'(0)
						weights[1] = startGradient*K;
						/// ad+Kf'(0) = w
						/// a = (w-Kf'(0))/d
						weights[0] = (approxMidGradient - weights[1]) / currentApproxLen;
						finalize(currentApproxLen);
						return this->length < tightness*currentApproxLen;
					};

					uint32_t& it = reinterpret_cast<uint32_t&>(K);
					uint32_t first = 0u;
					int32_t count = 0x47800000;
					while (count > 0)
					{
						it = first;
						uint32_t step = count / 2u;
						it += step;
						if (comparelambda())
						{
							first = ++it;
							count -= step + 1;
						}
						else
							count = step;
					}
					it = first;
					comparelambda();
				}
				Segment(const Segment& previousSeg, const vectorSIMDf& startPt,const vectorSIMDf& endPt, float currentApproxLen, float tightness) :
					Segment(startPt,endPt,previousSeg.weights[0]*previousSeg.parameterLength*2.f+previousSeg.weights[1],currentApproxLen, tightness)
				{
				}
				static Segment createForBSpline(const vectorSIMDf& startPt, const vectorSIMDf& midCtrl, const vectorSIMDf& endPt)
				{
					Segment seg;
					seg.weights[2] = startPt;
					seg.weights[1] = (midCtrl-startPt)*2.f;
					seg.weights[0] = endPt+startPt-midCtrl*2.f;

					float bLen = dot(seg.weights[1],seg.weights[1]).X;
					if (bLen<0.000001f)
					{
						seg.weights[1] = endPt-startPt;
						seg.weights[0].set(0.f,0.f,0.f);
					}
					else if (std::abs(dot(seg.weights[2],seg.weights[1]).X)>std::sqrt(bLen*dot(seg.weights[2],seg.weights[2]).X)*0.999999f)
					{
						seg.weights[1] = endPt-startPt;
						seg.weights[0].set(0.f,0.f,0.f);
					}

					seg.finalize(1.f);
					return seg;
				}

				inline void finalize(const float &currentApproxLen)
				{
					parameterLength = currentApproxLen;

					lenASq       = dot(weights[0],weights[0]).x;
					double lenCSq= dot(weights[1],weights[1]).x;
					lenC         = std::sqrt(lenCSq);
					if (std::abs(lenASq)>0.000001f) //2.f*sqrt(lenASq)*lenC+term_b
					{
						/// integral sqrt(a x^2 + b x + c) dx =
						/// ((2 a x + b) sqrt(x (a x + b) + c))/(4 a) - ((b^2 - 4 a c) log(2 sqrt(a) sqrt(x (a x + b) + c) + 2 a x + b))/(8 a^(3/2))
						/// @ x=0
						/// (b sqrt(c))/(4 a) - ((b^2 - 4 a c) log(2 sqrt(a) sqrt(c) + b))/(8 a^(3/2))
						double term_b       = 4.f*dot(weights[1],weights[0]).x;
						double lenASq_4     = lenASq*16.f;
						double lenA_2       = std::sqrt(lenASq_4);

						double lenBSq       = dot(weights[1],weights[1]).x;

						/// integral sqrt(a x^2 + b x + c) dx =
						/// ((2 a x + b) sqrt(x (a x + b) + c))/(4 a) - ((b^2 - 4 a c) log(2 sqrt(a) sqrt(x (a x + b) + c) + 2 a x + b))/(8 a^(3/2))
						/// ((0.5 x + b/4a) sqrt(x (a x + b) + c)) - ((b*b/4a - c) log(2 sqrt(a) sqrt(x (a x + b) + c) + 2 a x + b))/(2 a^(1/2))
						/// differential
						/// ((2 a x + b) sqrt(x (a x + b) + c))/(4 a) - ((b^2 - 4 a c) log(2 sqrt(a) sqrt(x (a x + b) + c) + 2 a x + b))/(8 a^(3/2))
						arcCalcConstants[0] = term_b/lenASq_4;
						arcCalcConstants[1] = lenASq*4.f;
						arcCalcConstants[2] = term_b;
						arcCalcConstants[3] = lenCSq;
						arcCalcConstants[4] = (arcCalcConstants[3]-term_b*arcCalcConstants[0])/lenA_2;
						arcCalcConstants[5] = lenA_2;

						//lowerIntegralValue      = 0.f;
						//lowerIntegralValue      = getArcLenFromParameter(0.f);
						lowerIntegralValue      = arcCalcConstants[0]*lenC+arcCalcConstants[4]*logf(arcCalcConstants[5]*lenC+term_b);
						if (!isnan(lowerIntegralValue) && !isinf(lowerIntegralValue))
						{
							length = getArcLenFromParameter(parameterLength);
							return;
						}
					}

					length = lenC*parameterLength;
					lenC_reciprocal = reciprocal(lenC);
					lowerIntegralValue = NAN;
				}

				inline float getArcLenFromParameter(const float &parameter) const
				{
#ifdef _IRR_DEBUG
					assert(std::abs(lenASq)>0.000001f);
					assert(!isnan(lowerIntegralValue));
#endif
					double ax = arcCalcConstants[1]*parameter;
					double ax_b = ax+arcCalcConstants[2];
					double theSquareRoot = std::sqrt(parameter*ax_b+arcCalcConstants[3]);
					float higherIntTerm    = (0.5f*parameter+arcCalcConstants[0])*theSquareRoot+arcCalcConstants[4]*log(arcCalcConstants[5]*theSquareRoot+ax_b+ax);
	/*
					double a = dot(weights[0],weights[0]).x;
					double b = dot(weights[0],weights[1]).x*2.f;
					double c = dot(weights[1],weights[1]).x;
					double higherIntTerm = ((2.f* a* parameter + b)*std::sqrt(parameter* (a *parameter + b) + c))/(4.f* a) - ((b*b - 4*a*c)*log(2.f*std::sqrt(a)*std::sqrt(parameter*(a*parameter + b) + c) + 2.f*a*parameter + b))/(8.f*a*std::sqrt(a));
	*/
	/*
					/// extreme debug
					double checkLen = 0.0;
					vectorSIMDf prevPoint = weights[2];
					for (size_t i=1; i<=1024*16; i++)
					{
						double tmp = double(i)/double(1024*16);
						tmp *= parameter;
						vectorSIMDf nextPoint = posHelper(tmp);
						checkLen += (prevPoint - nextPoint).getLengthAsFloat();
						prevPoint = nextPoint;
					}
					float diff = std::abs(higherIntTerm-lowerIntegralValue-checkLen);
					assert(diff<0.001f);
	*/
					return higherIntTerm-lowerIntegralValue;
				}

				inline float getParameterFromArcLen(const float& arcLen, float parameterHint, const float& accuracyThresh) const
				{
					if (!isnan(lowerIntegralValue))
					{
						if (arcLen<=accuracyThresh)
							return arcLen;
						if (arcLen>=length-accuracyThresh)
							return parameterLength;
						if (parameterHint<0.f||parameterHint>parameterLength)
							parameterHint = parameterLength*(arcLen/length);
						/// dist = IndefInt(param) - lowerIntVal
						/// IndefInt^-1(dist+lowerIntVal) = param
						/// Newton-Raphson      f = arcLen - getArcLenFromParameter(parameterHint);
						/// Newton-Raphson      f' = -getArcLenFromParameter'(parameterHint);
						float arcLenDiffAtParamGuess = arcLen-getArcLenFromParameter(parameterHint);
						for (size_t i=0; std::abs(arcLenDiffAtParamGuess)>accuracyThresh&&i<32; i++)
						{
							float differentialAtGuess = directionHelper(parameterHint).getLengthAsFloat();
							parameterHint = parameterHint+arcLenDiffAtParamGuess/differentialAtGuess;
							arcLenDiffAtParamGuess = arcLen-getArcLenFromParameter(parameterHint);
						}
						return std::min(parameterHint,parameterLength);
					}
					else
						return arcLen*lenC_reciprocal;
				}

				inline vectorSIMDf posHelper(const float& parameter) const
				{
					return (weights[0]*parameter+weights[1])*parameter+weights[2];
				}
				inline vectorSIMDf directionHelper(const float& parameter) const
				{
					return weights[0]*parameter*2.f+weights[1];
				}/*
				inline float findNextBlockChange(const float& param) const
				{
					vectorSIMDf startingNegFrac = posHelper(param);
					startingNegFrac = floor(startingNegFrac)-startingNegFrac;
					vectorSIMDf dir = directionHelper(param);
					float changes[3];
					for (uint32_t i=0; i<3; i++)
						changes[i] = findChange(startingNegFrac.pointer[i],dir.pointer[i]);

					float smallest;
					if (reinterpret_cast<uint32_t*>(changes)[0]<=reinterpret_cast<uint32_t*>(changes)[1])
					{
						if (reinterpret_cast<uint32_t*>(changes)[2]<=reinterpret_cast<uint32_t*>(changes)[0])
							smallest = changes[2];
						else
							smallest = changes[0];
					}
					else if (reinterpret_cast<uint32_t*>(changes)[2]<=reinterpret_cast<uint32_t*>(changes)[1])
					{
						smallest = changes[2];
					}
					else
						smallest = changes[1];

					smallest_+= param;
					if (smallest<length)
						return smallest;

					return -1.f;
				}
				inline float findChange(const float& currentNegFrac, const float& changePerParam) const
				{
					if (currentNegFrac==0.f||currentNegFrac==1.f)
						return 0.f;

					if (changePerParam < -FLT_MIN)
						return currentNegFrac/changePerParam;
					else if (changePerParam > FLT_MIN)
						return (1.f+currentNegFrac)/changePerParam;

					return -1.f;
				}*/


				float length,parameterLength,lenASq,lowerIntegralValue;
				union
				{
					float lenC;
					float lenC_reciprocal;
				};
				float arcCalcConstants[6];
				vectorSIMDf weights[3];

            private:
                Segment() {}
        };

        core::vector<Segment> segments;
        float splineLen;
};


//! Loop code is wrong for now, unable to calculate A_0 so the gradients match all the way around the loop
class CQuadraticBSpline : public CQuadraticSpline
{
    public:
        CQuadraticBSpline(vectorSIMDf* controlPoints, const size_t& count, const bool loop = false) : CQuadraticSpline(loop)
        {
            //assert(count<0x80000000u && count);
            vectorSIMDf firstMidpoint = (controlPoints[0]+controlPoints[1])*0.5f;
            if (isLoop)
            {
                segments.push_back(Segment::createForBSpline((controlPoints[count-1]+controlPoints[0])*0.5f,controlPoints[0],firstMidpoint));
            }
            else
            {
                segments.push_back(Segment::createForBSpline(controlPoints[0],controlPoints[0],firstMidpoint));
            }

            vectorSIMDf midpoint = firstMidpoint;
            vectorSIMDf lastMid = midpoint;
            for (size_t i=2; i<count; i++)
            {
                midpoint = (controlPoints[i-1]+controlPoints[i])*0.5f;
                segments.push_back(Segment::createForBSpline(lastMid,controlPoints[i-1],midpoint));
                lastMid = midpoint;
            }

            if (isLoop)
            {
                segments.push_back(Segment::createForBSpline(lastMid,controlPoints[count-1],segments[0].weights[2]));
            }
            else
            {
                segments.push_back(Segment::createForBSpline(lastMid,lastMid,controlPoints[count-1]));
            }

            finalize();
        }
};

}
}

#endif
