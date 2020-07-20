#include "DetectionRegion.h"
#include <math.h>

DetectionRegion::DetectionRegion() : mXCenter(0.0f), mYCenter(0.0f), mBBXStart(0.0f), mBBYStart(0.0f), mBBWidth(0.0f), mBBHeight(0.0f), mScale(1.0), 
mConfidence(0.0f), mAppearanceCalculated(false), mClusterIndex(-1), mDetectionType(DetectionRegion::DETECTED)
{

}

DetectionRegion::DetectionRegion( const float x, const float y, const float scale) :
				mXCenter(x), mYCenter(y), mBBXStart(x), mBBYStart(y), mBBWidth(0.0f), mBBHeight(0.0f), mConfidence(0.0f), mScale(scale), 
				mDetectionType(DetectionRegion::DETECTED), mAppearanceCalculated(false), mClusterIndex(-1) 
{

}

DetectionRegion::DetectionRegion( const float xStart, const float yStart, const float width, const float height, const float scale) :
				mConfidence(0.0f), mScale(scale),mAppearanceCalculated(false), mClusterIndex(-1), mBBXStart(xStart), mBBYStart(yStart), 
				mBBWidth(width), mBBHeight(height), mDetectionType(DetectionRegion::DETECTED)
{
	mXCenter = xStart + width / 2.0;
	mYCenter = yStart + height / 2.0;
}

DetectionRegion::DetectionRegion(std::vector<DetectionRegion::Point> *points, const float scale) : 
				mConfidence(0.0f), mScale(scale),mAppearanceCalculated(false), mClusterIndex(-1), mDetectionType(DetectionRegion::DETECTED)
{
	mPointList = *points;
	updateBBFromPointList();
	mXCenter = mBBXStart + mBBWidth / 2.0;
	mYCenter = mBBYStart + mBBHeight / 2.0;
}

void DetectionRegion::clearParameters()
{
	//while(mParameters.size() > 0)
	//{
	//	std::list <RegionParameters *>::iterator it = mParameters.begin();
	//	RegionParameters * param = *it; 
	//	if(param)	delete param;
	//	mParameters.erase(it);
	//}
	mParameters.clear();
}

int DetectionRegion::addParameter(const std::string &keyvalue, const std::string &value)
{
	//RegionParameters * param = new RegionParameters;
	//param->first = keyvalue;
	//param->second = value;
	int i = 0;
	for(std::list <RegionParameters>::iterator iter = mParameters.begin(); iter != mParameters.end(); iter++) {
		if (iter->first == keyvalue) {
			iter->second = value;
			return i;
		}
		++i;
	}

	mParameters.push_back(RegionParameters(keyvalue, value));
	return mParameters.size()-1;
}

bool DetectionRegion::getParameter(int index, std::string &keyvalue, std::string &value)
{
	int i = 0;
	for(std::list <RegionParameters>::iterator iter = mParameters.begin(); iter != mParameters.end(); iter++)
	{		
		if(index == i)
		{
			keyvalue = iter->first;
			value = iter->second;
			return true;
		}
		i++;
	}
	return false;
}

bool DetectionRegion::getParameter(const std::string &keyvalue,std::string &value)
{
	std::list <RegionParameters>::iterator iter;

	iter = mParameters.begin(); 
	while (iter != mParameters.end() && iter->first != keyvalue)
		iter++;
	if (iter != mParameters.end())
	{
		value = iter->second;
		return true;
	}
	else
		return false;
}

void DetectionRegion::updateBBFromPointList()
{
	float xMin, yMin, xMax, yMax;
	xMin = mPointList[0].x;
	yMin = mPointList[0].y;
	xMax = mPointList[0].x;
	yMax = mPointList[0].y;
	for (int i = 1; i < mPointList.size(); i++)
	{
		if (mPointList[i].x < xMin)
			xMin = mPointList[i].x;
		if (mPointList[i].y < yMin)
			yMin = mPointList[i].y;
		if (mPointList[i].x > xMax)
			xMax = mPointList[i].x;
		if (mPointList[i].y > yMax)
			yMax = mPointList[i].y;
	}
	mBBXStart = xMin;
	mBBYStart = yMin;
	mBBWidth = xMax - xMin;
	mBBHeight = yMax - yMin;
}

void DetectionRegion::copyFromRegion(const DetectionRegion *region, bool copyAppearance)
{
	mXCenter = region->mXCenter;
	mYCenter = region->mYCenter;
	mBBWidth = region->mBBWidth;
	mBBHeight = region->mBBHeight;
	mBBXStart = region->mBBXStart;
	mBBYStart = region->mBBYStart; 
	mPointList = region->mPointList;
	mDetectionType = region->mDetectionType;
	mConfidence = region->mConfidence;
	mScale = region->mScale;

	mClusterIndex = region->mClusterIndex;
	mAppearanceCalculated = region->mAppearanceCalculated;

	mParameters = region->mParameters;
}

float DetectionRegion::matchDistance( const DetectionRegion &region )
{
	float fDist = sqrt( ( mXCenter - region.mXCenter ) * ( mXCenter - region.mXCenter ) + ( mYCenter - region.mYCenter ) * ( mYCenter - region.mYCenter ) );

	return fDist;
}

float DetectionRegion::mOverlap(float x1, float w1, float x2, float w2)
{
	float l1 = x1 - w1 / 2;
	float l2 = x2 - w2 / 2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1 / 2;
	float r2 = x2 + w2 / 2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

float DetectionRegion::getIou(const DetectionRegion &region)
{
	float x, y, width, height, i;

	((DetectionRegion)region).getBoundingBox(x, y, width, height);
	float w = mOverlap(mBBXStart, mBBWidth, x, width);
	float h = mOverlap(mBBYStart, mBBHeight, y, height);
	if (w < 0 || h < 0)
		i = 0;
	else
		i = w*h;

	float u = mBBWidth*mBBHeight + width*height - i;


	return i / u; // getIntersection(region) / getUnion(region);
}

void DetectionRegion::shiftRegion( const float x, const float y , const float scale)
{
	std::vector<Point>::iterator iter;
	for (iter = mPointList.begin(); iter != mPointList.end(); iter++)
	{
		iter->x = mXCenter + (iter->x - mXCenter) * scale + x;
		iter->y = mYCenter + (iter->y - mYCenter) * scale + y;
	}

	mXCenter += x;
	mYCenter += y;

	mBBXStart += x + mBBWidth*0.5f*(1.0 - scale);
	mBBYStart += y + mBBHeight*0.5f*(1.0 - scale);
	mBBWidth *= scale;
	mBBHeight *= scale;

	mAppearanceCalculated = false;
}

bool DetectionRegion::pointWithinRegion( const float x, const float y  )
{
	if( ( x > mBBXStart ) && ( x < mBBXStart + mBBWidth ) && ( y > mBBYStart ) && ( y < mBBYStart + mBBHeight ) ) 
		return true;
	else
		return false;
}