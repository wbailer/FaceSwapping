#include "FaceDetectionRegion.h"
#include <math.h>
#include <iostream>
#include <fstream>

int FaceDetectionRegion::mVersion = 100;

FaceDetectionRegion::FaceDetectionRegion()
{
	reset();
	mRegionImage = NULL;
}

FaceDetectionRegion::FaceDetectionRegion(const DetectionRegion *region)
{
	copyFromRegion(region);
}

FaceDetectionRegion::~FaceDetectionRegion()
{

}

void FaceDetectionRegion::reset()
{
	mFaceID = -1;
	mIsReliableClassified = false;
	mIsReliableDetected = true;
	mOriginTrajectoryIndex = -1;
	mAutoTrainCandidateNo = 0; // the default value indicating that there has no candidate number been set
	mTensorDetNumb = 0;
	mFaceOutOfImg = false;
	mIsFeatureVectSet = false;
	mSharpness = -1;
	mFaceOutOfImg = false;

	clearParameters();
	// mRegionImage can be reused
}

void FaceDetectionRegion::setBoundingBox(const float x, const float y, const float width, const float height)
{
	mBBXStart = x; mBBYStart = y; mBBWidth = width; mBBHeight = height;
	mXCenter = mBBXStart + mBBWidth / 2.0f;
	mYCenter = mBBYStart + mBBHeight / 2.0f;
}

float FaceDetectionRegion::matchAppearance(const DetectionRegion &region)
{
	const FaceDetectionRegion* cmpFaceDetRegion = dynamic_cast<const FaceDetectionRegion*>(&region);

	if (cmpFaceDetRegion && mFeatureVect.size() > 0 && cmpFaceDetRegion->mFeatureVect.size() > 0 && mFeatureVect.size() == cmpFaceDetRegion->mFeatureVect.size())
	{
		// calculate correlation
		int i;
		double corr,  mean, meanCmp, valMinusMean, valMinusMeanCmp, sum, standDev, standDevCmp;

		mean = 0.0;
		meanCmp = 0.0;
		for (i = 0; i < mFeatureVect.size(); i++)
		{
			mean += mFeatureVect[i];
			meanCmp += cmpFaceDetRegion->mFeatureVect[i];
		}
		mean /= mFeatureVect.size();
		meanCmp /= mFeatureVect.size();

		sum = 0.0;
		standDev = 0.0;
		standDevCmp = 0.0;
		for (i = 0; i < mFeatureVect.size(); i++)
		{
			valMinusMean = mFeatureVect[i] - mean;
			valMinusMeanCmp = cmpFaceDetRegion->mFeatureVect[i] - meanCmp;
			sum += valMinusMean*valMinusMeanCmp;
			standDev += valMinusMean*valMinusMean;
			standDevCmp += valMinusMeanCmp*valMinusMeanCmp;
		}

		standDev = sqrt(standDev);
		standDevCmp = sqrt(standDevCmp);

		if (standDev*standDevCmp == 0.0)
			corr = 0.0;
		else
		{
			corr = sum / (standDev*standDevCmp);
			corr = (corr + 1.0) / 2.0;
			//printf("\tCorr: %f\n", corr);
		}
		float iou = getIou(region);

		//printf("\t<matchAppearance> Correlation: %f\n", corr);
		//printf("\t<matchAppearance> Overlap: %f\n", iou);

		if (iou > 0.70 && corr < 0.85)
			corr += 0.15;


		return (float) 1.0f-(float)corr;
	}
	else
		return 1.0f- getIou(region);
}


float FaceDetectionRegion::matchDistance(const DetectionRegion &region)
{
	float x, y;

	region.getCenter(x, y);

	// calc relative distance
	float dx = (x - mXCenter) / mBBWidth;
	float dy = (y - mYCenter) / mBBHeight;

	return sqrt(dx*dx + dy*dy);
}


void FaceDetectionRegion::calcAppearance(IplImage **images)
{

}


void FaceDetectionRegion::adaptAppearance(DetectionRegion &region, float adaptFactor)
{

}


void FaceDetectionRegion::shiftRegion(const float x, const float y, const float scale)
{
	mXCenter += x;
	mYCenter += y;
	mBBWidth *= scale;
	mBBHeight *= scale;
	mBBXStart = mXCenter - mBBWidth / 2.0f;
	mBBYStart = mYCenter - mBBHeight / 2.0f;
}

bool FaceDetectionRegion::pointWithinRegion(const float x, const float y)
{
	// make rectangle 1.5 times smaller
	//float xStart = mBBXStart + mBBWidth / 6.0;
	//float yStart = mBBYStart + mBBHeight / 6.0;
	//float width = mBBWidth * 2 / 3.0;
	//float height = mBBHeight * 2 / 3.0;

	if ((x > mBBXStart) && (x < mBBXStart + mBBWidth) && (y > mBBYStart) && (y < mBBYStart + mBBHeight))
		return true;
	else
		return false;
}

void FaceDetectionRegion::copyFromRegion(const DetectionRegion *region, bool copyAppearance)
{
	DetectionRegion::copyFromRegion(region);

	const FaceDetectionRegion* fromFaceDetRegion = dynamic_cast<const FaceDetectionRegion*>(region);
	if (fromFaceDetRegion)
	{
		mIsReliableClassified = fromFaceDetRegion->mIsReliableClassified;
		mFaceUUID = fromFaceDetRegion->mFaceUUID;
		mFaceID = fromFaceDetRegion->mFaceID;
		mClassConfidence = fromFaceDetRegion->mClassConfidence;
		mCorrValue = fromFaceDetRegion->mCorrValue;
		mFeatureVect = fromFaceDetRegion->mFeatureVect;
	}
}

float FaceDetectionRegion::mOverlap(float x1, float w1, float x2, float w2)
{
	float l1 = x1 - w1 / 2;
	float l2 = x2 - w2 / 2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1 / 2;
	float r2 = x2 + w2 / 2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

float FaceDetectionRegion::getIntersection(const DetectionRegion &region)
{
	float x, y, width, height;

	((DetectionRegion)region).getBoundingBox(x, y, width, height);
	float w = mOverlap(mBBXStart, mBBWidth, x, width);
	float h = mOverlap(mBBYStart, mBBHeight, y, height);
	if (w < 0 || h < 0) return 0;
	float area = w*h;
	return area;
}

float FaceDetectionRegion::mGetUnion(const DetectionRegion &region)
{
	float x, y, width, height;

	((DetectionRegion)region).getBoundingBox(x, y, width, height);
	float i = getIntersection(region);
	float u = mBBWidth*mBBHeight + width*height - i;
	return u;
}

float FaceDetectionRegion::getIou(const DetectionRegion &region)
{
	float x, y, width, height, i;

	((DetectionRegion)region).getBoundingBox(x, y, width, height);
	float w = mOverlap(mBBXStart, mBBWidth, x, width);
	float h = mOverlap(mBBYStart, mBBHeight, y, height);
	if (w < 0 || h < 0)
		if (w < 0 || h < 0)
			i = 0;
		else
			i = w*h;

	float u = mBBWidth*mBBHeight + width*height - i;


	return i / u; // getIntersection(region) / getUnion(region);
}

bool FaceDetectionRegion::saveToFile(const char* filename)
{
	// save the content of the internal face-feature store
	std::ofstream fileStream(filename);

	if (fileStream.is_open())
	{
		saveToFile(fileStream);

		fileStream.close();
		return true;
	}
	else
		return false;
}

void FaceDetectionRegion::saveToFile(std::ofstream &fileStream)
{
	float x, y, width, height;
	std::string className;

	fileStream << mVersion << std::endl;

	fileStream << mConfidence << std::endl;
	fileStream << mClusterIndex << std::endl;

	fileStream << mFaceOutOfImg << std::endl;
	fileStream << mSharpness << std::endl;

	//if (getParameter(std::string("ClassName"), className))
	//	fileStream << className << std::endl;
	//else
	//	fileStream << std::string() << std::endl;

	//save bounding box
	getBoundingBox(x, y, width, height);
	fileStream << x << std::endl << y << std::endl << width << std::endl << height << std::endl;

	// save feature vector
	int featVectSize = (int)mFeatureVect.size();
	fileStream << featVectSize << std::endl;
	for (int t = 0; t < featVectSize; t++) {
		fileStream << mFeatureVect[t] << std::endl;
	}
}

bool  FaceDetectionRegion::loadFromFile(const char* filename)
{
	bool retValue = true;
	reset();

	std::ifstream fileStream(filename);
	
	if (fileStream.is_open())
	{
		retValue = loadFromFile(fileStream);

		fileStream.close();
	}
	else
		retValue = false;

	return retValue;
}

bool  FaceDetectionRegion::loadFromFile(std::ifstream &fileStream)
{
	float x, y, width, height;
	std::string className;
	int version; 

	fileStream >> version;

	if (mVersion != version)
		return false;

	fileStream >> mConfidence;
	fileStream >> mClusterIndex;

	fileStream >> mFaceOutOfImg;
	fileStream >> mSharpness;

	//fileStream >> className;
	//addParameter(std::string("ClassName"), className);

	//save bounding box
	fileStream >> x >> y >> width >> height;
	setBoundingBox(x, y, width, height);

	// save feature vector
	int featVectSize;

	fileStream >> featVectSize;
	if (mFeatureVect.size() != featVectSize)
		mFeatureVect.resize(featVectSize);

	for (int t = 0; t < featVectSize; t++) {
		fileStream >> mFeatureVect[t];
	}
	mIsFeatureVectSet = true;
	return true;
}

bool FaceDetectionRegion::checkIfRegionOnImageBorder(int imgWidth, int imgHeight, float relRegionMargin)
{
	float x, y, width, height;

	getBoundingBox(x, y, width, height);

	/// check if bounding box lies within image borders, othewise the detection is not further considered!
	if ( (x - (int)(relRegionMargin * width)) < 0 ||
		 (y - (int)(relRegionMargin * height) < 0) ||
		 (x + width - 1 + (int)(relRegionMargin * width)) > (imgWidth - 1) ||
		 (y + height + 1 + (int)(relRegionMargin * height)) > (imgHeight - 1))
	{
		setConfidence(-1);
		addParameter("RetCode", "FaceTooCloseToBorder");

		setFaceOutOfImgRegion(true);
		return true;
	}
	else
		return false;
}
