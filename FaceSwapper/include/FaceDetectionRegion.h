#ifndef _FACEDETECTIONREGION_H_
#define _FACEDETECTIONREGION_H_

#include "DetectionRegion.h"

// avoid compiler warning  C4275: non dll-interface class '???' used as base for dll-interface class '???' by exporting the base class
class DetectionRegion;

/// Class for the result of the face detector.
/// The rectangular region can be get by the functions getBoundingBox from the base class DetectionRegion.
/// The detection confidence value can be managed by the functions getConfidence and setConfidence from the base class DetectionRegion.
class FaceDetectionRegion : public DetectionRegion
{
public:
	FaceDetectionRegion();
	FaceDetectionRegion(const DetectionRegion *region);
	~FaceDetectionRegion();
	
	/// Resets the member variables for reuse (to avoid allocation and deallocation)
	void reset();

	/// Sets the bounding box of the region.
	/// @param x,y				position of the upper left corner of the bounding box
	/// @param width, height	size of the bounding box
	virtual void setBoundingBox(const float x, const float y, const float width, const float height);

	/// Match appearance similarity of two regions
	/// @param region	region to be matched
	/// @return			similarity of two regions
	virtual float matchAppearance(const DetectionRegion &region);

	/// Match euqlide distance between two regions
	/// @param region	region to calculate distance to
	/// @return			distance between two regions
	virtual float matchDistance(const DetectionRegion &region);

	/// Calculate the visual features which describes the region appearance
	/// @param images	array of input images with fixed size of 4 elements: RGB image, 1st channel, 2nd channel, 3d channel
	///					depending on the mode, some elements can be NULL (e.g. only RGB image is given, or only 3 channels
	///					or in case of gray image: [ NULL, imgGray, NULL, NULL ].
	///					The function has to take into account all of this cases
	virtual void calcAppearance(IplImage **images);

	/// Adapts the visual appearance features by the appearance description of an other region.
	/// @param region		the appearance features of this region is used for the adaptation
	/// @param adaptFactor	adaptation factor [0,1]
	virtual void adaptAppearance(DetectionRegion &region, float adaptFactor);

	/// Shift region by some vector
	/// @param x			x-displacement
	/// @param y			y-displacement
	/// @param scale		change of the scale
	virtual void shiftRegion(const float x, const float y, const float scale = 1.0);

	/// Check if input point is within the region (bounding box check is implemented.
	/// @param x	x-coordinate
	/// @param y	y-coordinate
	/// @return		boolean answer
	virtual bool pointWithinRegion(const float x, const float y);

	/// Copy all region information from the another region. Both regions has to be allocated outside
	/// @param region			region to be copied from
	/// @param copyAppearance	if this value is true then also the appearance feature values will be copied
	virtual void copyFromRegion(const DetectionRegion *region, bool copyAppearance = true);

	//// Calculates the size of the intersection area with the specified region.
	float getIntersection(const DetectionRegion &region);

	/// Calculates the intersection over union with the specfied region
	float getIou(const DetectionRegion &region);

	/// sets and gets the classification confidence measures
	void setClassificationConfidence(double NewClassificationConfidence) {mClassConfidence = NewClassificationConfidence;	}
	double getClassificationConfidence() { return mClassConfidence; }

	/// sets and gets the feature correlation value
	void setCorrelationValue(double newCorrValue) { mCorrValue = (float)newCorrValue; }
	double getCorrelationValue() { return mCorrValue; }

	/// Sets and gets Universally Unique Identifier for face classification.
	void setFaceUUID(std::string faceUUID) { mFaceUUID = faceUUID; }
	std::string getFaceUUID() { return mFaceUUID; }
	/// only used on detector version
	void setFaceID(int faceID) { mFaceID = faceID; }
	int getFaceID() { return mFaceID; }

	/// Sets and gets the feature vector which is used for face classification.
	void setFeatureVector(std::vector<double> featureVect) { mFeatureVect = featureVect; mIsFeatureVectSet = true; }
	std::vector<double>& getFeatureVector() { return mFeatureVect; }
	void featureVectorIsSet() { mIsFeatureVectSet = true; }

	/// Indicate if the feature vector has been set.
	bool isFeatureVectorSet() { return mIsFeatureVectSet; }

	/// Sets and gets the status of the reliablity flag for the classfication. E.g. the classification is set reliable after the face has been tracked.
	void setClassifiactionReliable(bool isReliable) { mIsReliableClassified = isReliable; }
	bool isClassificationReliable(){ return mIsReliableClassified; }

	/// Sets and gets the status of the flag indicating if this face region has been considered for the Auto-Train procedure
	void setAutoTrainCandidateNo( int AutoTrainCandidateNo) { mAutoTrainCandidateNo = AutoTrainCandidateNo; }
	int getAutoTrainCandidateNo() { return mAutoTrainCandidateNo; }

	/// Sets and gets the index of trajectory the region belongs to
	void setTrajectoryIndex(int Idx) { mOriginTrajectoryIndex = Idx; }
	int getTrajectoryIndex() { return mOriginTrajectoryIndex; }

	/// Sets and gets the status of the reliablity flag for the detection. Only the predicted regions produced by a tracker are not reliable detected.
	/// This flag has to be set outside of the tracker (because the base class DetectionRegion does not has this flag).
	void setDetectionReliable(bool isReliable) { mIsReliableDetected = isReliable; }
	bool isDetectionReliable(){ return mIsReliableDetected; }

	void setTensorDetNumb(int numb) {mTensorDetNumb = numb;}
	int getTensorDetNumb() {return mTensorDetNumb;}
	void setNumbTensorDet(int numb) { mTensorNrDetections = numb; }
	int getNumbTensorDet() { return mTensorNrDetections; }

	/// Sets and gets the image region of the face. Allocation and deallocation have to done outside of the class.
	IplImage* getRegionImage() {return mRegionImage; }
	void setRegionImage(IplImage *regionImage) { mRegionImage = regionImage; }


	/// Sets and gets the detection time of the regions in a video stream 
	/// This function should be moved in the base class (at the next version change of the base class)
	void setDetectionTime(int64_t time) { mDetTime = time; }
	int64_t getDetectionTime() { return mDetTime; }

	/// Saves and loads the region data to and from a file 
	/// @param filename		path and filename
	/// @return		indicate if the save or load function was successfull
	bool saveToFile(const char* filename);
	void saveToFile(std::ofstream &fileStream);
	bool loadFromFile(const char* filename);
	bool loadFromFile(std::ifstream &fileStream);

	bool isFaceOutOfImgRegion() { return mFaceOutOfImg; };
	void setFaceOutOfImgRegion(bool isOutOfImg) { mFaceOutOfImg = isOutOfImg; };

	double getSharpness() { return mSharpness; };
	void setSharpness(double sharpnessVal) { mSharpness = sharpnessVal; };

	static int getVersion() { return mVersion; }

	bool checkIfRegionOnImageBorder(int imgWidth, int imgHeight, float relRegionMargin);

protected:
	// version number for saving the region data
	static int mVersion;

	float mGetUnion(const DetectionRegion &region);
	float mOverlap(float x1, float w1, float x2, float w2);

	std::string mFaceUUID;				// ID for face classification
	int mFaceID;						//	 only used in old version
	double mClassConfidence;			// the classification confidence value
	float mCorrValue;					// feature correlation value (second check)
	bool mIsReliableDetected;			// predicted regions are not detected reliable, all other one are set as reliable detected
	bool mIsReliableClassified;			// indicate if the classfication is reliable
	int mAutoTrainCandidateNo;			// number indicating if this face region has been considered for the Auto-Train procedure as the n-th sample
 	int mTensorNrDetections;			// number of detections in the image
	int mTensorDetNumb;					// index of the detection (from mTensorNrDetections detections)


	IplImage*	mRegionImage;			// copy of the image region

	int64_t		mDetTime;					//detection time used for processing videos

	std::vector<double> mFeatureVect;	// feature vecture used for face classification
	bool mIsFeatureVectSet;				// indicate if there is a valid feature vector;
	int mOriginTrajectoryIndex;			// index of trajectory the region belongs to


	bool mFaceOutOfImg;					// if the boundingbox including the slight increasement is out of the image region
	double mSharpness;					// defines the sharpness of the detected face/bounding box, value is between 0 and 1.

};

#endif
