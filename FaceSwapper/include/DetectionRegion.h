//***************************************************************************
//
// Copyright (C) 2008 JOANNEUM RESEARCH - Institute of Information Systems & 
// Information Management, Graz, Austria. All rights reserved.
//
// Licensees holding a valid License Agreement may use this file in
// accordance with the rights, responsibilities and obligations
// contained therein.  Please consult your licensing agreement or
// contact iis@joanneum.at if any conditions of this licensing
// agreement are not clear to you.
//
//
// This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
// WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
//
//***************************************************************************

#ifndef _DETECTIONREGION_H_
#define _DETECTIONREGION_H_

//#include "ipl.h"
struct _IplImage;
typedef struct _IplImage IplImage;
#include <list>
#include <vector>

/// Abstract base class which has to be inherited by any detection class if it should be used in JRS application framework
/// @author Helmut Neuschmied
class DetectionRegion
{
public:
	enum DetectionType
	{
		DETECTED = 1,
		MANUAL = 2,
		PREDICTED = 4,
		INTERPOLATED = 8,
		EXTRAPOLATED = 16
	};

	struct Point
	{
		Point() {}
		Point(float xP, float yP) {x = xP; y = yP;}

		float x;
		float y;
	};

	DetectionRegion();
	DetectionRegion(const float x, const float y, const float scale = 1.0);
	DetectionRegion(const float xStart, const float yStart, const float width, const float height, const float scale = 1.0);
	DetectionRegion(std::vector<DetectionRegion::Point> *points, const float scale = 1.0);

	/// Get region center position
	/// @param x	x-cooridinate
	/// @param y	y-cooridinate
	void getCenter(float &x, float &y) const { x = mXCenter; y = mYCenter;}

	/// Set region center position
	/// @param x	x-cooridinate
	/// @param y	y-cooridinate
	void setCenter(const float x, const float y) {mXCenter = x;  mYCenter = y; }

	/// Gets the bounding box of the region.
	/// @param x,y				position of the upper left corner of the bounding box
	/// @param width, height	size of the bounding box
	virtual void getBoundingBox(float &x, float&y, float &width, float &height) 
					{x = mBBXStart; y = mBBYStart; width = mBBWidth; height = mBBHeight;} 

	/// Sets the bounding box of the region.
	/// @param x,y				position of the upper left corner of the bounding box
	/// @param width, height	size of the bounding box
	virtual void setBoundingBox(const float x, const float y, const float width, const float height)
					{mBBXStart = x; mBBYStart = y; mBBWidth = width; mBBHeight = height;} 

	/// Adds a point to the point list (e.g. polygon) of the region
	void addPoint(float x, float y) 
				{mPointList.push_back(DetectionRegion::Point(x,y));}
	void addPoint(DetectionRegion::Point &point) 
				{mPointList.push_back(point);}
	
	/// Set the point list(e.g. a polygon).
	void setPoints(std::vector<DetectionRegion::Point> *points)
				{mPointList = *points;}

	/// Gets the point list (e.g. a polygon).
	std::vector<DetectionRegion::Point>* getPoints() {return &mPointList;}

	/// Removes all points.
	void clearPoints() {mPointList.clear();}

	/// Calculates the bounding box from the specified points of the region.
	void updateBBFromPointList();

	/// Get region confidence value
	/// @return	confidence value
	float getConfidence() { return mConfidence; }

	/// Set region confidence value
	/// @param confidence	confidence value
	void setConfidence( float confidence )  { mConfidence = confidence; }

	/// Get region scale value
	/// @return	scale value
	float getScale() { return mScale; }

	/// Set region scale value
	/// @param scale scale value
	void setScale( float scale ) { mScale = scale; }

	/// Specifies how the region is detected.		
	void setDetectionType(DetectionRegion::DetectionType detectionType) {mDetectionType = detectionType;}

	/// Indicate how the region is detected.
	DetectionRegion::DetectionType getGetDetectionType() {return mDetectionType;}

	/// Sets the appearance features invalid.
	void resetAppearance() {mAppearanceCalculated = false;}

	/// Sets the index of the cluster region. If the detection regions are clustered then each region belongs to 
	/// a cluster region or is a cluster region itself. The cluster region is referenced by an index value. This 
	/// index value corresponds to a position in a detection region list.
	/// @param clusterIndex		index of the cluster region
	void setClusterIndex(int clusterIndex) {mClusterIndex = clusterIndex;}

	/// Gets the index of the cluster region. If the detection regions are clustered then each region belongs to 
	/// a cluster region or is a cluster region itself. The cluster region is referenced by an index value. This 
	/// index value corresponds to a position in a detection region list.
	/// @return		index of the cluster region
	int	getClusterIndex(){return mClusterIndex;}

	/// Clears all parameter.
	void clearParameters();

	/// Returns the number of parameters.
	int numbOfParameters() {return (int)mParameters.size();}

	/// Adds a parameter to the parameter list.
	/// @param	keyvalue	name of the parameter
	/// @param	value		parameter value
	/// @return		index of the paramater list
	int addParameter(const std::string &keyvalue,const  std::string &value);

	/// Gets a specific parameter.
	/// @param index	index of the parameter list
	/// @param keyvalue	name of the parameter
	/// @param value	parameter value
	bool getParameter(int index, std::string &keyvalue, std::string &value);
	bool getParameter(const std::string &keyvalue, std::string &value);

	//--------------------------------------------------------------------------------------------------------------
	// function which have to be implemented (in a derived class) in dependence of the used tracking algorithm

	/// Match appearance similarity of two regions
	/// @param region	region to be matched
	/// @return			similarity of two regions
	virtual float matchAppearance( const DetectionRegion &region ) {return 0.0f;}

	/// Match euqlide distance between two regions
	/// @param region	region to calculate distance to
	/// @return			distance between two regions
	virtual float matchDistance( const DetectionRegion &region );

	/// Calculates the intersection over union (IoU) with the specified region
	float getIou(const DetectionRegion &region);

	/// Calculate the visual features which describes the region appearance
	/// @param images	array of input images with fixed size of 4 elements: RGB image, 1st channel, 2nd channel, 3d channel
	///					depending on the mode, some elements can be NULL (e.g. only RGB image is given, or only 3 channels
	///					or in case of gray image: [ NULL, imgGray, NULL, NULL ].
	///					The function has to take into account all of this cases
	virtual void calcAppearance( IplImage **images) {}

	/// Adapts the visual appearance features by the appearance description of an other region.
	/// @param region		the appearance features of this region is used for the adaptation
	/// @param adaptFactor	adaptation factor [0,1]
	virtual void adaptAppearance(DetectionRegion &region, float adaptFactor) {}

	/// Shift region by some vector
	/// @param x			x-displacement
	/// @param y			y-displacement
	/// @param scale		change of the scale
	virtual void shiftRegion( const float x, const float y, const float scale = 1.0);

	/// Check if input point is within the region (bounding box check is implemented. 
	/// @param x	x-coordinate
	/// @param y	y-coordinate
	/// @return		boolean answer
	virtual bool pointWithinRegion( const float x, const float y );

	/// Copy all region information from the another region. Both regions has to be allocated outside
	/// @param region			region to be copied from
	/// @param copyAppearance	if this value is true then also the appearance feature values will be copied
	virtual void copyFromRegion( const DetectionRegion *region, bool copyAppearance = true );

	//--------------------------------------------------------------------------------------------------------------

protected:
	/// Calculates the overlap between two value ranges ([x1, x1+w1] and  [x2, x2+w2]).
	float mOverlap(float x1, float w1, float x2, float w2);

	/// center point of the region
	float				mXCenter;
	float				mYCenter;
	/// bounding box
	float				mBBWidth, mBBHeight;
	float				mBBXStart, mBBYStart;  // upper left corner point
	/// can be used to specify the region border by a polygon (inside the bounding box)
	std::vector<Point>	mPointList;
	
	DetectionType		mDetectionType;
	float				mConfidence;
	float				mScale;

	/// required for tracking
	int					mClusterIndex;
	bool				mAppearanceCalculated;

	/// parameter list
	typedef std::pair <std::string, std::string> RegionParameters;
	std::list <RegionParameters>	mParameters;
};

#endif
