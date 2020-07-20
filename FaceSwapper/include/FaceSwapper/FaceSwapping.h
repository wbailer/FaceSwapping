
#include <dlib/opencv.h>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>

#include <opencv2/core/mat.hpp>

#include "DetectionRegion.h"

namespace Jrs {
	namespace FaceSwapper{



class FaceSwapping {

public:
	FaceSwapping(std::string landmarksFile, bool triangulation = false);

	~FaceSwapping();

	void swapFaces(cv::Mat src, cv::Mat dst, cv::Mat faceSet, DetectionRegion* srcRegion, DetectionRegion* fsRegion);

protected:

	static bool reuseDetections(cv::InputArray image, cv::OutputArray faces, DetectionRegion *dr);

	void swapFacesAffine(cv::Mat src, cv::Mat dst, cv::Mat faceSet, DetectionRegion* srcRegion, DetectionRegion* fsRegion);

	void swapFacesTriangulated(cv::Mat src, cv::Mat dst, cv::Mat faceSet, DetectionRegion* srcRegion, DetectionRegion* fsRegion);

	void getLandmarks(cv::Mat img, DetectionRegion* dr, cv::Point2i* points, cv::Point2f* affine_transform_keypoints, cv::Size& feather_amount);
	
	void getWarpedMaskandFace(cv::Point2i* srcPoints, cv::Point2i* fsPoints, cv::Mat& trafo, cv::Mat fsImage, cv::Mat maskImage, cv::Mat warpedFaceImage);
	
	void colorCorrect(cv::Mat src, cv::Mat warped, cv::Mat maskImg, DetectionRegion* dr);

	void insertFaces(cv::Mat dst, cv::Mat warpedFace, cv::Mat maskImage, cv::Size& feather);


	///////

	const cv::Point2i FaceSwapping::getPoint(dlib::full_object_detection& shape, int part_index);

	void divideIntoTriangles(cv::Rect rect, std::vector<cv::Point2f> &points, std::vector< std::vector<int> > &delaunayTri);

	void warpTriangle(cv::Mat &img1, cv::Mat &img2, std::vector<cv::Point2f> &triangle1, std::vector<cv::Point2f> &triangle2);

	dlib::shape_predictor shapepred;
	std::string landmarksFile;
	bool triangulation;


};


}
}