#include "FaceSwapper/FaceSwapping.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/face.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/photo.hpp"

#include <iostream>


#include "dlib/ipl_image_hull.h"

using namespace cv;

namespace Jrs {
	namespace FaceSwapper {


FaceSwapping::FaceSwapping(std::string landmarksFile, bool triangulation) {
	this->landmarksFile = landmarksFile;
	this->triangulation = triangulation;
	if (!triangulation) {
		try
		{
			dlib::deserialize(landmarksFile) >> shapepred;
		}
		catch (std::exception& e)
		{
			std::cerr << "Error loading landmarks from " << landmarksFile << std::endl
				<< "You can download the file from http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2" << std::endl;
		}
	}

}

FaceSwapping::~FaceSwapping() {

}

void FaceSwapping::swapFaces(cv::Mat src, cv::Mat dst, cv::Mat faceSet, DetectionRegion* srcRegion, DetectionRegion* fsRegion)
{
	if (triangulation) {
		swapFacesTriangulated(src, dst, faceSet, srcRegion, fsRegion);
	}
	else {
		swapFacesAffine(src, dst, faceSet, srcRegion, fsRegion);
	}
}


void FaceSwapping::swapFacesAffine(cv::Mat src, cv::Mat dst, cv::Mat faceSet, DetectionRegion* srcRegion, DetectionRegion* fsRegion)
{
	
	cv::Point2i* srcPoints = new cv::Point2i[9];
	cv::Point2f* srcTransformPoints = new cv::Point2f[3];
	cv::Size srcFeather;
	
	cv::Point2i* fsPoints = new cv::Point2i[9];
	cv::Point2f* fsTransformPoints = new cv::Point2f[3];
	cv::Size fsFeather;

	getLandmarks(src, srcRegion, srcPoints, srcTransformPoints, srcFeather);
	getLandmarks(faceSet, fsRegion, fsPoints, fsTransformPoints, fsFeather);

	//drawPoints(src, srcPoints, "d:\\temp\\srcpoints.png",srcRegion);
	//drawPoints(faceSet, fsPoints, "d:\\temp\\fspoints.png",fsRegion);

	cv::Mat trafoMatrix = cv::getAffineTransform(fsTransformPoints, srcTransformPoints);

	cv::Mat mask =  cv::Mat(src.rows, src.cols, CV_8UC1);
	mask = Scalar(0);

	cv::Mat warpedFaceImage = src.clone();

	getWarpedMaskandFace(srcPoints, fsPoints, trafoMatrix, faceSet, mask, warpedFaceImage);
	
	colorCorrect(src, warpedFaceImage, mask, srcRegion);

	insertFaces(dst, warpedFaceImage, mask, srcFeather);
}

const cv::Point2i FaceSwapping::getPoint(dlib::full_object_detection& shape, int part_index)
{
	const auto &p = shape.part(part_index);
	return cv::Point2i(p.x(), p.y());
};

void FaceSwapping::getLandmarks(cv::Mat img, DetectionRegion* dr, cv::Point2i* points, cv::Point2f* affine_transform_keypoints, cv::Size& feather_amount) {
	float x, y, w, h;
	dr->getBoundingBox(x, y, w, h);
	dlib::rectangle rect = dlib::rectangle(x,y,x+w,y+h);

	IplImage iplImg = img;

	dlib::ipl_image_hull<dlib::rgb_pixel> dlibimg(&iplImg);

	dlib::full_object_detection shape = shapepred(dlibimg, rect);

	points[0] = getPoint(shape, 0);
	points[1] = getPoint(shape, 3);
	points[2] = getPoint(shape, 5);
	points[3] = getPoint(shape, 8);
	points[4] = getPoint(shape, 11);
	points[5] = getPoint(shape, 13);
	points[6] = getPoint(shape, 16);

	cv::Point2i nose_length = getPoint(shape, 27) - getPoint(shape, 30);
	points[7] = getPoint(shape, 26) + nose_length;
	points[8] = getPoint(shape, 17) + nose_length;

	affine_transform_keypoints[0] = points[3];
	affine_transform_keypoints[1] = getPoint(shape, 36);
	affine_transform_keypoints[2] = getPoint(shape, 45);

	feather_amount.width = feather_amount.height = (int)cv::norm(points[0] - points[6]) / 8;
}


void FaceSwapping::getWarpedMaskandFace(cv::Point2i* srcPoints, cv::Point2i* fsPoints, cv::Mat& trafo, cv::Mat fsImage, cv::Mat maskImage, cv::Mat warpedFaceImage) {

	// get mask
	cv::Mat fsmask = cv::Mat(fsImage.rows,fsImage.cols, CV_8UC1);
	fsmask = Scalar(0);

	cv::fillConvexPoly(fsmask, fsPoints, 9, cv::Scalar(255));
	
	CvSize sz = fsmask.size();
	
	cv::warpAffine(fsmask, maskImage, trafo, sz, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

	// get face image

	cv::Mat faceCutout = fsImage.clone();

	fsImage.copyTo(faceCutout, fsmask);

	cv::warpAffine(faceCutout, warpedFaceImage, trafo, sz, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));



}
	

void FaceSwapping::colorCorrect(cv::Mat src, cv::Mat warped, cv::Mat maskImg, DetectionRegion* dr)
{
	uint8_t LUT[3][256];
	int source_hist_int[3][256];
	int target_hist_int[3][256];
	float source_histogram[3][256];
	float target_histogram[3][256];

	float x, y, w, h;
	dr->getBoundingBox(x, y, w, h);
	cv::Rect rect = cv::Rect(x, y, w, h);

	cv::Mat source_image = src(rect);
	cv::Mat target_image = warped(rect);
	cv::Mat mask = mask(rect);

	std::memset(source_hist_int, 0, sizeof(int) * 3 * 256);
	std::memset(target_hist_int, 0, sizeof(int) * 3 * 256);

	for (size_t i = 0; i < mask.rows; i++)
	{
		auto current_mask_pixel = mask.row(i).data;
		auto current_source_pixel = source_image.row(i).data;
		auto current_target_pixel = target_image.row(i).data;

		for (size_t j = 0; j < mask.cols; j++)
		{
			if (*current_mask_pixel != 0) {
				source_hist_int[0][*current_source_pixel]++;
				source_hist_int[1][*(current_source_pixel + 1)]++;
				source_hist_int[2][*(current_source_pixel + 2)]++;

				target_hist_int[0][*current_target_pixel]++;
				target_hist_int[1][*(current_target_pixel + 1)]++;
				target_hist_int[2][*(current_target_pixel + 2)]++;
			}

			// Advance to next pixel
			current_source_pixel += 3;
			current_target_pixel += 3;
			current_mask_pixel++;
		}
	}

	// Calc CDF
	for (size_t i = 1; i < 256; i++)
	{
		source_hist_int[0][i] += source_hist_int[0][i - 1];
		source_hist_int[1][i] += source_hist_int[1][i - 1];
		source_hist_int[2][i] += source_hist_int[2][i - 1];

		target_hist_int[0][i] += target_hist_int[0][i - 1];
		target_hist_int[1][i] += target_hist_int[1][i - 1];
		target_hist_int[2][i] += target_hist_int[2][i - 1];
	}

	// Normalize CDF
	for (size_t i = 0; i < 256; i++)
	{
		source_histogram[0][i] = (source_hist_int[0][255] ? (float)source_hist_int[0][i] / source_hist_int[0][255] : 0);
		source_histogram[1][i] = (source_hist_int[1][255] ? (float)source_hist_int[1][i] / source_hist_int[1][255] : 0);
		source_histogram[2][i] = (source_hist_int[2][255] ? (float)source_hist_int[2][i] / source_hist_int[2][255] : 0);

		target_histogram[0][i] = (target_hist_int[0][255] ? (float)target_hist_int[0][i] / target_hist_int[0][255] : 0);
		target_histogram[1][i] = (target_hist_int[1][255] ? (float)target_hist_int[1][i] / target_hist_int[1][255] : 0);
		target_histogram[2][i] = (target_hist_int[2][255] ? (float)target_hist_int[2][i] / target_hist_int[2][255] : 0);
	}

	// Create lookup table

	auto binary_search = [&](const float needle, const float haystack[]) -> uint8_t
	{
		uint8_t l = 0, r = 255, m;
		while (l < r)
		{
			m = (l + r) / 2;
			if (needle > haystack[m])
				l = m + 1;
			else
				r = m - 1;
		}
		// TODO check closest value
		return m;
	};

	for (size_t i = 0; i < 256; i++)
	{
		LUT[0][i] = binary_search(target_histogram[0][i], source_histogram[0]);
		LUT[1][i] = binary_search(target_histogram[1][i], source_histogram[1]);
		LUT[2][i] = binary_search(target_histogram[2][i], source_histogram[2]);
	}

	// repaint pixels
	for (size_t i = 0; i < mask.rows; i++)
	{
		auto current_mask_pixel = mask.row(i).data;
		auto current_target_pixel = target_image.row(i).data;
		for (size_t j = 0; j < mask.cols; j++)
		{
			if (*current_mask_pixel != 0)
			{
				*current_target_pixel = LUT[0][*current_target_pixel];
				*(current_target_pixel + 1) = LUT[1][*(current_target_pixel + 1)];
				*(current_target_pixel + 2) = LUT[2][*(current_target_pixel + 2)];
			}

			// Advance to next pixel
			current_target_pixel += 3;
			current_mask_pixel++;
		}
	}
}



inline void FaceSwapping::insertFaces(cv::Mat dst, cv::Mat warpedFace, cv::Mat mask, cv::Size& feather)
{

	cv::erode(mask, mask, getStructuringElement(cv::MORPH_RECT, feather), cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));

	cv::blur(mask, mask, feather, cv::Point(-1, -1), cv::BORDER_CONSTANT);

	for (size_t i = 0; i < dst.rows; i++)
	{
		auto frame_pixel = dst.row(i).data;
		auto faces_pixel = warpedFace.row(i).data;
		auto masks_pixel = mask.row(i).data;

		for (size_t j = 0; j < dst.cols; j++)
		{
			if (*masks_pixel != 0)
			{
				*frame_pixel = ((255 - *masks_pixel) * (*frame_pixel) + (*masks_pixel) * (*faces_pixel)) >> 8; 
				*(frame_pixel + 1) = ((255 - *(masks_pixel + 1)) * (*(frame_pixel + 1)) + (*(masks_pixel + 1)) * (*(faces_pixel + 1))) >> 8;
				*(frame_pixel + 2) = ((255 - *(masks_pixel + 2)) * (*(frame_pixel + 2)) + (*(masks_pixel + 2)) * (*(faces_pixel + 2))) >> 8;
			}

			frame_pixel += 3;
			faces_pixel += 3;
			masks_pixel++;
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////
// OpenCV with triangulation 
/////////////////////////////////////////////////////////////////////////////////////////////

void FaceSwapping::swapFacesTriangulated(cv::Mat src, cv::Mat dst, cv::Mat faceSet, DetectionRegion* srcRegion, DetectionRegion* fsRegion){


	face::FacemarkKazemi::Params params;
	Ptr<face::FacemarkKazemi> facemark = face::FacemarkKazemi::create(params);
	facemark->loadModel(landmarksFile);
	
	//vector to store the faces detected in the image
	std::vector<Rect> faces1, faces2;
	std::vector< std::vector<Point2f> > shape1, shape2;

	Mat img1 = faceSet.clone();
	Mat img2 = src.clone();
	Mat img1Warped = img2.clone();

	facemark->setFaceDetector((face::FN_FaceDetector)FaceSwapping::reuseDetections, fsRegion);
	facemark->getFaces(img1, faces1);
	facemark->setFaceDetector((face::FN_FaceDetector)reuseDetections, srcRegion);
	facemark->getFaces(img2, faces2);


	//Initialise the shape of the faces
	facemark->fit(img1, faces1, shape1);
	facemark->fit(img2, faces2, shape2);
	int z = 0; // we have already limited to a single face

	std::vector<Point2f> points1 = shape1[z];
	std::vector<Point2f> points2 = shape2[z];
	img1.convertTo(img1, CV_32F);
	img1Warped.convertTo(img1Warped, CV_32F);
	// Find convex hull
	std::vector<Point2f> boundary_image1;
	std::vector<Point2f> boundary_image2;
	std::vector<int> index;
	convexHull(Mat(points2), index, false, false);
	for (size_t i = 0; i < index.size(); i++)
	{
		boundary_image1.push_back(points1[index[i]]);
		boundary_image2.push_back(points2[index[i]]);
	}
	// Triangulation for points on the convex hull
	std::vector< std::vector<int> > triangles;
	Rect rect(0, 0, img1Warped.cols, img1Warped.rows);
	divideIntoTriangles(rect, boundary_image2, triangles);
	// Apply affine transformation to Delaunay triangles
	for (size_t i = 0; i < triangles.size(); i++)
	{
		std::vector<Point2f> triangle1, triangle2;
		// Get points for img1, img2 corresponding to the triangles
		for (int j = 0; j < 3; j++)
		{
			triangle1.push_back(boundary_image1[triangles[i][j]]);
			triangle2.push_back(boundary_image2[triangles[i][j]]);
		}
		warpTriangle(img1, img1Warped, triangle1, triangle2);
	}
	// Calculate mask
	std::vector<Point> hull;
	for (size_t i = 0; i < boundary_image2.size(); i++)
	{
		Point pt((int)boundary_image2[i].x, (int)boundary_image2[i].y);
		hull.push_back(pt);
	}
	Mat mask = Mat::zeros(img2.rows, img2.cols, img2.depth());
	fillConvexPoly(mask, &hull[0], (int)hull.size(), Scalar(255, 255, 255));
	// Clone seamlessly.
	Rect r = boundingRect(boundary_image2);
	Point center = (r.tl() + r.br()) / 2;

	cv::Mat output = dst.clone();

	img1Warped.convertTo(img1Warped, CV_8UC3);
	seamlessClone(img1Warped, dst, mask, center, output, NORMAL_CLONE);


}

bool FaceSwapping::reuseDetections(InputArray image, OutputArray faces, DetectionRegion *dr)
{
	Mat gray;

	std::vector<Rect> faces_;
	
	Rect r;
	float x, y, w, h;
	dr->getBoundingBox(x, y, w, h);
	r.x = x;
	r.y = y;
	r.width = w;
	r.height = h;
	faces_.push_back(r);

	Mat(faces_).copyTo(faces);
	return true;
}

//Divide the face into triangles for warping
void FaceSwapping::divideIntoTriangles(Rect rect, std::vector<Point2f> &points, std::vector< std::vector<int> > &Tri) {

	// Create an instance of Subdiv2D
	Subdiv2D subdiv(rect);
	// Insert points into subdiv
	for (std::vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
		subdiv.insert(*it);
	std::vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	std::vector<Point2f> pt(3);
	std::vector<int> ind(3);
	for (size_t i = 0; i < triangleList.size(); i++)
	{
		Vec6f triangle = triangleList[i];
		pt[0] = Point2f(triangle[0], triangle[1]);
		pt[1] = Point2f(triangle[2], triangle[3]);
		pt[2] = Point2f(triangle[4], triangle[5]);
		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
			for (int j = 0; j < 3; j++)
				for (size_t k = 0; k < points.size(); k++)
					if (abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)
						ind[j] = (int)k;
			Tri.push_back(ind);
		}
	}
}

void FaceSwapping::warpTriangle(Mat &img1, Mat &img2, std::vector<Point2f> &triangle1, std::vector<Point2f> &triangle2)
{
	Rect rectangle1 = boundingRect(triangle1);
	Rect rectangle2 = boundingRect(triangle2);
	// Offset points by left top corner of the respective rectangles
	std::vector<Point2f> triangle1Rect, triangle2Rect;
	std::vector<Point> triangle2RectInt;
	for (int i = 0; i < 3; i++)
	{
		triangle1Rect.push_back(Point2f(triangle1[i].x - rectangle1.x, triangle1[i].y - rectangle1.y));
		triangle2Rect.push_back(Point2f(triangle2[i].x - rectangle2.x, triangle2[i].y - rectangle2.y));
		triangle2RectInt.push_back(Point((int)(triangle2[i].x - rectangle2.x), (int)(triangle2[i].y - rectangle2.y))); // for fillConvexPoly
	}
	// Get mask by filling triangle
	Mat mask = Mat::zeros(rectangle2.height, rectangle2.width, CV_32FC3);
	fillConvexPoly(mask, triangle2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);
	// Apply warpImage to small rectangular patches
	Mat img1Rect;
	img1(rectangle1).copyTo(img1Rect);
	Mat img2Rect = Mat::zeros(rectangle2.height, rectangle2.width, img1Rect.type());
	Mat warp_mat = getAffineTransform(triangle1Rect, triangle2Rect);
	warpAffine(img1Rect, img2Rect, warp_mat, img2Rect.size(), INTER_LINEAR, BORDER_REFLECT_101);
	multiply(img2Rect, mask, img2Rect);
	multiply(img2(rectangle2), Scalar(1.0, 1.0, 1.0) - mask, img2(rectangle2));
	img2(rectangle2) = img2(rectangle2) + img2Rect;
}




}
}
