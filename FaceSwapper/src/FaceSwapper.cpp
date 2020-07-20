
#include <stdio.h>
#include <fstream>

#ifdef _WINDOWS
#define WIN32_LEAN_AND_MEAN
#define WIN64_LEAN_AND_MEAN

#include <windows.h>

#else
#include <unistd.h>
#endif

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>

#include <iostream>

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv\cv.hpp>

#include "FaceSwapper/FaceSwapping.h"
#include "dlib/DlibFaceDetector.h"

std::vector<DetectionRegion*>* copyRegionList(std::vector<DetectionRegion*>* src) {
	std::vector<DetectionRegion*>* dst = new std::vector<DetectionRegion*>();

	for (int i = 0; i < src->size(); i++) {
		DetectionRegion* dr = new DetectionRegion();
		dr->copyFromRegion(src->at(i), true);
		dst->push_back(dr);
	}

	return dst;

}

int main(int argc, char** argv)
{

	if (argc < 4) {
		std::cerr << "Error: insufficient number of parameters" << std::endl;
		std::cerr << "Usage: FaceSwapper <inputImage> <faceImage> <outputImage>" << std::endl;
	}

	std::string inputImage = argv[1];
	std::string faceImage = argv[2];
	std::string outputImage = argv[3];

	std::cout << "loading " << inputImage << std::endl;
	
	cv::Mat inputImg = cv::imread(inputImage, CV_LOAD_IMAGE_COLOR);
	cv::Mat targetImg = inputImg.clone();

	std::cout << "loading " << faceImage << std::endl;

	cv::Mat faceImg = cv::imread(inputImage, CV_LOAD_IMAGE_COLOR);
	

	std::cout << "running face det on both images " << std::endl;

	FaceDetectorDlib faceDetector;
	
	faceDetector.doLazyInit("mmod_human_face_detector.dat");
	
	std::vector<DetectionRegion*>* detectedInputRegions = faceDetector.calculate(inputImg, 0.9);

	printf("Input: number of detected regions:%d\n", (int)(detectedInputRegions->size()));


	std::vector<DetectionRegion*>* detectedFaceRegions = faceDetector.calculate(faceImg, 0.98);
	printf("Face templates: number of detected regions:%d\n", (int)(detectedFaceRegions->size()));
	
	Jrs::FaceSwapper::FaceSwapping fswap("./models/shape_predictor_68_face_landmarks.dat");
	//Jrs::FaceSwapper::FaceSwapping fswap("./models/face_landmark_model.dat",true);

	std::cout << "replacing ... " << std::endl;

	srand((unsigned)time(0));

	for (int i = 0; i < detectedInputRegions->size(); i++) {

		int faceId = rand() % detectedFaceRegions->size();

		std::cout << "replacing face " << i << " with generated face " << faceId << std::endl;

		fswap.swapFaces(inputImg, targetImg, faceImg, detectedInputRegions->at(i), detectedFaceRegions->at(faceId));

		std::cout << "done " << std::endl;

	}

	try {

		cv::imwrite(outputImage, targetImg);
	}
	catch (std::exception e) {
		std::cerr << "failed to write image " << std::endl;
	}

	return 0;
}
