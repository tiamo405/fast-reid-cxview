#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "engine.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
    // if (argc!=3) {
	// 	cerr << "Usage: " << argv[0] << " engine.plan image.jpg" << endl;
	// 	return 1;
	// }

    cout << "Loading engine..." << endl;
    auto engine = yolo::Engine(argv[1]);

    cout << "Preparing data..." << endl;
    // auto image_raw = imread(argv[2], IMREAD_COLOR);
	// auto image = imread(argv[2], IMREAD_COLOR);
    auto inputSize = engine.getInputSize();
    // cv::resize(image, image, Size(inputSize[1], inputSize[0]));
	// cv::Mat pixels;
	// image.convertTo(pixels, CV_32FC3);

    int channels = 3;
	vector<float> img;
	vector<float> data (channels * inputSize[0] * inputSize[1]);

    // if (pixels.isContinuous())
	// 	img.assign((float*)pixels.datastart, (float*)pixels.dataend);
	// else {
	// 	cerr << "Error reading image " << argv[2] << endl;
	// 	return -1;
	// }

    vector<float> mean {0.0, 0.0, 0.0};

    for (int c = 0; c < channels; c++) {
		for (int j = 0, hw = inputSize[0] * inputSize[1]; j < hw; j++) {
			data[c * hw + j] = (img[channels * j + 2 - c] - mean[c]);
		}
	}

    // Create device buffers
	void *data_d, *scores_d, *boxes_d;
    auto num_det = engine.getMaxDetections();
    cudaMalloc(&data_d, 3 * inputSize[0] * inputSize[1] * sizeof(float));
	cudaMalloc(&scores_d, num_det * sizeof(float));
	cudaMalloc(&boxes_d, num_det * 4 * sizeof(float));

    // Copy image to device
	size_t dataSize = data.size() * sizeof(float);
	cudaMemcpy(data_d, data.data(), dataSize, cudaMemcpyHostToDevice);

    // Run inference n times
	cout << "Running inference..." << endl;
	const int count = 1000;
	auto start = chrono::steady_clock::now();
 	vector<void *> buffers = { data_d, scores_d, boxes_d };
	for (int i = 0; i < count; i++) {
		engine.infer(buffers, 1);
	}
	auto stop = chrono::steady_clock::now();
	auto timing = chrono::duration_cast<chrono::duration<double>>(stop - start);
	cout << "Took " << timing.count() / count << " seconds per inference." << endl;

	cudaFree(data_d);

    // Get back the bounding boxes
	unique_ptr<float[]> scores(new float[num_det]);
	unique_ptr<float[]> boxes(new float[num_det * 4]);
    cudaMemcpy(scores.get(), scores_d, sizeof(float) * num_det, cudaMemcpyDeviceToHost);
	cudaMemcpy(boxes.get(), boxes_d, sizeof(float) * num_det * 4, cudaMemcpyDeviceToHost);

    cudaFree(scores_d);
	cudaFree(boxes_d);

    // for (int i = 0; i < num_det; i++) {
	// 	cout << scores[i] << endl;
	// 	// Show results over confidence threshold
	// 	if (scores[i] >= 0.3f) {
	// 		float x = boxes[i*4+0];
	// 		float y = boxes[i*4+1];
	// 		float w = boxes[i*4+2];
	// 		float h = boxes[i*4+3];
	// 		cout << "Found box {" << x1 << ", " << y1 << ", " << x2 << ", " << y2
	// 			<< "} with score " << scores[i] << endl;

	// 		// Draw bounding box on image
	// 		cv::rectangle(image_raw, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 255, 0));
	// 	}
	// }

	// // Write image
	// string out_file = argc == 4 ? string(argv[3]) : "detections.png";
	// cout << "Saving result to " << out_file << endl;
	// imwrite(out_file, image_raw);

    return 0;
}
