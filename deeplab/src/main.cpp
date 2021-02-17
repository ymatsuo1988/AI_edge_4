/*
 * Copyright 2021 y.matsuo1988
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <chrono>

/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

/* header file for Vitis AI advanced API */
#include <dnndk/dnndk.h>

/* header file for Caffe input images APIs */
#include "dputils.h"

using namespace std;
using namespace cv;

/* DPU Kernel name for deeplab */
#define KRENEL_DEEPLAB "deeplab"
/* Input Node for Kernel deeplab */
#define INPUT_NODE      "MobilenetV2_Conv_conv2d_Conv2D"
/* Output Node for Kernel deeplab */
#define OUTPUT_NODE     "ResizeBilinear_2"

const string baseImagePath = "../dataset/AIedge/";
const string resultImagePath = "../dataset/AIedge/result/";



/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */

void ListImages(string const &path, vector<string> &images) {
    images.clear();
    struct dirent *entry;

    /*Check if path is a valid directory path. */
    struct stat s;
    lstat(path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
        exit(1);
    }

    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
        exit(1);
    }

    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
            string name = entry->d_name;
            string ext = name.substr(name.find_last_of(".") + 1);
            if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
                (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
                images.push_back(name);
            }
        }
    }

    closedir(dir);
    sort(images.begin(), images.end());
}



/**
 * @brief calculate argmax (fixed: height=384, width=384, channel=5)
 *
 * @param data - pointer to input buffer
 * @param result - calculation result
 *
 * @return none
 */

void Argmax(const float *data, Mat &result) {
    for (size_t i = 0; i < 384; i=i+2) { //height
        for (size_t j = 0; j < 384; j=j+2) { //width

            int max_idx=0;
            float max_val=data[i*5*384 + j*5 + 0];

            for (size_t k = 1; k < 5; k++) { //channel

                if(data[i*5*384 + j*5 + k]>max_val){
                    max_idx = k;
                    max_val = data[i*5*384 + j*5 + k];
                }
            }

            result.at<unsigned char>(i,j) = max_idx;
            result.at<unsigned char>(i,j+1) = max_idx;
            result.at<unsigned char>(i+1,j) = max_idx;
            result.at<unsigned char>(i+1,j+1) = max_idx;

        }
    }

}



/**
 * @brief Run DPU Task for Deeplab
 *
 * @param taskDeeplab - pointer to Deeplab Task
 *
 * @return none
 */
void runDeeplab(DPUTask *taskDeeplab) {

    assert(taskDeeplab);

    vector<string> images;

    /* Load all image names.*/
    ListImages(baseImagePath, images);
    if (images.size() == 0) {
        cerr << "\nError: No images existing under " << baseImagePath << endl;
        return;
    }

    /*Variables*/

    float *logit_L = new float[384*384*5];
    float *logit_R = new float[384*384*5];


    Mat image_L(1024, 1024, CV_8UC3);
    Mat image_R(1024, 1024, CV_8UC3);

    Mat image_L_resize = Mat::ones(384, 384, CV_8UC3);
    Mat image_R_resize = Mat::ones(384, 384, CV_8UC3);


    Mat result_L(384, 384, CV_8UC1);
    Mat result_R(384, 384, CV_8UC1);

    Mat result_L_resize(1024, 1024, CV_8UC1);
    Mat result_R_resize(1024, 1024, CV_8UC1);

    Mat result = Mat::ones(1216, 1936, CV_8UC1)*255;


    float mean[3] = {128.00,128.00,128.00};
    Size outsize_384 =Size(384, 384);
    Size outsize_1024 =Size(1024, 1024);
    int coeff = 128;
    Mat ave_column_vec, ave_point;
    int ave;


    double total_time = 0;
    chrono::system_clock::time_point  start, end; // 型は auto で可

    int idx = 0;


    /*Area*/
    Rect roi_ave(cv::Point(500, 600), cv::Size(936, 300));

    Rect roi_L(cv::Point(0, 100), cv::Size(1024, 1024));
    Rect roi_R(cv::Point(912, 100), cv::Size(1024, 1024));

    Rect roi_R_cp_to(cv::Point(968, 100), cv::Size(968, 1024));
    Rect roi_R_cp_from(cv::Point(56, 0), cv::Size(968, 1024));


    for (auto &imageName : images) {

        Mat image = imread(baseImagePath + imageName);

//MEASURE TIME FROM//////////////////////////////////////////////////////////////////////////////////////

        start = std::chrono::system_clock::now();

        /*pre-process*/
        reduce(image(roi_ave), ave_column_vec, 0, REDUCE_AVG);
        reduce(ave_column_vec, ave_point, 1, REDUCE_AVG);
        ave = (ave_point.data[0] + ave_point.data[1] + ave_point.data[2]) / 3;


        /*Infer Left image*/
        image_L = image(roi_L);

        resize(image_L, image_L_resize, outsize_384, 0, 0, INTER_LINEAR);

        if((coeff / ave)>0){
            image_L_resize = image_L_resize*(coeff / ave);
        }

        dpuSetInputImage(taskDeeplab, INPUT_NODE, image_L_resize, mean);
        dpuRunTask(taskDeeplab);
        dpuGetOutputTensorInHWCFP32(taskDeeplab, OUTPUT_NODE, logit_L, 384*384*5);

        Argmax(logit_L, result_L);

        resize(result_L,result_L_resize, outsize_1024, 0, 0, INTER_NEAREST);


        /*Infer Right image*/
        image_R = image(roi_R);

        resize(image_R,image_R_resize, outsize_384,0,0, INTER_LINEAR);

        if((coeff / ave)>0){
            image_R_resize = image_R_resize*(coeff / ave);
        }

        dpuSetInputImage(taskDeeplab, INPUT_NODE, image_R_resize, mean);
        dpuRunTask(taskDeeplab);
        dpuGetOutputTensorInHWCFP32(taskDeeplab, OUTPUT_NODE, logit_R, 384*384*5);

        Argmax(logit_R, result_R);

        resize(result_R,result_R_resize, outsize_1024, 0, 0, INTER_NEAREST);


        /*Merge Right, Left*/
        result_L_resize.copyTo(result(roi_L));
        result_R_resize(roi_R_cp_from).copyTo(result(roi_R_cp_to));

        end = std::chrono::system_clock::now();

//MEASURE TIME TO////////////////////////////////////////////////////////////////////////////////////////

        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

        total_time = total_time + elapsed;

        cout << "  idx= " << idx <<std::endl;

        if(idx<10){
            imwrite(resultImagePath + "test_00" + to_string(idx) + ".png",result);
        }else if(idx<100){
            imwrite(resultImagePath + "test_0" + to_string(idx) + ".png",result);
        }else{
            imwrite(resultImagePath + "test_" + to_string(idx) + ".png",result);
        }

        idx++;

    }

    cout << "  Total Execution time: " << total_time << "msec\n";
    cout << "  Execution time per image: " << total_time/idx << "msec\n";

    delete[] logit_L;
    delete[] logit_R;

    result_L.release();
    result_R.release();
    result.release();
    image_L.release();
    image_R.release();
    image_L_resize.release();
    image_R_resize.release();
    result_L_resize.release();
    result_R_resize.release();
}



int main(void) {

    DPUKernel *kernelDeeplab;
    DPUTask *taskDeeplab;

    dpuOpen();

    kernelDeeplab = dpuLoadKernel(KRENEL_DEEPLAB);
    taskDeeplab = dpuCreateTask(kernelDeeplab, 0);

    runDeeplab(taskDeeplab);

    dpuDestroyTask(taskDeeplab);
    dpuDestroyKernel(kernelDeeplab);

    dpuClose();

    return 0;
}
