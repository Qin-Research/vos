/*
  Copyright (c) 2012, Philipp Krähenbühl
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
  * Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
  * Neither the name of the Stanford University nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "saliency/saliency.h"
#include <cstdio>
#include <string>
#include <fstream>
#include <Eigen/Core>
#include <numpy.hpp>

using namespace Eigen;
using namespace aoba;
int main( int argc, char * argv[] ) {
    // if (argc < 2) {
    // 	printf("Usage: %s image\n", argv[0] );
    // 	return -1;
    // }


    const int n_frames = atoi(argv[1]);
    const std::string base(argv[2]);
    const std::string dir_name(base + "/frames");
    const std::string tsp_dir(base + "/tsp");
    const std::string vx_dir(base + "/vx");
    const std::string vy_dir(base + "/vy");
    const std::string out_dir(argv[3]);

    std::vector<Mat_<Vec3b>> imgs;
    std::cout << n_frames << std::endl;
    for(int i = 0; i < n_frames; ++i)
    {
        char str[9];
        sprintf(str, "%s/%05d.png", dir_name.c_str(), i);
        std::cout << str << std::endl;
        imgs.push_back(imread(str));
    }

    const int rows = imgs[0].rows;
    const int cols = imgs[0].cols;

    std::vector<int> shape = {rows, cols};

    std::vector<double> vx_buf(rows*cols);
    std::vector<double> vy_buf(rows*cols);
    std::vector<int> tsp_buf(rows*cols);
    std::vector<Mat_<int>> tsp;
    std::vector<MatrixXd> vx, vy;

    for(int i = 0; i < n_frames; ++i)
    {
        char str[9];
        sprintf(str, "%s/%05d.npy", tsp_dir.c_str(), i);
        LoadArrayFromNumpy(str, shape, tsp_buf);
        MatrixXi map = Eigen::Map<MatrixXi>(&tsp_buf[0], cols, rows).transpose();
        Mat_<int> sp(rows, cols);
        for(int y = 0; y < rows; ++y)
        {
            for(int x = 0; x < cols; ++x)
            {
                sp.at<int>(y,x) = map(y,x);
            }
        }
        tsp.push_back(sp);

        sprintf(str, "%s/%05d.npy", vx_dir.c_str(), i);
        LoadArrayFromNumpy(str, shape, vx_buf);
        vx.push_back(Eigen::Map<MatrixXd>(&vx_buf[0], cols, rows).transpose());
        sprintf(str, "%s/%05d.npy", vy_dir.c_str(), i);
        LoadArrayFromNumpy(str, shape, vy_buf);
        vy.push_back(Eigen::Map<MatrixXd>(&vy_buf[0], cols, rows).transpose());
    }

    Saliency saliency;
    auto saliencies = saliency.temporalSaliency(imgs,tsp, vx, vy);

    for(int i = 0; i < n_frames; ++i)
    {
        std::cout << i << std::endl;
        char str[9];
        sprintf(str, "%05d.txt", i);
        std::vector<float> buf(rows*cols);

        MatrixXf out(rows, cols);
        for(int y = 0; y < rows; ++y)
        {
            for(int x = 0; x < cols; ++x)
            {
                out(y,x) =  saliencies[i].at<float>(y,x) ;
            }
        }

        std::ofstream of(out_dir + '/' + str);
        of << out;
        //  aoba::SaveArrayAsNumpy(str, false, 2, &shape[0], &buf[0]);
        //imshow("sal",saliencies[i]);
        //waitKey();
        //    cv::imwrite(str,  saliencies[i]); //cv::Mat(rows, cols, CV_32FC1, saliencies[i]));
    }

    return 0;
}
