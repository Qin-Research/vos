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

#include <fstream>
#include <Eigen/Core>
#include <numpy.hpp>

using namespace Eigen;
int main( int argc, char * argv[] ) {
	if (argc < 2) {
		printf("Usage: %s image\n", argv[0] );
		return -1;
	}
	Mat im = imread( argv[1] );
  const int rows = im.rows;
  const int cols = im.cols;
    Saliency saliency;
	Mat_<float> sal = saliency.saliency( im );
	
	double adaptive_T = 2.0 * sum( sal )[0] / (sal.cols*sal.rows);
	while (sum( sal > adaptive_T )[0] == 0)
		adaptive_T /= 1.2;

      MatrixXf out(rows, cols);
      for(int y = 0; y < rows; ++y)
    {
      for(int x = 0; x < cols; ++x)
        {
          out(y,x) =  sal.at<float>(y,x) ;
          //buf.push_back(saliencies[i].at<float>(y,x));
        }
    }

      char str[9];
      sprintf(str, "%05d.txt", atoi(argv[2]));
      const std::string out_dir(argv[3]);
      std::ofstream of(out_dir + "/" + str);
      of << out;
    return 0;
}
