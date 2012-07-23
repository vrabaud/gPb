/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#include "watershed.h"

namespace cv
{
  void
  watershedFull(const cv::Mat & image, cv::Mat & regions)
  {
    // Find the local minima
    regions = cv::Mat::zeros(image.rows, image.cols, CV_32S);
    int index = 1;
    int window_size = 10;
    for(int y = window_size; y < image.rows-window_size; ++y)
      for(int x = window_size; x < image.cols-window_size; ++x) {
          bool local_minimum = true;
          for(int yy=y-window_size; yy<=y+window_size; ++yy)
            for(int xx=x-window_size; xx<=x+window_size; ++xx) {
              if ((yy==y) && (xx==x))
                continue;
              if (image.at<unsigned char>(yy,xx) < image.at<unsigned char>(y,x))
                local_minimum = false;
            }
          if (local_minimum) {
            regions.at<int>(y,x) = index;
            ++index;
          }
        }

    // Apply watershed to those
    cv::Mat image3, imageu;
    if (image.channels()==1) {
      image.convertTo(imageu,CV_8U);
      cv::cvtColor(imageu, image3, CV_GRAY2RGB);
    } else {
      image.convertTo(imageu,CV_8UC3);
      image3 = imageu;
    }

    cv::watershed(image3, regions);
  }
}
