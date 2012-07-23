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
#include <list>

#include <opencv2/imgproc/imgproc.hpp>

#include "watershed.h"

namespace
{
  template<typename T>
  void
  findLocalMinima(const cv::Mat_<T> &image, cv::Mat & regions)
  {
    regions = cv::Mat::ones(image.rows, image.cols, CV_32S);
    int window_size = 1;
    std::list<std::pair<int, int> > queue;
    for (int y = window_size; y < image.rows - window_size; ++y)
      for (int x = window_size; x < image.cols - window_size; ++x)
      {
        // If we already know it's not a maximum
        if (!regions.at<int>(y, x))
          continue;
        bool local_minimum = true;
        for (int yy = y - window_size; yy <= y + window_size; ++yy)
          for (int xx = x - window_size; xx <= x + window_size; ++xx)
          {
            if ((yy == y) && (xx == x))
              continue;
            if (image(yy, xx) < image(y, x))
            {
              local_minimum = false;
              break;
            }
            else if (image(yy, xx) == image(y, x))
              queue.push_back(std::pair<int, int>(xx, yy));
          }
        if (local_minimum)
        {
          regions.at<int>(y, x) = 1;
          // Make sure that ridges are not local extrema
          queue.push_back(std::pair<int, int>(x, y));
          while (!queue.empty())
          {
            int x2 = queue.front().first, y2 = queue.front().second;
            for (int yy = y2 - window_size; yy <= y2 + window_size; ++yy)
              for (int xx = x2 - window_size; xx <= x2 + window_size; ++xx)
              {
                if ((yy == y2) && (xx == x2))
                  continue;
                if ((image(yy, xx) == image(y, x)) && regions.at<int>(yy, xx))
                {
                  regions.at<int>(yy, xx) = 0;
                  queue.push_back(std::pair<int, int>(xx, yy));
                }
              }
            queue.pop_front();
          }
          queue.clear();
        }
        else
          regions.at<int>(y, x) = 0;
      }
    // Set a proper index to every local extremum
    int index = 1;
    for (int y = window_size; y < image.rows - window_size; ++y)
      for (int x = window_size; x < image.cols - window_size; ++x)
        if (regions.at<int>(y, x))
          regions.at<int>(y, x) = index++;
  }
}

namespace cv
{
  void
  watershedFull(const cv::Mat & image, cv::Mat & regions)
  {
    // Find the local minima
    switch (image.depth())
    {
      case CV_64F:
      {
        const cv::Mat_<double> & image_t = cv::Mat_<double>(image);
        findLocalMinima(image_t, regions);
        break;
      }
      default:
        std::cerr << "Type not implemented in watershed: " << image.depth() << std::endl;
    }

    // Apply watershed to those
    cv::Mat image3, imageu;
    if (image.channels() == 1)
    {
      image.convertTo(imageu, CV_8U);
      cv::cvtColor(imageu, image3, CV_GRAY2RGB);
    }
    else
    {
      image.convertTo(imageu, CV_8UC3);
      image3 = imageu;
    }

    cv::watershed(image3, regions);
    // OpenCV convention: -1 for boundaries, zone index start a 0
    // Matlab convention: 0 for boundaries, zone index start a 1
    regions = regions + 1;
  }
}
