//
// Created by Олег Аксененко on 01.03.2022.
//

#ifndef GEOMETRYOBJECTDETECTORLIB_LINEFINDER_H
#define GEOMETRYOBJECTDETECTORLIB_LINEFINDER_H

#include <vector>
#include <iostream>
#include <numeric>
#include <future>
#include <unordered_set>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/highgui.hpp>

namespace lineProc {

    class lineFinder {
    private:

        /*
         * минимальное значение для фильтра Canny
         */
        int min_thresh = 36;

        /*
         * максимальное значение для фильтра Canny
         */
        int max_thresh = 122;

        /*
         * пороговое значение длин линий для фильтра Canny
         */
        int length_threshold = 32;


        /*
         * пороговое значение отбрасывания линий, разрыв в которых больше чем значение параметра
         */
        int distance_threshold = 1;

        /*
         * размер фильтра Canny
         */
        int canny_size = 3;

        /*
         * минимальная длина линий
         */
        int line_length = 40;

    public:

        void setMinThresh(int minThresh) noexcept;

        void setMaxThresh(int maxThresh) noexcept;

        void setLengthThreshold(int lengthThreshold) noexcept;

        void setDistanceThreshold(int distanceThreshold) noexcept;

        void setCannySize(int cannySize) noexcept;

        void setLineLength(int lineLength) noexcept;

        cv::Mat readImage(const std::string &filename) const;

        std::vector<cv::Vec4i> findLines(const cv::Mat &target) const;

        void drawMatches(const cv::Mat& lhsImage, const cv::Mat& rhsImage,
                         const std::vector<cv::Vec4i>& lhsImageLines,
                         const std::vector<cv::Vec4i>& rhsImageLines,
                         const std::vector<std::pair<size_t, size_t>>& linesDMatch) const;

        void printParameters(std::ostream &os);
    };
}
#endif //GEOMETRYOBJECTDETECTORLIB_LINEFINDER_H
