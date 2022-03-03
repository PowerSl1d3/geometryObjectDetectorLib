//
// Created by Олег Аксененко on 01.03.2022.
//

#ifndef GEOMETRYOBJECTDETECTORLIB_LINEGROUPER_H
#define GEOMETRYOBJECTDETECTORLIB_LINEGROUPER_H

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

#include "lineProcess.h"

namespace lineProc {
    class lineGrouper {
    private:
        int extensionLengthFraction = 17;

        int maxAngleDiff = 7;

        int boundingRectangleThickness = 10;
    public:
        void setExtensionLengthFraction(int extensionLengthFraction) noexcept;

        void setMaxAngleDiff(int maxAngleDiff) noexcept;

        void setBoundingRectangleThickness(int boundingRectangleThickness) noexcept;

        std::vector<cv::Vec4i> reduceLines(const std::vector<cv::Vec4i>& lines) const;

        std::vector<std::pair<size_t, size_t>> computeMatch(const std::vector<cv::Vec4i>& lhsLines,
                                                            const std::vector<cv::Vec4i>& rhsLines) const;

        void printParameters(std::ostream &os);
    };
}

#endif //GEOMETRYOBJECTDETECTORLIB_LINEGROUPER_H
