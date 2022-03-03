//
// Created by Олег Аксененко on 01.03.2022.
//

#ifndef GEOMETRYOBJECTDETECTORLIB_LINEVISUALIZER_H
#define GEOMETRYOBJECTDETECTORLIB_LINEVISUALIZER_H

#include <opencv2/highgui.hpp>

#include <iomanip>

#include "lineFinder.h"
#include "lineGrouper.h"
#include "lineProcess.h"

namespace lineProc {

    //static void setImages(const cv::Mat& lhsImage, const cv::Mat& rhsImage);
    //void lineFinderCallback(int, void *);
    void startViz();

}

#endif //GEOMETRYOBJECTDETECTORLIB_LINEVISUALIZER_H
