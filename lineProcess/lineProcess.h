//
// Created by Олег Аксененко on 01.03.2022.
//

#ifndef GEOMETRYOBJECTDETECTORLIB_LINEPROCESS_H
#define GEOMETRYOBJECTDETECTORLIB_LINEPROCESS_H

#include <vector>
#include <cmath>
#include <unordered_set>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>

namespace lineProc
{

    enum class increaseMethod;

    bool operator==(const cv::Matx23d &lhs, const cv::Matx23d &rhs);

    /**
     * @param points - множество точек, представляющих один угол для которых необходимо вычислить центр масс или новую позицию угловой точки
    * @return новая угловая точка
    */
    cv::Point2i mergePoints(const std::vector<cv::Point2i> &points);

    /**
    * @param lines
    * @param radiusSearch - радиус в области которого каждый конец линий будет группироваться с концами другой линии
    * @return линии, сгрупированные в углах
    */
    void groupLinesInAngles(std::vector<cv::Vec4i> &lines, float radiusSearch);

    float lineLength(const cv::Vec4i &line);

    float lineAngle(const cv::Vec4i &line);

    cv::Vec2d linearParameters(cv::Vec4i line);

    cv::Vec4i increaseLineLength(const cv::Vec4i &line, int a, increaseMethod m);

    void increaseLinesLength(std::vector<cv::Vec4i> &lhsLines,
                             std::vector<cv::Vec4i> &rhsLines,
                             const std::vector<std::pair<size_t, size_t>> &lineDMatch);

    cv::Vec4i extendedLine(cv::Vec4i line, double d);

    std::vector<cv::Point2i> lineContext(cv::Vec4i line, float d);

    std::vector<bool> computeGoodMatches(const cv::Mat &lhsImage,
                                         const cv::Mat &rhsImage,
                                         const std::vector<cv::Vec4i> &lhsLines,
                                         const std::vector<cv::Vec4i> &rhsLines,
                                         const std::vector<std::pair<size_t, size_t>> &linesDMatch);

    std::vector<cv::Point2i> boundingRectangleContour(cv::Vec4i line, float d);

    bool extendedBoundingRectangleLineEquivalence(const cv::Vec4i &_l1, const cv::Vec4i &_l2,
                                                  float extensionLengthFraction,
                                                  float maxAngleDiff,
                                                  float boundingRectangleThickness);

    std::vector<cv::Vec4i> transformLines(const std::vector<cv::Vec4i> &lines, const cv::Matx23d &warpMat);

    cv::Point3d create3DPoint(cv::Point2d keyPoints1, cv::Point2d keyPoints2, int imageWidth, int imageHeight);

    /**
     * @warning сложность функции: 2^n, где n - count
     * @param line
     * @param count
     * @return
     */
    std::vector<cv::Point2i> generatePoints(const cv::Vec4i &line, int count);

    std::vector<std::vector<cv::Point3d>> createPointCloud(const std::vector<cv::Vec4i> &lhsImageLines,
                                                           const std::vector<cv::Vec4i> &rhsImageLines,
                                                           const std::vector<std::pair<size_t, size_t>> &lineDMatch,
                                                           const std::vector<bool> &mask,
                                                           int numberOfApproximatePoints,
                                                           int imageWidth,
                                                           int imageHeight);

    std::vector<cv::Vec4i> filterLines(const std::vector<cv::Vec4i> &lines,
                                       const std::vector<std::pair<size_t, size_t>> &match,
                                       const std::vector<bool> &mask);

    pcl::PointXYZ convertToPclPoint(const cv::Point3d &point);

    Eigen::Matrix4d computeTransformMatrix(
            const std::vector<cv::Vec4i> &firstTrajectoryLhsLines,
            const std::vector<cv::Vec4i> &firstTrajectoryRhsLines,
            const std::vector<std::pair<size_t, size_t>> &firstTrajectoryLinesDMatch,
            const std::vector<bool> &firstTrajectoryMask,
            int firstTrajectoryLhsImageWidth,
            int firstTrajectoryLhsImageHeight,
            const std::vector<cv::Vec4i> &secondTrajectoryLhsLines,
            const std::vector<cv::Vec4i> &secondTrajectoryRhsLines,
            const std::vector<std::pair<size_t, size_t>> &secondTrajectoryLinesDMatch,
            const std::vector<bool> &secondTrajectoryMask,
            int secondTrajectoryLhsImageWidth,
            int secondTrajectoryLhsImageHeight,
            int approximateNumber = 4
    );
}

#endif //GEOMETRYOBJECTDETECTORLIB_LINEPROCESS_H
