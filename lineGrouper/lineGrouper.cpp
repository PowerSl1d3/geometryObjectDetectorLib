//
// Created by Олег Аксененко on 01.03.2022.
//

#include "lineGrouper.h"

namespace lineProc {
    void lineGrouper::setExtensionLengthFraction(int extensionLengthFraction) noexcept {
       lineGrouper::extensionLengthFraction = extensionLengthFraction;
    }

    void lineGrouper::setMaxAngleDiff(int maxAngleDiff) noexcept {
       lineGrouper::maxAngleDiff = maxAngleDiff;
    }

    void lineGrouper::setBoundingRectangleThickness(int boundingRectangleThickness) noexcept {
       lineGrouper::boundingRectangleThickness = boundingRectangleThickness;
    }

    std::vector<cv::Vec4i> lineGrouper::reduceLines(const std::vector<cv::Vec4i>& lines) const {

       // partition via our partitioning function
       std::vector<int> labels;

       int equivalenceClassesCount = cv::partition(lines, labels, [=](const cv::Vec4i& l1, const cv::Vec4i& l2){
          return lineProc::extendedBoundingRectangleLineEquivalence(
                  l1, l2,
                  // line extension length - as fraction of original line width
                  static_cast<float>(extensionLengthFraction) / 10,
                  // maximum allowed angle difference for lines to be considered in same equivalence class
                  static_cast<float>(maxAngleDiff),
                  // thickness of bounding rectangle around each line
                  static_cast<float>(boundingRectangleThickness));
       });

       // build point clouds out of each equivalence classes
       std::vector<std::vector<cv::Point2i>> pointClouds(equivalenceClassesCount);
       for (int i = 0; i < lines.size(); i++){
          const cv::Vec4i& detectedLine = lines[i];
          pointClouds[labels[i]].push_back(cv::Point2i(detectedLine[0], detectedLine[1]));
          pointClouds[labels[i]].push_back(cv::Point2i(detectedLine[2], detectedLine[3]));
       }

       // fit line to each equivalence class point cloud
       std::vector<cv::Vec4i> reducedLines = std::accumulate(pointClouds.begin(),
                                                             pointClouds.end(),
                                                             std::vector<cv::Vec4i>{},
                                                             [](std::vector<cv::Vec4i> target, const std::vector<cv::Point2i>& _pointCloud){

          std::vector<cv::Point2i> pointCloud = _pointCloud;

          //lineParams: [vx,vy, x0,y0]: (normalized vector, point on our contour)
          // (x,y) = (x0,y0) + t*(vx,vy), t -> (-inf; inf)
          cv::Vec4f lineParams; cv::fitLine(pointCloud, lineParams, cv::DIST_L2, 0, 0.01, 0.01);

          // derive the bounding xs of point cloud
          decltype(pointCloud)::iterator minXP, maxXP;
          std::tie(minXP, maxXP) = std::minmax_element(pointCloud.begin(), pointCloud.end(), [](const cv::Point2i& p1, const cv::Point2i& p2){ return p1.x < p2.x; });

          // derive y coords of fitted line
          float m = lineParams[1] / lineParams[0];
          int y1 = ((minXP->x - lineParams[2]) * m) + lineParams[3];
          int y2 = ((maxXP->x - lineParams[2]) * m) + lineParams[3];

          target.emplace_back(minXP->x, y1, maxXP->x, y2);
          return target;
       });

       std::vector<cv::Vec4i> linesWithoutSmall;
       std::copy_if (reducedLines.begin(), reducedLines.end(), std::back_inserter(linesWithoutSmall), [](const cv::Vec4f& line){
          float length = std::sqrtf((line[2] - line[0]) * (line[2] - line[0])
                  + (line[3] - line[1]) * (line[3] - line[1]));
          //TODO: !!! вынести параметром
          return length > 40;
       });

       return linesWithoutSmall;
    }

    std::vector<std::pair<size_t, size_t>> lineGrouper::computeMatch(const std::vector<cv::Vec4i>& lhsLines,
                                                                                   const std::vector<cv::Vec4i>& rhsLines) const {

       std::vector<std::pair<size_t, size_t>> answer;
       std::unordered_set<size_t> usedRhsLines;

       for (size_t i = 0; i < lhsLines.size(); ++i) {
          const cv::Vec4i& first = lhsLines[i];
          for (size_t j = 0; j < rhsLines.size(); ++j) {
             const cv::Vec4i& second = rhsLines[j];
             //TODO: сделать учёт расположений этих линий на самой картинке и подбирать только линии в окрестности этого местоположения
             if (
                     std::abs(lineLength(first) - lineLength(second)) < 100 and
                     std::abs(lineAngle(first) - lineAngle(second)) < 5 and
                     cv::norm(first, second) < 200 and not
                     usedRhsLines.count(j)
                     ) {
                answer.emplace_back(i, j);
                usedRhsLines.insert(j);
                break;
             }
          }
       }

       return answer;
    }

    void lineGrouper::printParameters(std::ostream& os) {
       os << "Current extensionLengthFraction: " << static_cast<float>(extensionLengthFraction) / 10 << '\n';
       os << "Current maxAngleDiff: " << static_cast<float>(maxAngleDiff) << '\n';
       os << "Current boundingRectangleThickness: " << static_cast<float>(boundingRectangleThickness) << '\n';
    }
}