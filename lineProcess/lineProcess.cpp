//
// Created by Олег Аксененко on 01.03.2022.
//

#include "lineProcess.h"

namespace lineProc
{
    enum class increaseMethod
    {
        UP = 0,
        DOWN
    };

    bool operator==(const cv::Matx23d &lhs, const cv::Matx23d &rhs) {
       return std::fabs(cv::norm(lhs, rhs, cv::NORM_L1)) < 0.001;
    }

    cv::Point2i mergePoints(const std::vector<cv::Point2i> &points) {

       float x_acc = 0.0f;
       float y_acc = 0.0f;
       for (const auto &point: points)
       {
          x_acc += point.x;
          y_acc += point.y;
       }
       x_acc /= points.size();
       y_acc /= points.size();
       return {static_cast<int>(x_acc), static_cast<int>(y_acc)};
    }

    void groupLinesInAngles(std::vector<cv::Vec4i> &lines, float radiusSearch) {

       struct refPoint
       {
           int &x_;
           int &y_;

           refPoint(int &x, int &y) :
                   x_(x),
                   y_(y) {}
       };

       std::function<float(const refPoint &, const refPoint &)> distanceBetweenPoint = [](const refPoint &lhs,
                                                                                          const refPoint &rhs) {
           return sqrtf((rhs.x_ - lhs.x_) * (rhs.x_ - lhs.x_) + (rhs.y_ - lhs.y_) * (rhs.y_ - lhs.y_));
       };

       struct point2iHash
       {
           std::hash<int> intHash;

           uint32_t operator()(const cv::Point2i &point) const {
              return intHash(point.x) + intHash(point.y);
           }
       };

       std::vector<cv::Point2i> pointsToMerge;
       std::vector<refPoint> refPoints;
       std::unordered_set<cv::Point2i, point2iHash> usedPoints;

       for (size_t i = 0; i < lines.size(); ++i)
       {

          for (size_t start = 0; start < 3; start += 2)
          {

             refPoint pointToMove(lines[i][start], lines[i][start + 1]);
             pointsToMerge.clear();
             refPoints.clear();

             if (usedPoints.count(cv::Point2i(lines[i][start], lines[i][start + 1])))
             {
                continue;
             }

             for (size_t j = 0; j < lines.size(); ++j)
             {

                refPoint firstComparePoint(lines[j][0], lines[j][1]);
                refPoint secondComparePoint(lines[j][2], lines[j][3]);

                if (j == i) continue;

                bool isFirstPointAdded = false;

                if (distanceBetweenPoint(pointToMove, firstComparePoint) < radiusSearch)
                {
                   pointsToMerge.emplace_back(firstComparePoint.x_, firstComparePoint.y_);
                   refPoints.emplace_back(firstComparePoint.x_, firstComparePoint.y_);
                   isFirstPointAdded = true;
                }

                if (distanceBetweenPoint(pointToMove, secondComparePoint) < radiusSearch and not isFirstPointAdded)
                {
                   pointsToMerge.emplace_back(secondComparePoint.x_, secondComparePoint.y_);
                   refPoints.emplace_back(secondComparePoint.x_, secondComparePoint.y_);
                }

             }

             if (not pointsToMerge.empty())
             {
                pointsToMerge.emplace_back(pointToMove.x_, pointToMove.y_);
                refPoints.emplace_back(pointToMove.x_, pointToMove.y_);
                auto newPoint = mergePoints(pointsToMerge);

                for (auto &point: refPoints)
                {
                   point.x_ = newPoint.x;
                   point.y_ = newPoint.y;
                }

                usedPoints.insert(newPoint);
             }

          }

       }
    }

    float lineLength(const cv::Vec4i &line) {
       return std::sqrtf((line[2] - line[0]) * (line[2] - line[0]) + (line[3] - line[1]) * (line[3] - line[1]));
    }

    float lineAngle(const cv::Vec4i &line) {
       if (line[0] - line[2] == 0) return 0.0f;
       return std::atan((line[1] - line[3]) / (line[0] - line[2])) * 180 / CV_PI;
    }

    cv::Vec2d linearParameters(cv::Vec4i line) {
       cv::Mat a = (cv::Mat_<double>(2, 2) <<
                                           line[0], 1,
               line[2], 1);
       cv::Mat y = (cv::Mat_<double>(2, 1) <<
                                           line[1],
               line[3]);
       cv::Vec2d mc;
       solve(a, y, mc);
       return mc;
    }

    cv::Vec4i increaseLineLength(const cv::Vec4i &line, int a, increaseMethod m) {

       auto p = linearParameters(line);

       float dy = line[3] - line[1];
       float dx = line[2] - line[0];

       float rel = dx / dy;
       float b = a * rel;

       cv::Vec4i answer = line;

       int rel_sign = rel > 0 ? 1 : -1;

       size_t first_index = 1;
       size_t second_index = 0;

       if (rel_sign > 0 and increaseMethod::DOWN == m or rel_sign < 0 and increaseMethod::UP == m)
       {
          first_index += 2;
          second_index += 2;
       }

       if (increaseMethod::UP == m)
       {
          answer[first_index] -= a;
          answer[second_index] -= b;
       } else
       {
          answer[first_index] += a;
          answer[second_index] += b;
       }

       return answer;
    }

    void increaseLinesLength(std::vector<cv::Vec4i> &lhsLines,
                             std::vector<cv::Vec4i> &rhsLines,
                             const std::vector<std::pair<size_t, size_t>> &lineDMatch) {
       for (const auto &match : lineDMatch)
       {

          const cv::Vec4i &lhsRefLine = lhsLines[match.first];
          const cv::Vec4i &rhsRefLine = rhsLines[match.second];

          int sign = (lhsRefLine[3] - lhsRefLine[1]) > 0 ? 1 : -1;
          int startPointDelta = std::abs(lhsRefLine[1] - rhsRefLine[1]);
          int endPointDelta = std::abs(lhsRefLine[3] - rhsRefLine[3]);

          cv::Vec4i newLhsLine = lhsRefLine, newRhsLine = rhsRefLine;

          if (sign > 0 and newLhsLine[1] < newRhsLine[1])
          {
             newRhsLine = increaseLineLength(newRhsLine, startPointDelta, increaseMethod::UP);
          } else if (sign > 0 and newLhsLine[1] > newRhsLine[1])
          {
             newLhsLine = increaseLineLength(newLhsLine, startPointDelta, increaseMethod::UP);
          } else if (sign < 0 and newLhsLine[1] < newRhsLine[1])
          {
             newLhsLine = increaseLineLength(newLhsLine, startPointDelta, increaseMethod::DOWN);
          } else if (sign < 0 and newLhsLine[1] > newRhsLine[1])
          {
             newRhsLine = increaseLineLength(newRhsLine, startPointDelta, increaseMethod::DOWN);
          }

          if (sign > 0 and newLhsLine[3] < newRhsLine[3])
          {
             newLhsLine = increaseLineLength(newLhsLine, endPointDelta, increaseMethod::DOWN);
          } else if (sign > 0 and newLhsLine[3] > newRhsLine[3])
          {
             newRhsLine = increaseLineLength(newRhsLine, endPointDelta, increaseMethod::DOWN);
          } else if (sign < 0 and newLhsLine[3] < newRhsLine[3])
          {
             newRhsLine = increaseLineLength(newRhsLine, endPointDelta, increaseMethod::UP);
          } else if (sign < 0 and newLhsLine[3] > newRhsLine[3])
          {
             newLhsLine = increaseLineLength(newLhsLine, endPointDelta, increaseMethod::UP);
          }

          lhsLines[match.first] = newLhsLine;
          rhsLines[match.second] = newRhsLine;
       }
    }

    cv::Vec4i extendedLine(cv::Vec4i line, double d) {
       // oriented left-t-right
       cv::Vec4d _line =
               line[2] - line[0] < 0 ? cv::Vec4d(line[2], line[3], line[0], line[1]) : cv::Vec4d(line[0], line[1],
                                                                                                 line[2], line[3]);
       double m = linearParameters(_line)[0];
       // solution of pythagorean theorem and m = yd/xd
       double xd = std::sqrt(d * d / (m * m + 1));
       double yd = xd * m;
       return cv::Vec4d(_line[0] - xd, _line[1] - yd, _line[2] + xd, _line[3] + yd);
    }

    std::vector<cv::Point2i> lineContext(cv::Vec4i line, float d) {

       cv::Vec2f mc = linearParameters(line);
       float m = mc[0];
       float factor = sqrtf(
               (d * d) / (1 + (1 / (m * m)))
       );

       float x3, y3, x4, y4, x5, y5, x6, y6;
       // special case(vertical perpendicular line) when -1/m -> -infinity
       if (fabs(m) < 0.00001)
       {
          x3 = static_cast<float>(line[0]);
          y3 = static_cast<float>(line[1]) + (d > 0 ? d : 0);
          x4 = static_cast<float>(line[0]);
          y4 = static_cast<float>(line[1]) - (d > 0 ? 0 : d);
          x5 = static_cast<float>(line[2]);
          y5 = static_cast<float>(line[3]) + (d > 0 ? d : 0);
          x6 = static_cast<float>(line[2]);
          y6 = static_cast<float>(line[3]) - (d > 0 ? 0 : d);
       } else
       {
          // slope of perpendicular lines
          float m_per = -1 / m;

          // y1 = m_per * x1 + c_per
          float c_per1 = static_cast<float>(line[1]) - m_per * static_cast<float>(line[0]);
          float c_per2 = static_cast<float>(line[3]) - m_per * static_cast<float>(line[2]);

          // coordinates of perpendicular lines
          x3 = static_cast<float>(line[0]) + (d > 0 ? factor : 0);
          y3 = m_per * x3 + c_per1;
          x4 = static_cast<float>(line[0]) - (d > 0 ? 0 : factor);
          y4 = m_per * x4 + c_per1;
          x5 = static_cast<float>(line[2]) + (d > 0 ? factor : 0);
          y5 = m_per * x5 + c_per2;
          x6 = static_cast<float>(line[2]) - (d > 0 ? 0 : factor);
          y6 = m_per * x6 + c_per2;
       }

       return std::vector<cv::Point2i>{
               cv::Point2i(static_cast<int>(x3), static_cast<int>(y3)),
               cv::Point2i(static_cast<int>(x4), static_cast<int>(y4)),
               cv::Point2i(static_cast<int>(x6), static_cast<int>(y6)),
               cv::Point2i(static_cast<int>(x5), static_cast<int>(y5))
       };
    }

    std::vector<bool> computeGoodMatches(const cv::Mat &lhsImage,
                                         const cv::Mat &rhsImage,
                                         const std::vector<cv::Vec4i> &lhsLines,
                                         const std::vector<cv::Vec4i> &rhsLines,
                                         const std::vector<std::pair<size_t, size_t>> &linesDMatch) {

       std::vector<bool> goodMatches(linesDMatch.size(), true);

       int b_bins = 50, g_bins = 50, r_bins = 50;
       int histSize[] = {b_bins, g_bins, r_bins};

       float b_ranges[] = {0, 256};
       float g_ranges[] = {0, 256};
       float r_ranges[] = {0, 256};

       const float *ranges[] = {b_ranges, g_ranges, r_ranges};

       int channels[] = {0, 1, 2};

       cv::Mat lhsMask = cv::Mat(lhsImage.size(), CV_8U);
       cv::Mat rhsMask = lhsMask.clone();
       cv::Mat lhsImageHist, rhsImageHist;

       for (size_t i = 0; i < linesDMatch.size(); ++i)
       {

          const cv::Vec4i &lhsRefLine = lhsLines[linesDMatch[i].first];
          const cv::Vec4i &rhsRefLine = rhsLines[linesDMatch[i].second];

          int bounding_size = 10;

          std::vector<cv::Point2i> lhsUpperLineBoundingContour = boundingRectangleContour(lhsRefLine, bounding_size);
          std::vector<cv::Point2i> rhsUpperLineBoundingContour = boundingRectangleContour(rhsRefLine, bounding_size);

          lhsMask = 0;
          rhsMask = 0;

          cv::drawContours(lhsMask, std::vector<std::vector<cv::Point2i>>(1, lhsUpperLineBoundingContour), 0, 255,
                           cv::LineTypes::FILLED);
          cv::drawContours(rhsMask, std::vector<std::vector<cv::Point2i>>(1, rhsUpperLineBoundingContour), 0, 255,
                           cv::LineTypes::FILLED);

          cv::calcHist(&lhsImage, 1, channels, lhsMask, lhsImageHist, 3, histSize, ranges, true, false);
          cv::calcHist(&rhsImage, 1, channels, rhsMask, rhsImageHist, 3, histSize, ranges, true, false);

          cv::normalize(lhsImageHist, lhsImageHist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
          cv::normalize(rhsImageHist, rhsImageHist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

          double compareResult = cv::compareHist(lhsImageHist, rhsImageHist, cv::HistCompMethods::HISTCMP_CORREL);

          if (compareResult < 0.2)
          {
             goodMatches[i] = false;
          }

          std::vector<cv::Point2i> lhsLowerLineBoundingContour = boundingRectangleContour(lhsRefLine, -bounding_size);
          std::vector<cv::Point2i> rhsLowerLineBoundingContour = boundingRectangleContour(rhsRefLine, -bounding_size);

          lhsMask = 0;
          rhsMask = 0;

          cv::drawContours(lhsMask, std::vector<std::vector<cv::Point2i>>(1, lhsLowerLineBoundingContour), 0, 255,
                           cv::LineTypes::FILLED);
          cv::drawContours(rhsMask, std::vector<std::vector<cv::Point2i>>(1, rhsLowerLineBoundingContour), 0, 255,
                           cv::LineTypes::FILLED);

          cv::calcHist(&lhsImage, 1, channels, lhsMask, lhsImageHist, 3, histSize, ranges, true, false);
          cv::calcHist(&rhsImage, 1, channels, rhsMask, rhsImageHist, 3, histSize, ranges, true, false);

          cv::normalize(lhsImageHist, lhsImageHist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
          cv::normalize(rhsImageHist, rhsImageHist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

          compareResult = cv::compareHist(lhsImageHist, rhsImageHist, cv::HistCompMethods::HISTCMP_CORREL);

          if (compareResult < 0.2)
          {
             goodMatches[i] = false;
          }
       }

       return goodMatches;
    }

    std::vector<cv::Point2i> boundingRectangleContour(cv::Vec4i line, float d) {
       // finds coordinates of perpendicular lines with length d in both line points
       // https://math.stackexchange.com/a/2043065/183923

       cv::Vec2f mc = linearParameters(line);
       float m = mc[0];
       float factor = std::sqrtf(
               (d * d) / (1 + (1 / (m * m)))
       );

       float x3, y3, x4, y4, x5, y5, x6, y6;
       // special case(vertical perpendicular line) when -1/m -> -infinity
       if (std::fabs(m) < 0.00001)
       {
          x3 = static_cast<float>(line[0]);
          y3 = static_cast<float>(line[1]) + d;
          x4 = static_cast<float>(line[0]);
          y4 = static_cast<float>(line[1]) - d;
          x5 = static_cast<float>(line[2]);
          y5 = static_cast<float>(line[3]) + d;
          x6 = static_cast<float>(line[2]);
          y6 = static_cast<float>(line[3]) - d;
       } else
       {
          // slope of perpendicular lines
          float m_per = -1 / m;

          // y1 = m_per * x1 + c_per
          float c_per1 = static_cast<float>(line[1]) - m_per * static_cast<float>(line[0]);
          float c_per2 = static_cast<float>(line[3]) - m_per * static_cast<float>(line[2]);

          // coordinates of perpendicular lines
          x3 = static_cast<float>(line[0]) + factor;
          y3 = m_per * x3 + c_per1;
          x4 = static_cast<float>(line[0]) - factor;
          y4 = m_per * x4 + c_per1;
          x5 = static_cast<float>(line[2]) + factor;
          y5 = m_per * x5 + c_per2;
          x6 = static_cast<float>(line[2]) - factor;
          y6 = m_per * x6 + c_per2;
       }

       return std::vector<cv::Point2i>{
               cv::Point2i(static_cast<int>(x3), static_cast<int>(y3)),
               cv::Point2i(static_cast<int>(x4), static_cast<int>(y4)),
               cv::Point2i(static_cast<int>(x6), static_cast<int>(y6)),
               cv::Point2i(static_cast<int>(x5), static_cast<int>(y5))
       };
    }

    bool
    extendedBoundingRectangleLineEquivalence(const cv::Vec4i &_l1, const cv::Vec4i &_l2, float extensionLengthFraction,
                                             float maxAngleDiff, float boundingRectangleThickness) {

       cv::Vec4i l1(_l1), l2(_l2);
       // extend lines by percentage of line width
       float len1 = std::sqrtf((l1[2] - l1[0]) * (l1[2] - l1[0]) + (l1[3] - l1[1]) * (l1[3] - l1[1]));
       float len2 = std::sqrtf((l2[2] - l2[0]) * (l2[2] - l2[0]) + (l2[3] - l2[1]) * (l2[3] - l2[1]));
       cv::Vec4i el1 = extendedLine(l1, len1 * extensionLengthFraction);
       cv::Vec4i el2 = extendedLine(l2, len2 * extensionLengthFraction);

       // reject the lines that have wide difference in angles
       float a1 = std::atan(linearParameters(el1)[0]);
       float a2 = std::atan(linearParameters(el2)[0]);
       if (std::fabs(a1 - a2) > maxAngleDiff * M_PI / 180.0)
       {
          return false;
       }

       // calculate window around extended line
       // at least one point needs to inside extended bounding rectangle of other line,
       std::vector<cv::Point2i> lineBoundingContour = boundingRectangleContour(el1, boundingRectangleThickness / 2);
       return
               cv::pointPolygonTest(lineBoundingContour, cv::Point(el2[0], el2[1]), false) == 1 ||
               cv::pointPolygonTest(lineBoundingContour, cv::Point(el2[2], el2[3]), false) == 1;
    }

    std::vector<cv::Vec4i> transformLines(const std::vector<cv::Vec4i> &lines, const cv::Matx23d &warpMat) {
       std::vector<cv::Vec4i> translatedLines;
       translatedLines.reserve(lines.size());

       cv::Matx31d lineStartPoint(0, 0, 1);
       cv::Matx31d lineEndPoint(0, 0, 1);

       for (const auto &lhsImageLine: lines)
       {

          lineStartPoint(0) = lhsImageLine[0];
          lineStartPoint(1) = lhsImageLine[1];
          lineEndPoint(0) = lhsImageLine[2];
          lineEndPoint(1) = lhsImageLine[3];

          auto newLineStartPoint = warpMat * lineStartPoint;
          auto newLineEndPoint = warpMat * lineEndPoint;

          translatedLines.emplace_back(newLineStartPoint(0), newLineStartPoint(1), newLineEndPoint(0),
                                       newLineEndPoint(1));
       }
       return translatedLines;
    }

    cv::Point3d create3DPoint(cv::Point2d keyPoints1, cv::Point2d keyPoints2, int imageWidth, int imageHeight) {

       struct SRay2Params
       {
           cv::Point3d c1, c2;   // кратчайший отрезок между лучами
           double dist;   // длина отрезка
           cv::Point3d d1, d2; // направляющие единичные вектора
           double horzDeviation; // отклонение вектора, соединяющего реперы от горизонтали, в градусах
           int index;      // индекс пары в массиве
           SRay2Params() :
                   c1(0.0, 0.0, 0.0),
                   c2(0.0, 0.0, 0.0),
                   dist(0.0),
                   d1(0.0, 0.0, 0.0),
                   d2(0.0, 0.0, 0.0),
                   horzDeviation(0.0),
                   index(0) {}
       };

       cv::Point3d p1(0, 0, 0);              // точка фокуса левой камеры
       cv::Point3d p2(0.3f, 0, 0);                   // точка фокуса правой камеры

       SRay2Params rayParams;

       double x1, y1, fx1, fy1, cx1, cy1, x2, y2, fx2, fy2, cx2, cy2;

       fx1 = 389.7114317029974;
       fy1 = 389.7114317029974;
       cx1 = 300;
       cy1 = 225;
       fx2 = 389.7114317029974;
       fy2 = 389.7114317029974;
       cx2 = 300;
       cy2 = 225;

       /*fx1 = 779.42286340599480;
       fy1 = 779.42286340599480;
       cx1 = 600;
       cy1 = 450;
       fx2 = 779.42286340599480;
       fy2 = 779.42286340599480;
       cx2 = 600;
       cy2 = 450;*/

       x1 = keyPoints1.x;
       y1 = keyPoints1.y;

       x2 = keyPoints2.x;
       y2 = keyPoints2.y;

       if (x1 == x2)
       { // вектор вертикальный
          rayParams.horzDeviation = 180.0;
       } else
       {
          rayParams.horzDeviation = fabs(atan((y2 - y1) / (x2 - x1)) * 180 / CV_PI);
       }

       //cv::Point3d d1((x1 - cx1) / fx1, (cy1 - y1) / fy1, 1.0);  // (!) переворачиваем по оси Y
       //cv::Point3d d2((x2 - cx2) / fx2, (cy2 - y2) / fy2, 1.0);  // (!) переворачиваем по оси Y
       cv::Point3d d1((x1 - cx1) / fx1, (y1 - cy1) / fy1, 1.0);  // (!) переворачиваем по оси Y
       cv::Point3d d2((x2 - cx2) / fx2, (y2 - cy2) / fy2, 1.0);  // (!) переворачиваем по оси Y

       d1 /= cv::norm(d1);
       d2 /= cv::norm(d2);

       rayParams.d1 = d1;
       rayParams.d2 = d2;

       cv::Point3d n1 = d1.cross(d2.cross(d1));
       cv::Point3d n2 = d2.cross(d1.cross(d2));

       cv::Point3d c1 = p1 + (((p2 - p1).dot(n2)) / (d1.dot(n2))) * d1;
       cv::Point3d c2 = p2 + (((p1 - p2).dot(n1)) / (d2.dot(n1))) * d2;

       rayParams.c1 = c1;
       rayParams.c2 = c2;
       rayParams.dist = cv::norm(c2 - c1);

       cv::Point3d c3 = (rayParams.c1 + rayParams.c2) / 2.0;

       return c3;
    }

    std::vector<cv::Point2i> generatePoints(const cv::Vec4i &line, int count) {

       std::vector<cv::Point2i> answer;

       if (count)
       {
          std::vector<cv::Point2i> lhsPoints, rhsPoints;
          cv::Point2i newPoint((line[0] + line[2]) / 2, (line[1] + line[3]) / 2);

          lhsPoints = generatePoints({line[0], line[1], newPoint.x, newPoint.y}, count - 1);
          rhsPoints = generatePoints({newPoint.x, newPoint.y, line[2], line[3]}, count - 1);

          answer.insert(answer.end(), lhsPoints.begin(), lhsPoints.end());
          answer.push_back(newPoint);
          answer.insert(answer.end(), rhsPoints.begin(), rhsPoints.end());
       }

       return answer;
    }

    std::vector<std::vector<cv::Point3d>> createPointCloud(const std::vector<cv::Vec4i> &lhsImageLines,
                                                           const std::vector<cv::Vec4i> &rhsImageLines,
                                                           const std::vector<std::pair<size_t, size_t>> &lineDMatch,
                                                           const std::vector<bool> &mask,
                                                           int numberOfApproximatePoints,
                                                           int imageWidth,
                                                           int imageHeight) {

       std::vector<std::vector<cv::Point3d>> pointCloud(mask.size(), std::vector<cv::Point3d>());

       for (size_t i = 0; i < mask.size(); ++i)
       {
          if (mask[i])
          {
             auto lhsRefLine = lhsImageLines[lineDMatch[i].first];
             auto rhsRefLine = rhsImageLines[lineDMatch[i].second];

             std::vector<cv::Point2i> lhsLineApproximatePoints = generatePoints(lhsRefLine, numberOfApproximatePoints);
             std::vector<cv::Point2i> rhsLineApproximatePoints = generatePoints(rhsRefLine, numberOfApproximatePoints);

             cv::Point3d lineStart =
                     {(lhsRefLine[0] + rhsRefLine[0]) / 2.0,
                      static_cast<double>(lhsRefLine[1]),
                      create3DPoint(cv::Point2d(lhsRefLine[0], lhsRefLine[1]),
                                    cv::Point2d(rhsRefLine[0], rhsRefLine[1]),
                                    imageWidth, imageHeight).z
                     };

             pointCloud[i].emplace_back(std::move(lineStart));

             for (size_t j = 0; j < lhsLineApproximatePoints.size(); ++j)
             {
                pointCloud[i].emplace_back((lhsLineApproximatePoints[j].x + rhsLineApproximatePoints[j].x) / 2.0,
                                           static_cast<double>(lhsLineApproximatePoints[j].y),
                                           create3DPoint(cv::Point2d(lhsLineApproximatePoints[j].x,
                                                                     lhsLineApproximatePoints[j].y),
                                                         cv::Point2d(rhsLineApproximatePoints[j].x,
                                                                     rhsLineApproximatePoints[j].y),
                                                         imageWidth,
                                                         imageHeight).z);
             }

             cv::Point3d lineEnd = {(lhsRefLine[2] + rhsRefLine[2]) / 2.0,
                                    static_cast<double>(lhsRefLine[3]),
                                    create3DPoint(cv::Point2d(lhsRefLine[2], lhsRefLine[3]),
                                                  cv::Point2d(rhsRefLine[2], rhsRefLine[3]),
                                                  imageWidth, imageHeight).z
             };
             pointCloud[i].emplace_back(std::move(lineEnd));
          }
       }

       return pointCloud;
    }

    std::vector<cv::Vec4i> filterLines(const std::vector<cv::Vec4i> &lines,
                                       const std::vector<std::pair<size_t, size_t>> &match,
                                       const std::vector<bool> &mask) {

       std::vector<cv::Vec4i> filteredLines;

       for (size_t i = 0; i < match.size(); ++i)
       {
          if (mask[i])
          {
             filteredLines.push_back(lines[match[i].first]);
          }
       }
       return filteredLines;
    }

    pcl::PointXYZ convertToPclPoint(const cv::Point3d &point) {
       return {static_cast<float>(point.x), static_cast<float>(point.y), static_cast<float>(point.z)};
    }

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
            int approximateNumber
    ) {

       std::vector<std::vector<cv::Point3d>> firstScenePointCloud = lineProc::createPointCloud(
               firstTrajectoryLhsLines,
               firstTrajectoryRhsLines,
               firstTrajectoryLinesDMatch,
               firstTrajectoryMask,
               4,
               firstTrajectoryLhsImageWidth,
               firstTrajectoryLhsImageHeight
       );

       std::vector<std::vector<cv::Point3d>> secondScenePointCloud = lineProc::createPointCloud(
               secondTrajectoryLhsLines,
               secondTrajectoryRhsLines,
               secondTrajectoryLinesDMatch,
               secondTrajectoryMask,
               4,
               secondTrajectoryLhsImageWidth,
               secondTrajectoryLhsImageHeight
       );

       pcl::PointCloud<pcl::PointXYZ>::Ptr firstSceneTriangulatedPoints(new pcl::PointCloud<pcl::PointXYZ>);
       pcl::PointCloud<pcl::PointXYZ>::Ptr secondSceneTriangulatedPoints(new pcl::PointCloud<pcl::PointXYZ>);

       for (const auto &points: firstScenePointCloud)
       {
          for (const auto &point: points)
             firstSceneTriangulatedPoints->push_back(lineProc::convertToPclPoint(point));
       }
       for (const auto &points: secondScenePointCloud)
       {
          for (const auto &point: points)
             secondSceneTriangulatedPoints->push_back(lineProc::convertToPclPoint(point));
       }

       pcl::PointCloud<pcl::PointXYZ>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZ>);

       pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

       icp.setMaximumIterations(20);
       icp.setInputSource(firstSceneTriangulatedPoints);
       icp.setInputTarget(secondSceneTriangulatedPoints);
       icp.align(*transformedCloud);

       return icp.getFinalTransformation().cast<double>();

    }
}