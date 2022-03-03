//
// Created by Олег Аксененко on 01.03.2022.
//

#include "lineFinder.h"

namespace lineProc
{
    void lineFinder::setMinThresh(int minThresh) noexcept {
       min_thresh = minThresh;
    }

    void lineFinder::setMaxThresh(int maxThresh) noexcept {
       max_thresh = maxThresh;
    }

    void lineFinder::setLengthThreshold(int lengthThreshold) noexcept {
       length_threshold = lengthThreshold;
    }

    void lineFinder::setDistanceThreshold(int distanceThreshold) noexcept {
       distance_threshold = distanceThreshold;
    }

    void lineFinder::setCannySize(int cannySize) noexcept {
       canny_size = cannySize;
    }

    void lineFinder::setLineLength(int lineLength) noexcept {
       line_length = lineLength;
    }

    cv::Mat lineFinder::readImage(const std::string &filename) const {
       cv::Mat image, smallerImage;
       image = cv::imread(filename, cv::IMREAD_ANYCOLOR);
       resize(image, smallerImage, cv::Size(), 0.5, 0.5, cv::INTER_CUBIC);
       cv::blur(smallerImage, smallerImage, cv::Size(3, 3));
       return smallerImage.clone();
    }

    std::vector<cv::Vec4i> lineFinder::findLines(const cv::Mat &target) const {
       cv::Mat detectedLinesImg = cv::Mat::zeros(target.rows, target.cols, CV_8UC3);
       cv::Mat reducedLinesImg = detectedLinesImg.clone();

       cv::Mat grayscale;
       cv::cvtColor(target, grayscale, cv::COLOR_BGRA2GRAY);

       cv::Ptr<cv::ximgproc::FastLineDetector> detector = cv::ximgproc::createFastLineDetector(
               length_threshold,
               distance_threshold * 1.414213562f,
               min_thresh,
               max_thresh,
               canny_size,
               false
       );
       std::vector<cv::Vec4i> lines;
       detector->detect(grayscale, lines);

       // remove small lines
       std::vector<cv::Vec4i> linesWithoutSmall;
       std::copy_if(lines.begin(), lines.end(), std::back_inserter(linesWithoutSmall), [this](const cv::Vec4f &line) {
           float length = sqrtf((line[2] - line[0]) * (line[2] - line[0])
                                + (line[3] - line[1]) * (line[3] - line[1]));
           return length > this->line_length;
       });

       return linesWithoutSmall;

    }

    void lineFinder::drawMatches(const cv::Mat& lhsImage, const cv::Mat& rhsImage,
                                 const std::vector<cv::Vec4i>& lhsImageLines,
                                 const std::vector<cv::Vec4i>& rhsImageLines,
                                 const std::vector<std::pair<size_t, size_t>>& linesDMatch) const {
         cv::Mat lhsImageCopy = lhsImage.clone();
         cv::Mat rhsImageCopy = rhsImage.clone();
         cv::Mat result;

         cv::RNG rng(12345);

         std::vector<cv::Scalar> colors;
         std::generate_n(std::back_inserter(colors), linesDMatch.size(), [&rng]() {
            return cv::Scalar(rng.uniform(80, 255), rng.uniform(80, 255), rng.uniform(80, 255));
         });

         for (size_t i = 0; i < linesDMatch.size(); ++i) {
            const cv::Vec4i& lhsRefLine = lhsImageLines[linesDMatch[i].first];
            const cv::Vec4i& rhsRefLine = rhsImageLines[linesDMatch[i].second];
            cv::line(lhsImageCopy, cv::Point(lhsRefLine[0], lhsRefLine[1]), cv::Point(lhsRefLine[2], lhsRefLine[3]),
                     colors[i], 3);
            cv::line(rhsImageCopy, cv::Point(rhsRefLine[0], rhsRefLine[1]), cv::Point(rhsRefLine[2], rhsRefLine[3]),
                     colors[i], 3);
         }
         cv::hconcat(lhsImageCopy, rhsImageCopy, result);
         cv::imshow("result", result);
         cv::waitKey();
    }

    void lineFinder::printParameters(std::ostream& os) {
       os << "Current min thresh: " << min_thresh << '\n';
       os << "Current max thresh: " << max_thresh << '\n';
       os << "Current length thresh: " << length_threshold << '\n';
       os << "Current distance thresh: " << distance_threshold * 1.414213562f << '\n';
       os << "Current canny size: " << canny_size << '\n';
       os << "Current line length thresh: " << line_length << '\n';
    }
}