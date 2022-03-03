//
// Created by Олег Аксененко on 01.03.2022.
//

#include "lineVisualizer.h"

namespace lineProc
{
    static cv::RNG rng(12345);

    static std::vector<cv::Mat> lhsImages;
    static std::vector<cv::Mat> rhsImages;

    static cv::Mat lhsImage;
    static cv::Mat rhsImage;

    static cv::Mat result;

    static lineFinder lf;
    static lineGrouper lg;

    static int currentImagePairIndex = 0;

    static int min_thresh = 36;
    static int max_thresh = 122;
    static int length_threshold = 32;
    static int distance_threshold = 1;
    static int canny_size = 3;
    static int line_length = 40;

    static int extensionLengthFraction = 12;
    static int maxAngleDiff = 15;
    static int boundingRectangleThickness = 24;

    /*
     * static int min_thresh = 36;
    static int max_thresh = 122;
    static int length_threshold = 32;
    static int distance_threshold = 1;
    static int canny_size = 3;
    static int line_length = 40;

    static int extensionLengthFraction = 17;
    static int maxAngleDiff = 7;
    static int boundingRectangleThickness = 10;
     */

    std::pair<std::string, std::string> generateImagePair(const std::string& templ, const size_t index) {
       std::ostringstream oss;
       oss << std::setw(5) << std::setfill('0') << index;
       return {"/Users/olegaksenenko/Downloads/frames_KomplexSpiral_2021_03_30/" + templ +
       "_" + oss.str() + "_0.bmp",
       "/Users/olegaksenenko/Downloads/frames_KomplexSpiral_2021_03_30/" + templ +
       "_" + oss.str() + "_1.bmp"};
    }

    void lineFinderCallback(int, void *) {

      lhsImage = lhsImages[currentImagePairIndex];
      rhsImage = rhsImages[currentImagePairIndex];

       cv::Mat detectedLinesLhsImg = cv::Mat::zeros(lhsImage.rows, lhsImage.cols, CV_8UC3);
       cv::Mat detectedLinesRhsImg = cv::Mat::zeros(rhsImage.rows, rhsImage.cols, CV_8UC3);

       lf.setMinThresh(min_thresh);
       lf.setMaxThresh(max_thresh);
       lf.setLengthThreshold(length_threshold);
       lf.setDistanceThreshold(distance_threshold);
       if (canny_size < 3)
       {
          lf.setCannySize(3);
       } else if (canny_size == 4)
       {
          lf.setCannySize(5);
       } else if (canny_size == 6)
       {
          lf.setCannySize(7);
       } else
       {
          lf.setCannySize(canny_size);
       }
       lf.setLineLength(line_length);

       lf.printParameters(std::cout);

       auto lhsLines = lf.findLines(lhsImage);
       auto rhsLines = lf.findLines(rhsImage);

       auto linesDMatch = lg.computeMatch(lhsLines, rhsLines);

       std::vector<cv::Scalar> colors;
       std::generate_n(std::back_inserter(colors), linesDMatch.size(), []() {
           return cv::Scalar(rng.uniform(80, 255), rng.uniform(80, 255), rng.uniform(80, 255));
       });

       for (size_t i = 0; i < linesDMatch.size(); ++i)
       {
          const cv::Vec4i &lhsRefLine = lhsLines[linesDMatch[i].first];
          const cv::Vec4i &rhsRefLine = rhsLines[linesDMatch[i].second];

          cv::line(detectedLinesLhsImg,
                   cv::Point(lhsRefLine[0], lhsRefLine[1]),
                   cv::Point(lhsRefLine[2], lhsRefLine[3]),
                   colors[i],
                   3);
          cv::line(detectedLinesRhsImg,
                   cv::Point(rhsRefLine[0], rhsRefLine[1]),
                   cv::Point(rhsRefLine[2], rhsRefLine[3]),
                   colors[i],
                   3);
       }

       cv::hconcat(detectedLinesLhsImg, detectedLinesRhsImg, result);
       cv::imshow("Detected Lines", result);

    }

    void lineGrouperCallback(int, void *) {

       lhsImage = lhsImages[currentImagePairIndex];
       rhsImage = rhsImages[currentImagePairIndex];

       cv::Mat detectedLinesLhsImg = cv::Mat::zeros(lhsImage.rows, lhsImage.cols, CV_8UC3);
       cv::Mat detectedLinesRhsImg = cv::Mat::zeros(rhsImage.rows, rhsImage.cols, CV_8UC3);

       lg.setExtensionLengthFraction(extensionLengthFraction);
       lg.setMaxAngleDiff(maxAngleDiff);
       lg.setBoundingRectangleThickness(boundingRectangleThickness);

       lg.printParameters(std::cout);

       auto lhsLines = lf.findLines(lhsImage);
       auto rhsLines = lf.findLines(rhsImage);

       lhsLines = lg.reduceLines(lhsLines);
       rhsLines = lg.reduceLines(rhsLines);

       auto linesDMatch = lg.computeMatch(lhsLines, rhsLines);

       lineProc::increaseLinesLength(lhsLines, rhsLines, linesDMatch);

       lineProc::groupLinesInAngles(lhsLines, 40);
       lineProc::groupLinesInAngles(rhsLines, 40);

       std::vector<cv::Scalar> colors;
       std::generate_n(std::back_inserter(colors), linesDMatch.size(), []() {
          return cv::Scalar(rng.uniform(80, 255), rng.uniform(80, 255), rng.uniform(80, 255));
       });

       for (size_t i = 0; i < linesDMatch.size(); ++i)
       {
          const cv::Vec4i &lhsRefLine = lhsLines[linesDMatch[i].first];
          const cv::Vec4i &rhsRefLine = rhsLines[linesDMatch[i].second];

          cv::line(detectedLinesLhsImg,
                   cv::Point(lhsRefLine[0], lhsRefLine[1]),
                   cv::Point(lhsRefLine[2], lhsRefLine[3]),
                   colors[i],
                   3);
          cv::line(detectedLinesRhsImg,
                   cv::Point(rhsRefLine[0], rhsRefLine[1]),
                   cv::Point(rhsRefLine[2], rhsRefLine[3]),
                   colors[i],
                   3);
       }

       cv::hconcat(detectedLinesLhsImg, detectedLinesRhsImg, result);
       cv::imshow("Grouped Lines", result);

    }

    void callAll(int, void *) {
       std::cout << "---------\n";
       lineFinderCallback(0, nullptr);
       lineGrouperCallback(0, nullptr);
       std::cout << "Current frame: " << currentImagePairIndex << '\n';
       std::cout << "---------\n";
    }

    void startViz() {

       for (size_t i = 0; i < 340; ++i) {
          std::cout << "i = " << i << '\n';
          auto [lhsImageName, rhsImageName] = generateImagePair("frame", i);
          lhsImages.emplace_back(lf.readImage(lhsImageName));
          rhsImages.emplace_back(lf.readImage(rhsImageName));
       }

       cv::namedWindow("Image index", cv::WINDOW_NORMAL);
       cv::namedWindow("Detected Lines Parameters", cv::WINDOW_NORMAL);
       cv::namedWindow("Grouped Lines Parameters", cv::WINDOW_NORMAL);
       cv::namedWindow("Detected Lines", cv::WINDOW_NORMAL);
       cv::namedWindow("Grouped Lines", cv::WINDOW_NORMAL);

       cv::createTrackbar("min thresh:",
                          "Detected Lines Parameters",
                          &min_thresh,
                          255,
                          callAll);
       cv::createTrackbar("max thresh:",
                          "Detected Lines Parameters",
                          &max_thresh,
                          255,
                          callAll);
       cv::createTrackbar("length thresh:",
                          "Detected Lines Parameters",
                          &length_threshold,
                          100,
                          callAll);
       cv::createTrackbar("distance thresh:",
                          "Detected Lines Parameters",
                          &distance_threshold,
                          141,
                          callAll);
       cv::createTrackbar("canny size:",
                          "Detected Lines Parameters",
                          &canny_size,
                          7,
                          callAll);
       cv::createTrackbar("line length thresh:",
                          "Detected Lines Parameters",
                          &distance_threshold,
                          20,
                          callAll);

       cv::createTrackbar("image index:",
                          "Image index",
                          &currentImagePairIndex,
                          340,
                          callAll);

       cv::createTrackbar("image extension length fraction: ",
                          "Grouped Lines Parameters",
                          &extensionLengthFraction,
                          50,
                          callAll);
       cv::createTrackbar("image max angle diff: ",
                          "Grouped Lines Parameters",
                          &maxAngleDiff,
                          90,
                          callAll);
       cv::createTrackbar("image bounding rectangle thickness: ",
                          "Grouped Lines Parameters",
                          &boundingRectangleThickness,
                          40,
                          callAll);

       lineFinderCallback(0, nullptr);
       lineGrouperCallback(0, nullptr);
       cv::waitKey();
    }
}

/*
 * Current min thresh: 36
Current max thresh: 122
Current length thresh: 32
Current distance thresh: 4.24264
Current canny size: 3
Current line length thresh: 40
 */

/*
 * Current min thresh: 36
Current max thresh: 122
Current length thresh: 32
Current distance thresh: 1.41421
Current canny size: 3
Current line length thresh: 40
Current extensionLengthFraction: 1.2
Current maxAngleDiff: 15
Current boundingRectangleThickness: 24
 */