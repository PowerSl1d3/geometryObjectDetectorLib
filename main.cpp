//
// Created by Олег Аксененко on 01.03.2022.
//
#include "lineFinder.h"
#include "lineGrouper.h"
#include "lineProcess.h"
#include "lineVisualizer.h"

#include <opencv2/viz.hpp>

#include <vector>
#include <iomanip>

std::pair<std::string, std::string> generateImagePairName(const std::string& templ, const size_t index) {
   std::ostringstream oss;
   oss << std::setw(5) << std::setfill('0') << index;
   return {templ + "_" + oss.str() + "_0.bmp",templ + "_" + oss.str() + "_1.bmp"};
}

typedef std::tuple<
cv::Mat,
cv::Mat,
std::vector<cv::Vec4i>,
std::vector<cv::Vec4i>,
std::vector<std::pair<size_t, size_t>>
> ret_t;

lineProc::lineFinder lf;
lineProc::lineGrouper lg;

ret_t processImagePair(const std::string& lhsImageName, const std::string& rhsImageName) {
   cv::Mat lhsImage = lf.readImage(lhsImageName);
   cv::Mat rhsImage = lf.readImage(rhsImageName);

   auto lhsLines = lf.findLines(lhsImage);
   auto rhsLines = lf.findLines(rhsImage);

   if (lhsLines.size() < 5 or rhsLines.size() < 5) {
      return {};
   }

   lhsLines = lg.reduceLines(lhsLines);
   rhsLines = lg.reduceLines(rhsLines);

   auto match = lg.computeMatch(lhsLines, rhsLines);

   lineProc::increaseLinesLength(lhsLines, rhsLines, match);

   lineProc::groupLinesInAngles(lhsLines, 80);
   lineProc::groupLinesInAngles(rhsLines, 80);

   return {lhsImage, rhsImage, lhsLines, rhsLines, match};
}

bool next_iteration = false;

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void*) {
   if (event.getKeySym() == "space" && event.keyDown())
      next_iteration = true;
}

int main(int argc, char* argv[]) {

   //lineProc::startViz();

   size_t firstTrajectoryFrameMaxIndex = 300;
   size_t secondTrajectoryFrameMaxIndex = 300;

   int counter = 0;

   for (size_t firstTrajectoryIndex = 90; firstTrajectoryIndex < 91; ++firstTrajectoryIndex) {

      std::cout << "Process i = " << firstTrajectoryIndex << '\n';

      auto [firstTrajectoryLhsImageName, firstTrajectoryRhsImageName] = generateImagePairName("/Users/olegaksenenko/Desktop/clear_diplom_code_v2/input_data/frames_KomplexSpiral_2021_03_05/frame",
                                                                                              firstTrajectoryIndex);

      auto [firstTrajectoryLhsImage,
            firstTrajectoryRhsImage,
            firstTrajectoryLhsLines,
            firstTrajectoryRhsLines,
            firstTrajectoryLinesDMatch] =
      processImagePair(firstTrajectoryLhsImageName, firstTrajectoryRhsImageName);

      if (firstTrajectoryLinesDMatch.empty()) {
         continue;
      }

      auto firstTrajectoryMask = lineProc::computeGoodMatches(firstTrajectoryLhsImage, firstTrajectoryRhsImage,
                                                              firstTrajectoryLhsLines, firstTrajectoryRhsLines,
                                                              firstTrajectoryLinesDMatch);

      //firstTrajectoryLhsLines = lineProc::filterLines(firstTrajectoryLhsLines, firstTrajectoryLinesDMatch, firstTrajectoryMask);


      for (size_t secondTrajectoryIndex = 0; secondTrajectoryIndex < secondTrajectoryFrameMaxIndex; ++secondTrajectoryIndex) {

         std::cout << "Process j = " << secondTrajectoryIndex << '\n';

         auto [secondTrajectoryLhsImageName, secondTrajectoryRhsImageName] = generateImagePairName("/Users/olegaksenenko/Downloads/frames_KomplexSpiral_2021_03_30/frame",
                                                                                                 secondTrajectoryIndex);

         auto [secondTrajectoryLhsImage,
               secondTrajectoryRhsImage,
               secondTrajectoryLhsLines,
               secondTrajectoryRhsLines,
               secondTrajectoryLinesDMatch] =
         processImagePair(secondTrajectoryLhsImageName, secondTrajectoryRhsImageName);

         if (secondTrajectoryLinesDMatch.empty()) {
            continue;
         }

         auto secondTrajectoryMask = lineProc::computeGoodMatches(secondTrajectoryLhsImage, secondTrajectoryRhsImage,
                                                                  secondTrajectoryLhsLines, secondTrajectoryRhsLines,
                                                                  secondTrajectoryLinesDMatch);

         /*secondTrajectoryLhsLines = lineProc::filterLines(secondTrajectoryLhsLines, secondTrajectoryLinesDMatch,
                                                          secondTrajectoryMask);*/

         if (std::abs(static_cast<int>(firstTrajectoryLhsLines.size()) - static_cast<int>(secondTrajectoryLhsLines.size())) > 3) {
            continue;
         }

         auto warpMat = lineProc::findEssentialMatrix(firstTrajectoryLhsLines, secondTrajectoryLhsLines, lg);

         if (warpMat) {
            ++counter;
            //std::cout << firstTrajectoryIndex << ' ' << secondTrajectoryIndex << std::endl;
            //cv::Mat res1 = cv::imread(firstTrajectoryLhsImageName);
            //cv::Mat res2 = cv::imread(secondTrajectoryLhsImageName);
            //cv::hconcat(res1, res2, res1);
            //cv::imshow("res", res1);
            //cv::waitKey();

            std::cout << "Final transform:\n" << lineProc::computeTransformMatrix(
                    firstTrajectoryLhsLines,
                    firstTrajectoryRhsLines,
                    firstTrajectoryLinesDMatch,
                    firstTrajectoryMask,
                    firstTrajectoryLhsImage.cols,
                    firstTrajectoryLhsImage.rows,
                    secondTrajectoryLhsLines,
                    secondTrajectoryRhsLines,
                    secondTrajectoryLinesDMatch,
                    secondTrajectoryMask,
                    secondTrajectoryLhsImage.cols,
                    secondTrajectoryLhsImage.rows
                    ) << std::endl;

         }

      }
   }

   std::cout << "Found " << counter << " transform" << std::endl;

   return 0;
}