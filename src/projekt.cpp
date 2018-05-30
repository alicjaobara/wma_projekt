#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <dirent.h> //DIR

using namespace std;
using namespace cv;

static string path = string(PROJECT_SOURCE_DIR)+"/data/";

// Directory containing sample images
static string samplesDir= path+"asl_alphabet_train/";
// Set the file to write the features to
static string featuresFile = path+"hogFeatures/hog.txt";

//parametry do HOG
static const Size trainingPadding = Size(0, 0); //brzegi
static const Size winStride = Size(8, 8); //1,1 okno na każdym pikselu

static vector<string> validExtensions;

static void storeCursor(void) {
    printf("\033[s");
}

static void resetCursor(void) {
    printf("\033[u");
}

static string toLowerCase(const string& in) {
    string t;
    for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
        t += tolower(*i);
    }
    return t;
}

static void calculateFeaturesFromInput(const string& imageFilename, vector<float>& featureVector, HOGDescriptor& hog, Mat& descriptors, Ptr<Feature2D>& f2d) {
    Mat imageData = imread(imageFilename, IMREAD_GRAYSCALE);

    //-- Step 1: Detect the keypoints:
    std::vector<KeyPoint> keypoints;
    f2d->detect( imageData, keypoints );

    //-- Step 2: Calculate descriptors (feature vectors)
    f2d->compute( imageData, keypoints, descriptors);



    resize(imageData, imageData, Size(64, 128));
    if (imageData.empty()) {
        featureVector.clear();
        printf("Error: HOG image '%s' is empty, features calculation skipped!\n", imageFilename.c_str());
        return;
    }
    // Check for mismatching dimensions
    if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height) {
        featureVector.clear();
        printf("Error: Image '%s' dimensions (%u x %u) do not match HOG window size (%u x %u)!\n", imageFilename.c_str(), imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
        return;
    }
    vector<Point> locations;
    hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
    imageData.release(); // Release the image again after features are extracted
}

static void getFilesInDirectory(const string& dirName, vector<string>& fileNames, const vector<string>& validExtensions) {
    printf("Opening directory %s\n", dirName.c_str());
    struct dirent* ep;
    size_t extensionLocation;
    DIR* dp = opendir(dirName.c_str());
    if (dp != NULL) {
        while ((ep = readdir(dp))) {
            if (ep->d_type & DT_DIR) {
                continue;
            }
            extensionLocation = string(ep->d_name).find_last_of("."); // Assume the last point marks beginning of extension like file.ext
            // Check if extension is matching the wanted ones
            string tempExt = toLowerCase(string(ep->d_name).substr(extensionLocation + 1));
            if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end()) {
                printf("Found matching data file '%s'\n", ep->d_name);
                fileNames.push_back((string) dirName + ep->d_name);
            } else {
                printf("Found file does not match required file type, skipping: '%s'\n", ep->d_name);
            }
        }
        (void) closedir(dp);
    } else {
        printf("Error opening directory '%s'!\n", dirName.c_str());
    }
    return;
}

static void vectorF(vector<string> fileNames, vector<string>& newVector)
{
    int newSize=500;
    int licznik=1;
    for(int i=0;i<fileNames.size();++i)
    {
        if(licznik==(fileNames.size()/newSize))
        {
            newVector.push_back(fileNames.at(i));
            licznik=0;
        }
        licznik++;
    }
    return;
}

int toFile(string sign){
    HOGDescriptor hog( Size(64,128), Size(16,16), Size(16,16), Size(16,16), 9);
    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
    //    HOGDescriptor hog;
    static vector<string> TrainingImages0;
    static vector<string> TrainingImages;
    TrainingImages0.clear();
    TrainingImages.clear();
    string SamplesDirCurrent = samplesDir + sign + "/";
    getFilesInDirectory(SamplesDirCurrent, TrainingImages0, validExtensions);

    //moje vectory
    vectorF(TrainingImages0,TrainingImages);
    cout<<"TrainingImages0 "<<TrainingImages0.size()<<" TrainingImages "<<TrainingImages.size()<<endl;

    unsigned long overallSamples = TrainingImages.size();
    if (overallSamples == 0) {
        printf("No training sample files found, nothing to do!\n");
        return EXIT_SUCCESS;
    }
    printf("Reading files, generating HOG features and save them to file '%s':\n", featuresFile.c_str());
    float percent;
    fstream File;
    fstream FileSift;
    File.open(featuresFile.c_str(), ios::out|ios::app);
    if (File.good() && File.is_open()) {
        // Iterate over sample images
        for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
            storeCursor();
            vector<float> featureVector;
            Mat descriptors;
            const string currentImageFile = TrainingImages.at(currentFile);
            // Output progress
            if ( (currentFile+1) % 10 == 0 || (currentFile+1) == overallSamples ) {
                percent = ((currentFile+1) * 100 / overallSamples);
                printf("%5lu (%3.0f%%):\tFile '%s'", (currentFile+1), percent, currentImageFile.c_str());
                fflush(stdout);
                resetCursor();
            }
            // Calculate feature vector from current image file
            calculateFeaturesFromInput(currentImageFile, featureVector, hog, descriptors, f2d);
            if (!featureVector.empty()) {
                // Save feature vector components
                for (unsigned int feature = 0; feature < featureVector.size(); ++feature) {
                    if(feature == 0){
                        File << featureVector.at(feature);
                        //                        cout<<featureVector.at(feature)<<endl;
                    }
                    else
                        File << "," << featureVector.at(feature);
                }

                char s = sign[0];
                int isign = s;
                int i=0;
                if (isign>=65 && isign <= 90)
                    i=isign-64;
                else i=27;
                File << ", " << i << endl;
            }
            if (!descriptors.empty()) {
                // Save feature vector components
                for (unsigned int feature = 0; feature < descriptors.size(); ++feature) {
                    if(feature == 0){
                        FileSift << descriptors.at(feature);
                        //                        cout<<featureVector.at(feature)<<endl;
                    }
                    else
                        FileSift << "," << descriptors.at(feature);
                }

                char s = sign[0];
                int isign = s;
                int i=0;
                if (isign>=65 && isign <= 90)
                    i=isign-64;
                else i=27;
                FileSift << ", " << i << endl;
            }
        }
        printf("\n");
        File.flush();
        File.close();
        FileSift.flush();
        FileSift.close();
    }
    else {
        printf("Error opening file '%s'!\n", featuresFile.c_str());
        return EXIT_FAILURE;
    }
}

int show(Mat img,string okno)
{
    imshow(okno, img);
    waitKey(0);
    cout<<"wielkość "<<okno<<" "<<img.size()<<endl;
    destroyAllWindows();
    return 0;
}

int main(int argc, char** argv)
{
    validExtensions.push_back("jpg");
    fstream File;
    File.open(featuresFile.c_str(), ios::out);
    if (File.good() && File.is_open()) {
        //        File << "first line1,2,3,4,5,6,7,8,9" << endl;
        File.flush();
        File.close();
    }

    string sign="A";
    char s = sign[0];
    int isign = s;
    //    while (isign >= 65 && isign <= 90)
    //    {
    //        toFile(sign);
    //        s = sign[0];
    //        isign = s;
    //        isign++;
    //        s = isign;
    //        sign[0] = s;
    //    }
    toFile("B");
    //    toFile("C");







    return 0;
}
