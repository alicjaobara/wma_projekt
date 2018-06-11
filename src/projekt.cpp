#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <dirent.h> //DIR

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

// Directory containing data
static string dataPath = string(PROJECT_SOURCE_DIR)+"/data/";
// Directory containing sample images
static string samplesDir= dataPath+"asl_alphabet_train/";
// Set the files to write the features to
static string featuresFileHog = dataPath+"features/hog10.txt";
static string featuresFileSift = dataPath+"features/sift10.txt";
static string featuresFileSurf = dataPath+"features/surf10.txt";
static string namesFile = dataPath+"features/10.txt";

// HOG
HOGDescriptor hog( Size(64,128), Size(16,16), Size(16,16), Size(16,16), 9);
static const Size trainingPadding = Size(0, 0); //brzegi
static const Size winStride = Size(8, 8); //1,1 okno na każdym pikselu

// SIFT
cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

// SURF
cv::Ptr<Feature2D> f2dSURF = xfeatures2d::SURF::create();

// String to lower case (for getting files in directory)
static string toLowerCase(const string& in) {
    string t;
    for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
        t += tolower(*i);
    }
    return t;
}

// Getting files to fileNames with validExtensions in directory dirName
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
            extensionLocation = string(ep->d_name).find_last_of(".");
            // Assume the last point marks beginning of extension like file.ext
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

// resizes (cuts) oldVector to newVector
static void resizeVector(vector<string> oldVector, vector<string>& newVector, vector<string>& newVector2){
    int newSize=500;
    int counter=1;
    int counter2=1;
    for(int i=0;i<oldVector.size();++i){
        if(counter==(oldVector.size()/newSize)){
            newVector.push_back(oldVector.at(i));
            counter=0;
            if(counter2<=10){
                newVector2.push_back(oldVector.at(i-1));
                counter2++;
            }
        }
        counter++;
    }
    return;
}

// Calculates HOG features from imageFilename to featureVector using hog
static void calculateFeaturesFromInputHOG(const string& imageFilename, vector<float>& featureVector) {
    Mat imageData = imread(imageFilename, IMREAD_GRAYSCALE);
    resize(imageData, imageData, Size(64, 128));
    if (imageData.empty()) {
        featureVector.clear();
        printf("Error: HOG image '%s' is empty, features calculation skipped!\n", imageFilename.c_str());
        return;
    }
    // Check for mismatching dimensions
    if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height) {
        featureVector.clear();
        printf("Error: Image '%s' dimensions (%u x %u) do not match HOG window size (%u x %u)!\n",
               imageFilename.c_str(), imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
        return;
    }
    vector<Point> locations;
    hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
    imageData.release(); // Release the image again after features are extracted
}

int toFileHOG (string sign, vector<string> TrainingImages){
    unsigned long overallSamples = TrainingImages.size();
    if (overallSamples == 0) {
        printf("No training sample files found, nothing to do!\n");
        return EXIT_SUCCESS;
    }
    printf("Reading files, generating HOG features and save them to file '%s':\n", featuresFileHog.c_str());
    float percent;
    fstream File;
    File.open(featuresFileHog.c_str(), ios::out|ios::app);
    if (File.good() && File.is_open()) {
        // Iterate over sample images
        for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
            vector<float> featureVector;
            const string currentImageFile = TrainingImages.at(currentFile);
            // Output progress
            if ( (currentFile+1) % 10 == 0 || (currentFile+1) == overallSamples ) {
                percent = ((currentFile+1) * 100 / overallSamples);
                printf("%5lu (%3.0f%%):\tFile '%s'\n", (currentFile+1), percent, currentImageFile.c_str());
            }
            // Calculate feature vector from current image file
            calculateFeaturesFromInputHOG(currentImageFile, featureVector);
            if (!featureVector.empty()) {
                // Save feature vector components
                for (unsigned int feature = 0; feature < featureVector.size(); ++feature) {
                    if(feature == 0) {
                        File << featureVector.at(feature);
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
        }
        printf("\n");
        File.flush();
        File.close();
    }
    else {
        printf("Error opening file '%s'!\n", featuresFileHog.c_str());
        return EXIT_FAILURE;
    }
}

// Calculates SIFT features
static void toFileSIFT(string sign,vector<string> TrainingImages) {
    unsigned long overallSamples = TrainingImages.size();
    if (overallSamples == 0) {
        printf("No training sample files found, nothing to do!\n");
    }
    printf("Reading files, generating SIFT features and save them to file '%s':\n", featuresFileHog.c_str());
    float percent;
    fstream File;
    File.open(featuresFileSift.c_str(), ios::out|ios::app);
    if (File.good() && File.is_open() ) {
        cout << "Extracting SIFT Features of train set ...\n";
        Mat vocabulary;
        Mat all_descriptors;
        Mat descriptors;
        for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
            const string currentImageFile = TrainingImages.at(currentFile);
            Mat imageData= imread(currentImageFile, IMREAD_GRAYSCALE);
            if (imageData.empty()) {
                printf("Error: SIFT image '%s' is empty, features calculation skipped!\n", currentImageFile.c_str());
                return;
            }
            vector<cv::KeyPoint> keypoints;
            f2d->detect(imageData, keypoints);
            f2d->compute(imageData, keypoints, descriptors);
            all_descriptors.push_back(descriptors);
            if ( (currentFile+1) % 10 == 0 || (currentFile+1) == overallSamples ) {
                percent = ((currentFile+1) * 100 / overallSamples);
                cout << "["<<percent<<"%] Extracting image #"<<currentFile << "/" << overallSamples << "\r" << std::flush;
                fflush(stdout);
            }
        }
        cout <<endl << all_descriptors.size() << " features extracted for train set.\n";
        // Now cluster SIFT features using KNN
        int vocabulary_size = 288;
        // Clustering all SIFT descriptors
        TermCriteria terminate_criterion;
        terminate_criterion.epsilon = FLT_EPSILON;
        BOWKMeansTrainer bowTrainer( vocabulary_size, terminate_criterion, 3, KMEANS_PP_CENTERS );
        for (int i=0;i<all_descriptors.size().height;i++){
            Mat current_descriptor = all_descriptors.row(i);
            bowTrainer.add(current_descriptor);
            if ( (i+1) % 10 == 0 || (i+1) == all_descriptors.size().height ) {
                percent = ((i+1) * 100 / all_descriptors.size().height);
                cout <<"["<<percent<<"%]Adding Feature #" << i << " to Bag-Of-Words K-Means Trainer ...  \r" << std::flush;
            }

        }
        cout << "\nClustering... Please Wait ...\n";
        vocabulary = bowTrainer.cluster();

        cout << "\nSIFT Features Clustered in " << vocabulary.size() << " clusters." << endl;
        std::vector< DMatch > matches;
        std::vector<DMatch> trainin_set_matches;
        Ptr<FeatureDetector> featureDetector = xfeatures2d::SIFT::create();
        Ptr<DescriptorExtractor> descExtractor = xfeatures2d::SIFT::create();
        Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create( "BruteForce" );
        BOWImgDescriptorExtractor bowExtractor = BOWImgDescriptorExtractor( descExtractor, descMatcher);
        bowExtractor.setVocabulary(vocabulary);
        cout << "Building Histograms for training set :\n";
        for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
            const string currentImageFile = TrainingImages.at(currentFile);
            Mat imageData= imread(currentImageFile, IMREAD_GRAYSCALE);
            if (imageData.empty()) {
                printf("Error: SIFT image '%s' is empty, features calculation skipped!\n", currentImageFile.c_str());
                return;
            }
            imageData= imread(currentImageFile, IMREAD_GRAYSCALE);
            if (imageData.empty()) {
                printf("Error: SIFT image '%s' is empty, features calculation skipped!\n", currentImageFile.c_str());
                return;
            }
            vector<float> descriptors;
            vector<cv::KeyPoint> keypoints;
            // Each descriptor is histogram for the image
            featureDetector->detect( imageData, keypoints );
            bowExtractor.compute( imageData, keypoints, descriptors);
            if ( (currentFile+1) % 10 == 0 || (currentFile+1) == overallSamples ) {
                percent = ((currentFile+1) * 100 / overallSamples);
                cout << "["<<percent<<"%] Image #" << currentFile <<" Bag-Of-Words descriptors extracted , size = " << descriptors.size() << "\r" << std::flush;
            }
            if (!descriptors.empty()) {
                // Save feature vector components
                for (unsigned int feature = 0; feature < descriptors.size(); ++feature) {
                    if(feature == 0){
                        File << descriptors.at(feature);
                    }
                    else
                        File << "," << descriptors.at(feature);
                }
                char s = sign[0];
                int isign = s;
                int i=0;
                if (isign>=65 && isign <= 90)
                    i=isign-64;
                else i=27;
                File << ", " << i << endl;
            }
        }
    }
}

// Calculates SURF features
static void toFileSURF(string sign,vector<string> TrainingImages) {
    unsigned long overallSamples = TrainingImages.size();
    if (overallSamples == 0) {
        printf("No training sample files found, nothing to do!\n");
    }
    printf("Reading files, generating SURF features and save them to file '%s':\n", featuresFileHog.c_str());
    float percent;
    fstream File;
    File.open(featuresFileSurf.c_str(), ios::out|ios::app);
    if (File.good() && File.is_open() ) {
        cout << "Extracting SURF Features of train set ...\n";
        Mat vocabulary;
        Mat all_descriptors;
        Mat descriptors;
        for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
            const string currentImageFile = TrainingImages.at(currentFile);
            Mat imageData= imread(currentImageFile, IMREAD_GRAYSCALE);
            if (imageData.empty()) {
                printf("Error: SURF image '%s' is empty, features calculation skipped!\n", currentImageFile.c_str());
                return;
            }
            vector<cv::KeyPoint> keypoints;
            f2dSURF->detect(imageData, keypoints);
            f2dSURF->compute(imageData, keypoints, descriptors);
            all_descriptors.push_back(descriptors);
            if ( (currentFile+1) % 10 == 0 || (currentFile+1) == overallSamples ) {
                percent = ((currentFile+1) * 100 / overallSamples);
                cout << "["<<percent<<"%] Extracting image #"<<currentFile << "/" << overallSamples << "\r" << std::flush;
                fflush(stdout);
            }
        }
        cout <<endl << all_descriptors.size() << " features extracted for train set.\n";
        // Now cluster SURF features using KNN
        int vocabulary_size = 288;
        // Clustering all SURF descriptors
        TermCriteria terminate_criterion;
        terminate_criterion.epsilon = FLT_EPSILON;
        BOWKMeansTrainer bowTrainer( vocabulary_size, terminate_criterion, 3, KMEANS_PP_CENTERS );
        for (int i=0;i<all_descriptors.size().height;i++){
            Mat current_descriptor = all_descriptors.row(i);
            bowTrainer.add(current_descriptor);
            if ( (i+1) % 10 == 0 || (i+1) == all_descriptors.size().height ) {
                percent = ((i+1) * 100 / all_descriptors.size().height);
                cout <<"["<<percent<<"%]Adding Feature #" << i << " to Bag-Of-Words K-Means Trainer ...  \r" << std::flush;
            }

        }
        cout << "\nClustering... Please Wait ...\n";
        vocabulary = bowTrainer.cluster();

        cout << "\nSURF Features Clustered in " << vocabulary.size() << " clusters." << endl;
        std::vector< DMatch > matches;
        std::vector<DMatch> trainin_set_matches;
        Ptr<FeatureDetector> featureDetector = xfeatures2d::SURF::create();
        Ptr<DescriptorExtractor> descExtractor = xfeatures2d::SURF::create();
        Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create( "BruteForce" );
        BOWImgDescriptorExtractor bowExtractor = BOWImgDescriptorExtractor( descExtractor, descMatcher);
        bowExtractor.setVocabulary(vocabulary);
        cout << "Building Histograms for training set :\n";
        for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
            const string currentImageFile = TrainingImages.at(currentFile);
            Mat imageData= imread(currentImageFile, IMREAD_GRAYSCALE);
            if (imageData.empty()) {
                printf("Error: SURF image '%s' is empty, features calculation skipped!\n", currentImageFile.c_str());
                return;
            }
            imageData= imread(currentImageFile, IMREAD_GRAYSCALE);
            if (imageData.empty()) {
                printf("Error: SURF image '%s' is empty, features calculation skipped!\n", currentImageFile.c_str());
                return;
            }
            vector<float> descriptors;
            vector<cv::KeyPoint> keypoints;
            // Each descriptor is histogram for the image
            featureDetector->detect( imageData, keypoints );
            bowExtractor.compute( imageData, keypoints, descriptors);
            if ( (currentFile+1) % 10 == 0 || (currentFile+1) == overallSamples ) {
                percent = ((currentFile+1) * 100 / overallSamples);
                cout << "["<<percent<<"%] Image #" << currentFile <<" Bag-Of-Words descriptors extracted , size = " << descriptors.size() << "\r" << std::flush;
            }
            if (!descriptors.empty()) {
                // Save feature vector components
                for (unsigned int feature = 0; feature < descriptors.size(); ++feature) {
                    if(feature == 0){
                        File << descriptors.at(feature);
                    }
                    else
                        File << "," << descriptors.at(feature);
                }
                char s = sign[0];
                int isign = s;
                int i=0;
                if (isign>=65 && isign <= 90)
                    i=isign-64;
                else i=27;
                File << ", " << i << endl;
            }
        }
    }
}

int main(int argc, char** argv)
{
    fstream File;
    File.open(featuresFileHog.c_str(), ios::out);
    if (File.good() && File.is_open()) {
        File.flush();
        File.close();
    }
    fstream FileSift;
    FileSift.open(featuresFileSift.c_str(), ios::out);
    if (FileSift.good() && FileSift.is_open()) {
        FileSift.flush();
        FileSift.close();
    }
    fstream FileSurf;
    FileSurf.open(featuresFileSurf.c_str(), ios::out);
    if (FileSurf.good() && FileSurf.is_open()) {
        FileSurf.flush();
        FileSurf.close();
    }
    fstream FileNames;
    FileNames.open(namesFile.c_str(), ios::out);

    string sign="A";
    char s = sign[0];
    int isign = s;
    while (isign >= 65 && isign <= 90)
    {
        static vector<string> TrainingImages0;
        static vector<string> TrainingImages;
        static vector<string> TrainingImages2;
        TrainingImages0.clear();
        TrainingImages.clear();
        TrainingImages2.clear();
        string SamplesDirCurrent = samplesDir + sign + "/";
        static vector<string> validExtensions;
        validExtensions.push_back("jpg");
        validExtensions.push_back("png");
        getFilesInDirectory(SamplesDirCurrent, TrainingImages0, validExtensions);
        resizeVector(TrainingImages0, TrainingImages, TrainingImages2);
        toFileHOG(sign,TrainingImages2);
        toFileSIFT(sign,TrainingImages2);
        toFileSURF(sign,TrainingImages2);
        for(int i=0; i<TrainingImages2.size();i++){
            if (FileNames.good() && FileNames.is_open()) {
                //                FileNames.flush();

                FileNames<<TrainingImages2.at(i)<<endl;
            }
        }

        s = sign[0];
        isign = s;
        isign++;
        s = isign;
        sign[0] = s;
    }

    FileNames.close();
    return 0;
}
