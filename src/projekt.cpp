#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"
//#include "opencv2/features2d.hpp"
#include <dirent.h> //DIR


//#include "opencv2/opencv_modules.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
//#include "opencv2/features2d.hpp"
//#include "opencv2/xfeatures2d.hpp"
//#include "opencv2/ml.hpp"

//#include "opencv2/core.hpp"
//#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

// Directory containing data
static string dataPath = string(PROJECT_SOURCE_DIR)+"/data/";
// Directory containing sample images
static string samplesDir= dataPath+"asl_alphabet_train/";
// Set the files to write the features to
static string featuresFileHog = dataPath+"features/hog.txt";
static string featuresFileSift = dataPath+"features/sift.txt";

// HOG
HOGDescriptor hog( Size(64,128), Size(16,16), Size(16,16), Size(16,16), 9);
static const Size trainingPadding = Size(0, 0); //brzegi
static const Size winStride = Size(8, 8); //1,1 okno na każdym pikselu

// SIFT
cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

// Save cursor position (for getting files in directory)
static void storeCursor(void) {
    printf("\033[s");
}

// Restore cursor position (for getting files in directory)
static void resetCursor(void) {
    printf("\033[u");
}

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

//static bool writeVocabulary( const string& filename, const Mat& vocabulary )
//{
//    FileStorage fs( filename, FileStorage::WRITE );
//    if( fs.isOpened() )
//    {
//        fs << "vocabulary" << vocabulary;
//        return true;
//    }
//    return false;
//}

//static bool readVocabulary( const string& filename, Mat& vocabulary )
//{
//    FileStorage fs( filename, FileStorage::READ );
//    if( fs.isOpened() )
//    {
//        fs["vocabulary"] >> vocabulary;
//        // cout << "done" << endl;
//        return true;
//    }
//    return false;
//}

// resizes (cuts) oldVector to newVector
static void resizeVector(vector<string> oldVector, vector<string>& newVector){
    int newSize=500;
    int counter=1;
    for(int i=0;i<oldVector.size();++i){
        if(counter==(oldVector.size()/newSize)){
            newVector.push_back(oldVector.at(i));
            counter=0;
        }
        counter++;
    }
    return;
}

// Calculates SIFT features
static void toFileSIFT(string sign)//vector<string> TrainingImages)//,const string& imageFilename, //vector<float>& featureVector,unsigned long overallSamples)
{
    static vector<string> TrainingImages0;
    static vector<string> TrainingImages;
    TrainingImages0.clear();
    TrainingImages.clear();
    string SamplesDirCurrent = samplesDir + sign + "/";
    static vector<string> validExtensions;
    validExtensions.push_back("jpg");
    validExtensions.push_back("png");
    getFilesInDirectory(SamplesDirCurrent, TrainingImages0, validExtensions);
    resizeVector(TrainingImages0,TrainingImages);
    //    Mat imageData;


    //            //-- Step 1: Detect the keypoints:
    //            vector<KeyPoint> keypoints;
    //            f2d->detect( imageData, keypoints );

    //        //    //-- Step 2: Calculate descriptors (feature vectors)
    //            f2d->compute( imageData, keypoints, descriptors);


    //    cout<<"descr size "<<descriptors.size()<<endl;

    //    double centers = kmeans(descriptors,10,);


    //    bowTrain = BOWKMeansTrainer(10);
    //    bowTrain.add(descriptors);

    //    cout<<"centers size "<<centers.size()<<endl;
    //    cout<<"labels size "<<labels.size()<<endl;


    unsigned long overallSamples = TrainingImages.size();
    if (overallSamples == 0) {
        printf("No training sample files found, nothing to do!\n");
        //        return EXIT_SUCCESS;
    }
    printf("Reading files, generating HOG features and save them to file '%s':\n", featuresFileHog.c_str());
    float percent;
    fstream File;
    File.open(featuresFileSift.c_str(), ios::out|ios::app);
    if (File.good() && File.is_open() ) {
        // Iterate over sample images




        // Method 2 : SIFT
        cout << "===========================\n";
        cout << "Extracting SIFT Features of train set ...\n";

        // Check if we already have vocabulary written in file
        Mat vocabulary;
        //    if( !readVocabulary( "dataPath/vocabulary.dat", vocabulary) )
        //    {
        //        SiftDescriptorExtractor detector;
        //        Ptr<FeatureDetector> detector(new SiftFeatureDetector());
        //        Ptr<DescriptorExtractor> extractor(new cv::xfeatures2d::SiftDescriptorExtractor);

        //            cv::xfeatures2d::SiftFeatureDetector detector;
        //            cv::xfeatures2d::SiftDescriptorExtractor extractor;

        Mat all_descriptors;
        Mat descriptors;
        float percent;
        for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
            storeCursor();
            const string currentImageFile = TrainingImages.at(currentFile);
            Mat imageData= imread(currentImageFile, IMREAD_GRAYSCALE);
            if (imageData.empty()) {
                //                featureVector.clear();
                printf("Error: SIFT image '%s' is empty, features calculation skipped!\n", currentImageFile.c_str());
                return;
            }

            vector<cv::KeyPoint> keypoints;
            //                detector.detect(currentImageFile, keypoints);
            //                extractor.compute(currentImageFile, keypoints, descriptors);
            f2d->detect(imageData, keypoints);
            f2d->compute(imageData, keypoints, descriptors);
            all_descriptors.push_back(descriptors);

            if ( (currentFile+1) % 10 == 0 || (currentFile+1) == overallSamples ) {
                percent = ((currentFile+1) * 100 / overallSamples);
                //                    printf("%5lu (%3.0f%%):\tFile '%s'", (currentFile+1), percent, currentImageFile.c_str());
                //                cout << "["<<percent<<"%] Image #" << currentFile <<" Bag-Of-Words descriptors extracted , size = " << descriptors.size() << "\r" << std::flush;
                cout << "["<<percent<<"%] Extracting image #"<<currentFile << "/" << overallSamples << "\r" << std::flush;
                fflush(stdout);
                resetCursor();
            }

        }
        cout <<endl << all_descriptors.size() << " features extracted for train set.\n";
        // Now cluster SIFT features using KNN
        int vocabulary_size = 70;
        // Clustering all SIFT descriptors
        TermCriteria terminate_criterion;
        terminate_criterion.epsilon = FLT_EPSILON;
        BOWKMeansTrainer bowTrainer( vocabulary_size, terminate_criterion, 3, KMEANS_PP_CENTERS );
        for (int i=0;i<all_descriptors.size().height;i++){
            //            storeCursor();
            Mat current_descriptor = all_descriptors.row(i);
            //cout << "Size of current_descriptor = " << current_descriptor.size() << endl;  getchar();
            bowTrainer.add(current_descriptor);
            if ( (i+1) % 10 == 0 || (i+1) == all_descriptors.size().height ) {
                percent = ((i+1) * 100 / all_descriptors.size().height);
                //                    printf("%5lu (%3.0f%%):\tFile '%s'", (currentFile+1), percent, currentImageFile.c_str());
                //                cout << "["<<percent<<"%] Image #" << currentFile <<" Bag-Of-Words descriptors extracted , size = " << descriptors.size() << "\r" << std::flush;
                cout <<"["<<percent<<"%]Adding Feature #" << i << " to Bag-Of-Words K-Means Trainer ...  \r" << std::flush;
                //                fflush(stdout);
                //                resetCursor();
            }

        }
        cout << "\nClustering... Please Wait ...\n";
        vocabulary = bowTrainer.cluster();

        cout << "\nSIFT Features Clustered in " << vocabulary.size() << " clusters." << endl;

        //        if( !writeVocabulary("dataPath/vocabulary.dat", vocabulary) )
        //        {
        //            cout << "Error: file " << "../vocabulary.dat" << " can not be opened to write" << endl;
        //            exit(-1);
        //        }
        //    }else
        //        cout << "Visual Vocabulary read from file successfully!\n";
        //    // Building Histograms
        //    cout << "===========================\n";

        std::vector< DMatch > matches;
        // Matching centroids with training set
        std::vector<DMatch> trainin_set_matches;

        //    cv::FeatureDetector * detector = new cv::SIFT();

        //    Ptr<FeatureDetector> featureDetector = FeatureDetector::create( "SIFT" );
        //    Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create( "SIFT" );
        Ptr<FeatureDetector> featureDetector = xfeatures2d::SIFT::create();
        Ptr<DescriptorExtractor> descExtractor = xfeatures2d::SIFT::create();

        Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create( "BruteForce" );

        BOWImgDescriptorExtractor bowExtractor = BOWImgDescriptorExtractor( descExtractor, descMatcher);

        bowExtractor.setVocabulary(vocabulary);



        Mat train_hist, test_hist;
        cout << "Building Histograms for training set :\n";
        //        if( !readVocabulary( "dataPath/train_hist.dat", train_hist) )
        //        {
        for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
            storeCursor();
            const string currentImageFile = TrainingImages.at(currentFile);
            Mat imageData= imread(currentImageFile, IMREAD_GRAYSCALE);
            if (imageData.empty()) {
                //                featureVector.clear();
                printf("Error: SIFT image '%s' is empty, features calculation skipped!\n", currentImageFile.c_str());
                return;
            }

            imageData= imread(currentImageFile, IMREAD_GRAYSCALE);
            if (imageData.empty()) {
                //                featureVector.clear();
                printf("Error: SIFT image '%s' is empty, features calculation skipped!\n", currentImageFile.c_str());
                return;
            }

            vector<float> descriptors;

            vector<cv::KeyPoint> keypoints;
            // Each descriptor is histogram for the image
            //            Mat descriptors;
            featureDetector->detect( imageData, keypoints );
            bowExtractor.compute( imageData, keypoints, descriptors);
            train_hist.push_back(descriptors);
            if ( (currentFile+1) % 10 == 0 || (currentFile+1) == overallSamples ) {
                percent = ((currentFile+1) * 100 / overallSamples);
                //                    printf("%5lu (%3.0f%%):\tFile '%s'", (currentFile+1), percent, currentImageFile.c_str());
                cout << "["<<percent<<"%] Image #" << currentFile <<" Bag-Of-Words descriptors extracted , size = " << descriptors.size() << "\r" << std::flush;
                fflush(stdout);
                resetCursor();
            }

            if (!descriptors.empty()) {
                // Save feature vector components
                for (unsigned int feature = 0; feature < descriptors.size(); ++feature) {
                    if(feature == 0){
                        File << descriptors.at(feature);
                        //                        cout<<featureVector.at(feature)<<endl;
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



        //            writeVocabulary("dataPath/train_hist.dat", train_hist);
        //        }else{
        //            cout << "Train Histograms read from file successfully!\n";
        //        }

        //    //    cout << "\nBuilding Histograms for test set :\n";
        //    //    if( !readVocabulary( "../test_hist.dat", test_hist) )
        //    //    {
        //    //    	for (int i=0;i<test_set.size();i++){
        //    //    		vector<cv::KeyPoint> keypoints;
        //    //  		// Each descriptor is histogram for the image
        //    //    		Mat descriptors;
        //    //    		featureDetector->detect( test_set.at(i), keypoints );
        //    //    		bowExtractor->compute( test_set.at(i), keypoints, descriptors);
        //    //    		test_hist.push_back(descriptors);
        //    //    		cout << "Image #" << i <<" Bag-Of-Words descriptors extracted , size = " << descriptors.size() << "\r" << std::flush;
        //    //    	}
        //    //    	writeVocabulary("../test_hist.dat", test_hist);
        //    //    }else{
        //    //    cout << "Test Histograms read from file successfully!\n";
        //    //    }

        ////    delete bowExtractor;
    }
}



int toFileHOG (string sign){
    static vector<string> TrainingImages0;
    static vector<string> TrainingImages;
    TrainingImages0.clear();
    TrainingImages.clear();
    string SamplesDirCurrent = samplesDir + sign + "/";
    static vector<string> validExtensions;
    validExtensions.push_back("jpg");
    validExtensions.push_back("png");
    getFilesInDirectory(SamplesDirCurrent, TrainingImages0, validExtensions);
    resizeVector(TrainingImages0,TrainingImages);
    //    cout<<"TrainingImages0 "<<TrainingImages0.size()<<" TrainingImages "<<TrainingImages.size()<<endl;

    unsigned long overallSamples = TrainingImages.size();
    if (overallSamples == 0) {
        printf("No training sample files found, nothing to do!\n");
        return EXIT_SUCCESS;
    }



    //    calculateFeaturesFromInputSIFT(TrainingImages);

    printf("Reading files, generating HOG features and save them to file '%s':\n", featuresFileHog.c_str());
    float percent;
    fstream File;
    fstream FileSift;
    File.open(featuresFileHog.c_str(), ios::out|ios::app);
    FileSift.open(featuresFileSift.c_str(), ios::out|ios::app);
    if (File.good() && File.is_open() && FileSift.good() && FileSift.is_open()) {
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
            calculateFeaturesFromInputHOG(currentImageFile, featureVector);



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

            //            std::vector<float> descriptorsVector;
            //            if (descriptors.isContinuous()) {
            //                descriptorsVector.assign((float*)descriptors.datastart, (float*)descriptors.dataend);
            //            } else {
            //                for (int i = 0; i < descriptors.rows; ++i) {
            //                    descriptorsVector.insert(descriptorsVector.end(), descriptors.ptr<float>(i),
            //                                             descriptors.ptr<float>(i)+descriptors.cols);
            //                }
            //            }

            //            if (!descriptorsVector.empty()) {
            //                // Save feature vector components
            //                for (unsigned int feature = 0; feature < descriptorsVector.size(); ++feature) {
            //                    if(feature == 0){
            //                        FileSift << descriptorsVector.at(feature);                    }
            //                    else
            //                        FileSift << "," << descriptorsVector.at(feature);
            //                }

            //                char s = sign[0];
            //                int isign = s;
            //                int i=0;
            //                if (isign>=65 && isign <= 90)
            //                    i=isign-64;
            //                else i=27;
            //                FileSift << ", " << i << endl;
            //            }
        }
        printf("\n");
        File.flush();
        File.close();
        FileSift.flush();
        FileSift.close();
    }
    else {
        printf("Error opening file '%s'!\n", featuresFileHog.c_str());
        return EXIT_FAILURE;
    }
}

//int show(Mat img,string okno)
//{
//    imshow(okno, img);
//    waitKey(0);
//    cout<<"wielkość "<<okno<<" "<<img.size()<<endl;
//    destroyAllWindows();
//    return 0;
//}

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

    string sign="A";
    char s = sign[0];
    int isign = s;
    while (isign >= 65 && isign <= 90)
    {
        toFileHOG(sign);
        toFileSIFT(sign);
        s = sign[0];
        isign = s;
        isign++;
        s = isign;
        sign[0] = s;
    }
    //    toFileHOG("B");
    //    toFileHOG("C");
    //    toFileSIFT("B");
    //    toFileSIFT("C");

    return 0;
}
