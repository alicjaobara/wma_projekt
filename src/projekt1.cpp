#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/ml/ml.hpp>
//#include <stdexcept>
//#include <ios>
#include <dirent.h> //DIR

using namespace std;
using namespace cv;

// Directory containing positive sample images
static string posSamplesDir = "/home/alicja/gnu/projekt_wma/data/asl_alphabet_train/A/";
// Set the file to write the features to
static string featuresFile = "/home/alicja/gnu/projekt_wma/data/hogFeatures/hog_A.txt";

//parametry do HOG
static const Size trainingPadding = Size(0, 0);
static const Size winStride = Size(8, 8);

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

// pokazywanie obrazów
int show(Mat img,string okno)
{
    imshow(okno, img);
    waitKey(0);
    cout<<"wielkość "<<okno<<" "<<img.size()<<endl;
    destroyAllWindows();
    return 0;
}
/**
 * This is the actual calculation from the (input) image data to the HOG descriptor/feature vector using the hog.compute() function
 * @param imageFilename file path of the image file to read and calculate feature vector from
 * @param descriptorVector the returned calculated feature vector<float> ,
 *      I can't comprehend why openCV implementation returns std::vector<float> instead of cv::MatExpr_<float> (e.g. Mat<float>)
 * @param hog HOGDescriptor containin HOG settings
 */
static void calculateFeaturesFromInput(const string& imageFilename, vector<float>& featureVector, HOGDescriptor& hog) {
    /** for imread flags from openCV documentation,
     * @see http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#Mat imread(const string& filename, int flags)
     * @note If you get a compile-time error complaining about following line (esp. imread),
     * you either do not have a current openCV version (>2.0)
     * or the linking order is incorrect, try g++ -o openCVHogTrainer main.cpp `pkg-config --cflags --libs opencv`
     */
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
        printf("Error: Image '%s' dimensions (%u x %u) do not match HOG window size (%u x %u)!\n", imageFilename.c_str(), imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
       return;
    }
    vector<Point> locations;
    hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
    imageData.release(); // Release the image again after features are extracted
}
/**
 * For unixoid systems only: Lists all files in a given directory and returns a vector of path+name in string format
 * @param dirName
 * @param fileNames found file names in specified directory
 * @param validExtensions containing the valid file extensions for collection in lower case
 */
static void getFilesInDirectory(const string& dirName, vector<string>& fileNames, const vector<string>& validExtensions) {
    printf("Opening directory %s\n", dirName.c_str());
#ifdef __MINGW32__
    struct stat s;
#endif
    struct dirent* ep;
    size_t extensionLocation;
    DIR* dp = opendir(dirName.c_str());
    if (dp != NULL) {
        while ((ep = readdir(dp))) {
            // Ignore (sub-)directories like . , .. , .svn, etc.
#ifdef __MINGW32__
            stat(ep->d_name, &s);
            if (s.st_mode & S_IFDIR) {
                continue;
            }
#else
            if (ep->d_type & DT_DIR) {
                continue;
            }
#endif
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
int main(int argc, char** argv)
{
    /*
    string path = posSamplesDir+"A1.jpg";
//    string path = string(PROJECT_SOURCE_DIR) + "/data/asl_alphabet_train/Atest/A1.jpg";
    Mat input = imread(path); //,IMREAD_GRAYSCALE)
    if (input.empty()) // Sprawdzenie, czy udalo sie otworzyc obraz z sciezka
    {
        cout << "ERROR: Can't open image" << endl;
        return -1;
    }
    show(input,"obraz");

    // przycinanie, resize, convert
    Rect myROI(50, 0, 100, 200);
    Mat img = input(myROI);
    show(img,"przyciety");
    resize(img, img, cv::Size(), 0.64, 0.64);
    show(img,"resize");
    img.convertTo(img, CV_32F, 1/255.0);
    show(img,"convert");

    // Calculate gradients gx, gy
    Mat gx, gy;
    Sobel(img, gx, CV_32F, 1, 0, 1);
    Sobel(img, gy, CV_32F, 0, 1, 1);

    // Calculate gradient magnitude and direction (in degrees)
    Mat mag, angle;
    cartToPolar(gx, gy, mag, angle, 1);

    show(gx,"gx");
    show(gy,"gy");
    show(mag,"mag");
    */

    // "/home/alicja/gnu/projekt_wma/data/asl_alphabet_test"


    HOGDescriptor hog;
    // Get the files to train from somewhere
        static vector<string> positiveTrainingImages;
//        static vector<string> negativeTrainingImages;
        static vector<string> validExtensions;
        validExtensions.push_back("jpg");
        validExtensions.push_back("png");
        validExtensions.push_back("ppm");

        getFilesInDirectory(posSamplesDir, positiveTrainingImages, validExtensions);
//        getFilesInDirectory(negSamplesDir, negativeTrainingImages, validExtensions);
        /// Retrieve the descriptor vectors from the samples
        unsigned long overallSamples = positiveTrainingImages.size();  //+ negativeTrainingImages.size();

        // Make sure there are actually samples to train
        if (overallSamples == 0) {
            printf("No training sample files found, nothing to do!\n");
            return EXIT_SUCCESS;
        }


        /*
        /// @WARNING: This is really important, some libraries (e.g. ROS) seems to set the system locale which takes decimal commata instead of points which causes the file input parsing to fail
        setlocale(LC_ALL, "C"); // Do not use the system locale
        setlocale(LC_NUMERIC,"C");
        setlocale(LC_ALL, "POSIX");
        */


        printf("Reading files, generating HOG features and save them to file '%s':\n", featuresFile.c_str());
        float percent;
        /**
         * Save the calculated descriptor vectors to a file in a format that can be used by SVMlight for training
         * @NOTE: If you split these steps into separate steps:
         * 1. calculating features into memory (e.g. into a cv::Mat or vector< vector<float> >),
         * 2. saving features to file / directly inject from memory to machine learning algorithm,
         * the program may consume a considerable amount of main memory
         */
        fstream File;
        File.open(featuresFile.c_str(), ios::out);
        if (File.good() && File.is_open()) {
            File << "# Use this file to train, e.g. SVMlight by issuing $ svm_learn -i 1 -a weights.txt " << featuresFile.c_str() << endl; // Remove this line for libsvm which does not support comments
            // Iterate over sample images
            for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
                storeCursor();
                vector<float> featureVector;
                // Get positive or negative sample image file path
//                const string currentImageFile = (currentFile < positiveTrainingImages.size() ? positiveTrainingImages.at(currentFile) : negativeTrainingImages.at(currentFile - positiveTrainingImages.size()));
                const string currentImageFile = positiveTrainingImages.at(currentFile);
                // Output progress
                if ( (currentFile+1) % 10 == 0 || (currentFile+1) == overallSamples ) {
                    percent = ((currentFile+1) * 100 / overallSamples);
                    printf("%5lu (%3.0f%%):\tFile '%s'", (currentFile+1), percent, currentImageFile.c_str());
                    fflush(stdout);
                    resetCursor();
                }
                // Calculate feature vector from current image file
                calculateFeaturesFromInput(currentImageFile, featureVector, hog);
                if (!featureVector.empty()) {
                    /* Put positive or negative sample class to file,
                     * true=positive, false=negative,
                     * and convert positive class to +1 and negative class to -1 for SVMlight
                     */
                    File << ((currentFile < positiveTrainingImages.size()) ? "+1" : "-1");
                    // Save feature vector components
                    for (unsigned int feature = 0; feature < featureVector.size(); ++feature) {
                        File << " " << (feature + 1) << ":" << featureVector.at(feature);
                    }
                    File << endl;
                }
            }
            printf("\n");
            File.flush();
            File.close();
        } else {
            printf("Error opening file '%s'!\n", featuresFile.c_str());
            return EXIT_FAILURE;
        }

    return 0;
}
