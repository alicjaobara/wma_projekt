// Method 2 : SIFT
cout << "===========================\n";
cout << "Extracting SIFT Features of train set ...\n";

    // Check if we already have vocabulary written in file
Mat vocabulary;
if( !readVocabulary( "../vocabulary.dat", vocabulary) )
{
    SiftFeatureDetector detector;
    SiftDescriptorExtractor extractor;

    Mat all_descriptors;
    Mat descriptors;

    for (int i=0;i<train_set.size();i++){
            vector<cv::KeyPoint> keypoints;
            detector.detect(train_set.at(i), keypoints);
            extractor.compute(train_set.at(i), keypoints, descriptors);
            all_descriptors.push_back(descriptors);
            cout << "Extracting image #"<<i << "/" << train_set.size() << "\r" << std::flush;
    }
    cout <<endl << all_descriptors.size() << " features extracted for train set.\n";
// Now cluster SIFT features using KNN
    int vocabulary_size = 70;
// Clustering all SIFT descriptors
    TermCriteria terminate_criterion;
    terminate_criterion.epsilon = FLT_EPSILON;
    BOWKMeansTrainer bowTrainer( vocabulary_size, terminate_criterion, 3, KMEANS_PP_CENTERS );
    for (int i=0;i<all_descriptors.size().height;i++){
            Mat current_descriptor = all_descriptors.row(i);
    //cout << "Size of current_descriptor = " << current_descriptor.size() << endl;  getchar();
            bowTrainer.add(current_descriptor);
            cout << "Adding Feature #" << i << " to Bag-Of-Words K-Means Trainer ...  \r" << std::flush;
    }
    cout << "\nClustering... Please Wait ...\n";
    vocabulary = bowTrainer.cluster();

    cout << "\nSIFT Features Clustered in " << vocabulary.size() << " clusters." << endl;
    if( !writeVocabulary("../vocabulary.dat", vocabulary) )
    {
            cout << "Error: file " << "../vocabulary.dat" << " can not be opened to write" << endl;
            exit(-1);
    }
}else
cout << "Visual Vocabulary read from file successfully!\n";
// Building Histograms
cout << "===========================\n";

std::vector< DMatch > matches;
    // Matching centroids with training set
std::vector<DMatch> trainin_set_matches;

Ptr<FeatureDetector> featureDetector = FeatureDetector::create( "SIFT" );
Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create( "SIFT" );
Ptr<BOWImgDescriptorExtractor> bowExtractor;

Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create( "BruteForce" );

bowExtractor = new BOWImgDescriptorExtractor( descExtractor, descMatcher );

bowExtractor->setVocabulary(vocabulary);


Mat train_hist,test_hist;
cout << "Building Histograms for training set :\n";
if( !readVocabulary( "../train_hist.dat", train_hist) )
{
    for (int i=0;i<train_set.size();i++){
            vector<cv::KeyPoint> keypoints;
            // Each descriptor is histogram for the image
            Mat descriptors;
            featureDetector->detect( train_set.at(i), keypoints );
            bowExtractor->compute( train_set.at(i), keypoints, descriptors);
            train_hist.push_back(descriptors);
            cout << "Image #" << i <<" Bag-Of-Words descriptors extracted , size = " << descriptors.size() << "\r" << std::flush;
    }
    writeVocabulary("../train_hist.dat", train_hist);
}else{
cout << "Train Histograms read from file successfully!\n";
}

cout << "\nBuilding Histograms for test set :\n";
if( !readVocabulary( "../test_hist.dat", test_hist) )
{
    for (int i=0;i<test_set.size();i++){
            vector<cv::KeyPoint> keypoints;
            // Each descriptor is histogram for the image
            Mat descriptors;
            featureDetector->detect( test_set.at(i), keypoints );
            bowExtractor->compute( test_set.at(i), keypoints, descriptors);
            test_hist.push_back(descriptors);
            cout << "Image #" << i <<" Bag-Of-Words descriptors extracted , size = " << descriptors.size() << "\r" << std::flush;
    }
    writeVocabulary("../test_hist.dat", test_hist);
}else{
cout << "Test Histograms read from file successfully!\n";
}


cout << "\n===========================\n";
cout << "Classifying using SIFT and K-Nearest Neighbour:\n";

 misclassified_count = 0;
for (int i=0;i<test_set.size();i++){
    Mat current_test_hist = test_hist.row(i);
    int current_best_ind =-1;
    float current_best_distance = 100000;
    for (int j=0;j<train_set.size();j++){
            Mat current_train_hist = train_hist.row(j);
            float dist = norm(current_train_hist -current_test_hist);
            if (dist < current_best_distance ){
                    current_best_distance = dist;
                    current_best_ind = j;
            }
    }

    if ( !isequals(test_labels.at(i) , train_labels.at(current_best_ind) ) ){
            misclassified_count++;
    }
    cout << "Accuracy = " << RED << 100 * (1 - misclassified_count/(double)i )<< Color_Off<< "\r" << std::flush;

}
cout << endl;
