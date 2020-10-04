// WYOpenCv.cpp: 定义应用程序的入口点。
//

#include "WYOpenCv.h"


//级联分类器：用于训练和检测；
CascadeClassifier faceCascadeClassifier;

//简单的摄像头调用
void simpleShow();

//静态人脸检测，这种方式适用于检测静态的图片
void staticFaceCheck();

//动态人脸检测，这种方式适用于视频检测
void dynamicFaceCheck();

//计算LBP图谱
void calculateLBP(Mat src, Mat& dst);

//采集人脸训练正样本
void collectSamples(Mat frame, Rect face, int number);

//跟踪器测试
void trackerTest();

//测试
void test() {
	Mat ones = Mat::ones(4, 4, CV_8UC3);
	Mat gray;
	cvtColor(ones, gray, COLOR_BGR2GRAY);
	cout << "通道数：" << ones.channels() << endl;
	cout << ones.rows << "-" << ones.cols << endl;
	cout << ones << endl;
	cout << gray.rows << "-" << gray.cols << endl;
	cout << gray << endl;
	//Mat的rows/cols仅仅代表的是图像的高/宽，即行/列
	//因此遍历多通道的Mat可以这样做
	for (int h = 0; h < ones.rows; h++) {
		for (int w = 0; w < ones.cols; w++) {
			cout << (int)ones.at<Vec3b>(h, w)[0] << endl;
			cout << (int)ones.at<Vec3b>(h, w)[1] << endl;
			cout << (int)ones.at<Vec3b>(h, w)[2] << endl;
		}
	}

	Mat graySrc = Mat::ones(4, 4, CV_8UC1);
	Mat colorDst;
	cvtColor(graySrc, colorDst, COLOR_GRAY2BGR);
	cout << colorDst << endl;
}

int main()
{
	//simpleShow();       //显示摄像头图像
	//staticFaceCheck();  //静态检测
	dynamicFaceCheck(); //动态检测
	//trackerTest();
	//test();

	cout << "arrived here after tacker runing" << endl;

	return 0;
}

void simpleShow() {

#ifdef ENABLE_LBP

	VideoCapture capture(0);
	Mat mat;
	Mat gray;
	Mat lbp;
	while (1) {
		capture >> mat;
		//转成灰度图
		cvtColor(mat, gray, COLOR_BGR2GRAY);
		if (lbp.empty()) {
			lbp = Mat(gray.rows, gray.cols, CV_8UC1);
		}
		//lpb转换
		calculateLBP(gray, lbp);
		//显示
		imshow("img-color", mat);
		imshow("img-lbp", lbp);
		waitKey(30);
	}

#else

	VideoCapture capture(0);
	Mat mat;
	while (1) {
		capture >> mat;
		imshow("img", mat);
		waitKey(30);
	}

#endif // ENABLE_LBP

}

//静态人脸检测，这种方式适用于检测静态的图片，
//但是这里我们仍然选择使用摄像头动态捕捉的画面进行检测，
//你会发现卡顿严重
void staticFaceCheck() {
	//加载人脸模型
	const string path = "E:/OPENCV/opencv4.1.2/INSTALL/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml";
	if (!faceCascadeClassifier.load(path)) {
		cout << "人脸模型加载失败！" << endl;
		return;
	}

	//打开摄像头
	VideoCapture capture;
	capture.open(0);
	//opencv中用Mat矩阵表示图像，
	//frame用于保存原始图像，opencv彩色图像默认使用bgr格式
	Mat frame;
	//灰度图像
	Mat gray;
	while (1) {
		capture >> frame;
		if (frame.empty()) {
			cout << "图像采集失败！";
			continue;
		}
		//转成灰度图
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		//直方图均衡化，增强对比度
		equalizeHist(gray, gray);
		//检测向量
		vector<Rect> faces;
		//开始检查
		faceCascadeClassifier.detectMultiScale(gray, faces);
		//画人脸框
		//vs特有的for循环
		for each (Rect face in faces)
		{
			//在原始图上画一个红色的框
			rectangle(frame, face, Scalar(0, 0, 255));
		}
		//显示图像
		imshow("人脸检测", frame);
		//30ms刷新一次，如果27：ESC按下则break；
		if (waitKey(30) == 27) {
			break;
		}
	}
}

//动态人脸检测，这种方式适用于视频检测
//动态检测需要用到适配器
void dynamicFaceCheck() {
	//人脸模型路径
	//两个种方式都可以
	//const string path = "E:/OPENCV/opencv4.1.2/INSTALL/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml";
	const string path = "E:/VisualStudio/FILE/cascadeClassifier/model/face/cascade.xml";
	const char* pathP = path.c_str();
	//创建主适配器
	Ptr<CascadeClassifier> mainCascadeClassifier = makePtr<CascadeClassifier>(pathP);
	Ptr<CascadeDetectorAdapter> mainDetector = makePtr<CascadeDetectorAdapter>(mainCascadeClassifier);
	//创建追踪检测适配器
	Ptr<CascadeClassifier> trackerCascadeClassifier = makePtr<CascadeClassifier>(pathP);
	Ptr<CascadeDetectorAdapter> trackingDetector = makePtr<CascadeDetectorAdapter>(trackerCascadeClassifier);
	//创建追踪器
	Ptr<DetectionBasedTracker> detectorTracker;
	DetectionBasedTracker::Parameters detectorParams;
	detectorTracker = makePtr<DetectionBasedTracker>(mainDetector, trackingDetector, detectorParams);
	//启动追踪器
	detectorTracker->run(); 

	//打开摄像头
	VideoCapture capture;
	capture.open(0);
	//opencv中用Mat矩阵表示图像，
	//frame用于保存原始图像，opencv彩色图像默认使用bgr格式
	Mat frame;
	//灰度图像
	Mat gray;
	while (1) {
		capture >> frame;
		if (frame.empty()) {
			cout << "图像采集失败！";
			continue;
		}
		//转成灰度图
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		//直方图均衡化，增强对比度
		equalizeHist(gray, gray);
		//检测向量
		vector<Rect> faces;
		//人脸检测处理
		detectorTracker->process(gray);
		//获取检测结果
		detectorTracker->getObjects(faces);
		//画人脸框
		//vs特有的for循环
		for each (Rect face in faces)
		{
			//在原始图上画一个红色的框
			rectangle(frame, face, Scalar(0, 0, 255));

#ifdef ENABLE_SAMPLES
			collectSamples(frame,face,20);
#endif // ENABLE_SAMPLES

		}
		//显示图像
		imshow("人脸检测", frame);
		//30ms刷新一次，如果27：ESC按下则break；
		if (waitKey(30) == 27) {
			break;
		}
	}
}

//跟踪器测试
void trackerTest() {
	//人脸模型路径
//两个种方式都可以
	const string path = "E:/OPENCV/opencv4.1.2/INSTALL/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml";
	const char* pathP = path.c_str();
	//创建主适配器
	Ptr<CascadeClassifier> mainCascadeClassifier = makePtr<CascadeClassifier>(pathP);
	Ptr<CascadeDetectorAdapter> mainDetector = makePtr<CascadeDetectorAdapter>(mainCascadeClassifier);
	//创建追踪检测适配器
	Ptr<CascadeClassifier> trackerCascadeClassifier = makePtr<CascadeClassifier>(pathP);
	Ptr<CascadeDetectorAdapter> trackingDetector = makePtr<CascadeDetectorAdapter>(trackerCascadeClassifier);
	//创建追踪器
	Ptr<DetectionBasedTracker> detectorTracker;
	DetectionBasedTracker::Parameters detectorParams;
	detectorTracker = makePtr<DetectionBasedTracker>(mainDetector, trackingDetector, detectorParams);
	//启动追踪器
	detectorTracker->run();
	//detectorTracker->stop();
}

//计算LBP图谱
//Mat(高，宽)  /  Mat(y，x)
void calculateLBP(Mat src,Mat &dst) {
	//从左往右，从上往下遍历
	for (int y = 1; y < src.rows - 1; y++) {
		for (int x = 1; x < src.cols - 1; x++) {
			uchar lbp = 0;
			uchar center = src.at<uchar>(y, x);

			//顺时针遍历周围的点
			if (src.at<uchar>(y-1, x-1) >= center)lbp += 1 << 7; //左上
			if (src.at<uchar>(y-1,   x) >= center)lbp += 1 << 6; //正上
			if (src.at<uchar>(y-1, x+1) >= center)lbp += 1 << 5; //右上
			if (src.at<uchar>(y,   x+1) >= center)lbp += 1 << 4; //正右
			if (src.at<uchar>(y+1, x+1) >= center)lbp += 1 << 3; //右下
			if (src.at<uchar>(y+1,   x) >= center)lbp += 1 << 2; //正下
			if (src.at<uchar>(y+1, x-1) >= center)lbp += 1 << 1; //左下
			if (src.at<uchar>(y,   x-1) >= center)lbp += 1 << 0; //正左

			dst.at<uchar>(y, x) = lbp;
		}
	}
}

//采集人脸训练正样本
void collectSamples(Mat frame,Rect face,int number) {
	static int count = 0;
	if (count == number) {
		return;
	}
	count++;
	Mat sample;
	//从frame中把人脸face抠出来保存到sample中
	frame(face).copyTo(sample);
	//归一化大小
	resize(sample, sample, Size(24, 24));
	//转成灰度图
	cvtColor(sample, sample, COLOR_BGR2GRAY);
	//生成路径
	char p[100];
	sprintf(p, "E:/VisualStudio/FILE/samples/face/pos/face_%d.jpg", count);
	//保存样本图片
	imwrite(p, sample);
}
